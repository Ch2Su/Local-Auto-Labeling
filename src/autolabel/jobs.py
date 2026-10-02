"""웹 UI에서 띄우는 백그라운드 작업(자동 라벨링 / 학습) 관리.

학습·추론은 오래 걸리므로 별도 스레드에서 돌리고, 그 스레드의 표준 출력만
골라내어 UI 로그로 보여준다. 같은 시각에 도는 작업은 하나로 제한한다
(GPU 8GB에서 학습과 추론을 동시에 돌리면 OOM 나기 쉽다).
"""

from __future__ import annotations

import io
import re
import sys
import threading
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

_MAX_LOG_LINES = 400

# GPU 는 하나뿐이라 **프로젝트가 달라도** 학습·추론이 겹치면 안 된다. 넘치면 에러 없이
# 조용히 느려지기만 한다(CLAUDE.md 함정 9). JobManager 는 프로젝트마다 따로 있으므로
# 이 잠금만 프로세스 전역이다 (2026-09-21, 탭으로 프로젝트를 나누면서 추가).
_GPU_KINDS = frozenset({"predict", "train", "resume", "train_wsl", "resume_wsl"})
_gpu_lock = threading.Lock()
_gpu_owner: tuple[str, str] | None = None          # (프로젝트 이름, 작업 종류)


def _claim_gpu(label: str, kind: str) -> None:
    """GPU 를 쓰는 작업을 등록한다. 이미 다른 작업이 쓰고 있으면 RuntimeError."""
    global _gpu_owner
    with _gpu_lock:
        if _gpu_owner is not None:
            owner_label, owner_kind = _gpu_owner
            raise RuntimeError(
                f"'{owner_label}' 프로젝트에서 '{owner_kind}' 작업이 GPU 를 쓰는 중입니다. "
                "끝난 뒤 다시 시도하세요."
            )
        _gpu_owner = (label, kind)


def _release_gpu() -> None:
    global _gpu_owner
    with _gpu_lock:
        _gpu_owner = None


def gpu_busy() -> bool:
    """어느 프로젝트에서든 GPU 작업(학습·자동 라벨링)이 도는 중인가.

    검수 UI 의 '예측 비교' 는 이때 CPU 로 돈다 — 한 장쯤은 CPU 로도 된다.
    """
    return _gpu_owner is not None

# 한 줄 + 그 줄을 끝낸 개행문자. '\r' 로만 끝났으면 진행 표시줄의 중간 상태다.
_LINE = re.compile(r"([^\r\n]*)([\r\n]+)")
# ANSI escape (색·커서 이동·줄 지우기). 웹 로그에는 '[K' 같은 찌꺼기로 보인다.
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


class _ThreadTee:
    """작업 스레드의 출력을 작업 로그와 원래 스트림 **양쪽**으로 보낸다.

    양쪽에 보내는 이유: ultralytics 의 LOGGER 는 import 시점의 sys.stdout 을 핸들러에
    붙잡아 두는 반면, TQDM 은 호출할 때마다 sys.stdout 을 다시 본다. 그래서 한쪽으로만
    보내면 import 시점에 따라 콘솔과 웹 로그에 서로 다른 절반씩만 나온다
    (헤더는 콘솔에, epoch 진행줄은 웹에). 양쪽에 모두 흘려보내면 그 순서에 상관없이
    두 곳이 같은 내용을 본다.
    """

    def __init__(self, original):
        self._original = original
        self._sinks: dict[int, Callable[[str], None]] = {}
        self._lock = threading.Lock()

    def register(self, thread_id: int, sink: Callable[[str], None]) -> None:
        with self._lock:
            self._sinks[thread_id] = sink

    def unregister(self, thread_id: int) -> None:
        with self._lock:
            self._sinks.pop(thread_id, None)

    def write(self, text: str) -> int:
        sink = self._sinks.get(threading.get_ident())
        if sink is not None:
            sink(text)
        # 콘솔이 닫혀 있어도(창을 닫은 경우 등) 작업이 죽지는 않게 한다.
        try:
            return self._original.write(text)
        except Exception:
            return len(text)

    def flush(self) -> None:
        self._original.flush()

    def isatty(self) -> bool:
        return False

    def __getattr__(self, item):
        return getattr(self._original, item)


_tee_out: _ThreadTee | None = None
_tee_err: _ThreadTee | None = None


def _install_tees() -> tuple[_ThreadTee, _ThreadTee]:
    global _tee_out, _tee_err
    if _tee_out is None:
        _tee_out = _ThreadTee(sys.stdout)
        _tee_err = _ThreadTee(sys.stderr)
        sys.stdout = _tee_out  # type: ignore[assignment]
        sys.stderr = _tee_err  # type: ignore[assignment]
    return _tee_out, _tee_err  # type: ignore[return-value]


@dataclass
class Job:
    id: int
    kind: str
    status: str = "running"  # running | done | error
    started_at: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))
    finished_at: str | None = None
    progress: str = ""
    log: list[str] = field(default_factory=list)
    result: dict | None = None
    error: str | None = None
    cancel_requested: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "progress": self.progress,
            "log": self.log[-80:],
            "result": self.result,
            "error": self.error,
            "cancel_requested": self.cancel_requested,
        }


class JobManager:
    def __init__(self, label: str = "이 프로젝트") -> None:
        # 첫 작업이 아니라 지금 설치한다. 서버의 다른 경로(예: /api/resume-point)가
        # 먼저 ultralytics 를 import 하면, 그 LOGGER 가 '설치 전' 스트림을 붙잡아
        # 콘솔에만 출력이 남고 웹 로그에는 안 들어간다.
        _install_tees()
        self.label = label                      # GPU 를 누가 쓰는지 알릴 때 쓰는 이름
        self._jobs: dict[int, Job] = {}
        self._next_id = 1
        self._current: Job | None = None
        self._lock = threading.Lock()

    @property
    def busy(self) -> bool:
        return self._current is not None and self._current.status == "running"

    def current(self) -> Job | None:
        return self._current

    def get(self, job_id: int) -> Job | None:
        return self._jobs.get(job_id)

    def cancel(self, job_id: int) -> bool:
        """중단을 요청한다. 실제 중단은 작업이 이 플래그를 확인하는 시점에 일어난다.

        스레드를 강제로 죽이지 않는다. 학습은 배치 경계에서 빠져나와 최종 검증과
        가중치 저장까지 마치고 정상 종료된다.
        """
        job = self._jobs.get(job_id)
        if job is None or job.status != "running":
            return False
        job.cancel_requested = True
        job.log.append("[중단 요청] 진행 중인 단계를 마치는 대로 종료합니다...")
        return True

    def start(self, kind: str, fn: Callable[[Job], dict]) -> Job:
        with self._lock:
            if self.busy:
                raise RuntimeError(
                    f"이미 '{self._current.kind}' 작업이 실행 중입니다. 끝난 뒤 다시 시도하세요."
                )
            if kind in _GPU_KINDS:
                _claim_gpu(self.label, kind)     # 다른 프로젝트가 GPU 를 쓰면 여기서 막힌다
            job = Job(id=self._next_id, kind=kind)
            self._jobs[job.id] = job
            self._next_id += 1
            self._current = job

        out_tee, err_tee = _install_tees()
        buf = io.StringIO()

        # 직전에 쌓은 줄이 '\r' 로 끝난 진행 표시줄이면, 다음 줄이 그 자리를 덮어쓴다.
        transient = False

        def sink(text: str) -> None:
            """작업 스레드의 출력을 로그 줄로 쌓는다.

            tqdm 진행 표시줄은 '\\r' 로 같은 줄을 계속 덮어쓰며 갱신한다. 그걸 매번
            새 줄로 쌓으면 epoch 하나에 수십 줄이 생겨, 로그 상한(400줄)을 진행
            표시줄이 다 차지해 정작 필요한 메시지가 밀려난다. 화면에서도 같은 자리를
            맴도는 것처럼 보여 학습이 멈춘 것으로 오해하게 된다.

            그래서 '\\r' 로 끝난 조각은 '갱신'으로 보고 마지막 줄을 교체하고,
            '\\n' 으로 끝난 조각만 새 줄로 쌓는다. ('\\r\\n' 은 진짜 줄바꿈이다.)
            """
            nonlocal transient
            buf.write(text)
            value = buf.getvalue()
            idx = max(value.rfind("\n"), value.rfind("\r"))
            if idx < 0:
                return
            chunk, rest = value[: idx + 1], value[idx + 1 :]
            buf.seek(0)
            buf.truncate(0)
            buf.write(rest)

            for m in _LINE.finditer(chunk):
                line = _ANSI.sub("", m.group(1)).rstrip()
                if not line:
                    continue
                if transient and job.log:
                    job.log[-1] = line
                else:
                    job.log.append(line)
                transient = "\n" not in m.group(2)
                job.progress = line[:200]
            del job.log[:-_MAX_LOG_LINES]

        def runner() -> None:
            tid = threading.get_ident()
            out_tee.register(tid, sink)
            err_tee.register(tid, sink)
            try:
                job.result = fn(job)
                job.status = "done"
            except Exception as exc:
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                job.log.extend(traceback.format_exc().splitlines()[-15:])
            finally:
                out_tee.unregister(tid)
                err_tee.unregister(tid)
                job.finished_at = datetime.now().strftime("%H:%M:%S")
                if kind in _GPU_KINDS:
                    _release_gpu()

        threading.Thread(target=runner, name=f"job-{job.id}-{kind}", daemon=True).start()
        return job
