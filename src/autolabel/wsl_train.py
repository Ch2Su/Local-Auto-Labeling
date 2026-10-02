"""웹 UI 'WSL 학습' 버튼의 뒷단. WSL 에서 학습을 돌리고 결과를 가져온다.

흐름

1. DB 로 데이터셋 폴더(config 의 data.dataset)를 만든다 (Windows, `trainer.build_dataset` 그대로)
2. WSL 홈으로 복사한다 (`wsl/sync_dataset.sh` — 이미지·라벨만, `.npy`·DB 는 안 간다)
3. `wsl` 로 `wsl/train_wsl.py` 를 띄우고 출력을 작업 로그로 중계한다
4. 끝나면 best.pt 를 운영 가중치로 올린다 (`trainer.promote_weights` — 이전 것은
   `best.prev.pt`. Windows 학습 버튼과 같은 규칙이다)

왜 WSL 인가 (2026-09-12 실측, yolo26s / 640 / batch 20 / train 4,792장, 양쪽 모두
Ultralytics 8.4.142·같은 인자): epoch 258초 → 62초 (중앙값). 288 epoch 을 5.0시간에
끝냈다(Windows 로는 약 20시간). Linux 라 `cache: ram` 을 쓸 수
있고(Windows 는 spawn 이라 불가, CLAUDE.md 함정 2) 워커가 fork 로 뜨기 때문이다.

**중단.** 웹 UI 의 [중단] 은 `job.cancel_requested` 만 세운다. 여기서 그걸 보고 WSL
쪽 STOP 파일을 만들면, train_wsl.py 가 다음 배치에서 멈추고 resume.pt 를 남긴다.
학습 전 단계(데이터셋 빌드·복사)에서 누르면 그냥 거기서 그만둔다.

**서버를 죽이면 WSL 학습은 계속 돈다.** 별개 프로세스라서다. 그때는 STOP 파일을
직접 만들어 멈출 것 (`\\\\wsl.localhost\\Ubuntu\\...\\runs_wsl\\STOP`).
"""

from __future__ import annotations

import codecs
import csv
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .store import Store
from .trainer import (
    ResumePoint, build_dataset, promote_weights, resume_point_in, train_args, uses_dataset,
)

REPO = Path(__file__).resolve().parents[2]          # 저장소 루트
SCRIPT_TRAIN = REPO / "wsl" / "train_wsl.py"
SCRIPT_SYNC = REPO / "wsl" / "sync_dataset.sh"
WSL_EXE = shutil.which("wsl") or r"C:\Windows\System32\wsl.exe"

# wsl.exe 가 직접 내는 메시지(배포판 없음 등)는 기본이 UTF-16 이라 그대로 읽으면 깨진다.
_ENV = {**os.environ, "WSL_UTF8": "1"}
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def to_linux(p: Path) -> str:
    """Windows 경로를 WSL 이 보는 경로로. `E:\\a\\b` -> `/mnt/e/a/b`."""
    p = Path(p).resolve()
    drive = p.drive.rstrip(":").lower()
    return f"/mnt/{drive}{p.as_posix().split(':', 1)[1]}"


# --------------------------------------------------------------------- 설정

_home_cache: dict[str, str] = {}


def _wsl_home(distro: str) -> str:
    if distro not in _home_cache:
        r = subprocess.run([WSL_EXE, "-d", distro, "--", "sh", "-c", "echo $HOME"],
                           capture_output=True, timeout=60, env=_ENV,
                           creationflags=_NO_WINDOW)
        home = r.stdout.decode("utf-8", "replace").strip()
        if r.returncode != 0 or not home.startswith("/"):
            raise RuntimeError(f"WSL({distro}) 의 홈 경로를 알 수 없습니다: "
                               f"{r.stderr.decode('utf-8', 'replace').strip()}")
        _home_cache[distro] = home
    return _home_cache[distro]


@dataclass
class WslSettings:
    distro: str
    python: str        # 전부 WSL 안의 절대경로
    project: str
    dataset: str
    name: str
    workers: int
    cache: str

    @classmethod
    def from_cfg(cls, cfg: Config) -> "WslSettings":
        distro = str(cfg.get("wsl.distro", "Ubuntu"))

        def path(key: str, default: str) -> str:
            v = str(cfg.get(f"wsl.{key}", default))
            if v.startswith("~"):
                v = _wsl_home(distro) + v[1:]
            return v.rstrip("/")

        return cls(
            distro=distro,
            python=path("python", "~/miniconda3/envs/yolo_final/bin/python"),
            project=path("project", "~/new_env/yolo/runs_wsl"),
            dataset=path("dataset", "~/datasets/visible"),
            name=str(cfg.get("wsl.name", "visible")),
            workers=int(cfg.get("wsl.workers", 8)),
            cache=str(cfg.get("wsl.cache", "ram")),
        )

    def unc(self, linux_path: str) -> Path:
        """WSL 안의 경로를 Windows 에서 여는 경로로 (`\\\\wsl.localhost\\<배포판>\\...`)."""
        return Path(rf"\\wsl.localhost\{self.distro}" + linux_path.replace("/", "\\"))

    @property
    def stop_file(self) -> Path:
        return self.unc(f"{self.project}/STOP")

    def cmd(self, *args: str) -> list[str]:
        return [WSL_EXE, "-d", self.distro, "--", *args]


# ------------------------------------------------------------ 사용 가능 여부

_check_cache: dict[str, tuple[float, tuple[bool, str]]] = {}
_CHECK_TTL = 60.0


def check(cfg: Config) -> tuple[bool, str]:
    """WSL 학습을 쓸 수 있는가. 서브프로세스를 띄우므로 60초 동안 결과를 재사용한다."""
    if not bool(cfg.get("wsl.enabled", True)):
        return False, "config.yaml 의 wsl.enabled 가 꺼져 있습니다."
    key = str(cfg.path)
    hit = _check_cache.get(key)
    if hit and time.time() - hit[0] < _CHECK_TTL:
        return hit[1]
    result = _check_uncached(cfg)
    _check_cache[key] = (time.time(), result)
    return result


def _check_uncached(cfg: Config) -> tuple[bool, str]:
    if not Path(WSL_EXE).exists():
        return False, "wsl.exe 를 찾을 수 없습니다."
    try:
        s = WslSettings.from_cfg(cfg)
        r = subprocess.run(s.cmd("test", "-x", s.python), capture_output=True,
                           timeout=60, env=_ENV, creationflags=_NO_WINDOW)
    except Exception as exc:
        return False, f"WSL 을 확인하지 못했습니다: {exc}"
    if r.returncode != 0:
        return False, f"WSL 에서 파이썬을 찾을 수 없습니다: {s.python}"
    return True, ""


# ---------------------------------------------------------------- 이어하기

_resume_cache: dict[str, tuple[tuple, ResumePoint | None]] = {}


def find_resume(cfg: Config) -> ResumePoint | None:
    """WSL 쪽에서 이어 학습할 체크포인트를 찾는다 (가장 최근 폴더부터).

    체크포인트를 여는 건 비싸다 — 60MB 파일을 `\\\\wsl.localhost` 로 읽는다. 웹 UI 는
    박스를 저장할 때마다 상태를 새로 고치므로, 파일의 (크기, 수정 시각) 이 그대로면
    직전 결과를 재사용한다.
    """
    s = WslSettings.from_cfg(cfg)
    proj = s.unc(s.project)
    try:
        dirs = sorted((d for d in proj.iterdir() if (d / "weights").is_dir()),
                      key=lambda d: d.stat().st_mtime, reverse=True)
    except OSError:
        return None

    sig = []
    for d in dirs:
        for n in ("resume.pt", "last.pt", "results.csv"):
            f = (d / "weights" / n) if n.endswith(".pt") else (d / n)
            try:
                st = f.stat()
                sig.append((str(f), st.st_size, st.st_mtime))
            except OSError:
                pass
    # runs_wsl/ 는 프로젝트가 여럿이면 같이 쓴다. 이 설정의 데이터셋으로
    # 돈 학습만 고른다 — 아니면 끝날 때 다른 프로젝트 결과가 이쪽 운영 가중치를 덮는다.
    want = f"{s.dataset}/data.yaml"
    key = f"{proj}|{want}"
    sig_t = tuple(sig)
    hit = _resume_cache.get(key)
    if hit and hit[0] == sig_t:
        return hit[1]

    point = None
    for d in dirs:
        p = resume_point_in(d)
        if p is not None and uses_dataset(p, want):
            point = p
            break
    _resume_cache[key] = (sig_t, point)
    return point


# --------------------------------------------------------------- 출력 중계

_RESULT = re.compile(r"^RESULT_DIR=(.+?)\s*$", re.M)
_STOPPED = re.compile(r"^STOPPED=(\d)\s*$", re.M)


def _stream(cmd: list[str], job, on_cancel=None) -> tuple[int, str, bool]:
    """명령을 띄워 출력을 그대로 흘려보내고 (종료 코드, 끝부분 출력, 중단 여부) 를 돌려준다.

    이 함수는 작업 스레드에서 돈다. 여기서 sys.stdout 에 쓰면 jobs.py 의 tee 가
    작업 로그로 옮겨 준다 — 진행 표시줄의 '\\r' 갱신 처리도 거기서 한다.
    읽기는 별도 스레드로 해서, 출력이 뜸한 동안에도 중단 요청을 0.5초 안에 본다.
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env=_ENV, creationflags=_NO_WINDOW)
    q: queue.Queue = queue.Queue()

    def reader() -> None:
        while True:
            chunk = proc.stdout.read1(8192)
            if not chunk:
                break
            q.put(chunk)
        q.put(None)

    threading.Thread(target=reader, daemon=True).start()
    # 한글이 청크 경계에서 잘려도 깨지지 않게 증분 디코더를 쓴다.
    dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
    tail = ""
    cancelled = False
    while True:
        try:
            chunk = q.get(timeout=0.5)
        except queue.Empty:
            chunk = b""
        if chunk is None:
            break
        if chunk:
            text = dec.decode(chunk)
            sys.stdout.write(text)
            sys.stdout.flush()
            tail = (tail + text)[-20000:]
        if job is not None and job.cancel_requested and not cancelled:
            cancelled = True
            if on_cancel is not None:
                on_cancel()
            else:
                proc.terminate()
    rest = dec.decode(b"", final=True)
    if rest:
        sys.stdout.write(rest)
        tail = (tail + rest)[-20000:]
    return proc.wait(), tail, cancelled


# ---------------------------------------------------------------------- 학습

# train_args() 중 WSL 쪽이 따로 정하는 것. 나머지는 전부 --hyp 로 넘긴다.
#   data·project·name : WSL 안의 경로 (config 의 wsl: 설정)
#   workers·cache     : Linux 는 fork 라 워커가 싸고 cache: ram 을 쓸 수 있다 (wsl: 설정)
#   device            : WSL 은 GPU 0.   exist_ok·plots : train_wsl.py 가 고정한다
_WSL_OWN = frozenset({"data", "project", "name", "workers", "cache", "device", "exist_ok", "plots"})
# wsl.exe 는 명령줄을 bash 로 넘긴다 — `$HOME` 이 풀리고, 공백 없는 `a;b` 는 명령 두 개가
# 된다(2026-09-11 확인). 그래서 영숫자와 `_ . + - =` 만 쓰는 조각만 넘긴다.
_SAFE_TOKEN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=[A-Za-z0-9_.+\-]*$")


def hyp_tokens(cfg: Config) -> list[str]:
    """Windows 학습 버튼의 인자(`trainer.train_args`)를 `KEY=VALUE` 조각으로 바꾼다.

    train_wsl.py 가 `--bare --hyp ...` 로 받는다. --bare 는 스크립트 자체 기본값(MuSGD
    고정 등)을 끄라는 뜻이라, 두 버튼이 **같은 인자**로 학습한다. config 에 적지 않은 키는
    양쪽 모두 Ultralytics 기본값이다(예: optimizer=auto).
    값은 JSON 으로 적는다(true·1e-05 …). 받는 쪽이 JSON 으로 읽고 안 되면 문자열로 둔다.
    """
    out = []
    for k, v in train_args(cfg, "-").items():
        if k in _WSL_OWN:
            continue
        tok = f"{k}={v if isinstance(v, str) else json.dumps(v)}"
        if not _SAFE_TOKEN.match(tok):
            raise RuntimeError(
                f"config 의 train.overrides.{k} = {v!r} 는 WSL 로 넘길 수 없습니다. "
                "숫자·true/false·영문 단어만 됩니다 (wsl.exe 가 명령줄을 bash 로 넘겨서)."
            )
        out.append(tok)
    return out


def train_cmd(cfg: Config, s: WslSettings) -> list[str]:
    """새 학습을 시작하는 train_wsl.py 명령 (WSL 안에서 돌 인자 목록)."""
    # 사전학습 가중치는 Windows 쪽 weights/ 의 것을 /mnt/e 경로로 넘긴다. 이름만
    # 넘기면 WSL 의 Ultralytics 가 실행 폴더에서 찾다가 없으면 새로 받는다.
    base = Path(cfg.base_weights)
    model_arg = to_linux(base) if base.exists() else base.name
    return [s.python, to_linux(SCRIPT_TRAIN),
            "--project", s.project, "--name", s.name,
            "--data", f"{s.dataset}/data.yaml",
            "--model", model_arg,
            "--workers", str(s.workers), "--cache", s.cache,
            # epochs·batch·증강·손실 게인 등은 Windows 학습 버튼과 같은 인자로
            "--bare", "--hyp", *hyp_tokens(cfg)]


def _read_metrics(run_dir: Path) -> dict:
    """results.csv 에서 best epoch 과 마지막 epoch 을 뽑는다.

    best 는 Ultralytics 가 best.pt 를 고르는 기준과 같아야 한다. 8.4.142 는 **mAP50-95
    하나**다(utils/metrics.py `Metric.fitness`, w = [0, 0, 0, 1]). 옛 버전의
    0.1×mAP50 + 0.9×mAP50-95 로 고르면 2026-09-12 학습에서 epoch 172 를 가리켰는데,
    best.pt 는 epoch 168 이었다.
    """
    f = run_dir / "results.csv"
    try:
        rows = list(csv.DictReader(open(f, encoding="utf-8")))
    except OSError:
        return {}
    if not rows:
        return {}
    num = lambda r, k: float(r.get(k, "nan") or "nan")
    fit = [num(r, "metrics/mAP50-95(B)") for r in rows]
    b = max(range(len(rows)), key=lambda i: fit[i])
    best, last = rows[b], rows[-1]
    return {
        "best_epoch": int(float(best["epoch"])),
        "metrics/mAP50(B)": num(best, "metrics/mAP50(B)"),
        "metrics/mAP50-95(B)": num(best, "metrics/mAP50-95(B)"),
        "metrics/precision(B)": num(best, "metrics/precision(B)"),
        "metrics/recall(B)": num(best, "metrics/recall(B)"),
        "last_epoch": int(float(last["epoch"])),
    }


def train(cfg: Config, store: Store, job, *, resume: bool = False) -> dict:
    """WSL 에서 학습한다. job 은 jobs.Job — 중단 요청을 보려고 받는다."""
    ok, why = check(cfg)
    if not ok:
        raise RuntimeError(why)
    s = WslSettings.from_cfg(cfg)

    if resume:
        point = find_resume(cfg)
        if point is None:
            raise RuntimeError(
                f"WSL 쪽에 이어 학습할 체크포인트가 없습니다 ({s.project}).\n"
                "  정상 종료된 학습의 last.pt 는 옵티마이저가 지워져 이어갈 수 없습니다.")
        print(point.summary().replace(str(s.unc(s.project)), s.project), flush=True)
        args = [s.python, to_linux(SCRIPT_TRAIN), "--project", s.project,
                "--name", point.run_dir.name, "--resume", "--workers", str(s.workers)]
    else:
        args = train_cmd(cfg, s)     # 넘길 수 없는 값이면 빌드·복사 전에 멈춘다
        print(f"1/3 데이터셋 만드는 중 (DB -> {cfg.dataset}) ...", flush=True)
        report = build_dataset(cfg, store)
        print(report.summary(), flush=True)
        if job is not None and job.cancel_requested:
            return {"stopped_early": True, "note": "학습 시작 전에 중단했습니다.", "where": "wsl"}

        print(f"\n2/3 WSL 로 복사 중 ({s.dataset}) ...", flush=True)
        rc, _, cancelled = _stream(
            s.cmd("bash", to_linux(SCRIPT_SYNC), to_linux(cfg.dataset), s.dataset), job)
        if cancelled:
            return {"stopped_early": True, "note": "복사 중에 중단했습니다.", "where": "wsl"}
        if rc != 0:
            raise RuntimeError(f"WSL 로 복사하지 못했습니다 (종료 코드 {rc}).")

    def request_stop() -> None:
        print(f"\n[중단] WSL 쪽에 {s.stop_file} 를 만듭니다. "
              "다음 배치에서 멈추고 검증·저장합니다.", flush=True)
        s.stop_file.parent.mkdir(parents=True, exist_ok=True)
        s.stop_file.write_text("stop", encoding="utf-8")

    print(f"\n3/3 WSL 에서 학습 ({s.distro}) ...", flush=True)
    rc, tail, _ = _stream(s.cmd(*args), job, on_cancel=request_stop)

    m = _RESULT.findall(tail)
    if rc != 0 or not m:
        raise RuntimeError(f"WSL 학습이 실패했습니다 (종료 코드 {rc}). 위 로그를 확인하세요.")
    run_linux = m[-1]
    stopped = bool(int(_STOPPED.findall(tail)[-1])) if _STOPPED.findall(tail) else False
    run_dir = s.unc(run_linux)

    prev = promote_weights(run_dir / "weights" / "best.pt", cfg)
    snap = run_dir / "weights" / "resume.pt"
    _resume_cache.clear()        # 방금 체크포인트가 바뀌었다
    return {
        "where": "wsl",
        "run_dir": run_linux,
        "weights": str(cfg.weights),
        "previous_weights": str(prev) if prev else None,
        "stopped_early": stopped,
        "resume_snapshot": f"{run_linux}/weights/resume.pt" if stopped and snap.exists() else None,
        "metrics": _read_metrics(run_dir),
    }
