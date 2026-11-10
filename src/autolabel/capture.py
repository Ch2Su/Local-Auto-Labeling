"""영상 캡처 탭 (2026-11-10, 사용자 요청) — 영상에서 몇 초마다 한 장 뽑아, 사람이 고른 것만 `unlabeled/<카메라>/` 에 넣는다.

흐름 (검수 UI 상단 '영상 캡처' 보기, `web/capture.js`):
1. 브라우저에서 고른 영상을 그대로 올린다(POST 본문 = 파일) → `<DB 폴더>/capture/<세션>/video.<확장자>` 사본.
2. 몇 초마다 한 장 뽑아 같은 세션 폴더에 원본 해상도 JPEG(품질 95)과 썸네일을 쓴다 — 뽑는 동안 화면에 차례로 뜬다.
3. 화면에서 넣을 것·뺄 것을 고르고 카메라(cameras.yaml)와 촬영 시작 시각을 정해 '넣기' → 고른 프레임을
   `unlabeled/<카메라>/<날짜>/<카메라>_<YYYYMMDD_HHMMSS>_<밀리초>.jpg` 로 **옮기고**(다시 인코딩하지 않는다) 스캔한다.
   그다음은 보통 수집분과 같다(자동 라벨링 → 검수). 넣거나 닫으면 세션 폴더(영상 사본·안 고른 프레임)를 지운다.

왜 이렇게 했나:
- **영상은 서버(OpenCV)가 읽는다.** 브라우저는 고른 파일의 경로를 알려 주지 않고, 브라우저 안에서 풀면 브라우저가 아는
  코덱만 된다(AVI·일부 MKV·HEVC 는 안 될 수 있다). 그래서 사본을 한 번 만들고(E: 에 1TB 넘게 남아 있다, 11-10)
  `tools/extract_frames.py` 와 같은 방식 — 처음부터 차례로 grab, 시각은 `CAP_PROP_POS_MSEC`, 0·N·2N초 격자 — 으로 뽑는다.
  사본 이름은 ASCII 라, 한글 경로를 OpenCV 가 못 여는 문제도 없다. 1080p H.264 는 이 PC 의 CPU 로 초당 약 640프레임을
  푼다(VIRAT_S_000001, 11-10) — 1시간짜리 30fps 영상이면 약 3분.
- **이름·변화량 계산은 이 모듈 한 곳에 둔다**(`frame_name`·`parse_start`·`changed_pct` — extract_frames·import_roadplus 도
  여기서 가져간다). 이름의 시각을 scan 이 `captured_at` 으로 읽는다(provenance.py). **시작 시각을 모르면 넣지 않는다** —
  영상 이름에 있으면 그것, 없으면 사람이 적는다. 파일 수정 시각 − 길이는 화면이 버튼으로 제안만 한다(OBS 녹화처럼 끝날 때
  닫힌 파일에서만 맞고, 복사한 파일이면 틀린다 — provenance.py 머리말).
- **작업 큐(jobs.JobManager)를 쓰지 않는다.** 그 큐는 프로젝트에 작업 하나만 돌려서, 몇 시간 걸리는 학습 중에 캡처를 못
  하게 된다. 뽑기는 CPU 만 쓰는 따로 된 스레드이고, 프로젝트 안에서 한 번에 한 영상씩 뽑는다(나머지는 기다린다).
- **넣기의 스캔은 '폴더 스캔' 작업과 같은 잠금으로 겹치지 않게 한다**(서버가 `ingest` 로 넘긴다). 정리·휴지통 비우기가
  도는 동안은 넣지 않는다(파일을 옮기는 중이다 — 서버가 `blocker` 로 넘긴다).
- 같은 이름의 파일이 이미 있으면 덮지 않고 건너뛴다 — 같은 영상을 두 번 넣어도 그대로다(extract_frames 와 같다).
- 프레임마다 직전 프레임과 바뀐 픽셀 비율(extract_frames `--min-change` 와 같은 값)을 남기고, 앞의 프레임과 **픽셀까지
  같은** 프레임(멈춘 화면 — 웹 CCTV 에서 흔하다)은 처음부터 '빼기' 로 둔다. 넣어도 scan 의 중복 검사가 막는다.
- 서버를 다시 켜도 세션은 `session.json` 으로 이어진다(뽑던 것은 '멈춤', 올리다 끊긴 것은 지운다).
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import secrets
import shutil
import threading
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

from .provenance import _from_name, load_cameras, tz_text

SID_RE = re.compile(r"^\d{8}_\d{6}_[0-9a-f]{4}$")
JPEG_QUALITY = 95            # extract_frames 의 기본값과 같다
THUMB_W = 384                # 썸네일 폭 (화면 칸은 140~380px)
THUMB_QUALITY = 80
MIN_EVERY, MAX_EVERY = 0.2, 3600.0
MAX_FRAMES = 5000            # 한 번에 뽑는 최대 장수 — 간격을 너무 좁게 잡은 실수로 디스크를 채우지 않게
MIN_FREE = 2 * 1024 ** 3     # 올린 뒤에도 이만큼은 남아야 받는다
_SAVE_EVERY_S = 2.0          # 뽑는 동안 session.json 을 남기는 간격
RUNNING = ("queued", "extracting")

_START_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f",
                  "%Y%m%d_%H%M%S")


# ───────────────────────────── 이름·시각·변화량 (extract_frames·import_roadplus 와 함께 쓴다) ─────────────────────────────

def frame_name(camera: str, t: datetime) -> str:
    """`<카메라>_<YYYYMMDD_HHMMSS>_<밀리초>.jpg` — scan 이 이 시각을 `captured_at` 으로 읽는다(provenance._COMPACT)."""
    return f"{camera}_{t:%Y%m%d_%H%M%S}_{t.microsecond // 1000:03d}.jpg"


def parse_start(text: str) -> datetime:
    """영상 첫 프레임의 현지 시각. 못 읽으면 ValueError."""
    s = str(text or "").strip()
    for fmt in _START_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"시작 시각 '{s}' 를 읽지 못했습니다 — 예: 2026-11-03 14:00:00")


def fmt_start(t: datetime) -> str:
    """parse_start 가 다시 읽는 꼴 (밀리초가 있으면 붙인다)."""
    return f"{t:%Y-%m-%d %H:%M:%S}" + (f".{t.microsecond // 1000:03d}" if t.microsecond else "")


def small_gray(frame):
    """비교용 축소 흑백 (가로 320px)."""
    import cv2

    h, w = frame.shape[:2]
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (320, max(1, round(h * 320 / w))), interpolation=cv2.INTER_AREA)


def changed_pct(a, b) -> float:
    """두 축소 흑백에서 밝기가 20 넘게 바뀐 픽셀 비율(%)."""
    import numpy as np

    return float((np.abs(a.astype(np.int16) - b.astype(np.int16)) > 20).mean() * 100)


# ───────────────────────────── 세션 ─────────────────────────────

class CaptureError(Exception):
    """화면에 그대로 보여 줄 오류 — status 는 HTTP 상태 코드."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class Session:
    """불러온 영상 하나. 상태: uploading → ready → (queued → extracting → done | stopped | error) …

    frames 의 항목은 {i: 번호, t: 영상 안 시각(초), c: 직전 프레임과 바뀐 픽셀 %, s: 픽셀까지 같은 앞 프레임 번호, k: 넣기}.
    """

    def __init__(self, root: Path, sid: str):
        self.sid = sid
        self.dir = root / sid
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.deleted = False
        self.created = datetime.now().isoformat(timespec="seconds")
        self.video: dict = {}
        self.state = "uploading"
        self.error: str | None = None
        self.note: str | None = None
        self.settings: dict | None = None
        self.frames: list[dict] = []
        self.run = 0
        self.pos = 0.0
        self.camera: str | None = None
        self.start: str | None = None
        self.start_guess: str | None = None
        self.start_mtime: str | None = None
        self._saved_at = 0.0

    # --- 파일 ---
    @property
    def video_path(self) -> Path:
        return self.dir / self.video.get("file", "video.bin")

    def frame_path(self, i: int, thumb: bool = False) -> Path:
        return self.dir / ("thumbs" if thumb else "frames") / f"{i:05d}.jpg"

    _FIELDS = ("created", "video", "state", "error", "note", "settings", "frames", "run", "pos", "camera", "start",
               "start_guess", "start_mtime")

    def save(self) -> None:
        """session.json 을 바꿔 끼운다. 다시 켤 때 이어 보기 위한 것이라 실패해도 작업은 계속한다."""
        with self.lock:
            if self.deleted:
                return
            data = {k: getattr(self, k) for k in self._FIELDS}
            tmp = self.dir / f"session.{secrets.token_hex(3)}.tmp"
            try:
                tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                for attempt in range(5):
                    try:
                        os.replace(tmp, self.dir / "session.json")
                        break
                    except PermissionError:      # Windows: 다른 프로세스(백신 등)가 잠깐 열어 둔 때
                        if attempt == 4:
                            raise
                        time.sleep(0.05)
            except OSError as exc:
                tmp.unlink(missing_ok=True)
                print(f"[캡처] {self.sid}: session.json 을 쓰지 못했습니다 — {exc}")
            self._saved_at = time.monotonic()

    @classmethod
    def load(cls, root: Path, sid: str) -> "Session":
        s = cls(root, sid)
        data = json.loads((s.dir / "session.json").read_text(encoding="utf-8"))
        for k in cls._FIELDS:
            if k in data:
                setattr(s, k, data[k])
        return s

    # --- 화면에 보내는 꼴 ---
    def summary(self) -> dict:
        with self.lock:
            v = self.video
            return {
                "sid": self.sid, "name": v.get("name"), "size": v.get("size"), "state": self.state,
                "created": self.created, "duration": v.get("duration"), "width": v.get("width"),
                "height": v.get("height"), "n": len(self.frames), "keep": sum(1 for f in self.frames if f["k"]),
                "camera": self.camera,
            }

    def detail(self, since: int = 0) -> dict:
        with self.lock:
            out = self.summary()
            since = max(0, int(since))
            out.update(video=dict(self.video), settings=self.settings, run=self.run, pos=self.pos, error=self.error,
                       note=self.note, start=self.start, start_guess=self.start_guess, start_mtime=self.start_mtime,
                       since=since, frames=[dict(f) for f in self.frames[since:]])
            return out


class CaptureManager:
    """한 프로젝트의 캡처 세션들 (`<DB 폴더>/capture/`). 서버가 프로젝트마다 하나 만든다.

    ingest: 스캔(잠금을 쥐고 `ingest.ingest` 를 부른다), blocker: 넣으면 안 되는 이유(정리·휴지통 비우기 중) 또는 None.
    """

    def __init__(self, cfg, *, ingest: Callable[[], object], blocker: Callable[[], str | None] | None = None,
                 known_paths: Callable[[], set[str]] | None = None):
        self.cfg = cfg
        self.root = cfg.db.parent / "capture"
        self.ingest = ingest
        self.blocker = blocker or (lambda: None)
        self.known_paths = known_paths
        self.sessions: dict[str, Session] = {}
        self._lock = threading.Lock()            # sessions 사전
        self._extract_lock = threading.Lock()    # 한 번에 한 영상만 뽑는다
        self._commit_lock = threading.Lock()     # 넣기는 한 번에 하나
        self._load()

    # ----------------------------------------------------------------- 불러오기·목록

    def _load(self) -> None:
        if not self.root.exists():
            return
        for d in sorted(self.root.iterdir()):
            if not d.is_dir() or not SID_RE.match(d.name):
                continue
            try:
                s = Session.load(self.root, d.name)
            except (OSError, ValueError) as exc:
                # session.json 이 없다 = 올리다 끊겼거나 쓰다 멈췄다. 이 폴더는 이 모듈이 만든 임시 사본뿐이다.
                print(f"[캡처] 읽지 못한 세션 폴더를 지웁니다: {d.name} ({type(exc).__name__})")
                shutil.rmtree(d, ignore_errors=True)
                continue
            if s.state == "uploading":
                print(f"[캡처] 올리다 끊긴 영상을 지웁니다: {s.video.get('name') or d.name}")
                shutil.rmtree(d, ignore_errors=True)
                continue
            if s.state in RUNNING:
                s.state = "stopped"
                s.note = "서버가 다시 켜져 뽑기가 멈췄습니다 — 이어 뽑으려면 다시 캡처하세요"
                s.save()
            elif s.state == "committing":
                s.state = "stopped"
                s.note = ("넣는 도중 서버가 꺼졌습니다 — 이미 옮긴 프레임은 unlabeled 에 있으니 '폴더 스캔' 으로 등록하고, "
                          "남은 프레임만 다시 넣으세요(같은 이름은 건너뜁니다)")
                s.save()
            self.sessions[s.sid] = s

    def list(self) -> list[dict]:
        with self._lock:
            items = list(self.sessions.values())
        return [s.summary() for s in sorted(items, key=lambda s: s.created, reverse=True)]

    def get(self, sid: str) -> Session:
        if not SID_RE.match(str(sid)):
            raise CaptureError(404, "그런 영상이 없습니다.")
        with self._lock:
            s = self.sessions.get(sid)
        if s is None:
            raise CaptureError(404, "그런 영상이 없습니다 — 이미 넣었거나 닫았을 수 있습니다.")
        return s

    @property
    def commit_lock(self) -> threading.Lock:
        """넣기와 겹치면 안 되는 일(카메라 등록 지우기 — server)이 쥔다. 넣기는 이것을 쥔 채 ingest 잠금을 쥔다 — 같은 순서로."""
        return self._commit_lock

    def forget_cameras(self, keys) -> None:
        """등록을 지운 카메라를 골라 둔 세션은 카메라를 비운다(넣을 때 다시 고르게)."""
        gone = set(keys)
        with self._lock:
            items = list(self.sessions.values())
        for s in items:
            with s.lock:
                hit = s.camera in gone
                if hit:
                    s.camera = None
            if hit:
                s.save()

    def cameras(self) -> tuple[list[dict], str | None]:
        try:
            cams = load_cameras(self.cfg.cameras)
        except ValueError as exc:
            return [], str(exc)
        return [{"key": c.key, "kind": c.kind, "place": c.place, "mount": c.mount, "split": c.split, "tz": tz_text(c.tz)}
                for c in sorted(cams.values(), key=lambda c: c.key)], None

    # ----------------------------------------------------------------- 올리기

    def begin_upload(self, name: str, size: int | None, mtime_ms: int | None) -> tuple[Session, Path]:
        name = Path(str(name or "").replace("\\", "/")).name.strip() or "video"
        self.root.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(self.root).free
        if size is not None and size < 0:
            raise CaptureError(400, "크기가 잘못됐습니다.")
        if size is not None and free - size < MIN_FREE:
            raise CaptureError(507, f"디스크에 남은 공간이 모자랍니다 — 남은 {free / 1e9:.1f}GB, 영상 {size / 1e9:.1f}GB "
                                    f"(올린 뒤에도 {MIN_FREE / 1e9:.0f}GB 는 남아야 합니다): {self.root}")
        ext = Path(name).suffix.lower()
        if not re.fullmatch(r"\.[a-z0-9]{1,5}", ext):
            ext = ".bin"
        with self._lock:
            while True:
                sid = f"{datetime.now():%Y%m%d_%H%M%S}_{secrets.token_hex(2)}"
                if sid not in self.sessions and not (self.root / sid).exists():
                    break
            s = Session(self.root, sid)
            s.video = {"name": name, "size": size, "mtime_ms": mtime_ms, "file": f"video{ext}"}
            s.dir.mkdir(parents=True)
            s.save()
            self.sessions[sid] = s
        return s, s.video_path

    def finish_upload(self, s: Session, written: int) -> dict:
        """올린 사본을 OpenCV 로 열어 본다. 못 열면 지운다."""
        import cv2

        cap = cv2.VideoCapture(str(s.video_path))
        try:
            ok = cap.isOpened()
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) if ok else 0.0
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if ok else 0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) if ok else 0
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) if ok else 0
            first = cap.read()[0] if ok else False
        finally:
            cap.release()
        if not ok or not first:
            self.discard(s.sid, force=True)
            raise CaptureError(415, f"영상을 열지 못했습니다: {s.video.get('name')} — OpenCV(FFmpeg)가 모르는 형식이거나 "
                                    "깨진 파일입니다")
        if not (0 < fps < 1000):
            fps = 0.0
        dur = n / fps if fps and n > 0 else None
        with s.lock:
            s.video.update(size=written, fps=round(fps, 3), frames=n if n > 0 else None,
                           duration=round(dur, 3) if dur else None, width=w, height=h)
            guess = _from_name(Path(s.video["name"]))
            s.start_guess = fmt_start(guess.replace(tzinfo=None)) if guess else None
            mt = s.video.get("mtime_ms")
            if mt and dur:
                try:
                    s.start_mtime = fmt_start(datetime.fromtimestamp(mt / 1000) - timedelta(seconds=dur))
                except (OverflowError, OSError, ValueError):
                    s.start_mtime = None
            s.start = s.start_guess
            s.camera = self._camera_from_name(s.video["name"])
            s.state = "ready"
        s.save()
        return s.detail()

    def _camera_from_name(self, name: str) -> str | None:
        """영상 이름이 `<등록된 카메라 키>_…` 로 시작하면 그 키 (ROAD PLUS 구간 영상처럼)."""
        cams, _ = self.cameras()
        stem = Path(name).stem.lower()
        hits = [c["key"] for c in cams if stem.startswith(c["key"] + "_") or stem == c["key"]]
        return max(hits, key=len) if hits else None

    # ----------------------------------------------------------------- 뽑기

    def start_extract(self, sid: str, every: float, t_from: float = 0.0, t_to: float | None = None) -> dict:
        s = self.get(sid)
        try:
            every = float(every)
            t_from = float(t_from or 0.0)
            t_to = None if t_to is None else float(t_to)
        except (TypeError, ValueError):
            raise CaptureError(400, "간격·구간은 숫자(초)로 주세요.") from None
        if not (MIN_EVERY <= every <= MAX_EVERY):
            raise CaptureError(400, f"간격은 {MIN_EVERY:g}~{MAX_EVERY:g}초 사이로 주세요.")
        if t_from < 0 or (t_to is not None and t_to <= t_from):
            raise CaptureError(400, "구간의 끝이 시작보다 뒤여야 합니다.")
        dur = s.video.get("duration")
        if dur and t_from >= dur:
            raise CaptureError(400, f"시작({t_from:g}초)이 영상 길이({dur:g}초)를 넘습니다.")
        if dur:
            end = min(t_to, dur) if t_to is not None else dur
            est = int((end - t_from) // every) + 1
            if est > MAX_FRAMES:
                raise CaptureError(400, f"약 {est}장이 나옵니다 — 한 번에 {MAX_FRAMES}장까지입니다. 간격을 늘리거나 구간을 "
                                        "줄이세요.")
        with s.lock:
            if s.state == "uploading":
                raise CaptureError(409, "아직 올리는 중입니다.")
            if s.state == "committing":
                raise CaptureError(409, "넣는 중입니다.")
            if s.state in RUNNING:
                raise CaptureError(409, "이미 뽑는 중입니다 — 멈춘 뒤 다시 하세요.")
            # 앞 run 의 스레드가 아직 잠금을 기다리는 중일 수는 있다 — 깨어나면 run 이 달라 아무것도 안 하고 끝난다.
            # 파일을 만지는 'extracting' 은 위에서 막았다(그 상태를 벗어날 때는 영상도 이미 닫았다).
            for sub in ("frames", "thumbs"):
                shutil.rmtree(s.dir / sub, ignore_errors=True)
                (s.dir / sub).mkdir(exist_ok=True)
            s.frames = []
            s.run += 1
            s.settings = {"every": every, "t_from": t_from, "t_to": t_to}
            s.state, s.error, s.note, s.pos = "queued", None, None, t_from
            s.stop.clear()
            run = s.run
            s.thread = threading.Thread(target=self._extract, args=(s, run), name=f"capture-{sid}", daemon=True)
        s.save()
        s.thread.start()
        return s.detail()

    def stop_extract(self, sid: str) -> dict:
        s = self.get(sid)
        with s.lock:
            if s.state not in RUNNING:
                raise CaptureError(409, "뽑는 중이 아닙니다.")
            s.stop.set()
            if s.state == "queued":          # 아직 시작 전(다른 영상을 뽑는 중) — 기다리던 스레드는 깨어나면 그냥 끝난다
                s.state = "stopped"
        s.save()
        return s.detail()

    def _extract(self, s: Session, run: int) -> None:
        with self._extract_lock:
            with s.lock:
                if s.deleted or s.run != run or s.state != "queued":
                    return
                if s.stop.is_set():
                    s.state = "stopped"
                    s.save()
                    return
                s.state = "extracting"
            try:
                self._extract_frames(s)
            except Exception as exc:  # noqa: BLE001 — 스레드 밖으로 내보낼 곳이 없다. 화면에 보인다
                with s.lock:
                    s.state, s.error = "error", f"{type(exc).__name__}: {exc}"
            finally:
                s.save()

    def _extract_frames(self, s: Session) -> None:
        import cv2

        st = s.settings
        every, t_from, t_to = st["every"], st["t_from"], st["t_to"]
        fps = s.video.get("fps") or 0.0
        cap = cv2.VideoCapture(str(s.video_path))
        if not cap.isOpened():
            raise RuntimeError("영상을 열지 못했습니다")
        seen: dict[str, int] = {}
        last_small = None
        next_t = t_from
        idx = -1
        stopped = full = False
        try:
            while True:
                if s.stop.is_set() or s.deleted:
                    stopped = True
                    break
                if not cap.grab():
                    break
                idx += 1
                pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if pos <= 0 and idx > 0 and fps:
                    pos = idx / fps                       # 백엔드가 시각을 안 주면 프레임 번호로 (extract_frames 와 같다)
                s.pos = pos
                if t_to is not None and pos > t_to + 1e-6:
                    break
                if pos + 1e-6 < next_t:
                    continue
                next_t = t_from + (int((pos - t_from + 1e-6) // every) + 1) * every   # t_from·+N·+2N초… 격자
                ok, frame = cap.retrieve()
                if not ok:
                    continue
                i = len(s.frames)
                if i >= MAX_FRAMES:
                    full = True
                    break
                okj, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if not okj:
                    continue
                data = buf.tobytes()
                digest = hashlib.sha1(data).hexdigest()   # 같은 픽셀이면 같은 JPEG 바이트다
                same = seen.setdefault(digest, i)
                small = small_gray(frame)
                change = None if last_small is None else round(changed_pct(small, last_small), 3)
                last_small = small
                h, w = frame.shape[:2]
                tw = min(THUMB_W, w)
                thumb = cv2.resize(frame, (tw, max(1, round(h * tw / w))), interpolation=cv2.INTER_AREA)
                okt, tbuf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, THUMB_QUALITY])
                s.frame_path(i).write_bytes(data)
                if okt:
                    s.frame_path(i, thumb=True).write_bytes(tbuf.tobytes())
                with s.lock:
                    if s.deleted:
                        return
                    s.frames.append({"i": i, "t": round(pos, 3), "c": change,
                                     "s": None if same == i else same, "k": same == i})
                if time.monotonic() - s._saved_at > _SAVE_EVERY_S:
                    s.save()
        finally:
            cap.release()
        with s.lock:
            s.state = "stopped" if stopped else "done"
            if full:
                s.note = f"{MAX_FRAMES}장에서 멈췄습니다 — 간격을 늘리거나 구간을 나눠 다시 뽑으세요"
            elif not s.frames and not stopped:
                s.note = "뽑은 프레임이 없습니다 — 구간이 영상 밖이거나 영상을 끝까지 읽지 못했습니다"

    # ----------------------------------------------------------------- 고르기·넣기·닫기

    def update(self, sid: str, *, camera: str | None = None, start: str | None = None,
               drop: list[int] | None = None, upto: int | None = None) -> dict:
        """화면에서 고른 것(빼기 목록·카메라·시작 시각)을 남긴다 — 다시 열면 이어진다. 확인은 넣을 때 한다.

        drop 은 화면이 아는 앞 `upto` 장(없으면 전부) 안에서 뺄 번호 — 뽑는 중이라 화면이 아직 못 받은 뒤쪽 프레임은
        그대로 둔다(픽셀까지 같은 프레임은 처음부터 '빼기' 인데, 그것까지 '넣기' 로 바꾸면 안 된다)."""
        s = self.get(sid)
        with s.lock:
            if camera is not None:
                s.camera = camera or None
            if start is not None:
                s.start = start.strip() or None
            if drop is not None:
                off = set(int(i) for i in drop)
                limit = len(s.frames) if upto is None else max(0, int(upto))
                for f in s.frames[:limit]:
                    f["k"] = f["i"] not in off
        s.save()
        return s.summary()

    def commit(self, sid: str, camera: str, start: str, keep: list[int]) -> dict:
        s = self.get(sid)
        with self._commit_lock:
            with s.lock:
                if s.deleted:            # 잠금을 기다리는 사이 다른 화면이 넣었거나 닫았다
                    raise CaptureError(404, "이 영상은 이미 넣었거나 닫았습니다.")
                if s.state in RUNNING or s.state == "uploading":
                    raise CaptureError(409, "뽑는 중입니다 — 끝나거나 멈춘 뒤 넣으세요.")
                frames = list(s.frames)
            why = self.blocker()
            if why:
                raise CaptureError(409, why)
            try:
                cams = load_cameras(self.cfg.cameras)
            except ValueError as exc:
                raise CaptureError(400, f"카메라 등록 파일을 읽지 못했습니다:\n{exc}") from exc
            if camera not in cams:
                raise CaptureError(400, f"카메라 '{camera}' 가 {self.cfg.cameras.name} 에 없습니다 — 먼저 등록하세요"
                                        " (docs/provenance.md).")
            try:
                t0 = parse_start(start)
            except ValueError as exc:
                raise CaptureError(400, str(exc)) from None
            idx = sorted({int(i) for i in keep})
            if not idx:
                raise CaptureError(400, "넣을 프레임을 하나도 고르지 않았습니다.")
            if idx[0] < 0 or idx[-1] >= len(frames):
                raise CaptureError(400, "없는 프레임 번호가 있습니다 — 화면을 새로 고치세요.")
            with s.lock:
                if s.deleted or s.state in RUNNING or s.state in ("uploading", "committing"):
                    raise CaptureError(409, "그사이 다시 뽑기를 시작했거나 닫았습니다 — 화면을 새로 고치세요.")
                before, s.state = s.state, "committing"   # 그동안 다시 뽑기·닫기가 프레임을 지우지 않게

            root = self.cfg.unlabeled / camera
            moved: list[Path] = []
            exists: list[str] = []
            missing: list[int] = []
            try:
                for i in idx:
                    src = s.frame_path(i)
                    t = t0 + timedelta(seconds=frames[i]["t"])
                    dst = root / f"{t:%Y%m%d}" / frame_name(camera, t)
                    if dst.exists():
                        exists.append(dst.name)
                        continue
                    if not src.exists():
                        missing.append(i)
                        continue
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        _rename(src, dst)         # 같은 디스크라 옮기기만 한다. Windows 의 rename 은 덮어쓰지 않는다
                    except FileExistsError:
                        exists.append(dst.name)
                        continue
                    moved.append(dst)
            except OSError as exc:
                with s.lock:
                    s.state = before
                    s.note = f"넣다가 멈췄습니다 — {len(moved)}장은 {root} 에 옮겼습니다: {exc}"
                s.save()
                raise CaptureError(500, f"{len(moved)}장을 옮긴 뒤 멈췄습니다 ({root}): {exc}") from exc

            out = {"camera": camera, "moved": len(moved), "exists": len(exists), "missing": len(missing),
                   "exists_names": exists[:5], "dest": str(root), "first": moved[0].name if moved else None,
                   "last": moved[-1].name if moved else None}
            if moved:
                try:
                    report = self.ingest()
                except Exception as exc:  # noqa: BLE001 — 파일은 이미 옮겼다. 사람이 고친 뒤 '폴더 스캔' 하면 된다
                    out["scan_error"] = f"{type(exc).__name__}: {exc}"
                else:
                    mine = {os.path.normcase(str(p.resolve())) for p in moved}
                    dups = [d for d in report.duplicates if os.path.normcase(d["path"]) in mine]
                    out["duplicates"] = len(dups)
                    out["unreadable"] = len([u for u in report.unreadable if any(u.startswith(p.name) for p in moved)])
                    new_total = report.unlabeled_images - report.unlabeled_already
                    if self.known_paths is not None:
                        known = self.known_paths()
                        out["registered"] = sum(1 for p in mine if p in known)
                    else:
                        out["registered"] = len(moved) - out["duplicates"] - out["unreadable"]
                    out["other_new"] = max(0, new_total - out["registered"])
                    out["summary"] = report.summary()
            left = self.discard(sid, force=True)
            if left:
                out["left"] = left
            return out

    def discard(self, sid: str, *, force: bool = False) -> str | None:
        """세션을 닫고 폴더(영상 사본·프레임)를 지운다. 다 못 지웠으면 그 이유를 돌려준다."""
        s = self.get(sid)
        with s.lock:
            if not force and s.state in ("uploading", "committing"):
                raise CaptureError(409, "올리는 중입니다 — 올리기를 취소하세요." if s.state == "uploading"
                                   else "넣는 중입니다.")
            s.stop.set()
            th = s.thread if s.state == "extracting" else None
        if th is not None and th.is_alive() and th is not threading.current_thread():
            # 뽑던 스레드가 영상·파일을 놓을 때까지 (Windows 는 열린 파일을 못 지운다). 잠금을 기다리기만 하던 스레드는
            # 파일을 만지지 않으니 기다리지 않는다 — 깨어나면 deleted 를 보고 끝난다.
            th.join(timeout=15)
        with s.lock:
            s.deleted = True
        with self._lock:
            self.sessions.pop(sid, None)
        errors = _rmtree(s.dir)
        if errors:
            return f"임시 폴더를 다 지우지 못했습니다({len(errors)}개 — {s.dir}): {errors[0]}"
        return None


def _rename(src: Path, dst: Path, tries: int = 20) -> None:
    """os.rename — Windows 에서 화면이 그 프레임을 받아 가는 중(파일이 열려 있다)이면 잠깐 기다렸다 다시 한다.
    미라벨 폴더가 DB 와 다른 드라이브면(설정으로 가능하다) 복사한 뒤 지운다 — 그때도 있는 파일은 덮지 않는다."""
    for k in range(tries):
        try:
            os.rename(src, dst)
            return
        except FileExistsError:
            raise
        except PermissionError:
            if k == tries - 1:
                raise
            time.sleep(0.05)
        except OSError as exc:
            if getattr(exc, "winerror", None) != 17 and exc.errno != errno.EXDEV:   # 17 = 다른 드라이브
                raise
            try:
                with open(src, "rb") as fi, open(dst, "xb") as fo:      # xb: 있으면 FileExistsError
                    shutil.copyfileobj(fi, fo, 1 << 20)
            except FileExistsError:
                raise
            except BaseException:
                dst.unlink(missing_ok=True)
                raise
            os.unlink(src)
            return


def _rmtree(path: Path) -> list[str]:
    """폴더를 지운다. 화면이 파일을 받아 가는 중이라 못 지운 것이 있으면 한 번 더 해 보고, 그래도 남은 것을 돌려준다."""
    for attempt in range(2):
        errors: list[str] = []
        shutil.rmtree(path, onexc=lambda fn, p, exc: errors.append(f"{Path(p).name}: {exc}"))
        if not errors or not path.exists():
            return []
        time.sleep(0.3)
    return errors
