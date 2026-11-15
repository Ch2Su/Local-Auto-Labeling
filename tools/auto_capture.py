"""영상에서 학습할 프레임을 자동으로 골라 '영상 캡처' 로 넣는다 (2026-11-12, 사용자 요청 — 녹화부터 자동화의 첫 단계).

지금은 검수 UI '영상 캡처' 탭에서 사람이 간격을 정해 뽑고 눈으로 고른다. 이 도구는 그 탭의 서버 API 를 그대로 쓰고
**고르기만 대신한다**:

    영상 → 올리기·뽑기(서버, --every 초마다) → 후보마다 운영 가중치로 검출·화질 측정(이 도구) → 고르기(src/autolabel/autopick.py)
         → 기본: 고른 결과를 탭에 표시(넣기는 사람이 화면에서) / --apply: 넣기(unlabeled/<카메라>/ + 스캔) → 자동 라벨링 작업

    python tools/auto_capture.py 영상.mp4                    # 탭에 표시만 — 화면에서 확인하고 '넣기'
    python tools/auto_capture.py D:\\videos --apply           # 폴더의 영상을 차례로 넣고, 끝나면 자동 라벨링 작업을 건다
    python tools/auto_capture.py D:\\rec --watch --apply      # 다 쓴 영상이 생길 때마다 (녹화 폴더 — 아래 '폴더 보기')
    python tools/auto_capture.py 영상.mp4 --dry-run --sheet   # 서버에 아무것도 남기지 않고 고른 결과만 (대조 시트)

**검수 서버가 켜져 있어야 한다.** 이름(`frame_name`·`notime_name`)·카메라 추정(영상 이름 `<키>_…`)·시작 시각(이름의 시각)·
자동 카메라(`vid-NN`)·넣기의 스캔 잠금(`ingest_lock`)을 서버의 영상 캡처 코드(capture.py) 한 곳에 두려고 같은 API 를 부른다 —
탭에서 넣은 것과 이름·출처가 바이트까지 같다. 영상은 서버에 사본으로 올라가고(원본은 읽기만), 넣거나 닫으면 서버가 지운다.

**기본은 보기만이다** — 고른 결과를 세션의 넣기/빼기로 남기고 끝난다(`PUT /api/capture/<세션>/state`). 탭에서 그 영상을 열면
고른 칸만 '넣기' 로 보이니 고쳐서 넣거나 닫는다. `--apply` 면 바로 넣는다 — 카메라·시작 시각은 `--camera`·`--start`, 없으면
탭과 같이 영상 이름에서, 그래도 없으면 자동 카메라·시각 없는 이름. 넣은 것이 있으면 마지막에 자동 라벨링 작업을 건다
(`--no-label` 로 끈다, `--tiles` 면 타일 추론). **자동 승인은 없다** — 검수 대기까지만 간다.

**같은 카메라의 영상 여러 개는 `--camera` 로 묶을 것.** 카메라를 못 정하면 영상마다 자동 카메라(vid-NN)가 생기고, 분할이
카메라째라 같은 장소의 영상이 train 과 test 로 갈릴 수 있다.

검출은 이 프로세스가 운영 가중치(`model.weights`)로 한다 — `predict.imgsz`·`iou`·`max_det` 은 자동 라벨링과 같고, 신뢰도만
놓칠 뻔한 객체까지 보려고 `--det-conf`(0.10)로 낮춘다. **서버에서 GPU 작업(학습·자동 라벨링)이 도는 중이면 CPU 로 돈다**
(`/api/state` — 예측 비교 P 와 같은 규칙, `--recheck` 초마다 다시 본다). 이 프로세스의 GPU 사용은 서버의 GPU 잠금에 잡히지
않는다 — 처리 중에 학습을 시작하면 다음 확인(기본 20초)까지 둘이 GPU 를 함께 쓴다. 그동안 이 도구가 쥐는 VRAM 은
약 0.9GB 다(2026-11-16 실측 — yolo26m·640·묶음 8: 할당 정점 460MB, 프로세스 전체로 nvidia-smi +857MiB). 16GB 카드에서는
상관없고, 8GB 에서 한도까지 채운 학습(m 640 b10)과 겹치면 그 20초 동안 학습이 느려질 수 있다(Windows 는 OOM 대신 공유
메모리로 넘긴다 — CLAUDE.md 함정 9). 그런 PC 에서는 `--recheck` 를 줄이거나 `--device cpu` 로 돌린다. 서버의 GPU 잠금을
함께 쥐게 하지는 않았다 — 그러면 이 도구가 도는 동안 학습을 시작할 수 없게 된다(11-16 Gemini 제안 검토).

**폴더 보기(`--watch`)** — `--poll` 초마다 폴더를 다시 보고, 크기가 있고 지난번 확인 때와 크기·수정 시각이 같으며 마지막으로
바뀐 지 `--settle` 초가 지난 영상만 다 쓴 것으로 본다. 수정 시각만 보면 안 된다 — 시험에서 ffmpeg 가 mkv 조각을 몇 초씩 버퍼에
쥐고 있어 0바이트 파일을 올렸다. 서버가 꺼지면 다시 켜질 때까지 기다린다. `--watch` 없이 돌 때는 30초 안에 바뀐 파일(내려받기
직후·녹화 중인 조각)을 30초 동안 그대로일 때까지 기다리고(최대 90초), 그래도 바뀌면 건너뛴다.

처리한 영상은 `<DB 폴더>/autocapture/videos.jsonl` 에 영상 내용 sha1 로 남겨 다시 돌리면 건너뛴다(`--redo` 면 다시).
후보 프레임마다 잰 값·검출·고른 이유는 `autocapture/frames/<지문>_<세션>.jsonl` — 기본값을 이 데이터에 맞출 때 쓴다.
`--sheet` 면 고른 프레임 대조 시트(`autocapture/sheets/`, 새 객체는 초록·이미 본 객체는 회색 박스)도 남긴다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # tools/ -> 저장소 루트

import numpy as np  # noqa: E402

from autolabel import autopick  # noqa: E402
from autolabel.autopick import Cand, Det, Params  # noqa: E402
from autolabel.capture import RUNNING, small_gray  # noqa: E402
from autolabel.config import Config  # noqa: E402
from autolabel.jobs import _GPU_KINDS  # noqa: E402

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".m4v", ".ts", ".mts", ".m2ts", ".webm", ".flv", ".wmv", ".mpg", ".mpeg",
              ".3gp"}
DONE = ("proposed", "committed", "empty")     # 다시 돌리지 않는 기록 (error·dry-run 은 다시 한다)
FRESH = 30.0                                   # --watch 없이 돌 때 이보다 최근에 바뀐 파일은 건너뛴다(초)


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class ServerDown(Exception):
    pass


# ───────────────────────────── 서버 ─────────────────────────────

class Server:
    """검수 서버 API (urllib — tools/scheduled_stop.py 와 같은 방식. Origin 없는 요청이라 _RequestGuard 를 지난다)."""

    def __init__(self, base: str):
        self.base = base.rstrip("/")
        self.prefix = ""                 # 탭 서버면 /p/<프로젝트>
        self.prefixes: list[str] = [""]  # GPU 확인용 — 모든 프로젝트

    def call(self, method: str, path: str, body=None, *, data=None, headers=None, timeout: float = 60):
        h = dict(headers or {})
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            h["Content-Type"] = "application/json"
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=h)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read() or b"{}").get("detail")
            except (ValueError, AttributeError):
                detail = None
            raise ApiError(exc.code, str(detail or exc.reason)) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise ServerDown(f"검수 서버({self.base})에 닿지 않습니다 — {getattr(exc, 'reason', exc)}") from None
        return json.loads(raw) if raw else None

    def api(self, method: str, path: str, body=None, **kw):
        return self.call(method, f"{self.prefix}/api/{path}", body, **kw)

    def connect(self, project: str | None) -> str | None:
        """프로젝트를 골라 주소 앞부분을 정한다. 탭 서버면 프로젝트 이름, `--config` 로 하나만 띄운 서버면 None."""
        try:
            projects = self.call("GET", "/api/projects")["projects"]
        except ApiError as exc:
            if exc.status != 404:
                raise
            if project:
                raise SystemExit("이 서버는 프로젝트 탭 없이 설정 하나로 떠 있습니다 — --project 를 빼고 --config 로 고르세요.")
            return None
        self.prefixes = [p["base"].rstrip("/") for p in projects]
        names = [p["name"] for p in projects]
        name = project or names[0]
        if name not in names:
            raise SystemExit(f"서버에 '{name}' 프로젝트가 없습니다 — {', '.join(names)}")
        self.prefix = self.prefixes[names.index(name)]
        return name

    def gpu_busy(self) -> str | None:
        """어느 프로젝트에서든 GPU 작업이 도는 중이면 그 작업 이름."""
        for pre in self.prefixes:
            try:
                job = (self.call("GET", f"{pre}/api/state", timeout=15) or {}).get("job")
            except (ApiError, ServerDown):
                continue
            if job and job.get("status") == "running" and job.get("kind") in _GPU_KINDS:
                return f"{pre or '/'} {job['kind']}"
        return None

    def upload(self, path: Path, size: int, mtime_ms: int) -> dict:
        """영상을 본문 그대로 올린다(탭과 같은 `POST capture/upload` — 리눅스 시험에서 300MB 가 1초 안쪽)."""
        q = urllib.parse.urlencode({"name": path.name, "size": size, "mtime": mtime_ms})
        with open(path, "rb") as f:
            return self.api("POST", f"capture/upload?{q}", data=f, timeout=600,
                            headers={"Content-Length": str(size), "Content-Type": "application/octet-stream"})


# ───────────────────────────── 검출 ─────────────────────────────

class Detector:
    """운영 가중치로 후보를 검출한다. 장치는 --device(auto 면 서버의 GPU 작업을 보고) — 바뀌면 다시 올린다
    (Ultralytics 는 첫 추론 때 장치를 정하고 바꾸지 않는다 — predictor.LiveDetector 와 같다)."""

    def __init__(self, cfg: Config, classes: list[str], det_conf: float, device: str, srv: Server, recheck: float):
        self.cfg = cfg
        self.classes = classes
        self.det_conf = det_conf
        self.pref = device
        self.srv = srv
        self.recheck = recheck
        self.weights = cfg.active_weights()
        self.model = None
        self.cmap: dict[int, int] = {}
        self.device = None
        self._checked = 0.0
        self.note = ""

    def _want(self):
        if self.pref != "auto":
            return int(self.pref) if self.pref.isdigit() else self.pref
        import torch

        if not torch.cuda.is_available():
            return "cpu"
        busy = self.srv.gpu_busy()
        self.note = f"서버에서 {busy} 이 GPU 를 쓰는 중 — CPU 로" if busy else ""
        return "cpu" if busy else self.cfg.device

    def ensure(self) -> None:
        now = time.monotonic()
        if self.model is not None and now - self._checked < self.recheck:
            return
        self._checked = now
        want = self._want()
        if self.model is not None and str(want) == str(self.device):
            return
        self.release()
        from ultralytics import YOLO

        from autolabel.predictor import build_class_map

        model = YOLO(self.weights)
        # 빈 이미지로 예열 — 첫 추론의 CUDA 초기화가 NMS 시간 제한을 넘겨 검출이 잘리는 일을 막는다(LiveDetector 와 같다)
        model.predict(np.zeros((64, 64, 3), np.uint8), device=want, verbose=False)
        self.model, self.device = model, want
        self.cmap = build_class_map(dict(model.names), self.classes)
        if not self.cmap:
            raise SystemExit(f"가중치 {self.weights} 의 클래스가 프로젝트 클래스와 하나도 겹치지 않습니다.")

    def release(self) -> None:
        if self.model is None:
            return
        self.model = None
        self.device = None
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def __call__(self, images: list[np.ndarray]) -> list[list[tuple[int, float, tuple[float, float, float, float]]]]:
        self.ensure()
        res = self.model.predict(
            source=images,
            conf=self.det_conf,
            iou=float(self.cfg.get("predict.iou", 0.6)),
            max_det=int(self.cfg.get("predict.max_det", 100)),
            imgsz=int(self.cfg.get("predict.imgsz") or self.cfg.imgsz),
            device=self.device,
            verbose=False,
        )
        out = []
        for r in res:
            dets = []
            if r.boxes is not None and len(r.boxes):
                for box, c, cf in zip(r.boxes.xyxyn.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                    mapped = self.cmap.get(int(c))
                    if mapped is None:
                        continue
                    x1, y1, x2, y2 = (min(1.0, max(0.0, float(v))) for v in box)
                    if x2 > x1 and y2 > y1:
                        dets.append((mapped, float(cf), (x1, y1, x2, y2)))
            out.append(dets)
        return out


def read_image(path: Path) -> np.ndarray | None:
    """한글 경로도 되게 바이트로 읽어 푼다. 없거나 못 읽으면 None."""
    import cv2

    try:
        data = np.fromfile(str(path), np.uint8)
    except OSError:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    return img


# ───────────────────────────── 처리 기록 ─────────────────────────────

class Manifest:
    """처리한 영상 (videos.jsonl — 한 줄에 영상 하나, 뒤의 줄이 앞을 덮는다)."""

    def __init__(self, path: Path):
        self.path = path
        self.by_sha: dict[str, dict] = {}
        self.by_stat: dict[tuple, dict] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                self._index(rec)

    def _index(self, rec: dict) -> None:
        if rec.get("sha1"):
            self.by_sha[rec["sha1"]] = rec
        self.by_stat[(rec.get("path"), rec.get("size"), rec.get("mtime_ns"))] = rec

    def add(self, rec: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._index(rec)


def sha1_of(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_videos(inputs: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in inputs:
        if p.is_dir():
            out += sorted(q for q in p.rglob("*") if q.is_file() and q.suffix.lower() in VIDEO_EXTS
                          and not q.name.startswith((".", "~$")))
        elif p.is_file():
            out.append(p)
    seen: set[Path] = set()
    uniq = []
    for p in out:
        r = p.resolve()
        if r not in seen:
            seen.add(r)
            uniq.append(r)
    return uniq


# ───────────────────────────── 영상 하나 ─────────────────────────────

class Job:
    def __init__(self, args, cfg: Config, srv: Server, det: Detector, manifest: Manifest, classes: list[str],
                 weights: list[float], limits: dict):
        self.a = args
        self.cfg = cfg
        self.srv = srv
        self.det = det
        self.manifest = manifest
        self.classes = classes
        self.weights = weights
        self.limits = limits
        self.root = cfg.db.parent / "autocapture"
        self.registered = 0              # 이번 실행에서 넣어 등록된 장수 (자동 라벨링을 걸지 정한다)
        self.seen: set[tuple] = set()    # --watch: 이 프로세스에서 이미 본 (경로, 크기, 수정 시각) — 지문을 다시 세지 않게

    # --- 서버 세션 ---
    def _wait_extract(self, sid: str) -> dict:
        last = 0.0
        while True:
            d = self.srv.api("GET", f"capture/{sid}?since=1000000000")
            if d["state"] not in RUNNING:
                print(" " * 70, end="\r")
                return self.srv.api("GET", f"capture/{sid}?since=0")
            if time.monotonic() - last > 2:
                dur = d.get("duration")
                pos = f"{d.get('pos') or 0:.0f}/{dur:.0f}초" if dur else f"{d.get('pos') or 0:.0f}초"
                print(f"  뽑는 중 {d['n']}장 ({pos})", end="\r", flush=True)
                last = time.monotonic()
            time.sleep(1.0)

    def _every(self, duration: float | None) -> float:
        every = self.a.every
        mx = int(self.limits.get("max_frames") or 5000)
        if duration and int(duration // every) + 1 > mx:
            every = math.ceil(duration / (mx - 1) * 100) / 100
            print(f"  영상이 길어 간격을 {self.a.every:g}초 → {every:g}초로 늘립니다 (한 세션 {mx}장까지)")
        return max(every, float(self.limits.get("min_every") or 0.2))

    # --- 측정 ---
    def measure(self, sid: str, frames: list[dict]) -> list[Cand]:
        frames_dir = self.cfg.db.parent / "capture" / sid / "frames"
        if frames_dir.is_dir():
            def load(f: dict) -> np.ndarray | None:
                return read_image(frames_dir / f"{f['i']:05d}.jpg")
        else:
            print(f"  세션 폴더가 이 설정의 DB 폴더에 없습니다({frames_dir.parent}) — 서버에서 받아 읽습니다")
            load = self._loader_http(sid)

        cands: list[Cand] = []
        todo = [f for f in frames if f.get("s") is None]            # 픽셀까지 같은 프레임은 검출하지 않는다
        for f in frames:
            if f.get("s") is not None:
                cands.append(Cand(f["i"], float(f["t"]), [], same=int(f["s"])))
        t0 = time.monotonic()
        done = 0
        bs = max(1, self.a.batch)
        with ThreadPoolExecutor(max_workers=4) as pool:
            nxt = pool.map(load, todo[:bs])
            for k in range(0, len(todo), bs):
                chunk = todo[k:k + bs]
                imgs = list(nxt)
                if k + bs < len(todo):
                    nxt = pool.map(load, todo[k + bs:k + 2 * bs])        # 다음 묶음은 검출하는 동안 미리 읽는다
                metas, batch = [], []
                for f, img in zip(chunk, imgs):
                    if img is None:
                        continue
                    sharp, mean, flat = autopick.frame_stats(small_gray(img))
                    metas.append((f, sharp, mean, flat))
                    batch.append(img)
                results = self.det(batch) if batch else []
                for (f, sharp, mean, flat), img, dets in zip(metas, batch, results):
                    ds = [Det(c, conf, box, autopick.make_sig(img, box)) for c, conf, box in dets]
                    cands.append(Cand(f["i"], float(f["t"]), ds, sharp, mean, flat, None))
                done += len(chunk)
                rate = done / max(1e-6, time.monotonic() - t0)
                print(f"  검출 {done}/{len(todo)}장 ({self.det.device}, {rate:.0f}장/초)", end="\r", flush=True)
        print(" " * 70, end="\r")
        if self.det.note:
            print(f"  {self.det.note}")
        cands.sort(key=lambda c: c.i)
        return cands

    def _loader_http(self, sid: str):
        import cv2

        def load(f: dict) -> np.ndarray | None:
            url = f"{self.srv.base}{self.srv.prefix}/api/capture/{sid}/frame/{f['i']}"
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    data = np.frombuffer(r.read(), np.uint8)
            except (urllib.error.URLError, TimeoutError, OSError):
                return None
            return cv2.imdecode(data, cv2.IMREAD_COLOR)

        return load

    def _gray_fn(self, sid: str):
        """빈 화면·탐색 몫의 화면 비교용 축소 흑백 — 썸네일에서 (원본보다 싸다)."""
        thumbs = self.cfg.db.parent / "capture" / sid / "thumbs"

        def gray(i: int):
            img = read_image(thumbs / f"{i:05d}.jpg")
            return None if img is None else small_gray(img)

        return gray if thumbs.is_dir() else None

    # --- 기록 ---
    def write_frames(self, tag: str, sid: str, video: Path, cands: list[Cand], res: autopick.Result, every: float,
                     params: Params) -> Path:
        path = self.root / "frames" / f"{tag}_{sid}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        picks = {p.i: p for p in res.picks}
        dprev = _prev_match(cands, params.match_iou)
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"meta": {
                "video": video.name, "sid": sid, "every": every, "weights": self.det.weights, "device": str(self.det.device),
                "det_conf": self.a.det_conf, "classes": self.classes, "class_weights": [round(w, 3) for w in self.weights],
                "params": params.__dict__, "budget": res.budget, "budgets": res.budgets, "stop": res.stop,
                "at": datetime.now().isoformat(timespec="seconds")}}, ensure_ascii=False) + "\n")
            for c in cands:
                pk = picks.get(c.i)
                f.write(json.dumps({
                    "i": c.i, "t": c.t, "same": c.same,
                    "sharp": None if c.sharp is None else round(c.sharp, 1),
                    "mean": None if c.mean is None else round(c.mean, 1),
                    "flat": None if c.flat is None else round(c.flat, 3),
                    "drop": res.dropped.get(c.i), "pick": pk.kind if pk else None,
                    "gain": pk.gain if pk else None, "fresh": pk.fresh if pk else None,
                    # 검출마다 [클래스, 신뢰도, x1, y1, x2, y2, 앞 후보에서 IoU ≥ match_iou 인 검출과의 외형 차이(없으면 null)]
                    # — 마지막 값의 분포로 --match-sig 를 맞춘다(서 있는 객체는 작고, 다른 객체는 크다)
                    "dets": [[self.classes[d.cls], round(d.conf, 3), *(round(v, 4) for v in d.box), dp]
                             for d, dp in zip(c.dets, dprev.get(c.i) or [None] * len(c.dets))],
                }, ensure_ascii=False) + "\n")
        return path

    def write_sheet(self, tag: str, sid: str, cands: list[Cand], res: autopick.Result) -> Path | None:
        """고른 프레임 대조 시트 — 칸마다 썸네일, 새 객체 박스는 초록·이미 본 객체는 회색, 위에 시각·몫·값."""
        import cv2

        thumbs = self.cfg.db.parent / "capture" / sid / "thumbs"
        by_i = {c.i: c for c in cands}
        cells = []
        colors = {"main": (60, 200, 60), "background": (200, 160, 40), "explore": (40, 160, 230)}
        for pk in res.picks[:120]:
            img = read_image(thumbs / f"{pk.i:05d}.jpg")
            if img is None:
                continue
            img = cv2.resize(img, (320, max(1, round(img.shape[0] * 320 / img.shape[1]))), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]
            fresh = set(pk.fresh)
            for k, d in enumerate(by_i[pk.i].dets):
                x1, y1, x2, y2 = d.box
                col = (60, 220, 60) if k in fresh else (150, 150, 150)
                cv2.rectangle(img, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), col, 1)
            label = f"#{pk.i} {pk.t:.1f}s {pk.kind}" + (f" +{pk.gain:.1f}" if pk.kind == "main" else "")
            cv2.rectangle(img, (0, 0), (w, 16), (0, 0, 0), -1)
            cv2.putText(img, label, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[pk.kind], 1, cv2.LINE_AA)
            cells.append(img)
        if not cells:
            return None
        cols = 4
        ch = max(c.shape[0] for c in cells)
        rows = []
        for r in range(0, len(cells), cols):
            row = [np.pad(c, ((0, ch - c.shape[0]), (0, 0), (0, 0))) for c in cells[r:r + cols]]
            row += [np.zeros_like(row[0])] * (cols - len(row))
            rows.append(np.hstack(row))
        sheet = np.vstack(rows)
        path = self.root / "sheets" / f"{tag}_{sid}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            path.write_bytes(buf.tobytes())
            return path
        return None

    # --- 본체 ---
    def run(self, video: Path, n: int, total: int, *, watch: bool = False) -> None:
        a = self.a
        st = video.stat()
        stat_key = (str(video), st.st_size, st.st_mtime_ns)
        if watch and stat_key in self.seen:
            return
        self.seen.add(stat_key)
        again = a.redo or a.dry_run                  # --dry-run 은 아무것도 남기지 않으니 처리한 영상도 다시 본다(값 맞추기용)
        rec0 = self.manifest.by_stat.get(stat_key)
        if rec0 and not again and (rec0.get("state") in DONE or watch):
            if not watch:
                print(f"[{n}/{total}] {video.name} — 이미 처리함({rec0.get('state')}, {rec0.get('at')}) — 건너뜀 "
                      "(--redo 로 다시)")
            return
        sha1 = sha1_of(video)
        rec0 = self.manifest.by_sha.get(sha1)
        if rec0 and not again and rec0.get("state") in DONE:
            print(f"[{n}/{total}] {video.name} — 같은 내용의 영상을 이미 처리함({rec0.get('state')}, {rec0.get('name')}, "
                  f"{rec0.get('at')}) — 건너뜀 (--redo 로 다시)")
            return
        tag = sha1[:8]
        rec = {"at": datetime.now().isoformat(timespec="seconds"), "path": str(video), "name": video.name,
               "size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha1": sha1,
               "mode": "dry-run" if a.dry_run else ("apply" if a.apply else "propose")}
        print(f"[{n}/{total}] {video.name} ({st.st_size / 1e6:,.1f}MB)")
        sid = None
        try:
            d = self.srv.upload(video, st.st_size, st.st_mtime_ns // 1_000_000)
            sid = d["sid"]
            rec["sid"] = sid
            v = d.get("video") or {}
            every = self._every(v.get("duration"))
            dur = f"{v['duration']:.0f}초" if v.get("duration") else "길이 모름"
            print(f"  세션 {sid} · {v.get('width')}x{v.get('height')} · {dur} · {every:g}초마다 후보")
            self.srv.api("POST", f"capture/{sid}/extract", {"every": every})
            d = self._wait_extract(sid)
            if d["state"] == "error" or not d["frames"]:
                raise RuntimeError(d.get("error") or d.get("note") or "뽑은 프레임이 없습니다")
            if d.get("note"):
                print(f"  알림: {d['note']}")
            frames = d["frames"]
            cands = self.measure(sid, frames)
            params = self.params(int(sha1[:8], 16))
            res = autopick.select(cands, params, self.weights, every=every, gray=self._gray_fn(sid))
            fpath = self.write_frames(tag, sid, video, cands, res, every, params)
            sheet = self.write_sheet(tag, sid, cands, res) if a.sheet else None
            self.report(cands, res)
            rec.update(every=every, frames=len(frames), picked=res.counts(), budget=res.budget, stop=res.stop,
                       dropped=_count(res.dropped.values()), frames_log=str(fpath))
            if sheet:
                print(f"  대조 시트: {sheet}")
            keep = [p.i for p in res.picks]
            if a.dry_run or not keep:
                self.srv.api("DELETE", f"capture/{sid}")
                rec["state"] = "dry-run" if a.dry_run else "empty"
                print("  세션을 닫았습니다" + (" (--dry-run)" if a.dry_run else " — 고를 프레임이 없습니다"))
            elif not a.apply:
                kept = set(keep)
                body = {"drop": [f["i"] for f in frames if f["i"] not in kept], "upto": len(frames)}
                if a.camera:
                    body["camera"] = a.camera
                if a.start:
                    body["start"] = a.start
                self.srv.api("PUT", f"capture/{sid}/state", body)
                rec["state"] = "proposed"
                cam = a.camera or d.get("camera") or "(넣을 때 정함 — 비우면 vid-NN 자동)"
                start = a.start or d.get("start") or "(모름 — 시각 없는 이름)"
                print(f"  → '영상 캡처' 탭에 고른 {len(keep)}장을 표시했습니다(넣기 전) · 카메라 {cam} · 시작 {start}")
            else:
                camera = a.camera or d.get("camera")
                start = a.start or d.get("start")
                out = self.srv.api("POST", f"capture/{sid}/commit", {"camera": camera, "start": start, "keep": keep},
                                   timeout=600)
                rec["state"] = "committed"
                rec["commit"] = {k: out.get(k) for k in ("camera", "auto", "moved", "exists", "missing", "duplicates",
                                                          "unreadable", "registered", "no_time", "dest", "scan_error")}
                self.registered += int(out.get("registered") or 0)
                auto = {"new": " (자동 등록)", "reused": " (전에 자동 등록한 카메라)"}.get(out.get("auto"), "")
                print(f"  → 넣음 {out.get('moved')}장 → {out.get('dest')}{auto} · 등록 {out.get('registered')}장"
                      + (f" · 이미 있음 {out['exists']}" if out.get("exists") else "")
                      + (f" · 중복 {out['duplicates']}" if out.get("duplicates") else "")
                      + (" · 시각 없는 이름" if out.get("no_time") else ""))
                if out.get("scan_error"):
                    print(f"  ! 스캔 오류: {out['scan_error']} — 고친 뒤 '폴더 스캔'")
        except KeyboardInterrupt:
            if sid:
                print(f"\n  중단 — 세션 {sid} 이 '영상 캡처' 탭에 남아 있습니다(넣거나 닫으세요)")
            raise
        except ServerDown:
            self.seen.discard(stat_key)      # 서버가 돌아오면 다시 (--watch)
            if sid:
                print(f"\n  서버가 끊겼습니다 — 세션 {sid} 이 남아 있으면 탭에서 닫으세요")
            raise
        except (ApiError, RuntimeError, OSError) as exc:
            rec["state"] = "error"
            rec["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  ! {rec['error']}")
            if sid:
                try:
                    self.srv.api("DELETE", f"capture/{sid}")
                except (ApiError, ServerDown):
                    print(f"  ! 세션 {sid} 을 닫지 못했습니다 — 탭에서 닫으세요")
        self.manifest.add(rec)

    def params(self, seed: int) -> Params:
        a = self.a
        return Params(per_min=a.per_min, max_total=a.max, min_total=a.min, gap=a.gap, match_iou=a.match_iou,
                      match_sig=a.match_sig, uncertain_bonus=a.uncertain_bonus,
                      low_band=float(self.cfg.get("review.low_band", 0.30)),
                      high_band=float(self.cfg.get("review.high_band", 0.80)),
                      min_gain=a.min_gain, background=a.background, explore=a.explore,
                      explore_change=a.explore_change, blur_ratio=a.blur_ratio, seed=seed)

    def report(self, cands: list[Cand], res: autopick.Result) -> None:
        drops = _count(res.dropped.values())
        dtxt = ", ".join(f"{autopick.DROP_NAMES[k]} {v}" for k, v in drops.items()) or "없음"
        c = res.counts()
        stop = {"budget": "예산을 채움", "gain": "더 볼 새 객체 없음", "exhausted": "남은 후보가 간격 안",
                "none": "검출 있는 후보 없음"}.get(res.stop, res.stop)
        print(f"  후보 {len(cands)}장 · 버림 {dtxt}")
        print(f"  고름 {len(res.picks)}장 / 최대 {res.budget}장 (새 객체 {c['main']}/{res.budgets['main']} — {stop}, "
              f"빈 화면 {c['background']}/{res.budgets['background']}, 탐색 {c['explore']}/{res.budgets['explore']})")
        novel: dict[int, int] = {}
        for p in res.picks:
            for k, v in p.novel.items():
                novel[k] = novel.get(k, 0) + v
        if novel:
            print("  새 객체: " + ", ".join(f"{self.classes[k]} {v}" for k, v in sorted(novel.items(), key=lambda x: -x[1])))


def settled(videos: list[Path], fresh: float) -> list[Path]:
    """--watch 없이 돌 때 — 방금 바뀐 파일(내려받기·복사 직후, 녹화 중인 조각)은 크기·수정 시각이 fresh 초 동안 그대로일
    때까지 기다린다(최대 3배). 그래도 바뀌거나 비어 있으면 건너뛴다 — 녹화 폴더를 계속 볼 때는 --watch."""
    def key(v: Path):
        try:
            st = v.stat()
        except OSError:
            return None
        return st.st_size, st.st_mtime_ns, st.st_mtime

    keys = {v: key(v) for v in videos}
    wait = [v for v, k in keys.items() if k and (k[0] == 0 or time.time() - k[2] < fresh)]
    if wait:
        print(f"방금 바뀐 영상 {len(wait)}개 — {fresh:g}초 동안 그대로인지 봅니다(최대 {fresh * 3:g}초): "
              + ", ".join(v.name for v in wait[:3]) + (" …" if len(wait) > 3 else ""))
        end = time.time() + fresh * 3
        while wait and time.time() < end:
            time.sleep(2.0)
            still = []
            for v in wait:
                k = key(v)
                if k and k[:2] == keys[v][:2] and k[0] > 0 and time.time() - k[2] >= fresh:
                    continue                              # 다 썼다
                keys[v] = k
                still.append(v)
            wait = still
        for v in wait:
            print(f"{v.name} — 비었거나 아직 바뀌는 중이라 건너뜁니다(녹화 중이면 --watch 로)")
    return [v for v in videos if keys.get(v) and v not in wait]


def _prev_match(cands: list[Cand], match_iou: float) -> dict[int, list[float | None]]:
    """후보마다, 검출별로 바로 앞 후보(번호 순)에서 IoU ≥ match_iou 인 검출들과의 외형 차이 최솟값."""
    out: dict[int, list[float | None]] = {}
    prev: Cand | None = None
    for c in sorted(cands, key=lambda x: x.i):
        if c.same is not None:
            continue
        vals: list[float | None] = [None] * len(c.dets)
        if prev is not None and prev.dets and c.dets:
            iou = autopick._iou(np.array([d.box for d in c.dets], np.float32),
                                np.array([d.box for d in prev.dets], np.float32))
            for k, d in enumerate(c.dets):
                m = np.flatnonzero(iou[k] >= match_iou)
                if m.size:
                    vals[k] = round(min(float(np.abs(d.sig.astype(np.int16) - prev.dets[j].sig.astype(np.int16)).mean())
                                        for j in m), 1)
        out[c.i] = vals
        prev = c
    return out


def _count(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


# ───────────────────────────── main ─────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path, help="영상 파일 또는 폴더(하위 폴더까지)")
    g = ap.add_argument_group("넣기")
    g.add_argument("--apply", action="store_true", help="고른 것을 바로 넣는다(스캔까지). 없으면 탭에 표시만")
    g.add_argument("--dry-run", action="store_true", help="고르기만 하고 세션을 닫는다(서버에 아무것도 남기지 않음)")
    g.add_argument("--camera", help="cameras.yaml 의 키 — 없으면 영상 이름에서, 그래도 없으면 넣을 때 자동 카메라")
    g.add_argument("--start", help="영상 첫 프레임의 현지 시각(영상 하나일 때만) — 없으면 이름에서, 그래도 없으면 시각 없는 이름")
    g.add_argument("--no-label", action="store_true", help="--apply 뒤 자동 라벨링 작업을 걸지 않는다")
    g.add_argument("--tiles", action="store_true", help="자동 라벨링을 타일 추론으로")
    g.add_argument("--redo", action="store_true", help="처리 기록에 있는 영상도 다시")
    g = ap.add_argument_group("고르기 (src/autolabel/autopick.py)")
    g.add_argument("--every", type=float, default=1.0, help="후보 간격(초, 기본 1)")
    g.add_argument("--per-min", type=float, default=2.0, help="영상 1분당 최대 장수(기본 2)")
    g.add_argument("--max", type=int, default=300, help="영상 하나에서 최대 장수(기본 300)")
    g.add_argument("--min", type=int, default=1, help="짧은 영상도 이만큼은(기본 1)")
    g.add_argument("--gap", type=float, default=20.0,
                   help="고른 프레임 사이 최소 간격(초, 기본 20 — 1분에 60/gap 장을 넘게 못 고르니 --per-min 을 올리면 함께 줄일 것)")
    g.add_argument("--det-conf", type=float, default=0.10, help="고르기용 검출 신뢰도 하한(기본 0.10)")
    g.add_argument("--match-iou", type=float, default=0.5, help="같은 객체로 볼 IoU(기본 0.5)")
    g.add_argument("--match-sig", type=float, default=12.0, help="같은 객체로 볼 외형 차이 0~255(기본 12)")
    g.add_argument("--uncertain-bonus", type=float, default=1.0, help="애매한 검출 덧값(기본 1 — 최대 2배)")
    g.add_argument("--min-gain", type=float, default=1.0, help="본 고르기를 멈추는 새 객체 값(기본 1)")
    g.add_argument("--background", type=float, default=0.05, help="빈 화면 몫(기본 0.05)")
    g.add_argument("--explore", type=float, default=0.10, help="탐색(무작위) 몫(기본 0.10)")
    g.add_argument("--explore-change", type=float, default=0.3, help="빈 화면·탐색이 고른 것과 달라야 할 바뀐 픽셀 %%(기본 0.3)")
    g.add_argument("--blur-ratio", type=float, default=0.35, help="선명도가 영상 중앙값의 이 배 미만이면 흐림(기본 0.35, 0=끔)")
    g.add_argument("--weight", action="append", default=[], metavar="클래스=값",
                   help="클래스 가중치 덮어쓰기 (예: responder=8, person=0.5). 기본은 DB 박스 수로 — 적은 클래스일수록 크게")
    g = ap.add_argument_group("실행")
    g.add_argument("--server", help="검수 서버 주소(기본: config 의 server.host·port)")
    g.add_argument("--project", help="탭 서버의 프로젝트 이름(기본: 첫 프로젝트)")
    g.add_argument("--config", help="설정 파일(탭 없이 --config 로 띄운 서버일 때)")
    g.add_argument("--device", default="auto", help="auto(기본 — 서버 GPU 작업 중이면 cpu) · cpu · 0")
    g.add_argument("--recheck", type=float, default=20.0, help="GPU 작업을 다시 확인하는 간격(초)")
    g.add_argument("--batch", type=int, default=8, help="검출 묶음 크기(기본 8)")
    g.add_argument("--sheet", action="store_true", help="고른 프레임 대조 시트를 남긴다")
    g.add_argument("--watch", action="store_true", help="폴더를 계속 보며 다 쓴 영상이 생길 때마다 처리(Ctrl+C 로 끝)")
    g.add_argument("--settle", type=float, default=120.0, help="--watch: 이 초 동안 안 바뀐 파일만 다 쓴 것으로(기본 120)")
    g.add_argument("--poll", type=float, default=30.0, help="--watch: 폴더를 다시 보는 간격(초, 기본 30)")
    a = ap.parse_args()

    if a.apply and a.dry_run:
        ap.error("--apply 와 --dry-run 은 함께 쓸 수 없습니다")
    if not (0 <= a.background and 0 <= a.explore and a.background + a.explore <= 0.5):
        ap.error("--background·--explore 는 0 이상, 합이 0.5 이하여야 합니다")
    if a.every <= 0 or a.per_min <= 0 or a.max < 1 or a.min < 0 or a.gap < 0:
        ap.error("--every·--per-min 은 양수, --max 는 1 이상, --min·--gap 은 0 이상이어야 합니다")

    base_cfg = Config.load(a.config)
    srv = Server(a.server or f"http://{base_cfg.get('server.host', '127.0.0.1')}:{int(base_cfg.get('server.port', 7860))}")
    try:
        name = srv.connect(a.project)
    except ServerDown as exc:
        print(f"{exc}\n검수 서버를 먼저 켜세요 (run.bat 또는 labeler.py review).")
        return 1
    if name is None:
        cfg = base_cfg
    else:
        from autolabel.projects import load_projects

        hit = [p for p in load_projects() if p.name == name]
        if a.config:
            cfg = base_cfg
        elif hit:
            cfg = hit[0].cfg
        else:
            raise SystemExit(f"projects.yaml 에 '{name}' 이 없습니다 — --config 로 설정을 주세요.")

    state = srv.api("GET", "state")
    # DB 클래스(첫 스캔 전이면 비어 있다 — 그때는 스캔이 정할 config 의 classes)
    classes: list[str] = state["classes"] or [str(c) for c in (cfg.get("classes") or [])]
    if not classes:
        print("클래스 목록이 비어 있습니다 — config 의 classes 를 채우거나 '폴더 스캔' 을 먼저 하세요.")
        return 1
    counts = [int((cb or {}).get("total", 0)) for cb in state.get("class_boxes") or []]
    weights = autopick.class_weights(counts + [0] * (len(classes) - len(counts)))
    for item in a.weight:
        k, _, v = item.partition("=")
        if k not in classes:
            ap.error(f"--weight {item}: '{k}' 는 프로젝트 클래스가 아닙니다 ({', '.join(classes)})")
        weights[classes.index(k)] = float(v)
    cap = srv.api("GET", "capture")
    if cap.get("camera_error"):
        print(f"카메라 등록 파일 오류 — {cap['camera_error']}")
        return 1
    if a.camera and a.camera not in {c["key"] for c in cap.get("cameras") or []}:
        ap.error(f"--camera {a.camera}: cameras.yaml 에 없습니다 — '영상 캡처' 탭의 '＋ 새로' 로 먼저 등록하세요")

    videos = find_videos(a.inputs)
    if a.start and (a.watch or len(videos) != 1):
        ap.error("--start 는 영상 하나일 때만 씁니다(여러 영상은 이름의 시각으로)")
    if not videos and not a.watch:
        print("영상이 없습니다 — " + ", ".join(str(p) for p in a.inputs))
        return 1

    det = Detector(cfg, classes, a.det_conf, a.device, srv, a.recheck)
    det.ensure()                     # 가중치·클래스 대응을 영상을 올리기 전에 확인한다
    manifest = Manifest(cfg.db.parent / "autocapture" / "videos.jsonl")
    job = Job(a, cfg, srv, det, manifest, classes, weights, cap.get("limits") or {})
    wtxt = ", ".join(f"{c} {w:.2g}" for c, w in zip(classes, weights) if abs(w - 1) > 1e-9)
    mode = "보기만(--dry-run)" if a.dry_run else ("넣기(--apply)" if a.apply else "탭에 표시만")
    known = sorted(set(det.cmap.values()))
    unknown = [c for i, c in enumerate(classes) if i not in known]
    print(f"서버 {srv.base}{srv.prefix or ''} · 가중치 {Path(det.weights).name}({det.device}) · {mode}"
          + (f" · 클래스 가중치 {wtxt}" if wtxt else "")
          + (f"\n  이 가중치가 모르는 클래스: {', '.join(unknown)} — 그 객체는 탐색 몫으로만 들어온다" if unknown else ""))

    def label_if_needed() -> bool:
        """넣은 것이 있으면 자동 라벨링 작업을 건다. 걸었거나 걸 것이 없으면 True."""
        if not a.apply or a.no_label or job.registered <= 0:
            return True
        det.release()            # 서버의 자동 라벨링이 GPU 를 쓰게 내려놓는다
        try:
            j = srv.api("POST", "jobs/predict", {"tiles": a.tiles})
        except ApiError as exc:
            if exc.status == 409:
                print(f"자동 라벨링을 걸지 못했습니다(다른 작업 중: {exc}) — " +
                      ("다음 확인 때 다시 겁니다" if a.watch else "끝난 뒤 '모델 → 자동 라벨링'"))
                return False
            raise
        print(f"자동 라벨링 작업 #{j['id']} 을 걸었습니다 — 넣은 {job.registered}장 포함 미라벨 전부(검수 UI 에서 진행을 보세요)")
        job.registered = 0
        return True

    try:
        if not a.watch:
            videos = settled(videos, min(a.settle, FRESH))
            for k, v in enumerate(videos, 1):
                job.run(v, k, len(videos))
            label_if_needed()
            return 0
        print(f"폴더를 봅니다(--settle {a.settle:g}초, --poll {a.poll:g}초) — Ctrl+C 로 끝")
        down = False
        last: dict[Path, tuple[int, int]] = {}
        while True:
            # 다 쓴 파일 = 크기가 있고, 지난번 확인 때와 크기·수정 시각이 같고, 마지막으로 바뀐 지 settle 초가 지난 것.
            # 수정 시각만 보면 안 된다 — 2026-11-12 시험에서 ffmpeg 가 mkv 조각을 몇 초씩 버퍼에 쥐고 있어 0바이트 파일을
            # 올렸다(그 뒤 다 쓴 파일을 다시 처리했다). 복사 중인 파일도 크기가 늘어 걸러진다.
            now = time.time()
            ready, seen_now = [], {}
            for v in find_videos(a.inputs):
                try:
                    st = v.stat()
                except OSError:
                    continue
                key = (st.st_size, st.st_mtime_ns)
                seen_now[v] = key
                if st.st_size > 0 and last.get(v) == key and now - st.st_mtime >= a.settle:
                    ready.append(v)
            last = seen_now
            ready = [v for v in ready if (str(v), *last[v]) not in job.seen]
            try:
                for k, v in enumerate(ready, 1):
                    job.run(v, k, len(ready), watch=True)
                label_if_needed()
                if down:
                    print("검수 서버에 다시 닿았습니다.")
                down = False
            except ServerDown as exc:
                if not down:
                    print(f"{exc} — 서버가 다시 켜질 때까지 기다립니다")
                down = True
            time.sleep(a.poll)
    except KeyboardInterrupt:
        print("\n멈췄습니다.")
        label_if_needed()
        return 130
    except ServerDown as exc:
        print(f"\n{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
