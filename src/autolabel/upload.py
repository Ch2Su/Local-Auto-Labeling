"""웹 이미지 올리기 (2026-11-23, 사용자 요청 — 작업 배치) — 브라우저에서 고른 이미지(한 장·여러 장)를 `unlabeled/<카메라>/` 에
넣어 작업 배치 하나로 등록한다. 화면은 `작업 배치` 보기의 `＋ 이미지 넣기`(`web/work.js`).

흐름:
1. `begin(camera, name)` — 카메라(등록된 키, 또는 None = 이 묶음용 자동 카메라)와 배치 이름을 받아 세션을 만든다.
   세션 = `<DB 폴더>/upload/<YYYYMMDD_HHMMSS_hex4>/files/`. **여기까지는 unlabeled/ 에 아무것도 없다.**
2. 파일마다 `reserve` → (서버가 본문을 그 경로에 쓴다) → `accept`. 이름을 다듬고(경로 떼기·Windows 가 못 쓰는 글자),
   이미지로 열리는지 본다(`yolo_io.image_size` — 스캔과 같은 검사). 받는 형식은 `yolo_io.IMG_EXTS` 그대로다.
3. `commit` — 내용이 이미 있는 이미지와 바이트까지 같은 파일은 옮기지 않고(스캔의 중복 검사와 같은 `dupcheck.DupIndex` —
   옮긴 뒤에 막히면 등록 안 된 사본이 unlabeled/ 에 남아 스캔할 때마다 '중복' 으로 다시 뜬다), 나머지를
   `unlabeled/<카메라>/up<세션 시각>/` 로 **옮긴 뒤** 스캔한다. 서버는 그 파일들을 작업 배치 하나로 묶는다(workbatch.py).
   폴더를 올린 묶음마다 따로 두는 것은 이름이 같은 다른 그림(`frame_0001.jpg`)을 '이미 있다' 로 건너뛰지 않게 하려는 것이다.
4. 넣거나 버리면(`discard`) 세션 폴더를 지운다. **서버를 다시 켜면 넣지 않은 세션은 지운다** — 사본일 뿐이고 원본은 올린
   사람에게 그대로 있다(영상 캡처처럼 이어 가지 않는다: 브라우저가 고른 파일을 다시 알려 주지 않는다).

영상 캡처와 같은 약속:
- **출처(카메라) 없이는 넣지 않는다** — 분할이 카메라째라서다. 모르면 이 묶음용 카메라를 자동으로 등록한다(`vid-NN` —
  영상 캡처의 자동 카메라와 같은 번호줄, 장소 '모름 — 이미지 묶음 …'). 묶음 하나가 카메라 하나라, 서로 다른 장소의 그림을
  한 묶음으로 올리면 한 카메라로 묶인다(같은 분할로 간다 — 새는 쪽이 아니라 안전한 쪽). 알게 되면 이미지 보기의
  '옮기기'·영상 캡처 탭의 '고치기…' 로 나눈다.
- 넣기는 영상 캡처의 넣기·카메라 지우기와 같은 잠금(`commit_lock`)을 쥔 채 스캔 잠금을 쥔다(서버가 `ingest` 로 넘긴다).
  정리·휴지통 비우기가 도는 동안은 넣지 않는다(`blocker`).
- 같은 이름이 이미 있으면 덮지 않는다. 장소·파일 이름은 서버 로그에 찍지 않는다.
- 세션 id 는 `SID_RE` 로만 받고 파일은 세션 폴더 안의 다듬은 이름으로만 쓴다(경로 탈출 막기).
"""

from __future__ import annotations

import os
import re
import secrets
import shutil
import threading
import time
import unicodedata
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from PIL import Image

from .capture import MIN_FREE, SID_RE, _rename, _rmtree, next_auto_key, taken_camera_keys
from .dupcheck import DupIndex
from .provenance import CameraExists, add_camera, camera_fields, load_cameras, remove_cameras
from .yolo_io import IMG_EXTS, image_size

MAX_FILES = 5000                 # 한 묶음의 최대 장수 — 폴더를 잘못 끌어다 놓은 실수로 디스크를 채우지 않게
MAX_FILE_BYTES = 200 * 1024 ** 2  # 한 장의 최대 크기
NAME_MAX = 150                   # 다듬은 이름의 줄기(확장자 앞) 길이
STALE_S = 6 * 3600               # 이만큼 손대지 않은 세션은 새 묶음을 시작할 때 걷어 낸다(창을 닫아 버린 묶음)

_BAD = re.compile(r'[<>:"/\\|?*\x00-\x1f\x7f]')
_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}
# 이미지를 열거나 EXIF 를 읽다 나는 예외 (server._UNREADABLE 과 같다)
_UNREADABLE = (OSError, SyntaxError, ValueError, Image.DecompressionBombError)


class UploadError(Exception):
    """화면에 그대로 보일 오류. status 는 HTTP 상태 코드."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def clean_name(name: str) -> str:
    """올린 파일 이름을 폴더 안에 둘 이름으로 다듬는다 — 화면의 미리 보기(work.js `upCleanName`)와 같은 규칙.

    경로를 뗀다(폴더째 올리면 브라우저가 `a/b/c.jpg` 를 준다 · `..\\..\\x.jpg` 같은 이름도 마지막 조각만). Windows 가 파일
    이름에 못 쓰는 글자(`<>:"/\\|?*`·제어 문자)는 `_`, 끝의 점·공백은 뗀다. 한글은 NFC 로 모은다(macOS 는 자모로 풀어 준다).
    확장자가 `yolo_io.IMG_EXTS` 에 없으면 UploadError(415) — 스캔이 조용히 건너뛸 파일을 올리지 않게.
    이름의 시각(`…_YYYYMMDD_HHMMSS`)은 건드리지 않는다 — 스캔이 촬영 시각으로 읽는다.
    """
    base = unicodedata.normalize("NFC", str(name or "")).replace("\\", "/").rsplit("/", 1)[-1]
    base = _BAD.sub("_", base).strip().rstrip(". ")
    stem, dot, ext = base.rpartition(".")
    if not dot or ("." + ext.lower()) not in IMG_EXTS:
        raise UploadError(415, f"받지 않는 형식입니다: {base[:80] or '(이름 없음)'} — 이미지({' '.join(sorted(IMG_EXTS))})만 넣을 수 있습니다.")
    stem = stem.strip().rstrip(". ")[:NAME_MAX].rstrip(". ") or "image"
    if stem.lower() in _RESERVED:
        stem = "_" + stem
    return f"{stem}.{ext}"


class Session:
    def __init__(self, root: Path, sid: str, camera: str | None, name: str | None):
        self.sid = sid
        self.dir = root / sid
        self.files_dir = self.dir / "files"
        self.camera = camera               # None = 넣을 때 이 묶음용 카메라를 자동으로 등록한다
        self.name = name                   # 작업 배치 이름 (없으면 화면이 종류·시각으로 적는다)
        self.files: list[dict] = []        # 받은 것 [{name, size, width, height}]
        self.reserved: set[str] = set()    # 쓰는 중이거나 받은 이름(소문자) — 같은 이름을 둘이 쓰지 않게
        self.inflight: set[str] = set()    # 그 가운데 아직 쓰는 중인 이름(reserve ~ accept·reject) — 있으면 넣지 않는다
        self.state = "open"                # open | committing
        self.deleted = False
        self.created = datetime.now()
        self.touched = time.monotonic()
        self.lock = threading.Lock()

    def summary(self) -> dict:
        with self.lock:
            return {"sid": self.sid, "camera": self.camera, "name": self.name, "state": self.state,
                    "created": self.created.isoformat(timespec="seconds"),
                    "n": len(self.files), "bytes": sum(f["size"] for f in self.files),
                    "receiving": len(self.inflight)}


class UploadManager:
    """한 프로젝트의 이미지 올리기 세션들 (`<DB 폴더>/upload/`). 서버가 프로젝트마다 하나 만든다.

    ingest: 스캔 — `ingest(mine=[옮긴 파일 경로], note=…, name=배치 이름)`, 잠금을 쥐고 `workbatch.ingest_as_batch` 를 부른다.
    blocker: 넣으면 안 되는 이유(정리·휴지통 비우기 중) 또는 None. commit_lock: 영상 캡처의 넣기와 같은 잠금.
    dup_rows: 중복을 볼 DB 이미지 줄 [{id, path, …}] (없으면 중복을 미리 보지 않는다 — 스캔이 막는다).
    used_cameras: DB 이미지가 쓰는 카메라 키.
    """

    def __init__(self, cfg, *, ingest: Callable[..., object], blocker: Callable[[], str | None] | None = None,
                 commit_lock: threading.Lock | None = None, dup_rows: Callable[[], list[dict]] | None = None,
                 used_cameras: Callable[[], set[str]] | None = None):
        self.cfg = cfg
        self.root = cfg.db.parent / "upload"
        self.ingest = ingest
        self.blocker = blocker or (lambda: None)
        self.commit_lock = commit_lock or threading.Lock()
        self.dup_rows = dup_rows
        self.used_cameras = used_cameras
        self.sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._sweep()

    def _sweep(self) -> None:
        """서버를 켤 때: 지난번에 넣지 않은 세션 폴더를 지운다(그 이름 꼴의 폴더만 — 사본뿐이다)."""
        if not self.root.is_dir():
            return
        for d in sorted(self.root.iterdir()):
            if d.is_dir() and SID_RE.match(d.name):
                print(f"[올리기] 넣지 않은 묶음을 지웁니다: {d.name}")
                shutil.rmtree(d, ignore_errors=True)

    # ----------------------------------------------------------------- 세션

    def list(self) -> list[dict]:
        with self._lock:
            items = list(self.sessions.values())
        return [s.summary() for s in sorted(items, key=lambda s: s.created, reverse=True)]

    def get(self, sid: str) -> Session:
        if not SID_RE.match(str(sid)):
            raise UploadError(404, "그런 묶음이 없습니다.")
        with self._lock:
            s = self.sessions.get(sid)
        if s is None:
            raise UploadError(404, "그런 묶음이 없습니다 — 이미 넣었거나 버렸을 수 있습니다.")
        return s

    def _cameras(self) -> dict:
        try:
            return load_cameras(self.cfg.cameras)
        except ValueError as exc:
            raise UploadError(400, f"카메라 등록 파일을 읽지 못했습니다:\n{exc}") from exc

    def begin(self, camera: str | None, name: str | None = None) -> dict:
        """묶음 하나를 시작한다. camera 는 등록된 키이거나 None(자동 카메라 — 등록은 넣을 때 한다: 버린 묶음이 빈 등록을
        남기지 않게). 넣을 수 없는 때(정리 중·없는 카메라)는 파일을 받기 전에 여기서 알린다."""
        from .store import clean_work_name

        why = self.blocker()
        if why:
            raise UploadError(409, why)
        camera = (camera or "").strip() or None
        if camera is not None and camera not in self._cameras():
            raise UploadError(400, f"카메라 '{camera}' 가 {self.cfg.cameras.name} 에 없습니다 — 먼저 등록하세요"
                                   " (영상 캡처 탭의 '＋ 새로').")
        try:
            name = clean_work_name(name)
        except ValueError as exc:
            raise UploadError(400, str(exc)) from None
        self.root.mkdir(parents=True, exist_ok=True)
        self._drop_stale()
        for _ in range(20):
            sid = f"{datetime.now():%Y%m%d_%H%M%S}_{secrets.token_hex(2)}"
            s = Session(self.root, sid, camera, name)
            try:
                s.files_dir.mkdir(parents=True)
            except FileExistsError:
                continue
            with self._lock:
                self.sessions[sid] = s
            return {**s.summary(), "limits": {"max_files": MAX_FILES, "max_bytes": MAX_FILE_BYTES}}
        raise UploadError(500, "묶음 폴더를 만들지 못했습니다 — 다시 해 보세요.")

    def _drop_stale(self) -> None:
        """오래 손대지 않은 열린 세션을 걷어 낸다 — 올리다 창을 닫아 버린 묶음이 서버를 다시 켤 때까지 남지 않게."""
        now = time.monotonic()
        with self._lock:
            old = [s for s in self.sessions.values() if s.state == "open" and now - s.touched > STALE_S]
        for s in old:
            try:
                self.discard(s.sid)
            except UploadError:
                pass

    # ----------------------------------------------------------------- 파일 받기

    def reserve(self, sid: str, name: str, size: int | None) -> tuple[Session, Path, str]:
        """파일 하나를 받을 자리 — (세션, 쓸 경로, 다듬은 이름). 이름이 이 묶음에 이미 있으면 `_2`, `_3` … 을 붙인다
        (폴더째 올리면 다른 하위 폴더의 같은 이름이 온다). 부르는 쪽이 그 경로에 쓰고 `accept` 나 `reject` 를 부른다."""
        s = self.get(sid)
        final = clean_name(name)
        if size is not None and size < 0:
            raise UploadError(400, "크기가 잘못됐습니다.")
        if size is not None and size > MAX_FILE_BYTES:
            raise UploadError(413, f"{final}: 한 장은 {MAX_FILE_BYTES // 1024 ** 2}MB 까지입니다.")
        if size is not None and shutil.disk_usage(self.root).free - size < MIN_FREE:
            raise UploadError(507, "디스크에 남은 공간이 모자랍니다.")
        with s.lock:
            if s.deleted or s.state != "open":
                raise UploadError(409, "이 묶음은 넣는 중이거나 이미 닫혔습니다.")
            if len(s.reserved) >= MAX_FILES:
                raise UploadError(413, f"한 묶음은 {MAX_FILES:,}장까지입니다 — 나눠서 넣으세요.")
            stem, ext = final.rsplit(".", 1)
            k = 1
            while final.lower() in s.reserved:
                k += 1
                final = f"{stem}_{k}.{ext}"
            s.reserved.add(final.lower())
            s.inflight.add(final.lower())
            s.touched = time.monotonic()
        return s, s.files_dir / final, final

    def reject(self, s: Session, path: Path) -> None:
        """받다 만 파일을 지우고 자리를 돌려준다. **자리를 잡은(`reserve`) 쪽은 어떻게 끝나든 `accept` 나 이것을 부른다** —
        안 부르면 그 이름이 '쓰는 중' 으로 남아 이 묶음을 넣을 수 없다(commit 의 409)."""
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        with s.lock:
            s.reserved.discard(path.name.lower())
            s.inflight.discard(path.name.lower())

    def accept(self, s: Session, path: Path, written: int, size: int | None) -> dict:
        """다 쓴 파일을 확인하고 묶음에 넣는다. 크기가 다르거나 이미지로 열리지 않으면 지우고 UploadError."""
        if not written or (size is not None and written != size):
            self.reject(s, path)
            raise UploadError(400, f"{path.name}: 받은 크기가 다릅니다({written:,} / {size:,} 바이트) — 다시 올리세요."
                              if written else f"{path.name}: 빈 파일입니다.")
        if written > MAX_FILE_BYTES:
            self.reject(s, path)
            raise UploadError(413, f"{path.name}: 한 장은 {MAX_FILE_BYTES // 1024 ** 2}MB 까지입니다.")
        try:
            w, h = image_size(path)
        except _UNREADABLE as exc:
            self.reject(s, path)
            raise UploadError(415, f"{path.name}: 이미지로 읽을 수 없습니다 ({type(exc).__name__}).") from None
        except BaseException:
            self.reject(s, path)                     # 뜻밖의 오류에도 자리는 돌려준다('쓰는 중' 으로 남지 않게)
            raise
        item = {"name": path.name, "size": written, "width": int(w), "height": int(h)}
        with s.lock:
            if s.deleted or s.state != "open":       # 쓰는 사이 버렸다(넣기는 쓰는 중인 파일이 있으면 시작하지 않는다)
                gone = True
            else:
                gone = False
                s.files.append(item)
                s.inflight.discard(path.name.lower())
                s.touched = time.monotonic()
        if gone:
            # 자리도 돌려준다 — 2026-11-25 전에는 파일만 지워서, 넣기가 실패해 묶음이 다시 열리면(reopen) 그 이름이
            # 예약된 채 남았다(같은 이름을 다시 올리면 `_2` 가 붙고 장수 한도를 한 자리 먹는다)
            self.reject(s, path)
            raise UploadError(409, "이 묶음은 넣는 중이거나 이미 닫혔습니다.")
        return item

    # ----------------------------------------------------------------- 넣기

    def _auto_camera(self, s: Session, cams: dict, n: int, first: str) -> str:
        """이 묶음용 카메라를 다음 번호 `vid-NN` 으로 등록한다(영상 캡처의 자동 카메라와 같은 번호줄). 넣기 잠금을 쥔 채 부른다."""
        fields = camera_fields(self.cfg.cameras)
        taken = taken_camera_keys(self.cfg, cams, fields, self.used_cameras() if self.used_cameras else ())
        first = re.sub(r"[\x00-\x1f\x7f]", " ", first).strip()[:80] or "?"
        attrs = {"kind": "cctv", "place": f"모름 — 이미지 묶음 {first}" + (f" 외 {n - 1}장" if n > 1 else ""),
                 "note": f"이미지 올리기 자동 등록 — 묶음 {s.sid}"}
        for _ in range(20):
            key = next_auto_key(taken)
            try:
                add_camera(self.cfg.cameras, key, attrs, backup_dir=self.cfg.db.parent / "backups",
                           comment=f"{datetime.now():%Y-%m-%d %H:%M} 검수 UI '이미지 넣기' 가 자동 등록(카메라를 고르지 않고 넣음)")
            except CameraExists:
                taken.add(key)               # 그사이 다른 넣기가 같은 키를 등록했다 — 다음 번호로
                continue
            except ValueError as exc:
                raise UploadError(400, f"이 묶음용 카메라를 등록하지 못했습니다: {exc}") from None
            print(f"[올리기] 카메라를 고르지 않은 묶음 — {key} 를 자동 등록 ({self.cfg.cameras.name})")
            return key
        raise UploadError(409, "이 묶음용 카메라 키를 정하지 못했습니다 — 다시 넣어 보세요.")

    def _drop_auto(self, key: str) -> None:
        try:
            remove_cameras(self.cfg.cameras, [key], backup_dir=self.cfg.db.parent / "backups")
        except (ValueError, OSError) as exc:
            print(f"[올리기] 자동 등록한 카메라 {key} 를 되돌리지 못했습니다(빈 등록이 남음): {exc}")

    def _duplicates(self, s: Session, files: list[dict]) -> dict[str, dict]:
        """받은 파일 가운데 내용(바이트)이 이미 있는 이미지나 같은 묶음의 앞 파일과 같은 것 — {이름: 같은 줄}.
        옮기기 **전에** 전부 본다(DupIndex 는 견줄 때 그 경로의 파일을 읽는다)."""
        if self.dup_rows is None or bool(self.cfg.get("data.allow_duplicates", False)):
            return {}
        idx = DupIndex(self.dup_rows())
        out: dict[str, dict] = {}
        for f in files:
            path = str(s.files_dir / f["name"])
            hit = idx.find(path)
            if hit is not None:
                out[f["name"]] = hit
            else:
                idx.add(path, {"id": None, "path": path, "name": f["name"]})
        return out

    def commit(self, sid: str) -> dict:
        """받은 파일을 unlabeled/<카메라>/up<세션 시각>/ 로 옮기고 스캔한다(작업 배치 하나). 결과 dict 를 돌려준다."""
        s = self.get(sid)

        def receiving() -> None:
            """쓰는 중인 파일이 있으면 넣지 않는다 (2026-11-25, Gemini 리뷰 묶음 6). 전에는 다 받은 것만 옮기고 세션 폴더를
            지웠다 — 쓰던 파일은 묶음에서 빠지고(그 요청은 409), Windows 에서는 열린 파일 때문에 폴더도 다 못 지웠다.
            s.lock 을 쥔 채 부른다."""
            if s.inflight:
                raise UploadError(409, f"아직 받는 중인 파일이 {len(s.inflight)}개 있습니다 — 다 받은 뒤 넣으세요.")

        with self.commit_lock:
            with s.lock:
                if s.deleted:
                    raise UploadError(404, "이 묶음은 이미 넣었거나 버렸습니다.")
                if s.state != "open":
                    raise UploadError(409, "넣는 중입니다.")
                receiving()
                if not s.files:
                    raise UploadError(400, "넣을 이미지가 없습니다.")
            why = self.blocker()
            if why:
                raise UploadError(409, why)
            cams = self._cameras()
            camera = s.camera
            if camera is not None and camera not in cams:
                raise UploadError(400, f"카메라 '{camera}' 가 {self.cfg.cameras.name} 에 없습니다 — 그사이 지워졌습니다.")
            with s.lock:
                if s.deleted or s.state != "open":
                    raise UploadError(409, "그사이 넣기를 시작했거나 버렸습니다.")
                receiving()                      # 위의 확인 뒤에 새로 받기 시작한 파일
                # 넣을 목록은 상태를 바꾸는 바로 이 자리에서 뜬다 — 위에서 떠 두면 그사이 다 받은 파일이 목록에서 빠진 채
                # (받았다는 응답은 이미 나갔다) 세션 폴더와 함께 지워진다
                files = [dict(f) for f in s.files]
                s.state = "committing"           # 그동안 파일을 더 받거나 버리지 않게

            def reopen(note: str | None = None) -> None:
                with s.lock:
                    s.state = "open"
                    s.touched = time.monotonic()

            try:
                dups = self._duplicates(s, files)
            except BaseException:
                reopen()
                raise
            todo = [f for f in files if f["name"] not in dups]
            auto = None
            if camera is None and todo:
                try:
                    camera = self._auto_camera(s, cams, len(todo), todo[0]["name"])
                except BaseException:
                    reopen()
                    raise
                auto = "new"

            out: dict = {"camera": camera, "auto": auto, "files": len(files), "moved": 0, "exists": 0, "missing": 0,
                         "duplicates": len(dups),
                         "duplicate_names": [{"name": n, "same_as": h.get("id")} for n, h in list(dups.items())[:5]]}
            if not todo:                          # 전부 이미 있는 그림이다 — 옮길 것도, 등록할 카메라도 없다
                out["left"] = self.discard(sid, force=True)
                return out

            dest = self.cfg.unlabeled / camera / f"up{sid[:15]}"
            moved: list[Path] = []
            mine: list[Path] = []                 # 이번 묶음의 파일 — 옮긴 것 + 앞선 넣기가 옮기다 멈춘 것
            exists: list[str] = []
            try:
                for f in todo:
                    src, dst = s.files_dir / f["name"], dest / f["name"]
                    if not src.exists():
                        if dst.exists():
                            mine.append(dst)      # 앞에서 옮기다 멈춘 넣기를 다시 하는 중이다
                        else:
                            out["missing"] += 1
                        continue
                    if dst.exists():
                        exists.append(dst.name)
                        continue
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        _rename(src, dst)         # 같은 디스크라 옮기기만 한다(덮어쓰지 않는다 — capture._rename)
                    except FileExistsError:
                        exists.append(dst.name)
                        continue
                    moved.append(dst)
                    mine.append(dst)
            except OSError as exc:
                if auto == "new" and not mine:
                    self._drop_auto(camera)
                reopen()
                raise UploadError(500, f"{len(moved)}장을 옮긴 뒤 멈췄습니다 ({dest}): {exc} — 다시 넣으면 남은 것만 옮깁니다.") from exc
            if auto == "new" and not mine:
                self._drop_auto(camera)           # 옮긴 것이 없다 — 빈 등록을 남기지 않는다
                out["auto"] = "dropped"
            out.update(moved=len(moved), exists=len(exists), exists_names=exists[:5], dest=str(dest))

            if mine:
                try:
                    report = self.ingest(mine=mine, note=f"이미지 {len(mine)}장", name=s.name)
                except Exception as exc:  # noqa: BLE001 — 파일은 이미 옮겼다. 사람이 고친 뒤 '폴더 스캔' 하면 된다
                    out["scan_error"] = f"{type(exc).__name__}: {exc}"
                else:
                    keys = {os.path.normcase(str(p.resolve())) for p in mine}
                    out["work"] = getattr(report, "work", None)
                    main = (out["work"] or {}).get("main")
                    out["registered"] = main["n"] if main else 0
                    out["scan_duplicates"] = len([d for d in report.duplicates if os.path.normcase(d["path"]) in keys])
                    out["summary"] = report.summary()
            out["left"] = self.discard(sid, force=True)
            return out

    def discard(self, sid: str, *, force: bool = False) -> str | None:
        """묶음을 버리고 세션 폴더(받은 사본)를 지운다. 다 못 지웠으면 그 이유를 돌려준다."""
        s = self.get(sid)
        with s.lock:
            if not force and s.state == "committing":
                raise UploadError(409, "넣는 중입니다.")
            s.deleted = True
        with self._lock:
            self.sessions.pop(sid, None)
        errors = _rmtree(s.dir)
        if errors:
            return f"임시 폴더를 다 지우지 못했습니다({len(errors)}개 — {s.dir}): {errors[0]}"
        return None
