"""검수 웹 UI 서버 (FastAPI).

이미지는 항상 DB에 등록된 id로만 서빙한다. 클라이언트가 보낸 경로를 그대로
열지 않으므로 경로 탈출(path traversal) 여지가 없다.
"""

from __future__ import annotations

import math
import os
import re
import secrets
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field
from starlette.datastructures import Headers
from starlette.responses import JSONResponse, PlainTextResponse

from .config import ROOT, Config, resolve
from .export import default_export_dir, export_yolo
from .ingest import discover_classes, ingest
from .jobs import JobManager
from .predictor import AutoLabeler, LiveDetector
from .masks import MAX_MASKS
from .masks import clean as clean_masks
from .provenance import SPLITS, load_cameras
from .scoring import image_uncertainty
from . import wsl_train
from .segmenter import Segmenter, SegmenterUnavailable
from .stopper import StopSignal
from .store import DOMAINS, NO_CAMERA, NO_DOMAIN, NO_SPLIT, NO_TAG, Store, TagFilter, TemplateExists
from .tidy import prune_missing, purge_trashed, tidy_approved
from .trainer import find_resume_point
from .trainer import resume_train as run_resume
from .trainer import train as run_train
from .yolo_io import Box, display_size

WEB_DIR = Path(__file__).parent / "web"

# 이미지를 열거나 EXIF 를 읽다 나는 예외 (2026-11-05). OSError(열 수 없는 파일)만 잡았더니 손상된 EXIF 는
# SyntaxError('not a TIFF file'), 아주 큰 이미지는 DecompressionBombError, 일부 손상은 ValueError 로 나와 500 이 됐다.
_UNREADABLE = (OSError, SyntaxError, ValueError, Image.DecompressionBombError)


class BoxIn(BaseModel):
    # NaN·Infinity 는 422 (2026-11-05). FastAPI 는 본문을 json.loads 로 읽어 NaN 리터럴을 받고, 그 박스는
    # Box.clamped 에서 '이미지 전체' 박스로 둔갑했다.
    model_config = ConfigDict(allow_inf_nan=False)

    cls: int = Field(ge=0)
    cx: float
    cy: float
    w: float = Field(gt=0)
    h: float = Field(gt=0)
    conf: float | None = None
    origin: str = "human"


class MaskIn(BaseModel):
    # 가릴 영역 하나 — 정규화 xyxy (2026-11-09, masks.py). NaN·Infinity 는 박스처럼 422.
    model_config = ConfigDict(allow_inf_nan=False)

    x1: float
    y1: float
    x2: float
    y2: float


class BoxesIn(BaseModel):
    boxes: list[BoxIn]
    # None(옛 화면·영역을 손대지 않은 저장)이면 영역은 그대로 둔다. [] 면 모두 지운다.
    masks: list[MaskIn] | None = None


class StatusIn(BaseModel):
    status: str


class DomainIn(BaseModel):
    domain: str | None          # cctv | noncctv | drone, None = 미분류로 되돌림


class TagIn(BaseModel):
    name: str


class ImageTagIn(BaseModel):
    tag_id: int
    on: bool = True             # True 붙이기, False 떼기


class BulkTagIn(BaseModel):     # 이미지 보기 탭에서 여러 장을 골라 한꺼번에 (2026-11-04)
    ids: list[int]
    tag_id: int
    on: bool = True


class BulkDomainIn(BaseModel):
    ids: list[int]
    domain: str | None


class MaskTemplateIn(BaseModel):    # 가릴 영역 템플릿 (2026-11-10) — 지금 이미지의 영역을 이름으로 남긴다
    name: str
    masks: list[MaskIn]
    width: int | None = Field(default=None, gt=0)       # 만든 이미지의 화면 크기 — 비율이 다른 그림을 가린다
    height: int | None = Field(default=None, gt=0)
    image_id: int | None = None
    overwrite: bool = False                             # 같은 이름이 있으면 409, true 면 덮어쓴다


class TemplateNameIn(BaseModel):
    name: str


class BulkMaskTemplateIn(BaseModel):
    ids: list[int]
    template_id: int
    mode: Literal["add", "remove"] = "add"


class ClassIn(BaseModel):
    name: str


class ClassesIn(BaseModel):
    names: list[str]


class PredictIn(BaseModel):
    # 이미 검수 대기(pending)인 것까지 다시 돌릴지 여부
    include_pending: bool = False
    # 타일 추론으로 돌릴지 (2026-11-10, 메뉴 '자동 라벨링 (타일)' — tiling.py). 기본은 통째 추론
    tiles: bool = False


class TrainIn(BaseModel):
    epochs: int | None = None


class ResumeIn(BaseModel):
    # VRAM 이 모자라 공유 메모리로 페이징될 때 줄여서 이어가기 위한 값.
    batch: int | None = None


class ExportIn(BaseModel):
    out_dir: str | None = None      # 없으면 export/<프로젝트> (export.default_export_dir)
    with_images: bool = False


class RestoreIn(BaseModel):
    # 이력 한 줄의 어느 쪽으로 되돌릴지. 'before' = 그 저장 직전 상태
    which: str = Field("before", pattern="^(before|after)$")


class SamePlaceIn(BaseModel):
    """같은 자리 일괄 수정. box 는 기준 박스의 정규화 (cx, cy, w, h)."""

    model_config = ConfigDict(allow_inf_nan=False)       # 좌표는 BoxIn 과 같이 유한한 수만

    box: list[float] = Field(min_length=4, max_length=4)
    to_cls: int = Field(ge=0)
    from_cls: list[int] | None = None       # None 이면 모든 클래스
    iou: float = Field(0.85, ge=0.3, le=1.0)
    expected: int | None = None             # 적용 때: 미리 보기에서 본 '바꿀 이미지 수'


class BatchIn(BaseModel):
    batch: str


class SegmentIn(BaseModel):
    """SAM 프롬프트. 좌표는 이 API 의 다른 곳과 같이 0~1 정규화 값이다.

    point 만 주면 '클릭 한 번으로 박스', box 만 주면 '대충 그린 박스 다듬기',
    둘 다 주면 박스로 범위를 가두고 점으로 대상을 가리킨다(겹친 객체에 가장 정확).
    """

    model_config = ConfigDict(allow_inf_nan=False)       # 좌표는 BoxIn 과 같이 유한한 수만

    point: list[float] | None = Field(None, min_length=2, max_length=2)
    box: list[float] | None = Field(None, min_length=4, max_length=4)
    cls: int = Field(0, ge=0)


_ID_MAX = 2**63 - 1      # SQLite INTEGER 의 상한


def _to_id(tok: str) -> int | None:
    """ASCII 숫자로만 된 2^63 미만의 수면 그 값, 아니면 None (2026-11-05).

    str.isdigit 은 '²'·'٣' 같은 유니코드 숫자도 참이라 int() 에서 500 이 나거나(²) 다른 수로 읽혔고(٣ → 3),
    2^63 이상은 SQLite 에 넘길 때 OverflowError(500)였다. 자릿수를 먼저 봐서 아주 긴 숫자로 int() 를 부르지 않는다.
    """
    if not (tok.isascii() and tok.isdigit()) or len(tok.lstrip("0")) > 19:
        return None
    v = int(tok)
    return v if v <= _ID_MAX else None


def _id_list(text: str | None, what: str) -> list[int]:
    """'3,7 12' → [3, 7, 12] (겹치면 하나만, 준 순서대로). 숫자가 아닌 조각이 있으면 400."""
    tokens = [t for t in (text or "").replace(",", " ").split() if t]
    ids = [_to_id(t) for t in tokens]
    if any(i is None for i in ids):
        raise HTTPException(400, f"{what}는 번호(0 이상 정수)를 쉼표로 구분해 주세요.")
    return list(dict.fromkeys(ids))


_DASHES = str.maketrans(dict.fromkeys("‐‑‒–—―−~〜", "-"))
IDS_MAX = 10_000


def _id_ranges(text: str) -> list[int]:
    """'id 로 열기' 의 id 목록 — '5532, 5493–5497 4435' (2026-11-05 범위 추가, app.js `parseIds` 와 같은 규칙).

    쉼표·공백·세미콜론·가운뎃점으로 나누고, 범위는 -·–·—·~ 어느 것이든 된다(거꾸로 쓰면 거꾸로).
    넣은 순서대로, 겹치는 id 는 처음 것만. 읽을 수 없는 조각이 있거나 IDS_MAX 개를 넘으면 400.
    """
    norm = re.sub(r"\s*-\s*", "-", unicodedata.normalize("NFKC", text).translate(_DASHES))
    out: dict[int, None] = {}
    for tok in re.split(r"[\s,;·、]+", norm):
        if not tok:
            continue
        # [0-9] 로 받는다 — \d 는 NFKC 뒤에도 남는 유니코드 숫자(٣ 등)까지 받는데 app.js 의 \d 는 ASCII 뿐이다
        m = re.fullmatch(r"([0-9]+)(?:-([0-9]+))?", tok)
        if not m:
            raise HTTPException(400, f"id 로 읽을 수 없는 조각: '{tok}' — 숫자나 범위(5493-5497)를 쉼표·공백으로 구분해 주세요.")
        a = _to_id(m.group(1))
        b = _to_id(m.group(2)) if m.group(2) else a
        if a is None or b is None:
            raise HTTPException(400, f"id 가 너무 큽니다: '{tok}' (2^63 미만)")
        if len(out) + abs(b - a) + 1 > IDS_MAX:
            raise HTTPException(400, f"id 는 한 번에 {IDS_MAX:,}개까지 열 수 있습니다.")
        step = 1 if a <= b else -1
        out.update(dict.fromkeys(range(a, b + step, step)))
    return list(out)


_LOOPBACK = ("127.0.0.1", "localhost", "::1")
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def _host_port(netloc: str, scheme: str) -> tuple[str, int] | None:
    """'127.0.0.1:7860'·'[::1]:7860'·'LOCALHOST' → (호스트 소문자, 포트 — 없으면 scheme 기본값). 못 읽으면 None."""
    try:
        u = urlsplit(f"//{netloc}")
        host, port = u.hostname, u.port
    except ValueError:              # '[::1' 처럼 깨졌거나 포트가 숫자가 아니다
        return None
    if not host:
        return None
    return host, port or (443 if scheme in ("https", "wss") else 80)


class _RequestGuard:
    """다른 사이트에서 부르기(CSRF)·DNS rebinding 을 막는다 (2026-11-05). 바깥 앱에 한 번만 건다.

    - 본문 없는 POST(학습·정리·휴지통 비우기·작업 중단 …)는 브라우저의 'simple request' 라, 아무 사이트의 페이지나
      사전 확인 없이 보내 200 을 받았다. 그래서 GET·HEAD·OPTIONS 가 아닌 요청은 `Sec-Fetch-Site` 가 있으면
      same-origin·none 만, `Origin` 이 있으면 그 host:port 가 요청의 Host 와 같아야 한다(아니면 403). 둘 다 없는 요청
      (urllib 의 tools/scheduled_stop.py, curl)은 브라우저가 아니라 그대로 받는다.
    - DNS rebinding: 남의 도메인을 127.0.0.1 로 돌려 '같은 출처' 로 들어오면 Host 가 그 도메인이다. Host 가 루프백·
      bind 주소가 아니면 400. 0.0.0.0·:: 로 열었으면(다른 PC 에서 접속) Host 는 보지 않는다.
    Starlette 의 TrustedHostMiddleware 는 Host 를 ':' 로 잘라 '[::1]:7860' 을 '[' 로 읽는다(IPv6 루프백이 400) —
    그래서 같은 일(허용 목록 밖이면 400 'Invalid host header')을 여기서 직접 한다.
    """

    def __init__(self, app, bind_host: str | None = None) -> None:
        self.app = app
        bind = (bind_host or "").strip().strip("[]").lower()
        self.allowed = None if bind in ("0.0.0.0", "::") else {*_LOOPBACK, *([bind] if bind else [])}

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        scheme = scope.get("scheme", "http")
        host = _host_port(headers.get("host", ""), scheme)
        if self.allowed is not None and (host is None or host[0] not in self.allowed):
            await PlainTextResponse("Invalid host header", status_code=400)(scope, receive, send)
            return
        if scope["type"] == "http" and scope["method"] not in _SAFE_METHODS:
            why = None
            site = headers.get("sec-fetch-site")
            if site is not None and site not in ("same-origin", "none"):
                why = f"다른 사이트에서 온 요청입니다 (Sec-Fetch-Site: {site})"
            origin = headers.get("origin")
            if why is None and origin is not None:
                o = urlsplit(origin)
                o_hp = _host_port(o.netloc, o.scheme) if o.scheme in ("http", "https") else None
                if o_hp is None or o_hp != host:
                    why = f"다른 출처에서 온 요청입니다 (Origin: {origin[:80]})"
            if why:      # HTTPException 과 같은 꼴({"detail": …}) — 검수 UI 의 토스트가 그대로 읽는다
                await JSONResponse({"detail": f"{why} — 검수 UI 에서만 바꿀 수 있습니다."}, status_code=403)(
                    scope, receive, send)
                return
        await self.app(scope, receive, send)


def _json_finite(v):
    """JSON 에 넣을 수 없는 수(NaN·±Infinity)를 글자로 바꾼다 — 422 응답의 detail.input 용."""
    if isinstance(v, float) and not math.isfinite(v):
        return str(v)
    if isinstance(v, dict):
        return {k: _json_finite(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_finite(x) for x in v]
    return v


async def _validation_error(request, exc: RequestValidationError) -> JSONResponse:
    """FastAPI 기본 422 와 같고, 받은 값(input)의 NaN·Infinity 만 글자로 바꾼다 (2026-11-05). 본문의 NaN 을 BoxIn 이
    거르면 그 값이 detail.input 에 그대로 실려 Starlette JSONResponse(allow_nan=False)가 422 대신 500 을 냈다."""
    return JSONResponse(status_code=422, content={"detail": _json_finite(jsonable_encoder(exc.errors()))})


def _tag_filter(tag: str | None, xtag: str | None, tagmode: str) -> TagFilter | None:
    """목록 API 의 태그 조건(store.TagFilter). `tag=__none__` 이면 태그 없는 것 — xtag·tagmode 는 보지 않는다."""
    if tag == NO_TAG:
        return TagFilter(none=True)
    if tagmode not in ("all", "any"):
        raise HTTPException(400, f"알 수 없는 tagmode: {tagmode} (all 또는 any)")
    inc, exc = _id_list(tag, "태그"), _id_list(xtag, "뺄 태그")
    if set(inc) & set(exc):
        raise HTTPException(400, "같은 태그를 넣기와 빼기에 함께 줄 수 없습니다.")
    return TagFilter(tuple(inc), tuple(exc), tagmode == "any") if inc or exc else None


def create_app(cfg: Config, *, label: str = "이 프로젝트", bind_host: str | None = None,
               guard: bool = True) -> FastAPI:
    """프로젝트 하나의 앱. guard 는 바깥 앱일 때만 켠다 — 탭 서버(create_multi_app)는 바깥 앱이 한 번 건다.
    bind_host 는 서버가 연 주소(labeler.py review 의 --host, 없으면 config 의 server.host) — _RequestGuard 참고."""
    store = Store(cfg.db)
    jobs = JobManager(label)          # label 은 'GPU 를 누가 쓰는 중' 안내에 쓰인다
    sam = Segmenter(cfg)
    live = LiveDetector(cfg)          # '예측 비교'(P). 처음 쓸 때 올린다
    app = FastAPI(title="Local Auto Labeling", docs_url="/api/docs")
    app.add_exception_handler(RequestValidationError, _validation_error)

    def free_gpu() -> None:
        """학습·자동 라벨링을 시작하기 전에 검수용 모델(SAM·예측 비교)을 내린다.
        8GB 카드에서는 몇백 MB 도 아깝고, 모자라면 에러 없이 느려지기만 한다(함정 9).
        다른 탭(프로젝트)에서 올린 것도 함께 내린다 — GPU 는 프로세스 전체가 하나다."""
        from . import predictor as _p, segmenter as _s

        for m in {sam, live, _s._active, _p._live_active}:
            if m is not None:
                m.release()

    # 서버를 켤 때 하루 한 번 DB 사본을 남긴다 (<DB 폴더>/backups/, 최근 backup.keep 개).
    # 실패해도 검수는 되어야 하므로 알리기만 한다.
    try:
        made = store.backup_daily(int(cfg.get("backup.keep", 14)))
        if made:
            print(f"[{label}] DB 백업: {made}")
    except Exception as exc:  # noqa: BLE001
        print(f"[{label}] DB 백업 실패 (검수는 계속됩니다): {exc}")

    def score_of(boxes: list[Box]) -> float:
        """박스 목록의 검수 우선순위(uncertainty). 같은 자리 일괄 수정·되돌리기도 이것으로 다시 매긴다 (2026-11-05)."""
        return image_uncertainty(
            [b.conf for b in boxes],
            low=float(cfg.get("review.low_band", 0.30)),
            high=float(cfg.get("review.high_band", 0.80)),
            empty=float(cfg.get("review.empty_score", 0.60)),
        )

    def save_boxes(image_id: int, boxes: list[Box], *, source: str, batch: str | None = None,
                   masks: list | None = None):
        """박스(와 masks 를 주면 가릴 영역)를 저장하고 검수 우선순위를 다시 매긴다. (바뀌기 전 개수, 뒤 개수)"""
        return store.set_boxes(image_id, boxes, uncertainty=score_of(boxes), source=source, batch=batch,
                               masks=masks)

    def mask_block(row, cams: dict | None = None) -> str | None:
        """이 이미지에 가릴 영역을 둘 수 없는 이유, 둘 수 있으면 None (2026-11-09, masks.py).

        test 는 평가용이라 실제 화면 그대로여야 한다 — 분할이 test 이거나, 아직 미배정이어도 cameras.yaml 에서 test 로
        고정한 카메라면 막는다. 학습 세트도 test 분할은 칠하지 않으니 여기를 빠져나간 영역이 평가를 바꾸지는 않는다.
        **이 규칙은 여기 한 곳에만 둔다** — 여러 장을 다루는 store(apply_mask_template·revert_batch)에는 이 함수를 넘긴다
        (mask_blocker — 등록 파일을 장마다 다시 읽지 않게 cams 를 한 번 읽어 준다). row 는 split·camera 만 있으면 된다.
        """
        if row["split"] == "test":
            return "test 이미지에는 가릴 영역을 둘 수 없습니다 — 평가는 실제 화면 그대로여야 합니다."
        if cams is None:
            cams = _cameras()[0]
        cam = cams.get(row["camera"]) if row["camera"] else None
        if cam is not None and cam.split == "test":
            return (f"test 로 고정한 카메라({row['camera']})의 이미지에는 가릴 영역을 둘 수 없습니다"
                    " — 평가는 실제 화면 그대로여야 합니다.")
        return None

    def mask_blocker():
        """여러 장에 쓸 mask_block — 카메라 등록 파일을 지금 한 번만 읽는다."""
        cams = _cameras()[0]
        return lambda row: mask_block(row, cams)

    def clean_mask_input(row, masks_in: list[MaskIn] | None) -> list | None:
        """요청의 영역을 정리한다. 너무 많으면 400, test 이미지에 영역을 남기려 하면 409(지우기만은 받는다)."""
        if masks_in is None:
            return None
        if len(masks_in) > MAX_MASKS:
            raise HTTPException(400, f"가릴 영역은 한 장에 {MAX_MASKS}개까지입니다.")
        masks = clean_masks([(m.x1, m.y1, m.x2, m.y2) for m in masks_in])
        if masks and (why := mask_block(row)):
            raise HTTPException(409, why + " 이 이미지의 가릴 영역을 모두 지워야 저장됩니다.")
        return masks

    # ------------------------------------------------------------- 상태 / 목록

    def _rel(p: Path) -> str:
        """확인창에 보여 줄 경로 — 저장소 안이면 상대경로, 밖이면 그대로."""
        try:
            return Path(p).resolve().relative_to(ROOT).as_posix()
        except ValueError:
            return str(p)

    # 정리·내보내기 확인창에 쓰는 이 프로젝트의 실제 폴더 (예전엔 문구에 옛 경로가 박혀 있었다).
    # 설정이 앱과 함께 고정이라 한 번만 계산한다 — /api/state 는 박스를 저장할 때마다 불린다.
    ui_paths = {
        "unlabeled": _rel(cfg.unlabeled),
        "labeled": _rel(cfg.labeled),
        "export": _rel(default_export_dir(cfg)),
    }

    # 엔드포인트의 `-> dict` 반환 주석은 지우지 말 것 (2026-11-05) — FastAPI 가 이 주석으로 Pydantic 직렬화를 거쳐
    # NaN(작업 결과의 지표 등)을 null 로 바꿔 준다. 빼면 Starlette JSONResponse(allow_nan=False)가 500 을 낸다.
    @app.get("/api/state")
    def get_state() -> dict:
        job = jobs.current()
        classes = store.get_classes()
        per_class = store.class_box_counts()
        return {
            "classes": classes,
            "class_boxes": [per_class.get(i, {"total": 0}) for i in range(len(classes))],
            "counts": store.counts(),
            "weights": cfg.active_weights(),
            "trained": cfg.weights.exists(),
            "confident_at": float(cfg.get("predict.confident_at", 0.90)),
            "job": job.to_dict() if job else None,
            "paths": ui_paths,
        }

    def _list_row(r: dict) -> dict:
        out = {
            "id": r["id"],
            "name": r["name"],
            "status": r["status"],
            "source": r["source"],
            "width": r["width"],
            "height": r["height"],
            "n_boxes": r["n_boxes"],
            "uncertainty": round(float(r["uncertainty"]), 4),
            "camera": r.get("camera"),
            "domain": r.get("domain"),
            "split": r.get("split"),
            "tags": sorted(int(t) for t in r["tag_ids"].split(",")) if r.get("tag_ids") else [],
        }
        if "rare_score" in r:      # '드문 클래스 먼저' 정렬일 때만
            out.update(rare_score=r["rare_score"], rare_cls=r["rare_cls"],
                       rare_groups=r["rare_groups"])
        return out

    @app.get("/api/images")
    def list_images(
        status: str = "pending", limit: int = 300, offset: int = 0, order: str = "uncertainty",
        ids: str | None = None, cls: str | None = None, empty: bool = False,
        camera: str | None = None, domain: str | None = None, split: str | None = None,
        tag: str | None = None, xtag: str | None = None, tagmode: str = "all",
    ) -> dict:
        """`cls=2,4` 는 **두 클래스를 모두 가진** 이미지만, `empty=1` 은 박스가 없는 이미지만.

        둘을 같이 주면 `empty` 가 이긴다 (store.list_images 참고). `domain=__none__` 은 미분류,
        `split=__none__` 은 미배정. `tag=3,7` 은 그 태그가 **모두** 붙은 것(`tagmode=any` 면 하나라도),
        `xtag=5,9` 는 그중 하나라도 붙은 것을 뺀다. `tag=__none__` 은 태그 없는 것(`xtag` 는 보지 않는다).
        """
        if ids is not None:
            # 'id 로 열기': 쉼표·공백으로 구분한 id 와 범위(5493-5497). 상태 필터를 무시하고 준 순서대로.
            wanted = _id_ranges(ids)
            found = store.list_images(ids=wanted, limit=len(wanted) or 1)
            found_ids = {r["id"] for r in found}
            return {
                "total": len(found),
                "offset": offset,
                "images": [_list_row(r) for r in found[offset:offset + limit]],
                "missing": [i for i in wanted if i not in found_ids],
            }

        wanted_cls: list[int] | None = None
        if cls:
            n_classes = len(store.get_classes())
            wanted_cls = sorted(_id_list(cls, "클래스"))
            if any(c >= n_classes for c in wanted_cls):
                raise HTTPException(400, f"없는 클래스 번호입니다: {wanted_cls}")

        camera = camera or None     # `camera=__none__` 은 출처 없는 이미지(legacy) — store.NO_CAMERA
        domain = domain or None
        if domain and domain not in (*DOMAINS, NO_DOMAIN):
            raise HTTPException(400, f"알 수 없는 도메인: {domain}")
        split = split or None
        if split and split not in (*SPLITS, NO_SPLIT):
            raise HTTPException(400, f"알 수 없는 분할: {split}")
        tag_f = _tag_filter(tag, xtag, tagmode)
        rows = store.list_images(
            status=status, limit=limit, offset=offset, order=order,
            classes=wanted_cls, empty=empty, camera=camera, domain=domain, split=split, tag=tag_f,
        )
        if empty or wanted_cls or camera or domain or split or tag_f is not None:
            # 거르면 상태별 개수를 쓸 수 없다. 목록 묶음이 이 수로 나뉘므로 같은 조건으로 센다.
            total = store.count_images(status=status, classes=wanted_cls, empty=empty,
                                       camera=camera, domain=domain, split=split, tag=tag_f)
        else:
            counts = store.counts()
            total = counts["total"] if status in ("all", None) else counts.get(status, 0)
        return {"total": total, "offset": offset, "images": [_list_row(r) for r in rows]}

    def _cameras() -> tuple[dict, str | None]:
        """등록 파일의 카메라와 읽기 오류. 화면 표시용이라 오류가 나도 목록은 DB 로 보여 준다."""
        try:
            return load_cameras(cfg.cameras), None
        except ValueError as exc:
            return {}, str(exc)

    def _camera_place(key: str | None) -> str | None:
        if not key:
            return None
        cam = _cameras()[0].get(key)
        return cam.place if cam else None

    @app.get("/api/cameras")
    def list_cameras() -> dict:
        """카메라 필터용: DB 에 있는 카메라(휴지통 제외)와 등록 파일의 장소·형태 (docs/provenance.md)."""
        reg, error = _cameras()
        counts = store.camera_counts()
        cams = [{"key": k, "n": n, "place": reg[k].place if k in reg else None,
                  "kind": reg[k].kind if k in reg else None,
                  "mount": reg[k].mount if k in reg else None,
                  "split": reg[k].split if k in reg else None}
                for k, n in sorted((k, n) for k, n in counts.items() if k)]
        return {"cameras": cams, "no_camera": counts.get(None, 0), "none_key": NO_CAMERA, "error": error}

    @app.get("/api/domains")
    def list_domains() -> dict:
        """도메인 필터용: 도메인별 이미지 수(휴지통 제외)와 미분류 수."""
        counts = store.domain_counts()
        return {"domains": [{"key": d, "n": counts.get(d, 0)} for d in DOMAINS],
                "none": counts.get(None, 0), "none_key": NO_DOMAIN}

    @app.get("/api/splits")
    def list_splits() -> dict:
        """분할 필터용: 분할별 이미지 수(휴지통 제외)와 미배정 수. 분할은 여기서 바꾸지 않는다."""
        counts = store.split_counts()
        return {"splits": [{"key": s, "n": counts.get(s, 0)} for s in SPLITS],
                "none": counts.get(None, 0), "none_key": NO_SPLIT}

    # ── 태그 (2026-11-03): 비슷한 이미지끼리 사람이 이름을 붙여 묶는다. 라벨이 아니다 ──

    @app.get("/api/tags")
    def list_tags() -> dict:
        """태그(이름순)와 붙은 이미지 수, 태그 없는 이미지 수(모두 휴지통 제외)."""
        return {"tags": store.list_tags(), "none": store.untagged_count(), "none_key": NO_TAG}

    @app.post("/api/tags")
    def create_tag(payload: TagIn) -> dict:
        try:
            tag_id = store.create_tag(payload.name)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"ok": True, "id": tag_id}

    @app.put("/api/tags/{tag_id}")
    def rename_tag(tag_id: int, payload: TagIn) -> dict:
        try:
            found = store.rename_tag(tag_id, payload.name)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if not found:
            raise HTTPException(404, "없는 태그입니다.")
        return {"ok": True}

    @app.delete("/api/tags/{tag_id}")
    def delete_tag(tag_id: int) -> dict:
        n = store.delete_tag(tag_id)
        if n is None:
            raise HTTPException(404, "없는 태그입니다.")
        return {"ok": True, "detached": n}

    @app.post("/api/images/{image_id}/tags")
    def post_image_tag(image_id: int, payload: ImageTagIn) -> dict:
        """이 이미지에 태그를 붙이거나 뗀다. (같은 장면 묶음째 하기는 2026-11-04 사용자 결정으로 뺐다.)"""
        if store.get_image(image_id) is None:
            raise HTTPException(404, "이미지를 찾을 수 없습니다.")
        try:
            n = store.set_image_tag([image_id], payload.tag_id, payload.on)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"ok": True, "changed": n}

    @app.post("/api/images/{image_id}/domain")
    def post_domain(image_id: int, payload: DomainIn) -> dict:
        """이 이미지의 도메인을 정한다. (같은 장면 묶음째 하기는 2026-11-04 사용자 결정으로 뺐다.)"""
        if store.get_image(image_id) is None:
            raise HTTPException(404, "이미지를 찾을 수 없습니다.")
        try:
            n = store.set_domain([image_id], payload.domain)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"ok": True, "changed": n}

    # ── 이미지 보기 탭(2026-11-04): 여러 장을 골라 태그·도메인을 한꺼번에 바꾼다 ──
    # /api/images/... 밑에 두지 않는다 — /api/images/{image_id}/... 와 겹치면 FastAPI 는 그 경로로 들어가
    # image_id 검사(422)에서 멈추고 다음 경로를 찾지 않는다.

    @app.post("/api/bulk/tag")
    def bulk_tag(payload: BulkTagIn) -> dict:
        """이미지 여러 장에 태그를 붙이거나 뗀다. changed = 실제로 바뀐 장수."""
        if not payload.ids:
            raise HTTPException(400, "이미지를 고르세요.")
        try:
            n = store.set_image_tag(sorted(set(payload.ids)), payload.tag_id, payload.on)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"ok": True, "changed": n}

    @app.post("/api/bulk/domain")
    def bulk_domain(payload: BulkDomainIn) -> dict:
        """이미지 여러 장의 도메인을 정한다(None = 미분류). changed = 바꾼 행 수."""
        if not payload.ids:
            raise HTTPException(400, "이미지를 고르세요.")
        try:
            n = store.set_domain(sorted(set(payload.ids)), payload.domain)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"ok": True, "changed": n}

    # ── 가릴 영역 템플릿 (2026-11-10, 사용자 요청): 고정 카메라 화면처럼 같은 자리를 여러 장에 덮을 때 이름(카메라A_1 …)을
    # 붙여 둔 영역 묶음. 적용하면 이미지의 영역으로 복사된다(store 의 mask_templates 주석). 한 장 적용은 검수 화면이 영역을
    # 더해 평소처럼 저장(PUT boxes)하고, 여러 장은 이미지 보기에서 /api/bulk/mask-template — 일괄 수정 묶음(source bulk)이라
    # 상단 변경 기록에서 묶음째 되돌린다(store.revert_batch 가 영역도 되돌린다).

    @app.get("/api/mask-templates")
    def list_mask_templates() -> dict:
        return {"templates": store.list_mask_templates()}

    @app.post("/api/mask-templates")
    def save_mask_template(payload: MaskTemplateIn) -> dict:
        """영역을 이름으로 남긴다. 같은 이름이 있으면 409 — overwrite=true 로 다시 보내면 덮어쓴다."""
        if len(payload.masks) > MAX_MASKS:
            raise HTTPException(400, f"가릴 영역은 {MAX_MASKS}개까지입니다.")
        camera = None
        if payload.image_id is not None and (row := store.get_image(payload.image_id)) is not None:
            camera = row["camera"]
        try:
            tid, replaced = store.save_mask_template(
                payload.name, [(m.x1, m.y1, m.x2, m.y2) for m in payload.masks],
                width=payload.width, height=payload.height, image_id=payload.image_id, camera=camera,
                overwrite=payload.overwrite)
        except TemplateExists as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"ok": True, "id": tid, "replaced": replaced}

    @app.put("/api/mask-templates/{template_id}")
    def rename_mask_template(template_id: int, payload: TemplateNameIn) -> dict:
        try:
            found = store.rename_mask_template(template_id, payload.name)
        except ValueError as exc:       # 겹치는 이름(TemplateExists)도 — 이름 바꾸기로 다른 템플릿을 덮지는 않는다
            raise HTTPException(400, str(exc)) from exc
        if not found:
            raise HTTPException(404, "없는 템플릿입니다(다른 창에서 지웠을 수 있습니다).")
        return {"ok": True}

    @app.delete("/api/mask-templates/{template_id}")
    def delete_mask_template(template_id: int) -> dict:
        """템플릿만 지운다 — 이미 적용한 이미지의 영역은 그대로."""
        if not store.delete_mask_template(template_id):
            raise HTTPException(404, "없는 템플릿입니다(다른 창에서 지웠을 수 있습니다).")
        return {"ok": True}

    def _display_ratio(row) -> float | None:
        """이 이미지의 화면(EXIF 회전 적용) 가로/세로. 파일을 못 열면 None."""
        try:
            w, h = display_size(Path(row["path"]))
        except _UNREADABLE:
            return None
        return w / h if w and h else None

    @app.post("/api/bulk/mask-template")
    def bulk_mask_template(payload: BulkMaskTemplateIn) -> dict:
        """고른 이미지들에 템플릿 영역을 더하거나(add) 뺀다(remove). 바뀐 장수와 건너뛴 이유별 장수를 돌려준다.

        더하기는 test 이미지(mask_block — store 가 쓰는 순간의 행으로 본다)와, 화면 비율이 템플릿을 만든 그림과 1% 넘게
        다른 이미지를 건너뛴다(다른 카메라·크기의 그림을 엉뚱하게 덮지 않게). 비율은 DB 의 크기로 먼저 보고, 어긋날 때만
        파일을 열어 EXIF 회전을 적용한 크기로 다시 본다(못 열면 unreadable) — 원시 비율은 같은데 90도 도는 그림은 가리지
        못하지만 장마다 파일을 열면 1천 장에 2초가 걸렸다. 빼기는 템플릿과 좌표가 같은 영역만 빼므로 비율을 보지 않는다.
        바뀐 이미지마다 이력(source bulk, 같은 batch)이 남는다.
        """
        if not payload.ids:
            raise HTTPException(400, "이미지를 고르세요.")
        tpl = store.get_mask_template(payload.template_id)
        if tpl is None:
            raise HTTPException(404, "없는 템플릿입니다(다른 창에서 지웠을 수 있습니다).")
        ids = list(dict.fromkeys(payload.ids))
        n_aspect = n_unreadable = 0
        if payload.mode == "add" and tpl["width"] and tpl["height"]:
            want = tpl["width"] / tpl["height"]
            near = lambda ratio: abs(ratio - want) <= 0.01 * want
            rows = store.image_sizes(ids)
            keep = []
            for i in ids:
                row = rows.get(i)                       # 없는 이미지는 store 가 missing 으로 센다
                if row is not None and not (row["width"] and row["height"] and near(row["width"] / row["height"])):
                    ratio = _display_ratio(row)         # 원시 크기로 어긋나면 EXIF 로 도는 그림일 수 있다
                    if ratio is None:
                        n_unreadable += 1
                        continue
                    if not near(ratio):
                        n_aspect += 1
                        continue
                keep.append(i)
            ids = keep
        verb = "더하기" if payload.mode == "add" else "빼기"
        batch = (f"{datetime.now():%m-%d %H:%M:%S} 가릴 영역 템플릿 '{tpl['name']}' {verb}"
                 f" (고른 {len(payload.ids)}장) #{secrets.token_hex(2)}")
        res = store.apply_mask_template(ids, tpl["rects"], mode=payload.mode, batch=batch, blocked=mask_blocker())
        return {**res, "aspect": n_aspect, "unreadable": n_unreadable,
                "batch": batch if res["changed"] else None}

    @app.get("/api/images/{image_id}")
    def get_image(image_id: int) -> dict:
        row = store.get_image(image_id)
        if row is None:
            raise HTTPException(404, "이미지를 찾을 수 없습니다.")
        boxes = store.get_boxes(image_id)
        return {
            "id": row["id"],
            "name": row["name"],
            "path": row["path"],
            "width": row["width"],
            "height": row["height"],
            "status": row["status"],
            "source": row["source"],
            "uncertainty": round(float(row["uncertainty"]), 4),
            "model_version": row["model_version"],
            # 이미지 URL 의 캐시 무효화 토큰으로 쓴다 (아래 /file 주석 참고).
            "updated_at": row["updated_at"],
            "boxes": [b.to_dict() for b in boxes],
            "next_id": store.next_image_id(image_id, status="pending"),
            "n_history": store.history_count(image_id),
            "grp": row["grp"],
            "split": row["split"],
            "camera": row["camera"],
            "captured_at": row["captured_at"],
            "camera_place": _camera_place(row["camera"]),
            "domain": row["domain"],
            "tags": store.image_tag_ids(image_id),
            # 가릴 영역(정규화 xyxy)과, 둘 수 없으면 그 이유(test 이미지) — masks.py
            "masks": [list(m) for m in store.get_masks(image_id)],
            "mask_block": mask_block(row),
        }

    @app.get("/api/images/{image_id}/file")
    def get_image_file(image_id: int):
        row = store.get_image(image_id)
        if row is None:
            raise HTTPException(404, "이미지를 찾을 수 없습니다.")
        path = Path(row["path"])
        if not path.exists():
            raise HTTPException(410, f"원본 파일이 사라졌습니다: {path}")
        # 예전 스키마는 images.id 가 rowid 별칭이라, 마지막 행을 지우면 같은 id 가 다음
        # 이미지에 다시 붙었다(CLAUDE.md 함정 11, 지금은 AUTOINCREMENT). 그러면 이 URL 은
        # 그대로인데 내용만 바뀌어 브라우저가 예전 이미지를 보여준다. 겹겹이 막는 한 겹이다.
        # no-cache = "쓰기 전에 반드시 물어봐라". ETag 가 같으면 304 라 재전송은 없다.
        return FileResponse(path, headers={"Cache-Control": "no-cache"})

    thumb_dir = cfg.db.parent / "thumbs"
    thumb_sizes = (160, 256, 384)       # 받는 크기를 정해 둔다 — 아무 값이나 받으면 캐시가 끝없이 는다

    @app.get("/api/images/{image_id}/thumb")
    def get_image_thumb(image_id: int, s: int = 256):
        """이미지 보기 탭의 썸네일(JPEG, 긴 변 s). 원본(폰 사진 4000×3000, PNG 크롭도 수 MB)을 수백 장 그대로
        받으면 느려서 줄여 보낸다. `<DB 폴더>/thumbs/` 에 남겨 다시 쓴다 — 이름에 원본의 수정 시각·크기를 넣어
        원본이 바뀌면 새로 만든다. 지워도 다시 만들어지는 캐시다(백업에서 뺀다)."""
        if s not in thumb_sizes:
            raise HTTPException(400, f"썸네일 크기는 {thumb_sizes} 가운데 하나입니다.")
        row = store.get_image(image_id)
        if row is None:
            raise HTTPException(404, "이미지를 찾을 수 없습니다.")
        path = Path(row["path"])
        try:
            st = path.stat()
        except OSError:
            raise HTTPException(410, f"원본 파일이 사라졌습니다: {path}") from None
        out = thumb_dir / f"{image_id}_{s}_{st.st_mtime_ns}_{st.st_size}.jpg"
        if not out.exists():
            thumb_dir.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(f"{out.stem}.{secrets.token_hex(4)}.tmp")
            try:
                with Image.open(path) as im:
                    im.draft("RGB", (s, s))                              # JPEG 은 1/2·1/4·1/8 로 풀어 빠르다
                    im = ImageOps.exif_transpose(im).convert("RGB")      # 화면처럼 회전(CLAUDE.md 'EXIF 회전')
                    im.thumbnail((s, s))
                    im.save(tmp, "JPEG", quality=82)
            except _UNREADABLE as exc:
                tmp.unlink(missing_ok=True)
                raise HTTPException(415, f"썸네일을 만들지 못했습니다: {type(exc).__name__}: {exc}") from exc
            try:
                os.replace(tmp, out)            # 반쯤 쓴 파일이 보이지 않게
            except PermissionError:
                # Windows: 다른 요청이 같은 썸네일을 보내는 중이면 바꿔 끼울 수 없다 — 그쪽이 만든 것을 쓴다
                tmp.unlink(missing_ok=True)
                if not out.exists():
                    raise
        return FileResponse(out, media_type="image/jpeg", headers={"Cache-Control": "no-cache"})

    # ------------------------------------------------------------------ 편집

    @app.put("/api/images/{image_id}/boxes")
    def put_boxes(image_id: int, payload: BoxesIn) -> dict:
        row = store.get_image(image_id)
        if row is None:
            raise HTTPException(404, "이미지를 찾을 수 없습니다.")
        n_classes = len(store.get_classes())
        boxes: list[Box] = []
        for b in payload.boxes:
            if b.cls >= n_classes:
                raise HTTPException(400, f"없는 클래스 인덱스입니다: {b.cls}")
            box = Box(b.cls, b.cx, b.cy, b.w, b.h, b.conf, b.origin).clamped()
            if box.is_valid():
                boxes.append(box)
        masks = clean_mask_input(row, payload.masks)
        n_before, n_after = save_boxes(image_id, boxes, source="ui", masks=masks)
        return {"ok": True, "n_boxes": n_after, "n_before": n_before,
                "n_history": store.history_count(image_id)}

    # ------------------------------------------------------------- 변경 이력

    @app.get("/api/images/{image_id}/history")
    def get_history(image_id: int) -> dict:
        if store.get_image(image_id) is None:
            raise HTTPException(404, "이미지를 찾을 수 없습니다.")
        return {"entries": store.box_history(image_id)}

    @app.post("/api/images/{image_id}/history/{entry_id}/restore")
    def restore_history(image_id: int, entry_id: int, payload: RestoreIn | None = None) -> dict:
        """이력 한 줄의 전(또는 후) 상태로 박스를 되돌린다. 되돌린 것도 이력에 남는다."""
        entry = store.history_entry(entry_id)
        if entry is None or entry["image_id"] != image_id:
            raise HTTPException(404, "그 이력을 찾을 수 없습니다.")
        which = payload.which if payload else "before"
        # 그림을 잘라 바꾼 뒤(source='crop')에는 그 전 박스가 통째 화면 좌표라 되돌리면 어긋난다 (2026-11-09).
        # 자르기 한 줄 자신은 '후' 만 지금 좌표다.
        if any(c > entry_id or (c == entry_id and which == "before") for c in store.crop_entries(image_id)):
            raise HTTPException(409, "이 기록 뒤에 그림을 잘라 박스 좌표가 달라져서 되돌릴 수 없습니다.")
        n_classes = len(store.get_classes())
        boxes = [Box(**b) for b in entry[which]]
        if any(b.cls >= n_classes for b in boxes):
            raise HTTPException(400, "지금 없는 클래스 번호가 들어 있어 되돌릴 수 없습니다.")
        # 그 저장에서 가릴 영역도 바뀌었으면 영역도 그때로 (아니면 None — 지금 영역을 그대로 둔다)
        masks = entry[f"masks_{which}"]
        if masks and (why := mask_block(store.get_image(image_id))):
            raise HTTPException(409, why)
        n_before, n_after = save_boxes(image_id, boxes, source="restore", masks=masks)
        return {"ok": True, "n_boxes": n_after, "n_before": n_before,
                "n_history": store.history_count(image_id),
                "boxes": [b.to_dict() for b in store.get_boxes(image_id)],
                "masks": [list(m) for m in store.get_masks(image_id)]}

    # ------------------------------------------------------- 같은 자리 일괄 수정

    def _same_place(image_id: int, p: SamePlaceIn) -> tuple[dict, list[dict]]:
        classes = store.get_classes()
        if p.to_cls >= len(classes) or any(c >= len(classes) for c in (p.from_cls or [])):
            raise HTTPException(400, "없는 클래스 번호입니다.")
        try:
            found = store.same_place(image_id, tuple(p.box), from_cls=p.from_cls, iou_min=p.iou)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        change = [m for m in found["matches"] if m["cls"] != p.to_cls]
        return found, change

    @app.post("/api/images/{image_id}/same-place")
    def same_place_preview(image_id: int, payload: SamePlaceIn) -> dict:
        """같은 묶음·같은 자리 박스를 찾아 **무엇이 바뀔지만** 알려준다 (DB 는 그대로)."""
        found, change = _same_place(image_id, payload)
        classes = store.get_classes()
        count = lambda key: dict(Counter(key(m) for m in change).most_common())
        return {
            "grp": found["grp"],
            "group_size": found["group_size"],
            "matched": len(found["matches"]),
            "to_change": len(change),
            "already": len(found["matches"]) - len(change),
            "by_class": count(lambda m: classes[m["cls"]] if m["cls"] < len(classes) else f"cls_{m['cls']}"),
            "by_split": count(lambda m: m["split"] or "미배정"),
            "by_status": count(lambda m: m["status"]),
            "ids": [m["image_id"] for m in change],
        }

    @app.post("/api/images/{image_id}/same-place/apply")
    def same_place_apply(image_id: int, payload: SamePlaceIn) -> dict:
        """미리 보기와 같은 조건으로 다시 찾아 클래스를 바꾼다. 그 사이 수가 달라졌으면 409."""
        found, change = _same_place(image_id, payload)
        if payload.expected is not None and len(change) != payload.expected:
            raise HTTPException(
                409, f"미리 본 뒤로 대상이 바뀌었습니다 ({payload.expected} -> {len(change)}장). "
                     "다시 확인하고 적용하세요.")
        if not change:
            return {"changed": 0, "batch": None}
        classes = store.get_classes()
        src = "·".join(classes[c] for c in payload.from_cls) if payload.from_cls else "모든 클래스"
        batch = (f"{datetime.now():%m-%d %H:%M:%S} {found['grp']} {src}→{classes[payload.to_cls]}"
                 f" ({len(change)}장) #{secrets.token_hex(2)}")
        by_image: dict[int, list[int]] = {}
        for m in change:
            by_image.setdefault(m["image_id"], []).append(m["id"])
        n = store.set_class_bulk(by_image, payload.to_cls, batch=batch, score=score_of)
        return {"changed": n, "batch": batch}

    @app.post("/api/history/batch/revert")
    def revert_batch(payload: BatchIn) -> dict:
        """일괄 수정 한 묶음을 되돌린다. 그 뒤 또 고친 이미지는 건너뛴다."""
        try:
            reverted, skipped = store.revert_batch(payload.batch, score=score_of, mask_blocked=mask_blocker())
        except ValueError as exc:   # 되돌리면 목록에 없는 클래스가 생긴다 (클래스를 합쳐 뺀 묶음)
            raise HTTPException(409, str(exc)) from exc
        if not reverted and not skipped:
            raise HTTPException(404, "그 일괄 수정을 찾을 수 없습니다.")
        return {"reverted": reverted, "skipped": skipped}

    @app.get("/api/history")
    def recent_history(limit: int = 200, shrunk: bool = False, source: str | None = None) -> dict:
        """최근 박스 변경. shrunk=1 이면 박스 수가 줄어든 저장만 (일괄 검수 뒤 점검용)."""
        return {
            "entries": store.recent_history(limit=min(limit, 2000), shrunk=shrunk, source=source),
            "batches": store.batches(),
        }

    @app.post("/api/images/{image_id}/segment")
    def post_segment(image_id: int, payload: SegmentIn) -> dict:
        """SAM 으로 박스 하나를 만들어 돌려준다. DB 에는 쓰지 않는다.

        저장은 기존 경로(PUT /boxes)를 그대로 쓴다. 사람이 확인하고 고친 뒤
        저장하는 흐름이어야 하고, 여기서 바로 쓰면 되돌리기와 어긋난다.
        """
        row = store.get_image(image_id)
        if row is None:
            raise HTTPException(404, "이미지를 찾을 수 없습니다.")
        if payload.point is None and payload.box is None:
            raise HTTPException(400, "point 또는 box 중 하나는 필요합니다.")
        n_classes = len(store.get_classes())
        if n_classes and payload.cls >= n_classes:
            raise HTTPException(400, f"없는 클래스 인덱스입니다: {payload.cls}")

        path = Path(row["path"])
        if not path.exists():
            raise HTTPException(410, f"원본 파일이 사라졌습니다: {path}")

        # 정규화 좌표 -> 픽셀. SAM 은 픽셀 좌표를 받는다.
        # DB 크기가 아니라 **EXIF 회전을 적용한 크기**로 바꾼다. SAM(Ultralytics)은 회전된
        # 이미지를 읽는데, DB 에는 회전 전 크기가 있어서 회전 사진에서는 엉뚱한 곳을 찍었다.
        try:
            w, h = (float(v) for v in display_size(path))
        except _UNREADABLE as exc:      # 썸네일과 같다 — 손상 EXIF·너무 큰 이미지는 500 이 아니라 415
            raise HTTPException(415, f"이미지를 읽지 못했습니다: {type(exc).__name__}: {exc}") from exc
        point = (payload.point[0] * w, payload.point[1] * h) if payload.point else None
        box = None
        if payload.box:
            x1, y1, x2, y2 = payload.box
            box = (min(x1, x2) * w, min(y1, y2) * h, max(x1, x2) * w, max(y1, y2) * h)

        try:
            result, score = sam.segment(path, point=point, box=box, cls=payload.cls)
        except SegmenterUnavailable as exc:
            raise HTTPException(503, str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(410, str(exc)) from exc
        if not result.is_valid():
            raise HTTPException(422, "SAM 이 만든 영역이 너무 작습니다.")
        return {"box": result.to_dict(), "score": round(score, 4)}

    @app.get("/api/images/{image_id}/predict")
    def get_predict(image_id: int) -> dict:
        """'예측 비교'(P): 운영 가중치로 이 이미지 한 장을 추론해 돌려준다. DB 에는 쓰지 않는다."""
        row = store.get_image(image_id)
        if row is None:
            raise HTTPException(404, "이미지를 찾을 수 없습니다.")
        path = Path(row["path"])
        if not path.exists():
            raise HTTPException(410, f"원본 파일이 사라졌습니다: {path}")
        try:
            return live.predict(path, store.get_classes())
        except Exception as exc:  # noqa: BLE001 — 모델·장치 문제를 그대로 알린다
            raise HTTPException(503, f"예측을 돌리지 못했습니다: {type(exc).__name__}: {exc}") from exc

    @app.get("/api/sam")
    def get_sam_status() -> dict:
        return sam.status()

    @app.post("/api/sam/release")
    def post_sam_release() -> dict:
        """SAM 을 내려 VRAM 을 돌려준다. 학습을 시작하기 전에 쓴다."""
        sam.release()
        return {"ok": True, **sam.status()}

    @app.post("/api/images/{image_id}/status")
    def post_status(image_id: int, payload: StatusIn) -> dict:
        if store.get_image(image_id) is None:
            raise HTTPException(404, "이미지를 찾을 수 없습니다.")
        try:
            store.set_status(image_id, payload.status)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"ok": True, "next_id": store.next_image_id(image_id, status="pending")}

    # ---------------------------------------------------------------- 클래스

    @app.post("/api/classes")
    def add_class(payload: ClassIn) -> dict:
        try:
            idx = store.add_class(payload.name)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        classes = store.get_classes()
        # config.yaml 의 classes 가 우선이라 이 파일은 평소에 읽히지 않지만, 이름을 잃었을 때
        # 되살릴 데가 여기뿐이다 (2026-09-20 에 실제로 잃었다).
        cfg.classes_txt.write_text("\n".join(classes) + "\n", encoding="utf-8")
        return {"index": idx, "classes": classes}

    @app.put("/api/classes")
    def set_classes(payload: ClassesIn) -> dict:
        names = [n.strip() for n in payload.names if n.strip()]
        if len(set(names)) != len(names):
            raise HTTPException(400, "클래스 이름이 중복됩니다.")
        used = store.max_class_id()
        if used >= len(names):
            raise HTTPException(
                400,
                f"클래스 {used} 번을 쓰는 박스가 있어 {len(names)}개로 줄일 수 없습니다.",
            )
        store.set_classes(names, drop_extra=True)
        cfg.classes_txt.write_text("\n".join(names) + "\n", encoding="utf-8")
        return {"classes": store.get_classes()}

    # ------------------------------------------------------------------ 작업

    @app.post("/api/jobs/scan")
    def job_scan() -> dict:
        def work(job) -> dict:
            report = ingest(cfg, store)
            discover_classes(cfg, store)
            print(report.summary())
            return {"counts": store.counts(), "classes": store.get_classes()}

        return _start(jobs, "scan", work)

    @app.post("/api/jobs/predict")
    def job_predict(payload: PredictIn) -> dict:
        def work(job) -> dict:
            free_gpu()
            labeler = AutoLabeler(cfg, store)
            ids = None
            if payload.include_pending:
                ids = [
                    r["id"]
                    for r in store.list_images(status="unlabeled", limit=10**9)
                ] + [r["id"] for r in store.list_images(status="pending", limit=10**9)]

            def progress(n: int, total: int, name: str) -> None:
                job.progress = f"{n}/{total}  {name}"

            report = labeler.run(ids, progress=progress, tiles=payload.tiles)
            print(report.summary())
            return {"processed": report.processed, "boxes": report.boxes,
                    "skipped_changed": report.skipped_changed, "tiled": report.tiled if payload.tiles else None}

        return _start(jobs, "predict", work)

    @app.post("/api/jobs/train")
    def job_train(payload: TrainIn) -> dict:
        def work(job) -> dict:
            # SAM·예측 비교 모델이 올라와 있으면 내린다. 8GB 카드에서는 0.55GB 도 아깝고,
            # VRAM 이 모자라면 에러 없이 몇 배 느려지기만 한다(CLAUDE.md 함정 9).
            free_gpu()
            # 중단 버튼과 정지 파일(runs/STOP) 둘 다 받는다. 작업 스레드라 신호
            # 핸들러는 못 걸지만 파일 확인은 어느 스레드에서든 된다.
            stop = StopSignal()
            try:
                return run_train(
                    cfg, store, epochs=payload.epochs,
                    should_stop=lambda: job.cancel_requested or stop(),
                )
            finally:
                stop.clear()

        return _start(jobs, "train", work)

    @app.post("/api/jobs/resume")
    def job_resume(payload: ResumeIn | None = None) -> dict:
        """중단된 학습을 이어서 진행한다."""
        batch = payload.batch if payload else None

        def work(job) -> dict:
            free_gpu()         # job_train 과 같은 이유
            stop = StopSignal()
            try:
                return run_resume(
                    cfg, store, batch=batch,
                    should_stop=lambda: job.cancel_requested or stop(),
                )
            finally:
                stop.clear()

        return _start(jobs, "resume", work)

    # -------------------------------------------------------------- WSL 학습

    @app.get("/api/wsl")
    def get_wsl() -> dict:
        """WSL 학습을 쓸 수 있는지, 이어 학습할 지점이 있는지."""
        ok, why = wsl_train.check(cfg)
        point = wsl_train.find_resume(cfg) if ok else None
        return {
            "available": ok,
            "reason": why,
            "resume": None if point is None else {
                "epoch": point.done,
                "total": point.total_epochs,
                "kind": point.kind,
                "lost": point.lost,
                "run": point.run_dir.name,
            },
        }

    @app.post("/api/jobs/train_wsl")
    def job_train_wsl() -> dict:
        def work(job) -> dict:
            # WSL 도 같은 RTX 를 쓴다. Windows 쪽 SAM·예측 비교가 잡은 몫도 돌려준다.
            free_gpu()
            return wsl_train.train(cfg, store, job)

        return _start(jobs, "train_wsl", work)

    @app.post("/api/jobs/resume_wsl")
    def job_resume_wsl() -> dict:
        def work(job) -> dict:
            free_gpu()
            return wsl_train.train(cfg, store, job, resume=True)

        return _start(jobs, "resume_wsl", work)

    @app.get("/api/resume-point")
    def resume_point() -> dict:
        """이어 학습이 가능한지, 가능하면 어디서부터인지 알려준다."""
        point = find_resume_point(cfg)
        if point is None:
            return {"available": False}
        return {
            "available": True,
            "epoch": point.done,
            "total": point.total_epochs,
            "lost": point.lost,
            "kind": point.kind,
            "summary": point.summary(),
        }

    @app.post("/api/jobs/tidy")
    def job_tidy() -> dict:
        def work(job) -> dict:
            def progress(n: int, total: int, name: str) -> None:
                job.progress = f"{n}/{total}  {name}"

            report = tidy_approved(cfg, store, progress=progress)
            print(report.summary())
            # 사람이 탐색기에서 지운 이미지의 DB 항목도 함께 걷어낸다.
            pruned = prune_missing(cfg, store)
            print(pruned.summary())
            return {
                "moved": report.moved,
                "dest": str(report.dest),
                "pruned": pruned.deleted,
                "prune_aborted": pruned.aborted,
            }

        return _start(jobs, "tidy", work)

    @app.post("/api/jobs/purge")
    def job_purge() -> dict:
        """휴지통 비우기: 원본 파일과 DB 항목을 영구 삭제한다."""

        def work(job) -> dict:
            def progress(n: int, total: int, name: str) -> None:
                job.progress = f"{n}/{total}  {name}"

            report = purge_trashed(cfg, store, progress=progress)
            print(report.summary())
            return {"deleted": report.rows_deleted, "files": report.files_deleted}

        return _start(jobs, "purge", work)

    @app.post("/api/jobs/export")
    def job_export(payload: ExportIn) -> dict:
        def work(job) -> dict:
            report = export_yolo(
                cfg, store, resolve(payload.out_dir) if payload.out_dir else None,
                with_images=payload.with_images,
            )
            print(report.summary())
            return {"out_dir": str(report.out_dir), "images": report.images}

        return _start(jobs, "export", work)

    @app.post("/api/jobs/{job_id}/stop")
    def stop_job(job_id: int) -> dict:
        if jobs.get(job_id) is None:
            raise HTTPException(404, "작업을 찾을 수 없습니다.")
        ok = jobs.cancel(job_id)
        if not ok:
            raise HTTPException(409, "이미 끝난 작업입니다.")
        return {"ok": True}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: int) -> dict:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "작업을 찾을 수 없습니다.")
        return job.to_dict()

    # ------------------------------------------------------------------ 정적

    @app.middleware("http")
    async def no_cache_web(request, call_next):
        # StaticFiles 는 Cache-Control 없이 ETag·Last-Modified 만 보낸다. 그러면 브라우저가
        # 휴리스틱으로 app.js 를 '신선하다'고 보고 서버에 묻지 않아, UI 를 고친 뒤 서버를
        # 재시작해도 옛 코드가 돌 수 있다. no-cache 는 매번 ETag 로 확인만 하니(304) 비용이 없다.
        response = await call_next(request)
        if not request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-cache")
        return response

    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    if guard:
        app.add_middleware(_RequestGuard, bind_host=bind_host or str(cfg.get("server.host", "127.0.0.1")))
    return app


def create_multi_app(projects: list, *, bind_host: str | None = None) -> FastAPI:
    """프로젝트마다 앱을 만들어 `/p/<name>` 에 붙인다 (검수 UI 상단 탭, 2026-09-21).

    앱이 통째로 따로라 Store·작업큐·SAM 이 섞이지 않는다. GPU 는 하나뿐이므로 학습·추론만
    `jobs._claim_gpu` 로 프로젝트를 가로질러 막는다. 다른 사이트에서 부르기 막기(_RequestGuard)는 바깥 앱에 한 번 건다.
    """
    from fastapi.responses import RedirectResponse

    root = FastAPI(title="Local Auto Labeling", docs_url=None)
    root.add_middleware(_RequestGuard,
                        bind_host=bind_host or str(projects[0].cfg.get("server.host", "127.0.0.1")))

    @root.get("/api/projects")
    def list_projects() -> dict:
        return {
            "projects": [
                {"name": p.name, "title": p.title, "base": f"/p/{p.name}/"} for p in projects
            ]
        }

    @root.get("/")
    def home() -> RedirectResponse:
        return RedirectResponse(f"/p/{projects[0].name}/")

    for p in projects:
        root.mount(f"/p/{p.name}", create_app(p.cfg, label=p.title, guard=False))
    return root


def _start(jobs: JobManager, kind: str, work) -> dict:
    try:
        job = jobs.start(kind, work)
    except RuntimeError as exc:
        # 이미 다른 작업이 도는 중. 프런트에서 토스트로 그대로 보여준다.
        raise HTTPException(409, str(exc)) from exc
    return job.to_dict()
