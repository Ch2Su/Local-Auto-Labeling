"""검수 웹 UI 서버 (FastAPI).

이미지는 항상 DB에 등록된 id로만 서빙한다. 클라이언트가 보낸 경로를 그대로
열지 않으므로 경로 탈출(path traversal) 여지가 없다.
"""

from __future__ import annotations

import secrets
from collections import Counter
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import ROOT, Config, resolve
from .export import default_export_dir, export_yolo
from .ingest import discover_classes, ingest
from .jobs import JobManager
from .predictor import AutoLabeler, LiveDetector
from .provenance import load_cameras
from .scoring import image_uncertainty
from . import wsl_train
from .segmenter import Segmenter, SegmenterUnavailable
from .stopper import StopSignal
from .store import DOMAINS, NO_CAMERA, NO_DOMAIN, Store
from .tidy import prune_missing, purge_trashed, tidy_approved
from .trainer import find_resume_point
from .trainer import resume_train as run_resume
from .trainer import train as run_train
from .yolo_io import Box, display_size

WEB_DIR = Path(__file__).parent / "web"


class BoxIn(BaseModel):
    cls: int = Field(ge=0)
    cx: float
    cy: float
    w: float = Field(gt=0)
    h: float = Field(gt=0)
    conf: float | None = None
    origin: str = "human"


class BoxesIn(BaseModel):
    boxes: list[BoxIn]


class StatusIn(BaseModel):
    status: str


class DomainIn(BaseModel):
    domain: str | None          # cctv | noncctv | drone, None = 미분류로 되돌림
    group: bool = False         # 같은 장면 묶음(grp)의 이미지까지 한꺼번에


class ClassIn(BaseModel):
    name: str


class ClassesIn(BaseModel):
    names: list[str]


class PredictIn(BaseModel):
    # 이미 검수 대기(pending)인 것까지 다시 돌릴지 여부
    include_pending: bool = False


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

    point: list[float] | None = Field(None, min_length=2, max_length=2)
    box: list[float] | None = Field(None, min_length=4, max_length=4)
    cls: int = Field(0, ge=0)


def create_app(cfg: Config, *, label: str = "이 프로젝트") -> FastAPI:
    store = Store(cfg.db)
    jobs = JobManager(label)          # label 은 'GPU 를 누가 쓰는 중' 안내에 쓰인다
    sam = Segmenter(cfg)
    live = LiveDetector(cfg)          # '예측 비교'(P). 처음 쓸 때 올린다
    app = FastAPI(title="Local Auto Labeling", docs_url="/api/docs")

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

    def save_boxes(image_id: int, boxes: list[Box], *, source: str, batch: str | None = None):
        """박스를 저장하고 검수 우선순위를 다시 매긴다. (바뀌기 전 개수, 뒤 개수)"""
        score = image_uncertainty(
            [b.conf for b in boxes],
            low=float(cfg.get("review.low_band", 0.30)),
            high=float(cfg.get("review.high_band", 0.80)),
            empty=float(cfg.get("review.empty_score", 0.60)),
        )
        return store.set_boxes(image_id, boxes, uncertainty=score, source=source, batch=batch)

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
        }
        if "rare_score" in r:      # '드문 클래스 먼저' 정렬일 때만
            out.update(rare_score=r["rare_score"], rare_cls=r["rare_cls"],
                       rare_groups=r["rare_groups"])
        return out

    @app.get("/api/images")
    def list_images(
        status: str = "pending", limit: int = 300, offset: int = 0, order: str = "uncertainty",
        ids: str | None = None, cls: str | None = None, empty: bool = False,
        camera: str | None = None, domain: str | None = None,
    ) -> dict:
        """`cls=2,4` 는 **두 클래스를 모두 가진** 이미지만, `empty=1` 은 박스가 없는 이미지만.

        둘을 같이 주면 `empty` 가 이긴다 (store.list_images 참고). `domain=__none__` 은 미분류.
        """
        if ids is not None:
            # 'id 로 열기': 쉼표·공백으로 구분한 id 목록. 상태 필터를 무시하고 준 순서대로.
            tokens = [t for t in ids.replace(",", " ").split() if t]
            if not all(t.isdigit() for t in tokens):
                raise HTTPException(400, "id 는 숫자만 쉼표로 구분해 주세요.")
            wanted = list(dict.fromkeys(int(t) for t in tokens))
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
            tokens = [t for t in cls.replace(",", " ").split() if t]
            if not all(t.isdigit() for t in tokens):
                raise HTTPException(400, "클래스는 번호를 쉼표로 구분해 주세요.")
            n_classes = len(store.get_classes())
            wanted_cls = sorted({int(t) for t in tokens})
            if any(c >= n_classes for c in wanted_cls):
                raise HTTPException(400, f"없는 클래스 번호입니다: {wanted_cls}")

        camera = camera or None     # `camera=__none__` 은 출처 없는 이미지(legacy) — store.NO_CAMERA
        domain = domain or None
        if domain and domain not in (*DOMAINS, NO_DOMAIN):
            raise HTTPException(400, f"알 수 없는 도메인: {domain}")
        rows = store.list_images(
            status=status, limit=limit, offset=offset, order=order,
            classes=wanted_cls, empty=empty, camera=camera, domain=domain,
        )
        if empty or wanted_cls or camera or domain:
            # 거르면 상태별 개수를 쓸 수 없다. 목록 묶음이 이 수로 나뉘므로 같은 조건으로 센다.
            total = store.count_images(status=status, classes=wanted_cls, empty=empty,
                                       camera=camera, domain=domain)
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

    @app.post("/api/images/{image_id}/domain")
    def post_domain(image_id: int, payload: DomainIn) -> dict:
        """도메인을 정한다. group 이면 같은 장면 묶음(휴지통 제외) 전체에."""
        if store.get_image(image_id) is None:
            raise HTTPException(404, "이미지를 찾을 수 없습니다.")
        ids = store.group_ids(image_id) if payload.group else [image_id]
        try:
            n = store.set_domain(ids, payload.domain)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"ok": True, "changed": n, "ids": ids}

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
            "grp_n": len(store.group_ids(image_id)),
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

    # ------------------------------------------------------------------ 편집

    @app.put("/api/images/{image_id}/boxes")
    def put_boxes(image_id: int, payload: BoxesIn) -> dict:
        if store.get_image(image_id) is None:
            raise HTTPException(404, "이미지를 찾을 수 없습니다.")
        n_classes = len(store.get_classes())
        boxes: list[Box] = []
        for b in payload.boxes:
            if b.cls >= n_classes:
                raise HTTPException(400, f"없는 클래스 인덱스입니다: {b.cls}")
            box = Box(b.cls, b.cx, b.cy, b.w, b.h, b.conf, b.origin).clamped()
            if box.is_valid():
                boxes.append(box)
        n_before, n_after = save_boxes(image_id, boxes, source="ui")
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
        n_classes = len(store.get_classes())
        boxes = [Box(**b) for b in entry[which]]
        if any(b.cls >= n_classes for b in boxes):
            raise HTTPException(400, "지금 없는 클래스 번호가 들어 있어 되돌릴 수 없습니다.")
        n_before, n_after = save_boxes(image_id, boxes, source="restore")
        return {"ok": True, "n_boxes": n_after, "n_before": n_before,
                "n_history": store.history_count(image_id),
                "boxes": [b.to_dict() for b in store.get_boxes(image_id)]}

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
        n = store.set_class_bulk(by_image, payload.to_cls, batch=batch)
        return {"changed": n, "batch": batch}

    @app.post("/api/history/batch/revert")
    def revert_batch(payload: BatchIn) -> dict:
        """일괄 수정 한 묶음을 되돌린다. 그 뒤 또 고친 이미지는 건너뛴다."""
        try:
            reverted, skipped = store.revert_batch(payload.batch)
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
        w, h = (float(v) for v in display_size(path))
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

            report = labeler.run(ids, progress=progress)
            print(report.summary())
            return {"processed": report.processed, "boxes": report.boxes}

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
    return app


def create_multi_app(projects: list) -> FastAPI:
    """프로젝트마다 앱을 만들어 `/p/<name>` 에 붙인다 (검수 UI 상단 탭, 2026-09-21).

    앱이 통째로 따로라 Store·작업큐·SAM 이 섞이지 않는다. GPU 는 하나뿐이므로 학습·추론만
    `jobs._claim_gpu` 로 프로젝트를 가로질러 막는다.
    """
    from fastapi.responses import RedirectResponse

    root = FastAPI(title="Local Auto Labeling", docs_url=None)

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
        root.mount(f"/p/{p.name}", create_app(p.cfg, label=p.title))
    return root


def _start(jobs: JobManager, kind: str, work) -> dict:
    try:
        job = jobs.start(kind, work)
    except RuntimeError as exc:
        # 이미 다른 작업이 도는 중. 프런트에서 토스트로 그대로 보여준다.
        raise HTTPException(409, str(exc)) from exc
    return job.to_dict()
