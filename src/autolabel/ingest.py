"""시드 데이터와 미라벨 이미지를 DB에 등록한다.

시드 라벨은 사람이 만든 정답으로 보고 곧바로 approved 상태로 넣는다.
따라서 첫 학습은 시드만으로 시작하고, 검수가 쌓일수록 학습 데이터가 늘어난다.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass

from .config import Config
from .provenance import Camera, camera_key, capture_time, load_cameras
from .store import Store
from .yolo_io import iter_images, image_size, label_path_for, read_yolo_txt


@dataclass
class IngestReport:
    seed_images: int = 0
    seed_boxes: int = 0
    seed_without_label: int = 0
    unlabeled_images: int = 0
    seed_already: int = 0       # 이미 등록되어 건너뛴 시드
    unlabeled_already: int = 0  # 이미 등록되어 건너뛴 미라벨
    unlabeled_status: dict = None  # type: ignore[assignment]  # 폴더 내 이미지의 실제 상태별 수
    coco: object | None = None    # COCO 로 읽은 경우의 상세 리포트
    unreadable: list[str] = None  # type: ignore[assignment]
    # 내용이 이미 있는 이미지와 같아 등록하지 않은 것 (dupcheck.py). {path, same_as, same_boxes}
    duplicates: list[dict] = None  # type: ignore[assignment]
    # 출처 (provenance.py): 새로 넣은 이미지의 카메라별 수, 그중 촬영 시각을 못 찾은 수,
    # 카메라 폴더 없이 unlabeled/ 바로 아래 있던 새 이미지 수
    by_camera: dict = None  # type: ignore[assignment]
    no_time: dict = None  # type: ignore[assignment]
    no_camera: int = 0

    def __post_init__(self) -> None:
        if self.unreadable is None:
            self.unreadable = []
        if self.unlabeled_status is None:
            self.unlabeled_status = {}
        if self.duplicates is None:
            self.duplicates = []
        if self.by_camera is None:
            self.by_camera = {}
        if self.no_time is None:
            self.no_time = {}

    def summary(self) -> str:
        new_seed = self.seed_images - self.seed_already
        new_unlab = self.unlabeled_images - self.unlabeled_already

        # 폴더에 있는 파일 수가 아니라 '지금 어떤 상태인지'로 보여준다.
        # 승인된 이미지는 원본이 이 폴더에 남아 있어도 미라벨이 아니다.
        labels = {
            "unlabeled": "미라벨",
            "pending": "검수 대기",
            "approved": "승인",
            "skipped": "건너뜀",
        }
        detail = ", ".join(
            f"{labels[k]} {self.unlabeled_status[k]}"
            for k in ("unlabeled", "pending", "approved", "skipped")
            if self.unlabeled_status.get(k)
        )
        lines = [
            f"시드 이미지        : {self.seed_images}장"
            f"  (신규 {new_seed}장 / 박스 {self.seed_boxes}개, 기존 {self.seed_already}장)",
            f"자동 라벨링 폴더   : {self.unlabeled_images}장"
            f"  (신규 {new_unlab}장" + (f", {detail}" if detail else "") + ")",
        ]
        if self.coco is not None:
            lines.insert(0, self.coco.summary())
        if self.seed_without_label:
            lines.append(
                f"  주의: 객체가 하나도 없는 시드 이미지 {self.seed_without_label}장 "
                "-> 배경 샘플로 학습에 포함됩니다."
            )
        for cam, n in sorted(self.by_camera.items()):
            miss = self.no_time.get(cam, 0)
            lines.append(f"  카메라 {cam}: 신규 {n}장" + (f" (촬영 시각 모름 {miss}장)" if miss else ""))
        if self.no_camera:
            lines.append(
                f"  주의: 카메라 폴더 없이 들어온 새 이미지 {self.no_camera}장 -> 출처 없음(source='unlabeled')."
                " 출처를 남기려면 unlabeled/<카메라키>/ 아래에 둘 것 (docs/provenance.md)")
        if self.unreadable:
            lines.append(f"  읽지 못한 파일 {len(self.unreadable)}개: {self.unreadable[:5]}")
        if self.duplicates:
            diff = sum(1 for d in self.duplicates if d["same_boxes"] is False)
            lines.append(
                f"  [중복] 이미 있는 이미지와 내용이 같아 등록하지 않은 파일 {len(self.duplicates)}장"
                + (f" — 그중 박스가 다른 것 {diff}장 (로보플로에서 라벨을 고친 것일 수 있음)"
                   if diff else "")
                + ". 목록: labeler.py dupcheck --folder <폴더>")
            for d in self.duplicates[:5]:
                s = d["same_as"]
                where = f"id {s['id']} ({s['status']}, {s['split'] or '미배정'})" if s["id"] else "이번 스캔의 다른 파일"
                tag = {True: "박스 같음", False: "박스 다름", None: ""}[d["same_boxes"]]
                lines.append(f"      {d['path']}\n        = {where} {s['path']} {tag}")
        return "\n".join(lines)


class _DupGuard:
    """새로 등록하려는 파일이 이미 있는 이미지와 **내용(바이트)이 같으면** 막는다 (2026-09-23).

    경로만 보던 예전에는 로보플로 재내보내기(새 `.rf.` 이름)가 전부 새 이미지로 들어가, 학습
    데이터가 겹치고 같은 사진이 train·test 에 동시에 들어갈 수 있었다. 막은 것은 리포트에
    남긴다 — 박스가 다른 중복을 건너뛸지 갱신할지는 사람이 정한다(dupcheck.py 머리말).
    config 의 `data.allow_duplicates: true` 로 끈다. 색인은 새 파일이 처음 나올 때 만든다.
    """

    def __init__(self, cfg: Config, store: Store, report: IngestReport):
        self.on = not bool(cfg.get("data.allow_duplicates", False))
        self.store, self.report = store, report
        self._idx = None

    def blocked(self, path: str, boxes=None) -> bool:
        if not self.on:
            return False
        from .dupcheck import DupIndex, boxes_equal

        if self._idx is None:
            self._idx = DupIndex([{k: r[k] for k in ("id", "path", "status", "split", "n_boxes")}
                                  for r in self.store.list_images(status="all", limit=10**9)])
        hit = self._idx.find(path)
        if hit is None:
            self._idx.add(path, {"id": None, "path": path, "status": "(이번 스캔)",
                                 "split": None, "n_boxes": len(boxes or [])})
            return False
        same = None
        if boxes is not None and hit["id"] is not None:
            same = boxes_equal(boxes, self.store.get_boxes(hit["id"]))
        self.report.duplicates.append({"path": path, "same_as": hit, "same_boxes": same})
        return True


def discover_classes(cfg: Config, store: Store) -> list[str]:
    """클래스 목록 확정. 우선순위: config.yaml > classes.txt(DB 옆) > 시드 라벨에서 추론.

    **config 목록보다 뒤에 있는 클래스는 건드리지 않는다** — 검수 UI 의 '클래스 추가'로
    붙인 것이 그 자리에 있기 때문이다 (`store.set_classes` 의 drop_extra 참고).
    """
    from_cfg = [str(c) for c in (cfg.get("classes") or [])]
    if from_cfg:
        store.set_classes(from_cfg)
        return store.get_classes()

    if cfg.classes_txt.exists():
        names = [
            ln.strip()
            for ln in cfg.classes_txt.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")
        ]
        if names:
            store.set_classes(names)
            return store.get_classes()

    existing = store.get_classes()
    if existing:
        return existing

    # 이름을 알 수 없으면 이미 등록된 박스의 최대 클래스 ID로 개수만 추론하고
    # 임시 이름을 붙인다. (ingest 뒤에 호출해야 의미가 있다.)
    max_id = store.max_class_id()
    names = [f"class_{i}" for i in range(max_id + 1)]
    if names:
        store.set_classes(names)
        cfg.classes_txt.parent.mkdir(parents=True, exist_ok=True)
        cfg.classes_txt.write_text("\n".join(names) + "\n", encoding="utf-8")
    return names


_BATCH = 500


def _key(path: str) -> str:
    """'이미 등록됐나' 를 볼 때 쓰는 경로 키 — Windows 는 대소문자와 / \\ 를 가리지 않으니 그렇게 맞춘다.

    2026-11-01 전에는 DB 문자열과 정확히 같아야 해서, 저장소를 옮기며 images.path 앞부분을 `d:\\...` 처럼
    표기만 다르게 고치면 다음 스캔이 시드·미라벨 이미지를 전부 새 행으로 다시 넣었다(DupIndex 는 normcase 로
    같은 파일이라 보고 막지 않았다). 시드는 검수 전 라벨을 단 approved 로 들어와 고친 라벨과 겹쳤다.
    """
    return os.path.normcase(path)


def _check_db_paths(paths: Iterable[str], *, max_ratio: float = 0.5) -> None:
    """DB 이미지의 절반 넘게 디스크에 없으면 스캔하지 않는다 (2026-11-01, prune 의 안전장치와 같은 비율).

    저장소를 다른 경로에 두고 images.path 를 고치기 전에 스캔하면, 옛 행은 '파일 없음' 이라 중복 검사에도
    걸리지 않고 새 경로의 시드가 검수 전 라벨로 다시 approved 로 들어간다 — 그 뒤엔 경로를 고치는 UPDATE 도
    UNIQUE 로 실패한다. 드라이브가 빠진 채 스캔한 경우도 같다.
    """
    paths = list(paths)
    if not paths:
        return
    missing = sum(1 for p in paths if not os.path.exists(p))
    if missing / len(paths) > max_ratio:
        raise RuntimeError(
            f"DB 의 이미지 {len(paths)}장 중 {missing}장({missing / len(paths):.0%})이 디스크에 없어 스캔하지 "
            "않았습니다. 저장소를 옮겼다면 README '새 PC 에서 시작하기' 5단계로 DB 경로부터 고치고, 드라이브가 "
            "빠졌다면 연결하세요 — 이대로 스캔하면 같은 사진이 검수 전 라벨로 다시 등록됩니다. "
            "그래도 스캔하려면 labeler.py scan --force."
        )


def ingest(cfg: Config, store: Store, *, force: bool = False) -> IngestReport:
    """폴더를 훑어 아직 DB에 없는 이미지만 등록한다.

    이미 등록된 이미지는 건너뛴다. 재스캔이 빨라지고, 검수 UI에서 손본 라벨이
    원본 .txt 로 되덮이지 않는다. DB 경로가 디스크와 맞지 않으면(절반 넘게 '파일 없음')
    force 가 아닌 한 멈춘다 (`_check_db_paths`).
    """
    raw = store.known_statuses()
    if not force:
        _check_db_paths(raw)
    report = IngestReport()
    statuses = {_key(p): st for p, st in raw.items()}
    known = set(statuses)
    pending: list[dict] = []
    guard = _DupGuard(cfg, store, report)
    # 카메라 확인은 **무엇이든 넣기 전에** 한다 — 등록 안 된 키가 있으면 시드도 미라벨도 넣지 않고 멈춘다.
    unlabeled = list(iter_images(cfg.unlabeled))
    cameras = _plan_cameras(cfg, unlabeled, known)

    def flush() -> None:
        store.ingest_batch(pending)
        pending.clear()

    def rest() -> IngestReport:
        return _ingest_unlabeled(cfg, store, known, pending, report, flush, statuses, guard,
                                 unlabeled, cameras)

    # --- 시드: 사람이 만든 정답이므로 곧바로 approved ---
    if cfg.seed_coco:
        _ingest_coco(cfg, store, known, pending, report, flush, guard)
        flush()
        return rest()

    for img in iter_images(cfg.seed_images):
        resolved = str(img.resolve())
        if _key(resolved) in known:
            report.seed_images += 1
            report.seed_already += 1
            continue
        try:
            w, h = image_size(img)
        except Exception as exc:  # 깨진 파일은 건너뛴다
            report.unreadable.append(f"{img.name}: {exc}")
            continue
        txt = label_path_for(img, cfg.seed_images, cfg.seed_labels)
        boxes = read_yolo_txt(txt, origin="seed")
        if guard.blocked(resolved, boxes):
            continue              # 내용이 같은 이미지가 이미 있다 (리포트에 남는다)
        if not txt.exists():
            report.seed_without_label += 1
        known.add(_key(resolved))  # 같은 스캔 안에서 두 번 나와도 한 번만 넣는다
        pending.append(
            {
                "path": resolved,
                "source": "seed",
                "width": w,
                "height": h,
                "status": "approved",
                "boxes": boxes,
                "uncertainty": 0.0,
                "model_version": "seed",
            }
        )
        report.seed_images += 1
        report.seed_boxes += len(boxes)
        if len(pending) >= _BATCH:
            flush()
    flush()

    return rest()


def _seed_root(cfg: Config):
    return (cfg.seed_coco or cfg.seed_images).resolve()


def _plan_cameras(cfg: Config, images: list, known: set[str]) -> dict[str, Camera]:
    """새로 들어올 미라벨 이미지의 카메라 폴더를 등록 파일과 맞춰 본다 (provenance.py 머리말).

    카메라 폴더가 하나도 없으면 등록 파일을 읽지도 않는다(옛 방식 그대로). 등록 파일이 잘못됐거나 등록 안 된
    키가 있으면 RuntimeError — 이미 등록된 이미지(재스캔)는 보지 않으므로, 예전 하위 폴더(`unlabeled/o365/` 등)
    는 걸리지 않는다.
    """
    root = cfg.unlabeled.resolve()
    seed_root = _seed_root(cfg)
    keys: dict[str, int] = {}
    for img in images:
        r = img.resolve()
        if seed_root in r.parents or _key(str(r)) in known:
            continue
        k = camera_key(r, root)
        if k:
            keys[k] = keys.get(k, 0) + 1
    if not keys:
        return {}
    try:
        cams = load_cameras(cfg.cameras)
    except ValueError as exc:
        raise RuntimeError(f"카메라 등록 파일을 고친 뒤 다시 스캔하세요. 아무것도 등록하지 않았습니다.\n{exc}") from exc
    unknown = sorted(k for k in keys if k not in cams)
    if unknown:
        listed = ", ".join(f"{k}({keys[k]}장)" for k in unknown)
        raise RuntimeError(
            f"등록 안 된 카메라 폴더가 있어 스캔을 멈췄습니다(아무것도 등록하지 않음): {listed}\n"
            f"  {cfg.cameras} 의 cameras: 아래에 추가하거나 폴더 이름을 고치세요 — 오타가 새 카메라로 들어가면 "
            "같은 카메라가 train·test 로 갈릴 수 있습니다 (docs/provenance.md)."
        )
    return cams


def _ingest_coco(cfg, store, known, pending, report, flush, guard) -> None:
    """COCO JSON 에서 시드를 읽는다. 폴리곤을 거치지 않으므로 박스가 잘리지 않는다."""
    from .coco_io import category_names, find_annotation_files, read_coco

    # COCO 는 카테고리 이름으로 매핑하므로 읽기 전에 클래스가 확정돼 있어야 한다.
    # (discover_classes 는 ingest 뒤에 도는 구조라 여기서 직접 정한다.)
    classes = [str(c) for c in (cfg.get("classes") or [])] or store.get_classes()
    if not classes:
        classes = category_names(find_annotation_files(cfg.seed_coco))
    if not classes:
        raise RuntimeError(
            f"클래스를 정할 수 없습니다. config.yaml 의 classes 를 채우거나 "
            f"{cfg.seed_coco} 의 COCO categories 를 확인하세요."
        )
    store.set_classes(classes)
    # UI 에서 추가한 클래스는 config 목록 뒤에 남아 있다. 그 이름이 COCO 카테고리에
    # 있으면 같은 인덱스로 받아야 하므로 합친 목록으로 매핑한다.
    classes = store.get_classes()
    index = {n.strip().lower(): i for i, n in enumerate(classes)}
    records, creport = read_coco(cfg.seed_coco, index)
    report.coco = creport

    for rec in records:
        resolved = str(rec.path.resolve())
        if _key(resolved) in known:
            report.seed_images += 1
            report.seed_already += 1
            continue
        if guard.blocked(resolved, rec.boxes):
            continue              # 내용이 같은 이미지가 이미 있다 (리포트에 남는다)
        if not rec.boxes:
            report.seed_without_label += 1
        # 같은 이미지를 가리키는 COCO 항목이 둘이면(json 두 개가 같은 images/ 를 볼 때 등)
        # 첫 것만 넣는다. 둘 다 넣으려 하면 DB 가 경로 중복으로 두 번째를 거른다.
        known.add(_key(resolved))
        pending.append({
            "path": resolved,
            "source": "seed",
            "width": rec.width,
            "height": rec.height,
            "status": "approved",
            "boxes": rec.boxes,
            "uncertainty": 0.0,
            "model_version": "seed-coco",
        })
        report.seed_images += 1
        report.seed_boxes += len(rec.boxes)
        if len(pending) >= _BATCH:
            flush()


def _ingest_unlabeled(cfg, store, known, pending, report, flush, statuses, guard,
                      images, cameras) -> "IngestReport":
    # --- 미라벨: 아직 추론 전 ---
    seed_root = _seed_root(cfg)
    root = cfg.unlabeled.resolve()
    for img in images:
        resolved_path = img.resolve()
        if seed_root in resolved_path.parents:
            continue  # 시드 폴더가 미라벨 폴더 안에 있는 경우 중복 방지
        resolved = str(resolved_path)
        if _key(resolved) in known:
            report.unlabeled_images += 1
            report.unlabeled_already += 1
            st = statuses.get(_key(resolved), "unlabeled")
            report.unlabeled_status[st] = report.unlabeled_status.get(st, 0) + 1
            continue
        try:
            w, h = image_size(img)
        except Exception as exc:
            report.unreadable.append(f"{img.name}: {exc}")
            continue
        if guard.blocked(resolved):
            continue              # 내용이 같은 이미지가 이미 있다 (리포트에 남는다)
        known.add(_key(resolved))
        # 카메라 폴더 아래면 source='collected' + 출처. 'unlabeled' 는 tools/test_baseline.py 에서 '비CCTV'
        # 를 뜻하므로 새 수집분을 거기 섞지 않는다. 바로 아래 파일은 옛 방식대로(출처 없음).
        key = camera_key(resolved_path, root)
        item = {"path": resolved, "source": "unlabeled", "width": w, "height": h,
                "status": "unlabeled", "boxes": []}
        if key:
            cam = cameras[key]  # _plan_cameras 가 등록 여부를 이미 확인했다
            item.update(source="collected", camera=key, captured_at=capture_time(resolved_path, cam.tz))
            report.by_camera[key] = report.by_camera.get(key, 0) + 1
            if item["captured_at"] is None:
                report.no_time[key] = report.no_time.get(key, 0) + 1
        else:
            report.no_camera += 1
        pending.append(item)
        report.unlabeled_images += 1
        report.unlabeled_status["unlabeled"] = report.unlabeled_status.get("unlabeled", 0) + 1
        if len(pending) >= _BATCH:
            flush()
    flush()

    return report
