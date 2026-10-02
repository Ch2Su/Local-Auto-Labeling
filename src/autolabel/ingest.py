"""시드 데이터와 미라벨 이미지를 DB에 등록한다.

시드 라벨은 사람이 만든 정답으로 보고 곧바로 approved 상태로 넣는다.
따라서 첫 학습은 시드만으로 시작하고, 검수가 쌓일수록 학습 데이터가 늘어난다.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config
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

    def __post_init__(self) -> None:
        if self.unreadable is None:
            self.unreadable = []
        if self.unlabeled_status is None:
            self.unlabeled_status = {}
        if self.duplicates is None:
            self.duplicates = []

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


def ingest(cfg: Config, store: Store) -> IngestReport:
    """폴더를 훑어 아직 DB에 없는 이미지만 등록한다.

    이미 등록된 이미지는 건너뛴다. 재스캔이 빨라지고, 검수 UI에서 손본 라벨이
    원본 .txt 로 되덮이지 않는다.
    """
    report = IngestReport()
    statuses = store.known_statuses()
    known = set(statuses)
    pending: list[dict] = []
    guard = _DupGuard(cfg, store, report)

    def flush() -> None:
        store.ingest_batch(pending)
        pending.clear()

    # --- 시드: 사람이 만든 정답이므로 곧바로 approved ---
    if cfg.seed_coco:
        _ingest_coco(cfg, store, known, pending, report, flush, guard)
        flush()
        return _ingest_unlabeled(cfg, store, known, pending, report, flush, statuses, guard)

    for img in iter_images(cfg.seed_images):
        resolved = str(img.resolve())
        if resolved in known:
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
        known.add(resolved)       # 같은 스캔 안에서 두 번 나와도 한 번만 넣는다
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

    return _ingest_unlabeled(cfg, store, known, pending, report, flush, statuses, guard)


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
        if resolved in known:
            report.seed_images += 1
            report.seed_already += 1
            continue
        if guard.blocked(resolved, rec.boxes):
            continue              # 내용이 같은 이미지가 이미 있다 (리포트에 남는다)
        if not rec.boxes:
            report.seed_without_label += 1
        # 같은 이미지를 가리키는 COCO 항목이 둘이면(json 두 개가 같은 images/ 를 볼 때 등)
        # 첫 것만 넣는다. 둘 다 넣으려 하면 DB 가 경로 중복으로 두 번째를 거른다.
        known.add(resolved)
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


def _ingest_unlabeled(cfg, store, known, pending, report, flush, statuses, guard) -> "IngestReport":
    # --- 미라벨: 아직 추론 전 ---
    seed_root = (cfg.seed_coco or cfg.seed_images).resolve()
    for img in iter_images(cfg.unlabeled):
        resolved_path = img.resolve()
        if seed_root in resolved_path.parents:
            continue  # 시드 폴더가 미라벨 폴더 안에 있는 경우 중복 방지
        resolved = str(resolved_path)
        if resolved in known:
            report.unlabeled_images += 1
            report.unlabeled_already += 1
            st = statuses.get(resolved, "unlabeled")
            report.unlabeled_status[st] = report.unlabeled_status.get(st, 0) + 1
            continue
        try:
            w, h = image_size(img)
        except Exception as exc:
            report.unreadable.append(f"{img.name}: {exc}")
            continue
        if guard.blocked(resolved):
            continue              # 내용이 같은 이미지가 이미 있다 (리포트에 남는다)
        known.add(resolved)
        pending.append(
            {
                "path": resolved,
                "source": "unlabeled",
                "width": w,
                "height": h,
                "status": "unlabeled",
                "boxes": [],
            }
        )
        report.unlabeled_images += 1
        report.unlabeled_status["unlabeled"] = report.unlabeled_status.get("unlabeled", 0) + 1
        if len(pending) >= _BATCH:
            flush()
    flush()

    return report
