"""데이터 점검의 수치가 학습 세트·내보내기가 **실제로 쓰는 라벨**과 같은가 (2026-11-15 리뷰 1·12번).

데이터 점검은 '가릴 영역이 반 이상 덮은 박스는 학습 라벨에서 빠진다' 를 따로 적어 세다가 test 분할의 예외(칠하지도, 박스를
빼지도 않는다)를 빠뜨렸다. 이제 세 곳(trainer.build_dataset·export_yolo·health)이 `masks.drops`·`masks.applies_to` 를 함께
쓰는데, 여기서는 그 약속을 결과로 지킨다 — 진짜 학습 세트와 내보내기를 임시 폴더에 만들어 라벨 줄 수를 분할·클래스별로 세고
`health.report` 와 견준다. 셋 중 하나가 판정을 따로 적기 시작하면 여기서 어긋난다.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from autolabel import health
from autolabel.export import export_yolo
from autolabel.trainer import build_dataset
from helpers import CLASSES, box, far_hashes

PERSON, CAR, TRUCK = CLASSES.index("person"), CLASSES.index("car"), CLASSES.index("truck")
SPLITS = ("train", "val", "test")


def label_counts(folder: Path) -> Counter:
    """라벨 폴더의 .txt 를 전부 읽어 클래스별 줄 수."""
    got: Counter = Counter()
    for txt in folder.glob("*.txt"):
        for line in txt.read_text(encoding="utf-8").splitlines():
            if line.strip():
                got[int(line.split()[0])] += 1
    return got


@pytest.fixture
def masked(store, add_image):
    """카메라 셋(train·val·test) — 장마다 통째로 덮인 차, 정확히 반이 덮인 트럭, 조금만 덮인 트럭, 안 덮인 사람.
    네 번째 장(train)은 영역이 없다. 영역은 test 장에도 있다(검수하며 먼저 덮고 분할은 나중에 정해진 때)."""
    boxes = [
        box(CAR, 0.25, 0.25, 0.2, 0.2),         # 0.15~0.35 — 영역 안에 통째로
        box(TRUCK, 0.5, 0.5, 0.5, 0.5),         # 0.25~0.75 — 왼쪽 반(0.25~0.5)이 덮인다: 정확히 반 → 뺀다
        box(TRUCK, 0.5625, 0.5, 0.5, 0.5),      # 0.3125~0.8125 — 덮인 몫 0.375 → 남는다
        box(PERSON, 0.9, 0.9, 0.05, 0.1),       # 안 덮인다
    ]
    rects = [(0.1, 0.1, 0.4, 0.4), (0.25, 0.25, 0.5, 0.75)]
    hashes = far_hashes(4)
    ids = {sp: add_image(f"{sp}.jpg", camera=f"cam-{k}", boxes=boxes, dhash=hashes[k], split=sp)
           for k, sp in enumerate(SPLITS)}
    ids["plain"] = add_image("plain.jpg", camera="cam-0", boxes=[box(CAR), box(PERSON, 0.1, 0.1, 0.05, 0.1)],
                             dhash=hashes[3], split="train")
    for image_id in ids.values():
        path = Path(store.get_image(image_id)["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (1280, 720), (90, 120, 150)).save(path, quality=90)
    for sp in SPLITS:
        store.set_masks(ids[sp], rects)
    return ids


def report(store, drop=True):
    return health.report(store.health_snapshot(), store.get_classes(), {}, scope="approved", imgsz=640,
                         val_ratio=0.2, test_ratio=0.15, drop_masked=drop)


def test_the_report_counts_what_the_training_set_labels(cfg, store, masked):
    r = report(store)
    built = build_dataset(cfg, store)
    assert (built.masked, built.masked_test, built.mask_dropped) == (2, 1, 4)     # train·val 을 칠하고 둘씩 뺐다, test 는 그대로

    for sp in SPLITS:
        got = label_counts(cfg.dataset / "labels" / sp)
        assert sum(got.values()) == r["splits"]["boxes"][sp], sp
        for row in r["classes"]:
            assert got.get(row["idx"], 0) == row["split"][sp], (sp, row["name"])
    # 손으로 센 값 — train: 덮인 장(트럭 1·사람 1) + 안 덮인 장(차 1·사람 1), val: 트럭 1·사람 1, test: 넷 다
    assert r["splits"]["boxes"] == {"train": 4, "val": 2, "test": 4, "none": 0}
    assert (r["masks"]["dropped"], r["masks"]["test_images"]) == (4, 1)
    assert not any(n["text"].startswith("test 에 박스가 없는 클래스") and "car" in n["text"] for n in r["notes"])


def test_the_report_counts_what_an_export_with_images_labels(cfg, store, masked, tmp_path):
    r = report(store)
    out = tmp_path / "exported"
    done = export_yolo(cfg, store, out, with_images=True)
    got = label_counts(out / "labels")
    assert (sum(got.values()), done.boxes) == (r["boxes"], r["boxes"])
    assert {row["idx"]: row["boxes"] for row in r["classes"] if row["boxes"]} == dict(got)
    assert done.mask_dropped == r["masks"]["dropped"]


def test_keeping_masked_boxes_counts_the_database(cfg, store, masked):
    """'넣고 셈' 은 DB 의 박스 전부다 — 라벨만 내보낼 때(그림을 칠하지 않으니 박스도 빼지 않는다)와 같다."""
    r = report(store, drop=False)
    out = cfg.dataset.parent / "labels_only"
    export_yolo(cfg, store, out, with_images=False)
    got = label_counts(out / "labels")
    assert sum(got.values()) == r["boxes"] == 14
    assert {row["idx"]: row["boxes"] for row in r["classes"] if row["boxes"]} == dict(got)
