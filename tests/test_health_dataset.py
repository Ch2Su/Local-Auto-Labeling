"""데이터 점검의 수치가 학습 세트·내보내기가 **실제로 쓰는 라벨**과 같은가 (2026-11-15 리뷰 1·12번).

데이터 점검은 '가릴 영역이 반 이상 덮은 박스는 학습 라벨에서 빠진다' 를 따로 적어 세다가 그때의 test 분할 예외를 빠뜨렸다.
이제 세 곳(trainer.build_dataset·export_yolo·health)이 `masks.drops`(분할과 상관없이 — 2026-11-24 부터 test 도)와
`masks.paints`(test 는 칠하지 않는다)를 함께 쓰는데, 여기서는 그 약속을 결과로 지킨다 — 진짜 학습 세트와 내보내기를 임시 폴더에 만들어 라벨 줄 수를 분할·클래스별로 세고
`health.report` 와 견준다. 셋 중 하나가 판정을 따로 적기 시작하면 여기서 어긋난다.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

import numpy as np

from autolabel import health
from autolabel.export import export_yolo
from autolabel.masks import paint_array
from autolabel.trainer import build_dataset
from helpers import CLASSES, ROOT, box, far_hashes

sys.path.insert(0, str(ROOT / "wsl"))
from eval_test import ignore_path, load_ignore  # noqa: E402  — 평가 도구가 그 그림을 찾는 규칙 그대로

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
    assert (built.masked, built.masked_test, built.mask_dropped) == (2, 1, 6)     # train·val 을 칠하고, 세 분할에서 둘씩 뺐다

    for sp in SPLITS:
        got = label_counts(cfg.dataset / "labels" / sp)
        assert sum(got.values()) == r["splits"]["boxes"][sp], sp
        for row in r["classes"]:
            assert got.get(row["idx"], 0) == row["split"][sp], (sp, row["name"])
    # 손으로 센 값 — train: 덮인 장(트럭 1·사람 1) + 안 덮인 장(차 1·사람 1), val·test: 트럭 1·사람 1
    assert r["splits"]["boxes"] == {"train": 4, "val": 2, "test": 2, "none": 0}
    assert (r["masks"]["dropped"], r["masks"]["test_images"]) == (6, 1)
    # test 장은 칠하지 않고(원본 색 그대로) 칠할 자리를 ignore/ 그림으로 남긴다 — 평가 도구가 읽는 자리에, 칠할 바로 그 픽셀로
    test_img = next((cfg.dataset / "images" / "test").glob("test__*"))
    with Image.open(test_img) as im:
        assert abs(im.getpixel((int(0.2 * 1280), int(0.2 * 720)))[2] - 150) <= 8
    ignore = ignore_path(test_img)
    assert ignore == cfg.dataset / "ignore" / "test" / f"{test_img.stem}.png" and ignore.is_file()
    want = paint_array(np.zeros((720, 1280), np.uint8), store.get_masks(masked["test"])) > 0
    assert (load_ignore(test_img) == want).all()
    assert not list((cfg.dataset / "ignore").glob("train/*")) and not list((cfg.dataset / "ignore").glob("val/*"))


def test_the_report_counts_what_an_export_with_images_labels(cfg, store, masked, tmp_path):
    r = report(store)
    out = tmp_path / "exported"
    done = export_yolo(cfg, store, out, with_images=True)
    got = label_counts(out / "labels")
    assert (sum(got.values()), done.boxes) == (r["boxes"], r["boxes"])
    assert {row["idx"]: row["boxes"] for row in r["classes"] if row["boxes"]} == dict(got)
    assert done.mask_dropped == r["masks"]["dropped"]


@pytest.fixture
def poly_masked(store, add_image):
    """다각형 영역 (2026-11-19) — train·val·test 장에 같은 박스와 같은 다각형 둘. 대각선으로 덮어 보는 자리:
    차는 왼쪽 위 삼각형(x+y≤0.8) 안에 통째로 → 뺀다, 큰 트럭은 x+y≤1.26 쪽이 68% → 뺀다, 작은 트럭은 43% → 남는다,
    사람은 안 덮인다. 사각형으로 재면(바깥 사각형) 작은 트럭과 사람까지 덮인 것으로 나온다."""
    boxes = [
        box(CAR, 0.25, 0.25, 0.2, 0.2),         # 0.15~0.35 — 첫 삼각형 안
        box(TRUCK, 0.6, 0.6, 0.3, 0.3),         # 0.45~0.75 — 둘째 삼각형이 68%
        box(TRUCK, 0.62, 0.2, 0.3, 0.2),        # x 0.47~0.77, y 0.1~0.3 — 첫 삼각형이 43%
        box(PERSON, 0.9, 0.9, 0.05, 0.1),       # 안 덮인다
    ]
    polys = [((0.0, 0.0), (0.8, 0.0), (0.0, 0.8)), ((0.4, 0.4), (0.86, 0.4), (0.4, 0.86))]
    hashes = far_hashes(3, seed=5)
    ids = {sp: add_image(f"p_{sp}.jpg", camera=f"pc-{k}", boxes=boxes, dhash=hashes[k], split=sp)
           for k, sp in enumerate(SPLITS)}
    for image_id in ids.values():
        path = Path(store.get_image(image_id)["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (1280, 720), (90, 120, 150)).save(path, quality=90)
        store.set_masks(image_id, polys)
    return ids


def test_polygons_paint_and_drop_the_same_in_training_set_and_report(cfg, store, poly_masked):
    r = report(store)
    built = build_dataset(cfg, store)
    assert (built.masked, built.masked_test, built.mask_dropped) == (2, 1, 6)     # 세 분할에서 둘씩, test 는 칠하지 않는다
    for sp in SPLITS:
        got = label_counts(cfg.dataset / "labels" / sp)
        assert sum(got.values()) == r["splits"]["boxes"][sp], sp
    assert r["splits"]["boxes"] == {"train": 2, "val": 2, "test": 2, "none": 0}
    assert (r["masks"]["dropped"], r["masks"]["rects"], r["masks"]["test_images"]) == (6, 6, 1)

    # 칠한 그림 — 다각형 안은 회색, 바깥 사각형 안이라도 모양 밖은 그대로 (픽셀 중심 규칙)
    px = lambda im, x, y: im.getpixel((int(x * 1280), int(y * 720)))
    with Image.open(next((cfg.dataset / "images" / "train").glob("p_train__*.jpg"))) as im:
        first, second = px(im, 0.1, 0.1), px(im, 0.5, 0.5)            # 두 삼각형 안
        corner, beyond = px(im, 0.75, 0.1), px(im, 0.7, 0.7)            # 바깥 사각형 안, 모양 밖(x+y 0.85 > 0.8, 1.4 > 1.26)
    assert all(abs(v - 114) <= 6 for v in first) and all(abs(v - 114) <= 6 for v in second)
    assert abs(corner[2] - 150) <= 8 and abs(beyond[2] - 150) <= 8     # 원래 색(90, 120, 150)
    # test 장은 칠하지 않고 원본을 그대로 쓴다
    test_img = next((cfg.dataset / "images" / "test").glob("p_test__*"))
    with Image.open(test_img) as im:
        assert abs(im.getpixel((int(0.1 * 1280), int(0.1 * 720)))[2] - 150) <= 8


def test_keeping_masked_boxes_counts_the_database(cfg, store, masked):
    """'넣고 셈' 은 DB 의 박스 전부다. 라벨만 내보낼 때 train·val 은 그림을 칠하지 않으니 박스도 빼지 않고, test 는 채점에서
    빼는 자리라 뺀다(2026-11-24 — 그 전에는 test 도 DB 박스 그대로였다)."""
    r = report(store, drop=False)
    out = cfg.dataset.parent / "labels_only"
    done = export_yolo(cfg, store, out, with_images=False)
    got = label_counts(out / "labels")
    assert r["boxes"] == 14 and sum(got.values()) == 12                   # test 장에서 둘이 빠졌다
    assert (done.masked_raw, done.masked_test, done.mask_dropped) == (2, 1, 2)
    assert len(list((out / "ignore").glob("*.png"))) == 1                  # 원본이 있으니 그 자리 그림도 남긴다


# ------------------------------------------- 학습을 시작할 때의 알림 (trainer.BuildReport.class_gaps, 2026-11-16)

def test_the_training_log_names_the_classes_a_split_has_no_boxes_for(cfg, store, masked, add_image):
    """분할이 카메라째라 드문 클래스는 한 분할에만 들어가기 쉽다 — 학습 로그(요약)가 그것을 시작할 때 말한다.

    전에는 요약이 클래스별 합계만 보여 줘서, 박스가 test 에만 있는 클래스(배우지 못한다)를 학습이 끝난 뒤에야 알았다.
    수는 라벨 파일에 **실제로 쓴** 박스다(가릴 영역이 뺀 박스는 세지 않는다) — 데이터 점검의 분할별 수와 같아야 한다.
    """
    cone = CLASSES.index("cone")
    only_test = add_image("cone.jpg", camera="cam-2", boxes=[box(cone, 0.9, 0.2, 0.05, 0.05)], dhash=far_hashes(5)[4], split="test")
    path = Path(store.get_image(only_test)["path"])
    Image.new("RGB", (1280, 720), (90, 120, 150)).save(path, quality=90)

    built = build_dataset(cfg, store)
    r = report(store)
    for row in r["classes"]:
        got = built.per_split.get(row["name"], {"train": 0, "val": 0, "test": 0})
        assert got == {sp: row["split"][sp] for sp in SPLITS}, row["name"]

    gaps = built.class_gaps()
    none = next(g for g in gaps if g.startswith("[알림] 박스가 없는 클래스"))
    assert "bicycle" in none and "scooter" in none and "cone" not in none and "car" not in none
    assert next(g for g in gaps if g.startswith("[경고] train 에")).startswith(
        "[경고] train 에 박스가 없는 클래스: cone(test 1) — 이 클래스는 배우지 못합니다")
    # val 장의 차는 영역에 통째로 덮여 라벨에서 빠졌다 — val 에는 차가 없다. 라바콘도 없다
    assert next(g for g in gaps if g.startswith("[알림] val 에")).startswith("[알림] val 에 박스가 없는 클래스: car, cone ")
    # test 장의 차도 영역에 통째로 덮여 빠졌다(2026-11-24 부터 — 채점에서 빼는 자리)
    assert next(g for g in gaps if g.startswith("[알림] test 에")).startswith("[알림] test 에 박스가 없는 클래스: car ")
    assert all(g in built.summary() for g in gaps)                           # 학습 로그에 찍히는 요약에 들어 있다


def test_no_val_warning_when_validation_borrows_a_training_image(cfg, store, add_image):
    """val 이 아예 없어 train 한 장을 빌려 쓸 때는 'val 에 없는 클래스' 를 따지지 않는다(전부 없다고 나올 뿐이다)."""
    i = add_image("one.jpg", camera="cam-0", boxes=[box(CAR)], dhash=far_hashes(1)[0], split="train")
    path = Path(store.get_image(i)["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (1280, 720), (90, 120, 150)).save(path, quality=90)
    built = build_dataset(cfg, store)
    assert built.val_borrowed and not any("val 에" in g or "test 에" in g or "train 에" in g for g in built.class_gaps())
