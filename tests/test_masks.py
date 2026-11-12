"""가릴 영역 (masks.py) — 덮는 몫이 틀리면 학습 라벨에서 박스가 잘못 빠지거나 남는다."""

from __future__ import annotations

import random

import pytest
from PIL import Image

from autolabel import masks
from autolabel.masks import DROP, GRAY, clean, covered, mask_key, merge, paint, pixel_rect, split_boxes, subtract, write_masked
from autolabel.yolo_io import Box

# x 0.25~0.75, y 0.25~0.75 — 2의 거듭제곱 분수라 넓이 계산에 반올림 오차가 없다
BOX = Box(2, 0.5, 0.5, 0.5, 0.5)


def reference_cover(box: Box, rects) -> float:
    """covered 와 따로 짠 느린 기준 — 경계로 칸을 나누고 칸 가운데가 어느 영역에든 들면 더한다."""
    bx1, by1, bx2, by2 = box.cx - box.w / 2, box.cy - box.h / 2, box.cx + box.w / 2, box.cy + box.h / 2
    area = (bx2 - bx1) * (by2 - by1)
    if area <= 0:
        return 0.0
    xs = sorted({bx1, bx2, *(min(max(v, bx1), bx2) for r in rects for v in (r[0], r[2]))})
    ys = sorted({by1, by2, *(min(max(v, by1), by2) for r in rects for v in (r[1], r[3]))})
    inside = 0.0
    for x1, x2 in zip(xs, xs[1:]):
        for y1, y2 in zip(ys, ys[1:]):
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            if any(r[0] < mx < r[2] and r[1] < my < r[3] for r in rects):
                inside += (x2 - x1) * (y2 - y1)
    return min(1.0, inside / area)


# ----------------------------------------------------------------- covered

def test_cover_of_half_is_exactly_half():
    assert covered(BOX, [(0.25, 0.25, 0.5, 0.75)]) == 0.5


def test_overlapping_rects_are_not_counted_twice():
    left = (0.25, 0.25, 0.5, 0.75)
    assert covered(BOX, [left, left]) == 0.5                                        # 같은 영역 두 번
    assert covered(BOX, [left, (0.375, 0.25, 0.625, 0.75)]) == 0.75                 # 반쯤 겹친 두 영역
    # 십자 — 가로 막대 0.5×0.125, 세로 막대 0.125×0.5, 겹친 0.125×0.125 는 한 번만
    cross = [(0.25, 0.4375, 0.75, 0.5625), (0.4375, 0.25, 0.5625, 0.75)]
    assert covered(BOX, cross) == pytest.approx((0.0625 + 0.0625 - 0.015625) / 0.25)


def test_cover_edges():
    assert covered(BOX, []) == 0.0
    assert covered(BOX, [(0.8, 0.8, 0.9, 0.9)]) == 0.0                              # 박스 밖
    assert covered(BOX, [(0.75, 0.25, 0.9, 0.75)]) == 0.0                           # 변만 닿는다
    assert covered(BOX, [(0.0, 0.0, 1.0, 1.0)]) == 1.0
    assert covered(Box(2, 0.5, 0.5, 0.0, 0.5), [(0.0, 0.0, 1.0, 1.0)]) == 0.0       # 넓이 없는 박스


def test_cover_matches_slow_reference_on_random_cases():
    rng = random.Random(20261112)
    grid = [i / 16 for i in range(17)]              # 경계가 겹치는 경우가 자주 나오게 성긴 격자도 섞는다
    for case in range(400):
        pick = (lambda: rng.choice(grid)) if case % 2 else rng.random
        b = Box(0, 0.2 + 0.6 * rng.random(), 0.2 + 0.6 * rng.random(), 0.05 + 0.3 * rng.random(), 0.05 + 0.3 * rng.random())
        rects = []
        for _ in range(rng.randrange(0, 12)):
            x1, x2 = sorted((pick(), pick()))
            y1, y2 = sorted((pick(), pick()))
            if x2 > x1 and y2 > y1:
                rects.append((x1, y1, x2, y2))
        assert covered(b, rects) == pytest.approx(reference_cover(b, rects), abs=1e-12), (case, b, rects)


# ----------------------------------------------------------------- 학습 라벨에서 뺄 박스

def test_split_boxes_drops_from_half_covered():
    assert DROP == 0.5
    half = Box(2, 0.5, 0.5, 0.5, 0.5)                   # 정확히 반이 덮인다 → 뺀다(이상)
    less = Box(2, 0.5625, 0.5, 0.5, 0.5)                # 덮인 몫 0.375 → 남긴다
    out = Box(2, 0.9, 0.9, 0.1, 0.1)
    keep, drop = split_boxes([half, less, out], [(0.25, 0.25, 0.5, 0.75)])
    assert drop == [half]
    assert keep == [less, out]


def test_split_boxes_without_rects_keeps_everything():
    boxes = [BOX, Box(0, 0.1, 0.1, 0.1, 0.1)]
    assert split_boxes(boxes, []) == (boxes, [])


# ----------------------------------------------------------------- 영역 목록 다루기

def test_clean_clamps_sorts_and_drops_bad_rects():
    got = clean([
        (0.6, 0.7, 0.2, 0.1),                   # 뒤집힌 좌표 → 바로 세운다
        (-0.5, 0.5, 1.5, 0.75),                 # 화면 밖 → 0~1 로 자른다
        (0.5, 0.5, 0.5, 0.75),                  # 폭 0
        (0.5, 0.5, 0.50005, 0.75),              # MIN_SIDE 보다 좁다
        (float("nan"), 0.0, 1.0, 1.0),
        (0.0, 0.0, float("inf"), 1.0),
        ("a", 0, 1, 1),
        (0.0, 0.0, 1.0),                        # 값이 셋
    ])
    assert got == [(0.2, 0.1, 0.6, 0.7), (0.0, 0.5, 1.0, 0.75)]


def test_merge_skips_same_place_and_keeps_order():
    cur = [(0.125, 0.125, 0.25, 0.25), (0.5, 0.5, 0.75, 0.75)]
    add = [(0.5 + 1e-9, 0.5, 0.75, 0.75),       # 부동소수 잡음 — 같은 자리
           (0.0, 0.0, 0.0625, 0.0625),
           (0.0, 0.0, 0.0625, 0.0625)]          # 더할 것 안에서 겹친 것도 한 번만
    assert merge(cur, add) == [*cur, (0.0, 0.0, 0.0625, 0.0625)]


def test_subtract_removes_only_same_place():
    cur = [(0.125, 0.125, 0.25, 0.25), (0.5, 0.5, 0.75, 0.75)]
    moved = (0.5, 0.5, 0.75, 0.75 + 1e-4)       # 손으로 옮긴 영역은 그 이미지의 것 — 남긴다
    assert subtract(cur, [(0.125 - 1e-9, 0.125, 0.25, 0.25), moved]) == [(0.5, 0.5, 0.75, 0.75)]


def test_mask_key_ignores_float_noise():
    assert mask_key([(0.1, 0.2, 0.3, 0.4)]) == mask_key([(0.1 + 1e-9, 0.2, 0.3 - 1e-9, 0.4)])
    assert mask_key([(0.1, 0.2, 0.3, 0.4)]) != mask_key([(0.1 + 1e-6, 0.2, 0.3, 0.4)])


# ----------------------------------------------------------------- 칠하기

def test_pixel_rect_round_trips_pixel_aligned_rects():
    w, h = 2560, 1440
    for x1, y1, x2, y2 in [(0, 0, 2560, 1440), (17, 3, 1001, 777), (2559, 1439, 2560, 1440)]:
        assert pixel_rect((x1 / w, y1 / h, x2 / w, y2 / h), w, h) == (x1, y1, x2, y2)


def test_paint_grays_only_the_rect():
    img = Image.new("RGB", (40, 20), (200, 10, 10))
    out = paint(img, [(0.25, 0.5, 0.75, 1.0)])          # x 10~30, y 10~20
    assert img.getpixel((15, 15)) == (200, 10, 10)      # 받은 그림은 그대로
    assert out.getpixel((10, 10)) == out.getpixel((29, 19)) == (GRAY, GRAY, GRAY)
    for xy in [(9, 15), (30, 15), (15, 9)]:
        assert out.getpixel(xy) == (200, 10, 10)


def test_write_masked_leaves_the_original_and_applies_rotation(tmp_path):
    src = tmp_path / "src.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6                                    # 화면에서는 세로 20×40 으로 보인다
    Image.new("RGB", (40, 20), (200, 10, 10)).save(src, quality=95, exif=exif)
    before = src.read_bytes()

    dst = tmp_path / "out" / "dst.jpg"
    write_masked(src, dst, [(0.0, 0.0, 1.0, 0.25)])     # 화면 기준 위쪽 1/4
    assert src.read_bytes() == before
    with Image.open(dst) as im:
        assert im.size == (20, 40)                      # 회전을 적용해 썼다
        assert im.getexif().get(0x0112) is None         # 그래서 회전 태그가 없다
        top, bottom = im.getpixel((10, 3)), im.getpixel((10, 30))
    assert all(abs(v - GRAY) <= 6 for v in top)
    assert bottom[0] > 150 and bottom[1] < 60
    assert not list(dst.parent.glob("*.tmp"))


def test_limits_are_what_the_docs_say():
    # 바꾸면 화면(app.js MASK_MAX·MASK_DROP)·문서도 함께 — 화면과의 대조는 test_js_parity.py
    assert masks.MAX_MASKS == 500
    assert masks.MIN_SIDE == 1e-4
