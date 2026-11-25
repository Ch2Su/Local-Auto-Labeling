"""가릴 영역 (masks.py) — 덮는 몫이 틀리면 학습 라벨에서 박스가 잘못 빠지거나 남는다."""

from __future__ import annotations

import random

import pytest
from PIL import Image

from autolabel import masks
from autolabel.masks import (DROP, GRAY, SAME, clean, covered, drops, has_rect, merge, paint, pixel_rect,
                             same_rect, same_rects, split_boxes, subtract, write_masked)
from autolabel.yolo_io import Box
from helpers import pixel_rects, screen_round_trip

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


def test_drops_is_the_one_rule_split_boxes_uses():
    """박스 하나의 판정(`drops`)과 목록을 가르는 `split_boxes` 는 같은 답을 낸다 — 데이터 점검(health)은 `drops` 를 쓴다."""
    rects = [(0.25, 0.25, 0.5, 0.75)]
    boxes = [Box(2, 0.5, 0.5, 0.5, 0.5), Box(2, 0.5625, 0.5, 0.5, 0.5), Box(2, 0.9, 0.9, 0.1, 0.1)]
    assert [drops(b, rects) for b in boxes] == [True, False, False]
    assert split_boxes(boxes, rects)[1] == [b for b in boxes if drops(b, rects)]
    assert drops(boxes[0], []) is False                 # 영역이 없으면 빠질 것도 없다


def test_every_split_but_test_is_painted():
    """test 는 평가라 그림은 실제 화면 그대로 — 칠하지 않는다. 박스를 빼는 것(drops)은 분할과 상관없다(2026-11-24 — test 의
    영역은 채점에서 빼는 자리). 이 예외는 `paints` 한 곳에 둔다(결과는 tests/test_health_dataset.py 가 견준다)."""
    from autolabel.masks import paints

    assert [paints(s) for s in ("train", "val", None, "", "test")] == [True, True, True, True, False]


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


# ----------------------------------------------------------------- 같은 자리

def test_same_place_tolerance_sits_between_rounding_error_and_one_step():
    """SAME 은 7자리 반올림으로 생기는 차(5e-8)보다 크고 7자리 한 칸(1e-7)보다 작아야 한다 — masks.py 의 SAME 설명."""
    assert 5e-8 < SAME < 1e-7
    r = (0.1, 0.2, 0.3, 0.4)
    assert same_rect(r, r) and same_rect(r, list(r))
    assert same_rect(r, (0.1 + 5e-8, 0.2 - 5e-8, 0.3 + 5e-8, 0.4 - 5e-8))
    assert same_rect((0.0, 0.0, 0.5, 0.5), (SAME, 0.0, 0.5, 0.5))      # 차가 정확히 SAME — '이하' 라 같은 자리
    for k in range(4):                                  # 네 좌표 가운데 하나만 한 칸 달라도 다른 자리
        other = list(r)
        other[k] += 1e-7
        assert not same_rect(r, other) and not same_rect(other, r)


def test_same_rects_compares_in_order():
    a, b = (0.0, 0.0, 0.25, 0.25), (0.5, 0.5, 0.75, 0.75)
    assert same_rects([], []) and same_rects([a, b], [list(a), (0.5 + 1e-9, 0.5, 0.75, 0.75)])
    assert not same_rects([a, b], [b, a])
    assert not same_rects([a], [a, b]) and not same_rects([a, b], [a])
    assert has_rect([a, b], (0.5, 0.5 - 1e-9, 0.75, 0.75)) and not has_rect([a], b) and not has_rect([], a)


@pytest.mark.parametrize("size", [(2560, 1440), (1280, 720), (1920, 1080), (3840, 2160), (640, 640)])
def test_pixel_aligned_rects_survive_the_screen_round_trip(size):
    """2026-11-12 — 반올림한 키를 견주던 때는 2560·1280 폭의 픽셀 좌표가 반올림 경계에 걸려, 화면을 한 번 오간 것만으로
    다른 자리가 됐다(8.4%·16.5%). 같은 템플릿을 두 번 더하면 겹쳐 들어가고 빼기가 그 영역을 남겼다."""
    moved = 0
    for r in pixel_rects(*size, 3000):
        back = screen_round_trip(r)
        moved += back != r
        assert same_rect(r, back), r
        assert merge([back], [r]) == [back]             # 다시 더해도 겹쳐 들어가지 않는다
        assert subtract([back], [r]) == []              # 빼기가 남기지 않는다
    assert moved > 1000                                 # 오가면 끝자리가 정말 달라지는 입력이다


def test_rects_read_back_from_history_are_the_same_place():
    """변경 이력은 영역을 7자리로 반올림해 둔다 — 되읽은 값이 지금 영역과 같은 자리여야 묶음 되돌리기가 그 이미지를
    '그 뒤 손대지 않았다' 고 알아본다. 화면을 한 번 오간 값과 견줘도 마찬가지다."""
    from autolabel.store import _masks_from_json, _masks_json

    rng = random.Random(5)
    rects = pixel_rects(2560, 1440, 2000) + pixel_rects(1280, 720, 2000)
    for _ in range(2000):
        x1, x2 = sorted(rng.random() for _ in range(2))
        y1, y2 = sorted(rng.random() for _ in range(2))
        rects.append((x1, y1, x2, y2))
    rects += [(1 / 3, 1 / 7, 2 / 3, 5 / 7), (0.00000005, 0.00000015, 0.99999995, 0.99999985)]
    back = _masks_from_json(_masks_json(rects))
    assert back != [list(r) for r in rects]             # 반올림으로 값이 정말 달라진다
    assert same_rects(rects, back)
    assert same_rects([screen_round_trip(r) for r in rects], back)


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
    # 바꾸면 화면(app.js MASK_MAX·MASK_DROP·MASK_MAX_PTS)·문서도 함께 — 화면과의 대조는 test_js_parity.py
    assert masks.MAX_MASKS == 500
    assert masks.MAX_POINTS == 100
    assert masks.MIN_SIDE == 1e-4


# ----------------------------------------------------------------- 다각형 (2026-11-19)

TRI = ((0.25, 0.25), (0.75, 0.25), (0.25, 0.75))        # BOX 의 왼쪽 위 절반(대각선 아래가 아니라 위)


def old_rect_cover(box: Box, rects) -> float:
    """다각형을 더하기 전의 masks.covered 를 그대로 옮긴 것 — 사각형만 걸린 박스의 값이 끝자리까지 그대로인지 본다."""
    bx1, by1 = box.cx - box.w / 2, box.cy - box.h / 2
    bx2, by2 = box.cx + box.w / 2, box.cy + box.h / 2
    area = (bx2 - bx1) * (by2 - by1)
    if area <= 0:
        return 0.0
    parts = []
    for x1, y1, x2, y2 in rects:
        a, t, c, d = max(x1, bx1), max(y1, by1), min(x2, bx2), min(y2, by2)
        if c > a and d > t:
            parts.append((a, t, c, d))
    if not parts:
        return 0.0
    xs = sorted({v for p in parts for v in (p[0], p[2])})
    ys = sorted({v for p in parts for v in (p[1], p[3])})
    xi = {v: i for i, v in enumerate(xs)}
    yi = {v: j for j, v in enumerate(ys)}
    ny = len(ys) - 1
    cell = bytearray((len(xs) - 1) * ny)
    for a, t, c, d in parts:
        j1, j2 = yi[t], yi[d]
        for i in range(xi[a], xi[c]):
            cell[i * ny + j1:i * ny + j2] = b"\x01" * (j2 - j1)
    inside = 0.0
    for i in range(len(xs) - 1):
        dx = xs[i + 1] - xs[i]
        row = i * ny
        for j in range(ny):
            if cell[row + j]:
                inside += dx * (ys[j + 1] - ys[j])
    return min(1.0, inside / area)


def sampled_cover(box: Box, ms, n: int = 400) -> float:
    """격자 n×n 점 가운데 영역(합집합, 다각형은 홀짝 규칙) 안에 드는 몫 — covered 와 따로 짠 거친 기준(오차 1/n 쯤)."""
    import numpy as np
    bx1, by1 = box.cx - box.w / 2, box.cy - box.h / 2
    X, Y = np.meshgrid(bx1 + (np.arange(n) + 0.5) / n * box.w, by1 + (np.arange(n) + 0.5) / n * box.h)
    hit = np.zeros(X.shape, bool)
    for m in ms:
        if masks.is_poly(m):
            P = np.array(m, float)
            inside = np.zeros(X.shape, bool)
            for (xa, ya), (xb, yb) in zip(P, np.roll(P, -1, axis=0)):
                with np.errstate(divide="ignore", invalid="ignore"):
                    xc = xa + (Y - ya) * (xb - xa) / (yb - ya)
                inside ^= ((ya > Y) != (yb > Y)) & (X < xc)
            hit |= inside
        else:
            x1, y1, x2, y2 = m
            hit |= (X >= x1) & (X < x2) & (Y >= y1) & (Y < y2)
    return float(hit.mean())


def test_rect_only_boxes_keep_the_old_value_to_the_last_digit():
    """'기존 사각형 영역은 그대로' — 다각형이 걸리지 않은 박스는 다각형을 더하기 전의 식과 끝자리까지 같다(문턱 0.5 의
    판정도 그래서 그대로다). 다각형이 있어도 그 박스에 안 걸리면 같다."""
    rng = random.Random(1119)
    far_poly = ((0.95, 0.95), (1.0, 0.95), (1.0, 1.0))
    for _ in range(3000):
        b = Box(0, rng.random(), rng.random(), 0.01 + 0.5 * rng.random(), 0.01 + 0.5 * rng.random())
        rects = []
        for _ in range(rng.choice([0, 1, 2, 5, 13])):
            x1, x2 = sorted((rng.random(), rng.random()))
            y1, y2 = sorted((rng.random(), rng.random()))
            rects.append((x1, y1, x2, y2))
        want = old_rect_cover(b, rects)
        assert covered(b, rects) == want
        if b.cx + b.w / 2 < 0.95 and b.cy + b.h / 2 < 0.95:
            assert covered(b, [*rects, far_poly]) == want


def test_polygon_cover_is_exact_on_known_shapes():
    assert covered(BOX, [TRI]) == 0.5                                               # 대각선 반
    assert covered(BOX, [((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))]) == 0.5             # 박스를 대각선으로 가른다
    assert covered(BOX, [TRI, TRI]) == 0.5                                          # 겹친 것을 두 번 세지 않는다
    assert covered(BOX, [TRI, ((0.75, 0.25), (0.75, 0.75), (0.25, 0.75))]) == 1.0   # 두 삼각형이 박스를 채운다
    # 삼각형 ∪ 왼쪽 절반 — 0.125 + 0.125 − 겹침 0.09375(왼쪽 절반 안의 x+y≤1) = 0.15625, 박스 0.25 의 0.625
    assert covered(BOX, [TRI, (0.25, 0.25, 0.5, 0.75)]) == 0.625
    square = ((0.25, 0.25), (0.5, 0.25), (0.5, 0.75), (0.25, 0.75))                 # 네모난 다각형 = 같은 사각형
    assert covered(BOX, [square]) == covered(BOX, [(0.25, 0.25, 0.5, 0.75)]) == 0.5
    assert covered(BOX, [((0.6, 0.8), (0.9, 0.8), (0.9, 0.9))]) == 0.0              # 박스 밖
    # 바깥 사각형은 박스를 덮지만 모양은 박스를 비켜 간다(대각선 띠의 빈 모서리) — 0 이어야 한다
    assert covered(Box(0, 0.2, 0.8, 0.2, 0.2), [((0.0, 0.0), (0.2, 0.0), (1.0, 0.8), (1.0, 1.0))]) == 0.0


def test_polygon_cover_matches_a_sampled_reference():
    rng = random.Random(7)
    from helpers import random_poly
    worst = 0.0
    for case in range(150):
        b = Box(0, 0.3 + 0.4 * rng.random(), 0.3 + 0.4 * rng.random(), 0.1 + 0.4 * rng.random(),
                0.1 + 0.4 * rng.random())
        ms = []
        for _ in range(rng.choice([1, 2, 3, 5])):
            if rng.random() < 0.7:
                ms.append(random_poly(rng, simple=case % 3 != 0, cx=b.cx, cy=b.cy))
            else:
                x1, x2 = sorted((rng.random(), rng.random()))
                y1, y2 = sorted((rng.random(), rng.random()))
                ms.append((x1, y1, x2, y2))
        worst = max(worst, abs(covered(b, ms) - sampled_cover(b, ms)))
    assert worst < 0.01                                 # 400×400 표본의 오차 안 — 근사 없이 잰 값이다


def test_self_crossing_polygon_uses_even_odd():
    """꼬인 다각형(8자) — 두 날개는 칠하고, 두 번 감싼 곳이 없으니 홀짝 규칙과 그냥 채우기가 같다. 오각별은 가운데가 빈다."""
    bowtie = ((0.25, 0.25), (0.75, 0.75), (0.75, 0.25), (0.25, 0.75))             # 좌우 날개
    assert covered(BOX, [bowtie]) == 0.5
    import math
    star = tuple((0.5 + 0.25 * math.cos(math.pi / 2 + k * 4 * math.pi / 5),
                  0.5 + 0.25 * math.sin(math.pi / 2 + k * 4 * math.pi / 5)) for k in range(5))
    center = Box(0, 0.5, 0.5, 0.02, 0.02)
    assert covered(center, [star]) == 0.0               # 가운데 오각형은 두 번 감싸여 빈다
    assert sampled_cover(BOX, [star]) == pytest.approx(covered(BOX, [star]), abs=0.01)


def test_drops_and_split_boxes_with_polygons():
    big = Box(2, 0.5, 0.5, 0.5, 0.5)
    small = Box(2, 0.3, 0.3, 0.05, 0.05)                # 삼각형의 왼쪽 위 꼭짓점 둘레 — 다 덮인다
    beside = Box(2, 0.7, 0.7, 0.05, 0.05)               # 삼각형 바깥(대각선 아래)
    assert drops(big, [TRI]) and drops(small, [TRI]) and not drops(beside, [TRI])
    keep, drop = split_boxes([big, small, beside], [TRI])
    assert keep == [beside] and drop == [big, small]


def test_clean_polygons():
    assert clean([TRI]) == [TRI]
    assert clean([[[0.25, 0.25], [0.75, 0.25], [0.25, 0.75]]]) == [TRI]           # JSON 의 리스트도 받는다
    # 0~1 로 자르고, 잇달아 같은 점(더블클릭)과 닫는 점(첫 점과 같은 끝)을 뺀다 — 순서는 그대로
    messy = [(-0.5, 0.25), (0.75, 0.25), (0.75, 0.25), (0.25, 1.5), (0.0, 0.25)]
    assert clean([messy]) == [((0.0, 0.25), (0.75, 0.25), (0.25, 1.0))]
    assert clean([((0.1, 0.1), (0.2, 0.2), (0.3, 0.3))]) == []                     # 한 줄로 늘어섰다
    assert clean([((0.1, 0.1), (0.2, 0.1))]) == []                                 # 점이 둘
    assert clean([((0.1, 0.1), (0.1 + 5e-5, 0.3), (0.1, 0.5))]) == []              # 너무 좁다(MIN_SIDE)
    assert clean([((0.1, float("nan")), (0.5, 0.1), (0.5, 0.5))]) == []
    assert clean([((0.1, 0.1), "x", (0.5, 0.5))]) == []
    mixed = [RECT := (0.1, 0.1, 0.2, 0.2), TRI, (0.3, 0.3, 0.2, 0.4)]
    assert clean(mixed) == [RECT, TRI, (0.2, 0.3, 0.3, 0.4)]                       # 섞여도 순서 그대로


def test_same_place_for_polygons():
    assert same_rect(TRI, [list(p) for p in TRI])
    nudged = ((0.25 + 5e-8, 0.25), (0.75, 0.25 - 5e-8), (0.25, 0.75))
    assert same_rect(TRI, nudged)                       # 이력 반올림(5e-8 까지) 안
    assert not same_rect(TRI, ((0.25 + 1e-7, 0.25), (0.75, 0.25), (0.25, 0.75)))
    assert not same_rect(TRI, TRI[1:] + TRI[:1])        # 같은 모양을 다른 점에서 시작 — 다른 영역으로 본다
    assert not same_rect(TRI, TRI[:2])
    square = ((0.25, 0.25), (0.5, 0.25), (0.5, 0.75), (0.25, 0.75))
    assert not same_rect(square, (0.25, 0.25, 0.5, 0.75)) and not same_rect((0.25, 0.25, 0.5, 0.75), square)
    assert same_rects([RECT_A := (0.0, 0.0, 0.1, 0.1), TRI], [list(RECT_A), [list(p) for p in TRI]])
    assert merge([TRI], [[list(p) for p in TRI], RECT_A]) == [TRI, RECT_A]
    assert subtract([RECT_A, TRI, square], [nudged]) == [RECT_A, square]
    assert has_rect([square, TRI], nudged)


def test_rows_and_wire_keep_polygons_exactly():
    pts = ((1 / 3, 1 / 7), (2 / 3, 1 / 7), (0.5, 5 / 7))
    row = masks.row_values(pts)
    assert row[:4] == (1 / 3, 1 / 7, 2 / 3, 5 / 7)     # 바깥 사각형 — poly 를 모르는 옛 코드는 이것을 읽는다
    assert masks.from_row(*row) == pts                  # JSON 이 끝자리까지 그대로 돌아온다
    assert masks.row_values((0.1, 0.2, 0.3, 0.4)) == (0.1, 0.2, 0.3, 0.4, None)
    assert masks.from_row(0.1, 0.2, 0.3, 0.4, None) == (0.1, 0.2, 0.3, 0.4)
    assert masks.to_wire(pts) == [list(p) for p in pts] and masks.as_mask(masks.to_wire(pts)) == pts
    assert masks.bbox(pts) == (1 / 3, 1 / 7, 2 / 3, 5 / 7)


def test_paint_polygon_by_pixel_centers():
    """픽셀 중심이 다각형 안이면 칠한다 — 대각선 위의 중심(경계)은 칠하지 않는다. 사각형 칠하기는 그대로다."""
    img = Image.new("RGB", (10, 10), (200, 10, 10))
    out = paint(img, [((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))])
    gray = {(x, y) for x in range(10) for y in range(10) if out.getpixel((x, y)) == (GRAY, GRAY, GRAY)}
    assert gray == {(x, y) for x in range(10) for y in range(10) if x + y < 9}    # 중심 (x+.5, y+.5) 가 x+y<1 쪽
    assert img.getpixel((0, 0)) == (200, 10, 10)        # 받은 그림은 그대로
    both = paint(img, [((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), (0.5, 0.5, 1.0, 1.0)])
    assert both.getpixel((9, 9)) == (GRAY, GRAY, GRAY) and both.getpixel((9, 0)) == (200, 10, 10)


def test_painted_share_follows_the_cover():
    """칠한 픽셀의 몫이 덮는 몫(covered)과 거의 같다 — 학습 세트가 칠하는 자리와 박스를 빼는 판정이 같은 모양을 본다."""
    import numpy as np
    rng = random.Random(3)
    from helpers import random_poly
    w, h = 640, 360
    for _ in range(40):
        ms = [random_poly(rng, simple=rng.random() < 0.7) for _ in range(rng.choice([1, 2, 3]))]
        arr = masks.paint_array(np.zeros((h, w), np.uint8), ms)
        share = float((arr == GRAY).mean())
        assert share == pytest.approx(covered(Box(0, 0.5, 0.5, 1.0, 1.0), ms), abs=0.01)


def test_clip_poly():
    tri = [(-1.0, -1.0), (5.0, -1.0), (5.0, 5.0)]
    got = masks.clip_poly(tri, (0, 0, 4, 4))           # 같은 삼각형 — 시작 꼭짓점은 자르는 차례에 따라 다르다
    (cut,) = clean([[(x / 4, y / 4) for x, y in got]])
    assert sorted(cut) == [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    inside = [(1.0, 1.0), (3.0, 1.0), (2.0, 3.0)]
    assert masks.clip_poly(inside, (0, 0, 4, 4)) == inside
    assert masks.clip_poly([(5.0, 5.0), (6.0, 5.0), (6.0, 6.0)], (0, 0, 4, 4)) == []
