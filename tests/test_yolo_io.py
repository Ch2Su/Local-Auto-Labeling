"""라벨 읽기·쓰기 (yolo_io.py)."""

from __future__ import annotations

import math

import pytest
from PIL import Image

from autolabel.yolo_io import Box, _row_to_box, display_size, image_size, read_yolo_txt, write_yolo_txt


def test_row_shapes():
    assert _row_to_box("3 0.5 0.5 0.2 0.1".split(), "seed") == Box(3, 0.5, 0.5, 0.2, 0.1, None, "seed")
    assert _row_to_box("3 0.5 0.5 0.2 0.1 0.87".split(), "auto") == Box(3, 0.5, 0.5, 0.2, 0.1, 0.87, "auto")
    # 7열 이상은 폴리곤 — 정점의 외접 사각형
    b = _row_to_box("1 0.25 0.25 0.75 0.25 0.5 0.75".split(), "seed")
    assert (b.cls, b.cx, b.cy, b.w, b.h, b.conf) == (1, 0.5, 0.5, 0.5, 0.5, None)


def test_polygon_with_odd_coordinates_is_an_error():
    with pytest.raises(ValueError):
        _row_to_box("1 0.1 0.1 0.2 0.2 0.3 0.3 0.4".split(), "seed")


def test_mixed_box_and_polygon_rows_are_both_read(tmp_path):
    """함정 4 — Ultralytics 는 한 파일에 두 형식이 섞이면 이미지째 버린다. 여기서는 행마다 살린다."""
    p = tmp_path / "a.txt"
    p.write_text("0 0.5 0.5 0.25 0.25\n"
                 "2 0.125 0.125 0.375 0.125 0.375 0.375 0.125 0.375\n"
                 "\n"
                 "5 0.5 0.5 0.0 0.25\n",                 # 폭 0 — 버린다
                 encoding="utf-8")
    got = read_yolo_txt(p)
    assert [(b.cls, b.cx, b.cy, b.w, b.h) for b in got] == [(0, 0.5, 0.5, 0.25, 0.25), (2, 0.25, 0.25, 0.25, 0.25)]
    assert all(b.origin == "seed" for b in got)


def test_short_row_names_the_file_and_line(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_text("0 0.5 0.5 0.25 0.25\n0 0.5 0.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.txt:2"):
        read_yolo_txt(p)


def test_clamped_cuts_to_the_image():
    b = Box(0, 0.95, 0.5, 0.25, 0.5).clamped()          # 오른쪽으로 0.075 나갔다
    assert (b.cx - b.w / 2, b.cx + b.w / 2) == pytest.approx((0.825, 1.0))
    assert b.is_valid()


@pytest.mark.parametrize("bad", [
    Box(0, math.nan, 0.5, 0.1, 0.1),
    Box(0, 0.5, 0.5, math.inf, 0.1),
    Box(0, 0.5, 0.5, 0.1, 0.1, conf=math.nan),
])
def test_nan_and_infinity_never_become_a_valid_box(bad):
    """2026-11-05 — NaN 좌표가 clamped 에서 '이미지 전체' 박스로 둔갑해 is_valid 를 통과했다."""
    assert not bad.clamped().is_valid()


def test_invalid_boxes():
    assert not Box(-1, 0.5, 0.5, 0.1, 0.1).is_valid()
    assert not Box(0, 0.5, 0.5, 1e-4, 0.1).is_valid()       # 하한은 '초과'


def test_write_then_read(tmp_path):
    p = tmp_path / "labels" / "x.txt"
    boxes = [Box(0, 0.5, 0.5, 0.25, 0.25, 0.9, "auto"), Box(9, 0.125, 0.875, 0.0625, 0.0625, None, "human"),
             Box(1, 0.5, 0.5, 0.0, 0.1)]                    # 쓸 수 없는 박스는 빠진다
    write_yolo_txt(p, boxes)
    assert p.read_text(encoding="utf-8").count("\n") == 2
    got = read_yolo_txt(p, origin="human")
    assert [(b.cls, b.cx, b.cy, b.w, b.h, b.conf) for b in got] == [
        (0, 0.5, 0.5, 0.25, 0.25, None), (9, 0.125, 0.875, 0.0625, 0.0625, None)]

    write_yolo_txt(p, [])                                   # 박스가 없으면 빈 파일(배경 샘플)
    assert p.read_text(encoding="utf-8") == ""


def test_display_size_swaps_for_rotated_exif(tmp_path):
    """DB 의 width/height 는 원시 크기, 화면·박스 좌표는 회전을 적용한 크기다 (CLAUDE.md 'EXIF 회전 이미지')."""
    p = tmp_path / "r.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.new("RGB", (40, 20)).save(p, exif=exif)
    assert image_size(p) == (40, 20)
    assert display_size(p) == (20, 40)

    q = tmp_path / "plain.jpg"
    Image.new("RGB", (40, 20)).save(q)
    assert display_size(q) == image_size(q) == (40, 20)
