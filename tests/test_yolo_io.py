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


def test_a_file_that_is_not_an_image_never_starts_a_package_install(tmp_path, monkeypatch):
    """Ultralytics(8.4.212)는 import 될 때 `PIL.Image.open` 을 제 함수로 바꾼다 — Pillow 가 파일을 못 알아보면 `pi-heif` 를
    지금 쓰는 파이썬 환경에 설치하려 들고, 못 깔면 ModuleNotFoundError 를 낸다. 2026-11-23 에 웹 이미지 올리기의 테스트
    (이름만 .jpg 인 글자 파일)가 전체 테스트에서만 그렇게 실패했다: 앞선 테스트가 Ultralytics 를 올려 둔 프로세스였다.
    `image_size` 는 원래의 `Image.open` 으로 연다 — 여기서는 그 갈아 끼우기를 흉내 낸다(진짜 Ultralytics 를 올리지 않는다)."""
    import sys
    import types

    import autolabel.yolo_io as yolo_io

    # 진짜 원래 함수 — 전체 테스트에서는 앞선 테스트가 Ultralytics 를 이미 올려 `Image.open` 이 갈아 끼워져 있다(그것을 원래
    # 함수로 잡으면 이 테스트가 진짜 설치를 돌린다 — 11-23 에 그렇게 겪었다)
    original = yolo_io.plain_image_open()
    installs: list = []

    def image_open(filename, *args, **kwargs):               # ultralytics.utils.patches.image_open 과 같은 꼴
        try:
            return original(filename, *args, **kwargs)
        except Image.UnidentifiedImageError:
            installs.append(str(filename))                   # 진짜는 여기서 check_requirements("pi-heif") — 설치를 돌린다
            raise ModuleNotFoundError("No module named 'pi_heif'") from None

    fake = types.ModuleType("ultralytics.utils.patches")
    fake._image_open = original
    monkeypatch.setitem(sys.modules, "ultralytics.utils.patches", fake)
    monkeypatch.setattr(Image, "open", image_open)

    junk = tmp_path / "notes.jpg"
    junk.write_bytes(b"this is not a jpeg at all")
    with pytest.raises(Image.UnidentifiedImageError):        # OSError 다 — 스캔·올리기가 '읽지 못한 파일' 로 거른다
        image_size(junk)
    assert installs == []
    good = tmp_path / "ok.jpg"
    Image.new("RGB", (40, 20)).save(good)
    assert image_size(good) == (40, 20)
    assert yolo_io.plain_image_open() is original
    # 그 모듈이 원래 함수를 다른 이름으로 들게 되면(Ultralytics 를 올렸다) 지금의 Image.open 으로 돌아간다 — 조용히 깨지지 않는다
    del fake._image_open
    assert yolo_io.plain_image_open() is image_open


def test_nothing_in_the_package_opens_images_through_the_patched_opener():
    """이 패키지에서 그림을 여는 곳은 전부 `plain_image_open` 을 쓴다 (2026-11-25, Gemini 리뷰 묶음 6) — `Image.open` 을 직접
    부르는 줄이 하나라도 생기면, Ultralytics 가 올라온 검수 서버에서 깨진 그림 하나가 패키지 설치를 일으킨다(위 테스트).
    그 전에는 `image_size` 만 이랬고 썸네일·`display_size`·dHash·정리의 점검·EXIF 시각·칠한 사본이 그대로였다."""
    from helpers import ROOT

    direct = [f"{p.name}:{n}" for p in sorted((ROOT / "src" / "autolabel").glob("*.py"))
              for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1) if "Image.open(" in line]
    assert direct == []


def test_the_rotated_size_and_the_scene_hash_also_use_the_plain_opener(tmp_path, monkeypatch):
    """갈아 끼운 `Image.open`(못 알아보는 파일에서 설치를 돌리는 것)을 흉내 내고, 그림을 여는 다른 길도 그것을 건드리지 않는지."""
    import sys
    import types

    import autolabel.yolo_io as yolo_io
    from autolabel.grouping import dhash
    from autolabel.masks import write_masked
    from autolabel.provenance import _from_exif

    original = yolo_io.plain_image_open()
    opened: list = []

    def image_open(filename, *args, **kwargs):
        opened.append(str(filename))
        raise ModuleNotFoundError("No module named 'pi_heif'")

    fake = types.ModuleType("ultralytics.utils.patches")
    fake._image_open = original
    monkeypatch.setitem(sys.modules, "ultralytics.utils.patches", fake)
    monkeypatch.setattr(Image, "open", image_open)

    junk = tmp_path / "notes.jpg"
    junk.write_bytes(b"this is not a jpeg at all")
    good = tmp_path / "ok.jpg"
    original_new = Image.new("RGB", (40, 20), (10, 200, 30))
    original_new.save(good)
    with pytest.raises(Image.UnidentifiedImageError):
        display_size(junk)
    assert display_size(good) == (40, 20)
    assert dhash(junk) is None and dhash(good)                 # 못 읽는 파일은 해시 없음(전과 같다), 설치는 안 돈다
    assert _from_exif(junk) is None and _from_exif(good) is None
    write_masked(good, tmp_path / "out" / "masked.jpg", [(0.0, 0.0, 0.5, 1.0)])
    assert (tmp_path / "out" / "masked.jpg").exists()
    assert opened == []


def test_importing_the_package_turns_off_ultralytics_auto_install_unless_told_otherwise():
    """`import autolabel` 이 `YOLO_AUTOINSTALL=false` 를 기본으로 둔다 — Ultralytics 가 모자란 패키지를 이 `.venv`(다른 프로젝트도
    쓴다)에 스스로 깔지 않게. 사람이 정해 둔 값은 그대로 둔다. conftest 가 먼저 정해 두므로 새 프로세스로 본다."""
    import os
    import subprocess
    import sys

    from helpers import ROOT

    code = "import os, autolabel; print(os.environ.get('YOLO_AUTOINSTALL'))"
    env = {k: v for k, v in os.environ.items() if k != "YOLO_AUTOINSTALL"}
    env["PYTHONPATH"] = str(ROOT / "src")

    def run(extra: dict) -> str:
        done = subprocess.run([sys.executable, "-c", code], env={**env, **extra}, capture_output=True, text=True, timeout=60)
        assert done.returncode == 0, done.stderr
        return done.stdout.strip()

    assert run({}) == "false"
    assert run({"YOLO_AUTOINSTALL": "true"}) == "true"
