"""CLAUDE.md '반드시 알아야 할 함정' 가운데 코드가 막아 주는 것들 — 막는 코드가 사라지면 여기서 걸린다.

(함정 4 는 test_yolo_io.py, 11·12 는 test_store.py)
"""

from __future__ import annotations

import hashlib
import sys
from types import SimpleNamespace

import pytest

from autolabel import trainer
from autolabel.grouping import base_name
from autolabel.tidy import _free_name


def _model(reg_max: int):
    return SimpleNamespace(model=SimpleNamespace(model=[SimpleNamespace(reg_max=reg_max)]))


# ----------------------------------------------------------------- 함정 1: dfl 을 0 으로 두지 말 것

def test_dfl_zero_warns_for_both_head_kinds():
    assert "L1" in trainer.check_box_distance_gain(_model(1), 0.0)          # yolo26 (reg_max 1)
    assert "DFL" in trainer.check_box_distance_gain(_model(16), 0.0)        # yolo11
    assert trainer.check_box_distance_gain(object(), 0) is not None         # 모델을 못 읽어도 경고는 한다


def test_dfl_default_is_quiet():
    assert trainer.check_box_distance_gain(_model(1), 1.5) is None


# ----------------------------------------------------------------- 함정 2: Windows 에서 cache: ram 금지

@pytest.mark.parametrize("cache, workers, warns", [
    ("ram", 8, True), (True, 8, True), ("RAM", 1, True),
    ("disk", 8, False), (False, 8, False), ("ram", 0, False),
])
def test_ram_cache_warning_on_windows(monkeypatch, cache, workers, warns):
    monkeypatch.setattr(sys, "platform", "win32")
    assert (trainer.check_ram_cache(cache, workers) is not None) == warns


def test_ram_cache_is_fine_on_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert trainer.check_ram_cache("ram", 8) is None


# ----------------------------------------------------------------- 함정 13: 정리가 같은 이름 파일을 덮어썼다

def test_free_name_keeps_a_free_name(tmp_path):
    img, lbl = tmp_path / "images", tmp_path / "labels"
    img.mkdir(), lbl.mkdir()
    src = tmp_path / "in" / "a.jpg"
    assert _free_name(img, lbl, src) == ("a", img / "a.jpg")


def test_free_name_never_points_at_an_existing_image_or_label(tmp_path):
    img, lbl = tmp_path / "images", tmp_path / "labels"
    img.mkdir(), lbl.mkdir()
    src = tmp_path / "in" / "a.jpg"
    tag = hashlib.sha1(str(src).encode("utf-8")).hexdigest()[:8]

    (img / "a.jpg").write_bytes(b"x")                       # 같은 이름의 이미지가 있다
    assert _free_name(img, lbl, src) == (f"a__{tag}", img / f"a__{tag}.jpg")

    (img / "a.jpg").unlink()
    (lbl / "a.txt").write_text("", encoding="utf-8")        # 이미지는 없어도 라벨(a.png 의 것)이 있다
    assert _free_name(img, lbl, src) == (f"a__{tag}", img / f"a__{tag}.jpg")

    (lbl / f"a__{tag}.txt").write_text("", encoding="utf-8")     # 꼬리표 붙인 이름도 차 있다
    stem, dst = _free_name(img, lbl, src)
    assert stem == f"a__{tag}_2"
    assert not dst.exists() and not (lbl / f"{stem}.txt").exists()

    (img / f"a__{tag}_2.jpg").write_bytes(b"x")
    assert _free_name(img, lbl, src)[0] == f"a__{tag}_3"


def test_names_made_by_tidy_still_group_with_the_original(tmp_path):
    """tidy 가 붙인 꼬리표를 grouping 이 뗀다 — 둘 중 하나의 꼴만 바꾸면 여기서 걸린다."""
    img, lbl = tmp_path / "images", tmp_path / "labels"
    img.mkdir(), lbl.mkdir()
    src = tmp_path / "in" / "cctv01_000100.jpg"
    (img / src.name).write_bytes(b"x")
    stem, _ = _free_name(img, lbl, src)
    (img / f"{stem}.jpg").write_bytes(b"x")
    stem2, _ = _free_name(img, lbl, src)
    assert stem != stem2 != src.stem
    assert base_name(f"{stem}.jpg") == base_name(f"{stem2}.jpg") == "cctv01_000100"
