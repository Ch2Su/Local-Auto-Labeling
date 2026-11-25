"""SAM 클릭 — 작게 나오면 프롬프트 주변만 잘라 다시 묻기 (2026-11-25, src/autolabel/segmenter.py 머리말).

모델은 돌리지 않는다. `_load` 를 막고 가짜 predictor 를 꽂는다 — 그림(경로 = 통째, 배열 = 잘라 낸 칸)과 프롬프트를 받아
'물체' 하나를 마스크로 돌려준다. 물체는 원본 픽셀 좌표의 사각형이고, 통째 그림에서는 일부러 다르게(작게·크게) 잡는다 —
잘라 다시 물은 결과가 쓰였는지, 칸 밖으로 넘친 결과를 버렸는지, 좌표를 원본으로 되돌렸는지를 본다.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from autolabel import segmenter as sg

torch = pytest.importorskip("torch")

W, H = 1280, 720


class FakePred:
    """set_image 한 그림과 프롬프트로 마스크 하나. full(x1,y1,x2,y2) — 통째 그림에서 잡는 사각형(원본 px),
    crop — 잘라 낸 칸에서 잡는 사각형(원본 px, 칸 밖은 칸으로 자른다). None 이면 마스크 없음."""

    def __init__(self, full, crop):
        self.full, self.crop = full, crop
        self.img = None
        self.sets = []          # set_image 에 들어온 것: 경로 문자열 또는 배열 꼴
        self.asks = []

    def reset_image(self):
        pass

    def set_image(self, img):
        self.img = img
        self.sets.append(img if isinstance(img, str) else img.shape)

    def __call__(self, **kw):
        self.asks.append(kw)
        if isinstance(self.img, str):
            rect, shape, off = self.full, (H, W), (0, 0)
        else:
            rect, shape = self.crop, self.img.shape[:2]
            off = self.img.off
        if rect is None:
            return [SimpleNamespace(masks=None, boxes=None)]
        m = np.zeros(shape, np.uint8)
        x1, y1, x2, y2 = (int(rect[0] - off[0]), int(rect[1] - off[1]), int(rect[2] - off[0]), int(rect[3] - off[1]))
        m[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)] = 1
        conf = torch.tensor([0.9 if isinstance(self.img, str) else 0.8])
        return [SimpleNamespace(masks=SimpleNamespace(data=torch.from_numpy(m)[None]), boxes=SimpleNamespace(conf=conf))]


class OffsetArray(np.ndarray):
    """잘라 낸 칸이 원본의 어디였는지(off)를 들고 다니는 배열 — 가짜 predictor 가 원본 좌표의 물체를 칸 좌표로 옮기려고."""


@pytest.fixture
def seg(cfg, tmp_path, monkeypatch):
    path = tmp_path / "frame.jpg"
    Image.new("RGB", (W, H), (90, 120, 150)).save(path, quality=90)
    s = sg.Segmenter(cfg)

    def install(full, crop):
        pred = FakePred(full, crop)

        def load(device):
            s._pred, s._device = pred, device
            return pred

        monkeypatch.setattr(s, "_load", load)
        monkeypatch.setattr("autolabel.jobs.gpu_busy", lambda: False)
        real = s._image_array

        def image_array(p):                      # 잘라 내는 쪽이 어디를 잘랐는지 가짜 predictor 가 알게
            arr = real(p)
            return None if arr is None else _Tracked(arr)

        monkeypatch.setattr(s, "_image_array", image_array)
        return pred

    return s, path, install


class _Tracked:
    """배열을 자를 때 잘린 자리(off)를 붙여 준다 — `_ask_crop` 의 `arr[y0:y0+side, x0:x0+side]` 를 그대로 받는다."""

    def __init__(self, arr):
        self.arr = arr
        self.shape = arr.shape

    def __getitem__(self, idx):
        ys, xs = idx
        out = self.arr[idx].view(OffsetArray)
        out.off = (xs.start or 0, ys.start or 0)
        return out


def px(box):
    """정규화 Box → 원본 px (x1, y1, x2, y2)."""
    return (round((box.cx - box.w / 2) * W, 3), round((box.cy - box.h / 2) * H, 3),
            round((box.cx + box.w / 2) * W, 3), round((box.cy + box.h / 2) * H, 3))


def test_big_result_is_kept_without_cropping(seg):
    s, path, install = seg
    pred = install(full=(100, 100, 400, 300), crop=(0, 0, 1, 1))
    box, score = s.segment(path, point=(250, 200))
    assert px(box) == (100, 100, 400, 300) and score == pytest.approx(0.9)
    assert pred.sets == [str(path)] and len(pred.asks) == 1


def test_small_result_is_asked_again_on_a_crop_around_the_point(seg):
    s, path, install = seg
    pred = install(full=(600, 340, 620, 360), crop=(598, 338, 624, 365))     # 통째는 20px, 칸에서는 26×27
    box, score = s.segment(path, point=(610, 350))
    assert px(box) == (598, 338, 624, 365) and score == pytest.approx(0.8)
    assert pred.sets == [str(path), (512, 512, 3)]
    assert pred.asks[1] == {"points": [[610 - 354, 350 - 94]], "labels": [1]}        # 칸 = (354, 94) 부터 512


def test_crop_window_is_pushed_inside_the_picture(seg):
    s, path, install = seg
    pred = install(full=(5, 5, 20, 20), crop=(4, 4, 22, 21))
    box, _ = s.segment(path, point=(10, 10))
    assert px(box) == (4, 4, 22, 21)
    assert pred.asks[1]["points"] == [[10, 10]]                               # 칸이 (0, 0) 에서 시작
    assert sg.crop_window(1270, 715, 512, W, H) == (768, 208, 512)
    assert sg.crop_window(600, 300, 5000, W, H) == (240, 0, 720)              # 그림의 짧은 변을 넘지 않는다


def test_result_touching_the_crop_edge_falls_back_to_the_whole_picture(seg):
    s, path, install = seg
    install(full=(600, 340, 650, 360), crop=(300, 340, 650, 360))            # 칸(354–866)의 왼쪽 끝까지 번짐
    box, score = s.segment(path, point=(610, 350))
    assert px(box) == (600, 340, 650, 360) and score == pytest.approx(0.9)


def test_no_mask_on_the_whole_picture_still_tries_the_crop(seg):
    s, path, install = seg
    install(full=None, crop=(605, 345, 615, 356))
    box, _ = s.segment(path, point=(610, 350))
    assert px(box) == (605, 345, 615, 356)
    install(full=None, crop=None)
    with pytest.raises(sg.SegmenterUnavailable):
        s.segment(path, point=(610, 350))


def test_box_prompt_window_covers_twice_the_box(seg):
    s, path, install = seg
    pred = install(full=(500, 300, 540, 330), crop=(495, 298, 545, 333))
    box, _ = s.segment(path, box=(300, 250, 700, 400))                        # 긴 변 400 → 칸 800 → 짧은 변 720 으로
    assert pred.sets[1] == (720, 720, 3)
    assert pred.asks[1] == {"bboxes": [[300 - 140, 250, 700 - 140, 400]]}     # 칸 = (140, 0)
    assert px(box) == (495, 298, 545, 333)


def test_after_a_crop_the_next_click_encodes_the_whole_picture_again(seg):
    s, path, install = seg
    pred = install(full=(600, 340, 620, 360), crop=(598, 338, 624, 365))
    s.segment(path, point=(610, 350))
    pred.full = (100, 100, 400, 300)
    s.segment(path, point=(250, 200))
    s.segment(path, point=(260, 210))
    assert pred.sets == [str(path), (512, 512, 3), str(path)]                # 셋째 클릭은 인코딩해 둔 것을 쓴다


def test_crop_below_zero_turns_it_off(seg):
    s, path, install = seg
    s.cfg.raw.setdefault("sam", {})["crop_below"] = 0
    pred = install(full=(600, 340, 620, 360), crop=(598, 338, 624, 365))
    box, _ = s.segment(path, point=(610, 350))
    assert px(box) == (600, 340, 620, 360) and pred.sets == [str(path)]


def test_crop_reads_the_picture_like_ultralytics(cfg, tmp_path):
    """잘라 낼 그림은 Ultralytics 가 경로로 읽는 함수로 — EXIF 로 돌려 보는 사진도 통째 마스크와 같은 꼴이어야 좌표가 맞는다."""
    from ultralytics.utils.patches import imread

    p = tmp_path / "rot.jpg"
    img = Image.new("RGB", (640, 360), (10, 20, 30))
    exif = img.getexif()
    exif[0x0112] = 6                                                          # 90도 돌려 보는 사진
    img.save(p, exif=exif)
    s = sg.Segmenter(cfg)
    arr = s._image_array(Path(p))
    assert arr.shape == imread(str(p)).shape
    assert s._image_array(Path(p)) is arr                                     # 같은 그림은 다시 읽지 않는다
