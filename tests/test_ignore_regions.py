"""test 이미지의 가릴 영역 = 채점에서 뺄 자리 (2026-11-24 사용자 결정 — src/autolabel/masks.py 머리말 'test 의 영역').

test 그림은 칠하지 않는다(평가는 실제 화면 그대로). 세트를 만드는 쪽(학습 세트·내보내기·`tools/test_baseline.py`·
`tools/build_subset_dataset.py`)이 반 넘게 덮인 정답 박스를 라벨에서 빼고 칠할 자리를 `ignore/…png` 로 남기면, 평가 쪽
(`wsl/eval_test.py` 의 `IgnoreValidator`·`drop_ignored` — test_baseline·test_tiles 도 쓴다)이 그 자리에 반 넘게 든 예측을
세지 않는다. 여기서는 그 둘이 **같은 자리**를 보는지(쓰는 곳 = 찾는 곳), 픽셀 몫을 맞게 세는지, 검증기가 Ultralytics 의 입력
좌표(레터박스)를 원본으로 되돌려 거르는지를 본다. 모델은 돌리지 않는다 — 검증기의 부모 `update_metrics` 를 받아 적는 대역으로 간다.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from autolabel.export import export_yolo
from autolabel.masks import paint_array, write_ignore
from helpers import CLASSES, ROOT, box, far_hashes

sys.path.insert(0, str(ROOT / "wsl"))
sys.path.insert(0, str(ROOT / "tools"))

import eval_test as et  # noqa: E402

torch = pytest.importorskip("torch")

CAR, PERSON = CLASSES.index("car"), CLASSES.index("person")
LEFT = (0.0, 0.0, 0.5, 1.0)                     # 그림 왼쪽 반을 덮는 영역


def jpeg(path: Path, size=(640, 360)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (90, 120, 150)).save(path, quality=90)
    return path


def test_the_tools_look_for_the_picture_where_the_writers_put_it():
    """라벨 파일 자리에 `labels` 대신 `ignore`, 확장자 .png — Ultralytics 가 라벨을 찾는 규칙(마지막 images → labels)과 같다."""
    assert et.ignore_path(Path("d/dataset/images/test/a__1.jpg")) == Path("d/dataset/ignore/test/a__1.png")
    assert et.ignore_path(Path("out/images/x.webp")) == Path("out/ignore/x.png")                   # 내보내기(분할 폴더 없음)
    assert et.ignore_path(Path("images/run/images/test/b.jpg")) == Path("images/run/ignore/test/b.png")   # 마지막 images 만
    assert et.ignore_path(Path("somewhere/b.jpg")) is None
    assert et.IGNORE_DIR == "ignore"            # src/autolabel/masks.py 의 IGNORE_DIR 와 같아야 한다
    from autolabel.masks import IGNORE_DIR
    assert IGNORE_DIR == et.IGNORE_DIR


def test_pixel_share_counts_pixel_centres_and_follows_scale():
    bm = np.zeros((10, 20), bool)
    bm[:, :10] = True                           # 왼쪽 반
    assert et.covered_px(bm, (0, 0, 10, 10), (10, 20)) == 1.0
    assert et.covered_px(bm, (5, 0, 15, 10), (10, 20)) == 0.5
    assert et.covered_px(bm, (10, 0, 20, 10), (10, 20)) == 0.0
    assert et.covered_px(bm, (2.2, 3.1, 2.4, 3.3), (10, 20)) == 1.0          # 픽셀 중심을 안 품는 작은 박스 — 가운데 칸
    assert et.covered_px(bm, (20, 0, 60, 40), (40, 80)) == 0.5               # 원본이 네 배 커도 비율로 맞춘다
    assert et.drop_ignored([(0, 0, 10, 10), (5, 0, 15, 10), (12, 0, 20, 10)], bm, (10, 20)) == [False, False, True]
    assert et.drop_ignored([(0, 0, 10, 10)], None, (10, 20)) == [True]       # 그림이 없으면 다 남긴다


def test_validator_drops_predictions_inside_ignored_places(tmp_path, monkeypatch):
    """검증기는 예측(레터박스 입력 좌표)을 원본 좌표로 되돌려, 뺄 자리가 반 이상 덮은 것만 부모에게 넘기기 전에 뺀다.
    뺄 자리 그림이 없는 장은 손대지 않는다."""
    from ultralytics.models.yolo.detect import DetectionValidator

    src = jpeg(tmp_path / "raw" / "a.jpg", size=(1280, 720))
    img_a, img_b = tmp_path / "set" / "images" / "test" / "a.jpg", tmp_path / "set" / "images" / "test" / "b.jpg"
    for p in (img_a, img_b):
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, p)
    write_ignore(src, et.ignore_path(img_a), [(0.0, 0.0, 1.0, 0.5)])        # 위쪽 반

    seen = []
    monkeypatch.setattr(DetectionValidator, "update_metrics", lambda self, preds, batch: seen.append(preds))
    cls = et.ignore_validator()
    v = cls.__new__(cls)                        # 부모의 __init__(데이터·모델 준비)은 부르지 않는다
    v.ignore_maps, v.ignore_images, v.ignored_conf = {}, 0, []

    # 1280×720 원본을 640×640 입력에 레터박스 — 배율 0.5, 위아래 140px 여백. 원본 y: 40–240(위쪽 — 뺀다),
    # 280–480(덮인 몫 0.4 — 남는다), 500–680(아래쪽). 배율·여백을 되돌리지 않고 재면 첫 박스가 남고 둘째가 빠진다
    native = torch.tensor([[200.0, 40, 400, 240], [200, 280, 400, 480], [600, 500, 800, 680]])
    letterboxed = native * 0.5 + torch.tensor([0.0, 140, 0, 140])

    def pred():
        return {"bboxes": letterboxed.clone(), "conf": torch.tensor([0.9, 0.8, 0.7]),
                "cls": torch.tensor([2.0, 2.0, 0.0]), "extra": torch.zeros((3, 0))}

    batch = {"im_file": [str(img_a), str(img_b)], "img": torch.zeros((2, 3, 640, 640)),
             "ori_shape": [(720, 1280), (720, 1280)], "ratio_pad": [((0.5, 0.5), (0.0, 140.0))] * 2}
    v.update_metrics([pred(), pred()], batch)
    (got,) = seen
    assert got[0]["bboxes"].tolist() == letterboxed[1:].tolist() and got[0]["conf"].tolist() == pytest.approx([0.8, 0.7])
    assert got[0]["cls"].tolist() == [2.0, 0.0] and got[0]["extra"].shape == (2, 0)
    assert got[1]["bboxes"].tolist() == letterboxed.tolist()                  # 그림이 없는 장은 그대로
    assert v.ignore_images == 1 and v.ignored_conf == pytest.approx([0.9])
    assert et.ignore_summary(cls, 0.25) == {}       # last 는 __init__ 이 적는다(여기선 안 불렀다) — 없으면 빈 기록
    cls.last = v
    assert et.ignore_summary(cls, 0.25) == {"images": 1, "preds_at_conf": 1}
    assert et.ignore_summary(cls, 0.95) == {"images": 1, "preds_at_conf": 0}


def test_validator_stops_on_a_prediction_shape_it_does_not_know(monkeypatch):
    from ultralytics.models.yolo.detect import DetectionValidator

    monkeypatch.setattr(DetectionValidator, "update_metrics", lambda self, preds, batch: None)
    cls = et.ignore_validator()
    v = cls.__new__(cls)
    v.ignore_maps, v.ignore_images, v.ignored_conf = {}, 0, []
    with pytest.raises(SystemExit, match="예측 꼴"):
        v.update_metrics([torch.zeros((1, 6))], {"im_file": ["x.jpg"]})


@pytest.fixture
def masked_test_image(store, add_image):
    """test 장 하나 — 왼쪽 반이 덮인 차(라벨에서 빠진다)와 오른쪽 사람. 그리고 영역 없는 test 장 하나."""
    i = add_image("t.jpg", camera="cam-1", split="test", dhash=far_hashes(2)[0],
                  boxes=[box(CAR, 0.25, 0.5, 0.2, 0.2), box(PERSON, 0.75, 0.5, 0.1, 0.2)])
    j = add_image("u.jpg", camera="cam-1", split="test", dhash=far_hashes(2)[1], boxes=[box(CAR, 0.25, 0.5, 0.2, 0.2)])
    for k in (i, j):
        jpeg(Path(store.get_image(k)["path"]))
    store.set_masks(i, [LEFT])
    return i, j


def test_baseline_set_drops_masked_labels_and_leaves_the_place(cfg, store, masked_test_image, tmp_path):
    """tools/test_baseline.py 의 build_split — 그림은 원본 그대로, 덮인 정답은 라벨에서 빼고 그 자리를 평가가 찾는 곳에."""
    import test_baseline as tb

    i, j = masked_test_image
    items = {it["id"]: it for it in tb.build_split(cfg, "test", tmp_path / "tb")}
    a, b = items[i], items[j]
    assert a["img"].read_bytes() == Path(store.get_image(i)["path"]).read_bytes()     # 칠하지 않았다
    assert (dict(a["counts"]), a["ignored"], a["masked"]) == ({PERSON: 1}, 1, True)
    assert (dict(b["counts"]), b["ignored"], b["masked"]) == ({CAR: 1}, 0, False)
    lines = (tmp_path / "tb" / "labels" / "test" / f"{a['img'].stem}.txt").read_text(encoding="utf-8").split()
    assert lines[0] == str(PERSON) and len(lines) == 5
    want = paint_array(np.zeros((360, 640), np.uint8), [LEFT]) > 0
    assert (et.load_ignore(a["img"]) == want).all() and et.load_ignore(b["img"]) is None


def test_export_keeps_test_pictures_and_leaves_the_place(cfg, store, masked_test_image, tmp_path):
    i, _ = masked_test_image
    out = tmp_path / "exported"
    done = export_yolo(cfg, store, out, with_images=True)
    assert (done.masked, done.masked_test, done.mask_dropped) == (0, 1, 1)
    (pic,) = [p for p in (out / "images").iterdir() if p.stem == "t"]
    assert pic.read_bytes() == Path(store.get_image(i)["path"]).read_bytes()          # 칠하지 않았다
    assert (out / "labels" / "t.txt").read_text(encoding="utf-8").split()[0] == str(PERSON)
    assert et.load_ignore(pic) is not None and et.load_ignore(out / "images" / "u.jpg") is None
    assert "ignore/" in (out / "data.yaml").read_text(encoding="utf-8")
