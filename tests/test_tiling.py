"""타일 추론의 합치기(`tiling.merge_cross`) — 클래스가 다른 겹침 (2026-11-17 사용자 화면).

클래스별 NMS 라서 통째가 car, 타일이 truck 이라고 낸 같은 차가 둘 다 남았다. 차량 클래스끼리는 클래스가 달라도
class_iou 이상 겹치면 신뢰도 높은 쪽만 남긴다. 사람과 탈것은 묶지 않는다. 모델·GPU 없이 검출 배열만으로 본다.
통째 자동 라벨링의 같은 정리(`cross_class_keep`·`plain_class_iou`, 2026-11-20)도 여기서 본다 — 같은 클래스는 건드리지 않고
차량 규칙만 merge_cross 와 같아야 한다.
"""

from __future__ import annotations

import numpy as np
import pytest

from autolabel import tiling

from helpers import CLASSES

CAR, MOTO, BUS, TRUCK, PERSON, PICKUP, RESP = (CLASSES.index(n) for n in ("car", "motorcycle", "bus", "truck", "person",
                                                                           "pickup", "responder"))
GROUP = tiling.vehicle_ids(CLASSES)


def det(x1, y1, x2, y2, conf, cls, comp):
    """열: x1 y1 x2 y2 conf cls touch comp(0 통째 / 1 타일)."""
    return [x1, y1, x2, y2, conf, cls, 0, comp]


def run(rows, **kw):
    d = tiling.merge_cross(np.array(rows, np.float32), 0.6, 0.3, 100, 0.25, **kw)
    return [(int(r[5]), round(float(r[4]), 2)) for r in d]


# 사용자 화면: 통째 car 0.82 와 그 안쪽의 타일 truck 0.26 (IoU 약 0.8)
SAME_CAR = [det(100, 100, 200, 240, 0.82, CAR, 0), det(104, 112, 196, 238, 0.26, TRUCK, 1)]


def test_vehicle_ids_are_the_vehicle_classes_only():
    assert {CLASSES[i] for i in GROUP} == {"car", "truck", "bus", "pickup", "responder"}
    assert tiling.vehicle_ids({0: "person", 1: "car"}) == {1}            # model.names 는 dict 다
    assert tiling.vehicle_ids(None) == frozenset()


def test_without_the_group_both_classes_survive_as_before():
    assert run(SAME_CAR) == [(CAR, 0.82), (TRUCK, 0.26)]


def test_same_vehicle_in_two_classes_keeps_the_more_confident_one():
    assert run(SAME_CAR, group=GROUP, iou_class=0.6) == [(CAR, 0.82)]
    # 같은 성분(타일끼리·통째끼리)에서 나온 것도 지운다
    both_whole = [det(100, 100, 200, 240, 0.5, PICKUP, 0), det(102, 102, 198, 238, 0.7, TRUCK, 0)]
    assert run(both_whole, group=GROUP, iou_class=0.6) == [(TRUCK, 0.7)]


def test_vehicles_that_overlap_less_than_the_threshold_both_stay():
    side_by_side = [det(100, 100, 200, 240, 0.8, CAR, 0), det(150, 100, 250, 240, 0.7, TRUCK, 1)]     # IoU 1/3
    assert run(side_by_side, group=GROUP, iou_class=0.6) == [(CAR, 0.8), (TRUCK, 0.7)]


def test_a_rider_and_the_bike_are_not_merged():
    rider = [det(100, 100, 160, 240, 0.8, PERSON, 0), det(100, 110, 160, 240, 0.7, MOTO, 1),
             det(100, 100, 160, 240, 0.6, CAR, 1)]                       # 사람과 겹친 차량도 사람 때문에 빠지지 않는다
    assert run(rider, group=GROUP, iou_class=0.6) == [(PERSON, 0.8), (MOTO, 0.7), (CAR, 0.6)]


def test_a_removed_box_does_not_remove_others():
    """car 0.9 가 truck 0.5 를 지우면, truck 0.5 와만 겹치는 bus 0.4 는 남는다(지워진 박스는 남을 지우지 못한다)."""
    rows = [det(0, 0, 100, 100, 0.9, CAR, 0), det(0, 30, 100, 130, 0.5, TRUCK, 0), det(0, 62, 100, 162, 0.4, BUS, 0)]
    assert run(rows, group=GROUP, iou_class=0.5) == [(CAR, 0.9), (BUS, 0.4)]


def test_boxes_under_the_confidence_floor_do_not_take_part():
    rows = [det(100, 100, 200, 240, 0.2, TRUCK, 1), det(100, 100, 200, 240, 0.3, CAR, 0)]
    assert run(rows, group=GROUP, iou_class=0.6) == [(CAR, 0.3)]


def test_settings_default_and_off(cfg):
    assert tiling.settings(cfg)["class_iou"] == tiling.DEFAULTS["class_iou"] == 0.6
    cfg.raw.setdefault("predict", {})["tiles"] = {"class_iou": 0}
    assert tiling.settings(cfg)["class_iou"] == 0.0                       # 0 은 '끔' — 없는 값으로 치지 않는다
    cfg.raw["predict"]["tiles"] = {"class_iou": 1.5}
    with pytest.raises(ValueError):
        tiling.settings(cfg)


def test_detect_passes_the_model_names(monkeypatch):
    """detect 가 model.names 로 차량 묶음을 만들어 넘긴다 — class_iou 0 이면 넘기지 않는다."""
    whole = np.array([[100, 100, 200, 240, 0.82, CAR, 0]], np.float32)
    tile = np.array([[104, 112, 196, 238, 0.26, TRUCK, 0]], np.float32)
    monkeypatch.setattr(tiling, "run_component", lambda model, img, comp, **kw: ((whole if comp == "full" else tile), 0.0))

    class Model:
        names = dict(enumerate(CLASSES))

    img = np.zeros((1000, 1000, 3), np.uint8)
    base = dict(imgsz=640, device="cpu", conf=0.25, iou=0.6, max_det=100, grid=2, overlap=0.2, edge_px=3, min_side=800, cross_iou=0.3)
    on, used = tiling.detect(Model(), img, class_iou=0.6, **base)
    off, _ = tiling.detect(Model(), img, class_iou=0.0, **base)
    assert used and [int(r[5]) for r in on] == [CAR] and [int(r[5]) for r in off] == [CAR, TRUCK]


# ── 통째 자동 라벨링의 차량 겹침 정리 (2026-11-20 사용자 결정, `tiling.cross_class_keep`) ──────────────────────
# 저장까지 가는 길(AutoLabeler.run)은 tests/test_predictor.py 가 본다.

def test_plain_keep_drops_only_the_other_class_vehicle():
    """평가 세트에서 본 꼴: 같은 차에 car 0.82 와 responder 0.5. responder 만 빠지고, car 끼리·사람은 그대로, 순서도 그대로."""
    boxes = [[100, 100, 200, 240], [104, 112, 196, 238], [100, 100, 200, 240], [102, 104, 198, 236], [100, 100, 160, 240]]
    conf = [0.5, 0.82, 0.3, 0.6, 0.9]
    cls = [RESP, CAR, CAR, CAR, PERSON]
    # car 0.3 은 responder 0.5 와 IoU 1 이지만 responder 가 이미 지워졌으므로 남는다(지워진 박스는 남을 지우지 못한다)
    assert tiling.cross_class_keep(boxes, conf, cls, GROUP, 0.6).tolist() == [False, True, True, True, True]
    # 신뢰도가 높은 쪽이 responder 면 car 가 빠진다 — 남는 클래스가 맞다는 보장은 없다
    assert tiling.cross_class_keep([[0, 0, 100, 100], [2, 2, 98, 100]], [0.4, 0.7], [CAR, RESP], GROUP, 0.6).tolist() == [False, True]


def test_plain_keep_threshold_and_off():
    two = ([[0, 0, 100, 100], [0, 0, 100, 50]], [0.9, 0.8], [CAR, TRUCK])          # IoU 정확히 0.5
    assert tiling.cross_class_keep(*two, GROUP, 0.5).tolist() == [True, False]      # 문턱 이상이면 지운다(merge_cross 와 같이)
    assert tiling.cross_class_keep(*two, GROUP, 0.51).tolist() == [True, True]
    assert tiling.cross_class_keep(*two, GROUP, 0.0).tolist() == [True, True]       # 0 은 끔
    assert tiling.cross_class_keep(*two, frozenset(), 0.5).tolist() == [True, True]  # 묶음이 없으면(차량 클래스를 모르는 모델) 그대로
    assert tiling.cross_class_keep([], [], [], GROUP, 0.6).tolist() == []
    assert tiling.cross_class_keep([[0, 0, 1, 1]], [0.5], [CAR], GROUP, 0.6).tolist() == [True]


def test_plain_keep_follows_the_tile_vehicle_rule():
    """같은 클래스 겹침을 빼면 merge_cross 의 차량 규칙과 남는 것이 같다 — 무작위 검출 400벌(같은 신뢰도도 나오게)로."""
    rng = np.random.default_rng(7)
    pool = np.array([CAR, TRUCK, BUS, PICKUP, RESP, PERSON, MOTO])
    removed = 0
    for _ in range(400):
        n = int(rng.integers(2, 12))
        base = rng.uniform(0, 300, (3, 2)).astype(np.float32)                    # 물체 셋 둘레에 검출을 흩는다(사슬 겹침도 생긴다)
        size = rng.uniform(40, 120, (3, 2)).astype(np.float32)
        k = rng.integers(0, 3, n)
        boxes = np.concatenate([base[k], base[k] + size[k]], 1)
        boxes += rng.uniform(-0.2, 0.2, (n, 4)).astype(np.float32) * np.concatenate([size[k], size[k]], 1)
        boxes[:, 2:] = np.maximum(boxes[:, 2:], boxes[:, :2] + 1)
        conf = np.round(rng.uniform(0.25, 1, n), 2).astype(np.float32)
        cls = rng.choice(pool, n)
        keep = tiling.cross_class_keep(boxes, conf, cls, GROUP, 0.5)
        d = np.concatenate([boxes, conf[:, None], cls[:, None].astype(np.float32), np.zeros((n, 2), np.float32)], 1)
        ref = tiling.merge_cross(d, 1.1, 1.1, 1000, 0.0, group=GROUP, iou_class=0.5)    # IoU 는 1.1 을 못 넘어 같은 클래스는 안 지운다
        assert sorted(map(tuple, d[keep].tolist())) == sorted(map(tuple, ref.tolist()))
        removed += n - int(keep.sum())
    assert removed > 100                                                          # 지울 것이 실제로 많은 표본이었다


def test_plain_class_iou_setting(cfg):
    assert tiling.plain_class_iou(cfg) == tiling.PLAIN_CLASS_IOU == 0.6          # 설정에 없으면 켠다
    cfg.raw["predict"] = {"class_iou": 0}
    assert tiling.plain_class_iou(cfg) == 0.0                                     # 0 은 '끔' — 없는 값으로 치지 않는다
    cfg.raw["predict"] = {"class_iou": 1.5}
    with pytest.raises(ValueError):
        tiling.plain_class_iou(cfg)
    cfg.raw["predict"] = {"tiles": {"class_iou": 0}}                              # 타일 설정은 통째에 쓰지 않는다
    assert tiling.plain_class_iou(cfg) == 0.6
