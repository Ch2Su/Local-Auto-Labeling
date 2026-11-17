"""타일 추론의 합치기(`tiling.merge_cross`) — 클래스가 다른 겹침 (2026-11-17 사용자 화면).

클래스별 NMS 라서 통째가 car, 타일이 truck 이라고 낸 같은 차가 둘 다 남았다. 차량 클래스끼리는 클래스가 달라도
class_iou 이상 겹치면 신뢰도 높은 쪽만 남긴다. 사람과 탈것은 묶지 않는다. 모델·GPU 없이 검출 배열만으로 본다.
"""

from __future__ import annotations

import numpy as np
import pytest

from autolabel import tiling

from helpers import CLASSES

CAR, MOTO, BUS, TRUCK, PERSON, PICKUP = (CLASSES.index(n) for n in ("car", "motorcycle", "bus", "truck", "person", "pickup"))
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
