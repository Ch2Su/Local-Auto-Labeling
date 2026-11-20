"""통째 자동 라벨링(`AutoLabeler.run`)의 차량 겹침 정리 (2026-11-20 사용자 결정).

새 풀 프레임 평가에서 통째도 같은 차에 car·responder 처럼 클래스만 다른 박스를 겹쳐 냈다. 그런 박스는 신뢰도 높은 쪽만
저장하고(`predict.class_iou`, `tiling.cross_class_keep`), 같은 클래스 겹침·사람은 모델이 낸 그대로 둔다. 모델은 가짜다 —
정해 둔 검출을 Ultralytics 의 `Boxes` 로 돌려줘서, GPU·가중치 없이 추론 결과를 읽는 데서 DB 에 저장하는 데까지 본다.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from autolabel.predictor import AutoLabeler

from helpers import CLASSES

torch = pytest.importorskip("torch")
results = pytest.importorskip("ultralytics.engine.results")

CAR, TRUCK, PERSON, RESP = (CLASSES.index(n) for n in ("car", "truck", "person", "responder"))
W, H = 1000, 800
DETS = [                                   # x1 y1 x2 y2 conf cls (픽셀)
    [100, 100, 300, 260, 0.80, CAR],
    [104, 104, 296, 262, 0.50, RESP],      # 같은 차(IoU 약 0.95) — 빠진다
    [100, 100, 300, 260, 0.30, CAR],       # 같은 클래스 겹침 — 그대로
    [110, 110, 180, 250, 0.70, PERSON],    # 사람은 차량 묶음 밖 — 그대로
    [600, 100, 800, 300, 0.60, TRUCK],     # 떨어진 차 — 그대로
]


class FakeModel:
    names = dict(enumerate(CLASSES))

    def __init__(self):
        self.calls = 0

    def predict(self, source=None, **kw):
        self.calls += 1
        return [SimpleNamespace(boxes=results.Boxes(torch.tensor(DETS, dtype=torch.float32), orig_shape=(H, W)))]


def labeler_for(cfg, store) -> AutoLabeler:
    labeler = AutoLabeler(cfg, store)
    labeler._model = FakeModel()            # YOLO 를 올리지 않는다
    return labeler


def run_on_new_image(cfg, store, add_image, name: str):
    image_id = add_image(name, status="unlabeled", size=(W, H))
    path = Path(store.get_image(image_id)["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")                  # run 은 파일이 있는지만 본다 — 읽는 것은 가짜 모델이다
    report = labeler_for(cfg, store).run([image_id])
    saved = sorted((b.cls, round(b.conf, 2)) for b in store.get_boxes(image_id))
    return report, saved, store.get_image(image_id)


def test_plain_autolabel_keeps_one_box_per_vehicle(cfg, store, add_image):
    report, saved, row = run_on_new_image(cfg, store, add_image, "a.jpg")
    assert saved == sorted([(CAR, 0.8), (CAR, 0.3), (PERSON, 0.7), (TRUCK, 0.6)])
    assert (report.processed, report.boxes, report.merged, report.class_iou) == (1, 4, 1, 0.6)
    assert "겹친 차량 박스 1개" in report.summary()
    assert row["status"] == "pending" and "+tiles" not in (row["model_version"] or "")


def test_class_iou_zero_saves_what_the_model_gave(cfg, store, add_image):
    cfg.raw["predict"] = {"class_iou": 0}
    report, saved, _ = run_on_new_image(cfg, store, add_image, "b.jpg")
    assert saved == sorted([(CAR, 0.8), (RESP, 0.5), (CAR, 0.3), (PERSON, 0.7), (TRUCK, 0.6)])
    assert (report.boxes, report.merged) == (5, 0)
    assert "겹친 차량" not in report.summary()


def test_a_vehicle_class_the_project_lacks_does_not_remove_others(cfg, store, add_image):
    """모델은 아는데 프로젝트 클래스 목록에 없는 차량 클래스(여기서는 responder 를 뺀 목록)의 박스는 저장 단계에서 버려진다 —
    그 박스가 같은 자리의 car 를 먼저 지우면 둘 다 사라지므로, 정리에 넣지 않는다."""
    store.set_classes([c for c in CLASSES if c != "responder"], drop_extra=True)
    DETS_HI = [[100, 100, 300, 260, 0.90, RESP], [104, 104, 296, 262, 0.60, CAR]]
    image_id = add_image("d.jpg", status="unlabeled", size=(W, H))
    path = Path(store.get_image(image_id)["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    labeler = labeler_for(cfg, store)
    labeler._model.predict = lambda source=None, **kw: [SimpleNamespace(
        boxes=results.Boxes(torch.tensor(DETS_HI, dtype=torch.float32), orig_shape=(H, W)))]
    report = labeler.run([image_id])
    car = store.get_classes().index("car")
    assert [b.cls for b in store.get_boxes(image_id)] == [car]
    assert (report.merged, report.dropped_unknown_class) == (0, 1)


def test_bad_class_iou_stops_before_the_model_runs(cfg, store, add_image):
    cfg.raw["predict"] = {"class_iou": 2}
    image_id = add_image("c.jpg", status="unlabeled")
    labeler = labeler_for(cfg, store)
    with pytest.raises(ValueError):
        labeler.run([image_id])
    assert labeler._model.calls == 0 and store.get_boxes(image_id) == []
