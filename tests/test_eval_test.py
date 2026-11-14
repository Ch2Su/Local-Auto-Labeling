"""wsl/eval_test.py — `at_conf`(놓침·헛 박스)를 세는 혼동 행렬의 문턱 고정 (2026-11-15).

Ultralytics 검증기는 혼동 행렬에 val 의 conf(기본 0.001)를 넘긴다. 8.4.212 는 그 값 그대로·IoU 0.5 로 세서, 같은 가중치·
같은 119장에서 헛 박스가 20 이 아니라 988 로 나왔다 — 에러는 없고 JSON 에는 `conf: 0.25` 라고 남았다. 여기서는 검증기가
부르는 모양 그대로(`conf=0.001`) 불러, 고정한 문턱(conf 0.25·IoU 0.45)으로 세는지 본다. Ultralytics 를 올린 뒤 이 테스트가
깨지면 `pin_confusion_thresholds` 가 더는 먹지 않는 것이다. 모델·GPU 는 쓰지 않는다(혼동 행렬 클래스만).
"""

from __future__ import annotations

import importlib.util

import pytest

from helpers import ROOT

torch = pytest.importorskip("torch")
metrics = pytest.importorskip("ultralytics.utils.metrics")

NAMES = {0: "car", 1: "truck"}
BG = len(NAMES)                 # 행렬의 마지막 줄·칸 = 배경 (m[예측, 정답])
GT_BOX = [0.0, 0.0, 100.0, 100.0]


def load_eval_test():
    spec = importlib.util.spec_from_file_location("eval_test", ROOT / "wsl" / "eval_test.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def ev(monkeypatch):
    """eval_test 모듈 — 테스트가 끝나면 Ultralytics 의 process_batch 를 원래대로 돌린다."""
    monkeypatch.setattr(metrics.ConfusionMatrix, "process_batch", metrics.ConfusionMatrix.process_batch)
    return load_eval_test()


def dets(*rows):
    """(box, conf, cls) 들을 검증기가 넘기는 모양으로."""
    return {"bboxes": torch.tensor([r[0] for r in rows], dtype=torch.float32).reshape(-1, 4),
            "conf": torch.tensor([r[1] for r in rows], dtype=torch.float32),
            "cls": torch.tensor([r[2] for r in rows], dtype=torch.float32)}


def truth(*rows):
    """(box, cls) 들을 정답 배치 모양으로."""
    return {"bboxes": torch.tensor([r[0] for r in rows], dtype=torch.float32).reshape(-1, 4),
            "cls": torch.tensor([r[1] for r in rows], dtype=torch.float32)}


def run(det, gt):
    cm = metrics.ConfusionMatrix(names=NAMES)
    cm.process_batch(det, gt, conf=0.001)          # 검증기가 부르는 그대로 — val 의 기본 conf
    return cm.matrix


def test_low_conf_stray_is_not_a_false_box(ev):
    """conf 0.25 아래의 예측은 헛 박스로 세지 않는다 — 8.4.212 를 그대로 두면 여기서 1 이 나온다."""
    ev.pin_confusion_thresholds(0.25)
    m = run(dets((GT_BOX, 0.9, 0), ([300, 300, 400, 400], 0.1, 0)), truth((GT_BOX, 0)))
    assert m[0, 0] == 1                             # 맞음
    assert m[:BG, BG].sum() == 0                    # 헛 박스 없음
    assert m[BG, :BG].sum() == 0                    # 놓침 없음


def test_low_conf_on_empty_image_is_not_a_false_box(ev):
    """정답이 없는 이미지도 같은 문턱을 탄다(코드의 다른 갈래)."""
    ev.pin_confusion_thresholds(0.25)
    m = run(dets(([10, 10, 50, 50], 0.1, 1), ([60, 60, 90, 90], 0.6, 1)), truth())
    assert m[1, BG] == 1                            # 0.6 하나만


def test_low_conf_does_not_hide_a_miss(ev):
    """정답 자리의 예측이 conf 0.25 아래면 놓친 것이다 — 0.001 로 세면 맞음이 된다."""
    ev.pin_confusion_thresholds(0.25)
    m = run(dets((GT_BOX, 0.1, 0)), truth((GT_BOX, 0)))
    assert m[0, 0] == 0
    assert m[BG, 0] == 1


def test_matches_at_iou_045(ev):
    """IoU 0.47 은 짝이다(11-12 기록의 기준 0.45) — 8.4.212 의 기본 0.5 로는 놓침 + 헛 박스가 된다."""
    ev.pin_confusion_thresholds(0.25)
    m = run(dets(([0, 0, 100, 47], 0.9, 0)), truth((GT_BOX, 0)))
    assert m[0, 0] == 1
    assert m[BG, 0] == 0 and m[0, BG] == 0


def test_below_match_iou_is_miss_and_false(ev):
    ev.pin_confusion_thresholds(0.25)
    m = run(dets(([0, 0, 100, 40], 0.9, 0)), truth((GT_BOX, 0)))      # IoU 0.40
    assert m[0, 0] == 0
    assert m[BG, 0] == 1 and m[0, BG] == 1


def test_wrong_class_is_counted_apart(ev):
    ev.pin_confusion_thresholds(0.25)
    m = run(dets((GT_BOX, 0.9, 1)), truth((GT_BOX, 0)))
    assert m[1, 0] == 1                             # 자리는 맞고 클래스가 다름
    assert m[BG, 0] == 0 and m[1, BG] == 0


def test_pinning_twice_keeps_the_last_values(ev):
    """두 번 불러도 겹쳐 감싸지 않는다 — 원래 함수는 하나만 남는다."""
    before = metrics.ConfusionMatrix.process_batch
    ev.pin_confusion_thresholds(0.25)
    ev.pin_confusion_thresholds(0.5, 0.3)
    assert metrics.ConfusionMatrix.process_batch._unpinned is getattr(before, "_unpinned", before)
    m = run(dets((GT_BOX, 0.4, 0), ([0, 0, 100, 35], 0.9, 0)), truth((GT_BOX, 0)))
    assert m[0, 0] == 1                             # conf 0.4 는 빠지고, IoU 0.35 가 문턱 0.3 을 넘어 짝이 된다
    assert m[:BG, BG].sum() == 0


def test_at_conf_records_the_thresholds(ev):
    """기록에 센 문턱이 남는다 — 숫자만 봐서는 어느 문턱인지 알 수 없다."""
    import numpy as np
    from types import SimpleNamespace

    m = np.array([[3, 1, 2], [0, 4, 1], [5, 0, 0]], dtype=float)       # m[예측, 정답], 끝 = 배경
    res = SimpleNamespace(box=SimpleNamespace(map=0.5, map50=0.7, mp=0.8, mr=0.6, class_result=lambda k: (0.8, 0.6, 0.7, 0.5)),
                          ap_class_index=[0, 1], nt_per_class=[8, 5], confusion_matrix=SimpleNamespace(matrix=m))
    a = ev.summarize(res, NAMES, 0.25, 0.45)["at_conf"]
    assert (a["conf"], a["match_iou"]) == (0.25, 0.45)
    assert (a["hit"], a["missed"], a["false"], a["wrong_class"]) == (7, 5, 3, 1)
