"""tools/import_aihub_pm.py — AI Hub '개인형 이동장치 안전 데이터' 프레임을 검수 대기로 넣는다 (2026-11-22).

AI Hub 의 PM 박스는 탄 사람 + 탈것을 한 박스로 그린다. 그대로 넣으면 우리 기준(탈것만 + person 따로)과 다른 라벨이 에러
없이 들어가므로, 단서로만 쓰는 규칙(`plan_hints`)과 버리는 줄(`clean_pm`)을 여기서 지킨다. 고르기·넣기는 임시 폴더에 만든
작은 가짜 데이터(그림 몇 장은 zip 안에)와 가짜 모델로 끝까지 돌린다 — 실제 데이터·GPU·가중치는 쓰지 않는다.
"""

from __future__ import annotations

import copy
import io
import json
import random
import sqlite3
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from autolabel.ingest import ingest
from autolabel.provenance import load_cameras

from helpers import CLASSES, ROOT

sys.path.insert(0, str(ROOT / "tools"))
import import_aihub_pm as ah  # noqa: E402

torch = pytest.importorskip("torch")
results = pytest.importorskip("ultralytics.engine.results")
pytest.importorskip("cv2")

PERSON, BICYCLE, CAR, MOTO, TRUCK, SCOOTER = (CLASSES.index(n) for n in
                                              ("person", "bicycle", "car", "motorcycle", "truck", "scooter"))
W, H = 320, 184                         # 가짜 프레임 크기 (8 의 배수 — 표식 칸이 JPEG 블록에 맞게)
STAGED, REAL = "CCTV/주간/맑음/연출", "CCTV/주간/맑음/실증"


# ───────────────────────────── PM 줄 ─────────────────────────────

def test_code_groups():
    assert [ah.pm_type(c) for c in (13, 19, "13")] == ["motorcycle"] * 3
    assert [ah.pm_type(c) for c in (20, 27, "24")] == ["bicycle"] * 3
    assert [ah.pm_type(c) for c in (28, 36, "30")] == ["scooter"] * 3
    assert [ah.pm_type(c) for c in (12, 37, None, "x", 1.5e9)] == [None] * 5


def test_clean_pm_keeps_good_rows_and_says_why_it_dropped_the_rest():
    raw = [
        ["30", 100.0, 50.0, 40.0, 80.0],              # 그대로
        ["30", 101.0, 50.0, 40.0, 80.0],              # 같은 물체에 또 하나 (IoU 0.95)
        ["13", 300.0, 100.0, 40.0, 60.0],             # 오른쪽으로 반 나감 — 화면 안에 반이 남아 자른다
        ["13", 310.0, 100.0, 40.0, 60.0],             # 넓이의 1/4 만 남음 — 버린다
        ["20", float("nan"), 10.0, 20.0, 20.0],       # 검증 라벨에 실제로 있던 NaN
        ["20", 10.0, 10.0, 0.0, 20.0],
        ["99", 10.0, 10.0, 20.0, 20.0],
        ["20", 10.0, 10.0, 20.0],                     # 값이 모자란 줄
        ["20", 10.0, "10", 20.0, 20.0],
    ]
    pm, why = ah.clean_pm(raw, W, H)
    assert [(p.code, p.kind, p.box) for p in pm] == [(30, "scooter", (100.0, 50.0, 140.0, 130.0)),
                                                    (13, "motorcycle", (300.0, 100.0, 320.0, 160.0))]
    assert why == {"같은 물체에 또 하나": 1, "화면 밖": 1, "좌표가 숫자가 아님": 2, "크기가 0 이하": 1,
                   "모르는 코드": 1, "꼴이 틀림": 1}


# ───────────────────────────── 단서 규칙 ─────────────────────────────

P = ah.PM(30, "scooter", 100.0, 40.0, 140.0, 140.0)           # 킥보드 PM 박스 40×100
RIDER = ("person", 0.9, 103.0, 41.0, 138.0, 128.0)            # 그 안의 탄 사람
SCOOT = ("scooter", 0.8, 106.0, 78.0, 136.0, 140.0)           # 그 안의 킥보드(아래쪽)


def test_nothing_is_added_when_the_model_found_vehicle_and_rider():
    h = ah.plan_hints([P], [RIDER, SCOOT])
    assert h.recls == {} and h.added == []
    assert h.notes == {"탈것 찾음": 1, "탄 사람 찾음": 1}


def test_a_vehicle_candidate_goes_where_our_labels_put_it():
    """탄 사람만 찾았으면 탈것 후보 하나 — 자리는 PM 박스에 VEH_IN_PM 을 곱한 것(킥보드는 아래 64%)."""
    h = ah.plan_hints([P], [RIDER])
    assert h.recls == {}
    assert [a[0] for a in h.added] == ["scooter"]
    assert h.added[0][1:] == pytest.approx((100 + 0.13 * 40, 40 + 0.36 * 100, 100 + 0.91 * 40, 40 + 1.00 * 100))


def test_rider_and_vehicle_candidates_when_the_model_found_neither():
    h = ah.plan_hints([P], [("car", 0.9, 200.0, 100.0, 300.0, 160.0)])
    assert sorted(a[0] for a in h.added) == ["person", "scooter"]
    person = next(a for a in h.added if a[0] == "person")
    assert person[1:] == pytest.approx((100 + 0.07 * 40, 40 + 0.01 * 100, 100 + 0.95 * 40, 40 + 0.89 * 100))


def test_model_class_gives_way_to_the_aihub_kind():
    """모델이 킥보드를 bicycle 로 잡은 곳 — 박스는 그대로 두고 클래스만 AI Hub 종류로. 후보를 또 넣지 않는다."""
    h = ah.plan_hints([P], [RIDER, ("bicycle", 0.6, *SCOOT[2:])])
    assert h.recls == {1: "scooter"} and h.added == []


def test_mixed_code_keeps_the_model_class():
    mixed = ah.PM(15, "motorcycle", *P.box)                    # 코드 15 는 자전거가 섞여 있다
    h = ah.plan_hints([mixed], [RIDER, ("bicycle", 0.6, *SCOOT[2:])])
    assert h.recls == {} and h.added == []


def test_one_model_box_serves_one_pm_box():
    """나란히 선 두 PM 박스에 모델 탈것이 하나뿐이면 다른 하나에는 후보가 들어간다."""
    q = ah.PM(30, "scooter", 150.0, 40.0, 190.0, 140.0)
    preds = [RIDER, SCOOT, ("person", 0.9, 153.0, 41.0, 188.0, 128.0)]
    h = ah.plan_hints([P, q], preds)
    assert [a[0] for a in h.added] == ["scooter"]
    assert h.added[0][1] > 150                                 # 후보는 모델이 못 찾은 쪽에


def test_a_parked_bike_outside_pm_boxes_is_left_alone():
    parked = ("bicycle", 0.9, 220.0, 60.0, 280.0, 120.0)
    h = ah.plan_hints([P], [RIDER, SCOOT, parked])
    assert h.recls == {} and h.added == []


def test_same_kind_box_drawn_differently_counts_as_the_vehicle():
    """탈것 박스가 PM 박스 밖으로 많이 나가 '안에 든' 것으로 못 세도, 같은 종류면 후보를 겹쳐 넣지 않는다."""
    wide = ("scooter", 0.5, 90.0, 80.0, 175.0, 140.0)          # 넓이의 반쯤만 PM 박스 안
    assert ah.inside(wide[2:], P.box) < ah.VEH_INSIDE
    h = ah.plan_hints([P], [RIDER, wide])
    assert h.added == [] and h.notes["탈것 모양만 다름"] == 1


def test_same_kind_box_overlapping_the_candidate_counts_too():
    """중심이 PM 박스 밖이어도 후보 자리와 IoU 0.1 이상 겹치는 같은 종류 박스면 그 탈것이다. 멀리 있는 같은 종류 박스는 아니다."""
    off = ("scooter", 0.5, 120.0, 78.0, 200.0, 140.0)           # 중심 x 160 — PM 박스(100–140) 밖
    assert ah.inside(off[2:], P.box) < ah.VEH_INSIDE and not (P.x1 <= 160.0 <= P.x2)
    assert ah.iou(ah.place(P.box, ah.VEH_IN_PM["scooter"]), off[2:]) >= ah.NEAR_IOU
    assert ah.plan_hints([P], [RIDER, off]).added == []
    far = ("scooter", 0.5, 220.0, 78.0, 300.0, 140.0)
    assert [a[0] for a in ah.plan_hints([P], [RIDER, far]).added] == ["scooter"]


def test_other_kind_box_nearby_is_not_relabelled():
    """PM 박스에 걸친 다른 종류(옆에 세워 둔 자전거)는 건드리지 않고 후보를 넣는다 — 클래스를 바꾸는 것은 안에 든 박스뿐."""
    bike = ("bicycle", 0.9, 90.0, 80.0, 175.0, 140.0)
    h = ah.plan_hints([P], [RIDER, bike])
    assert h.recls == {} and [a[0] for a in h.added] == ["scooter"]


# ───────────────────────────── 모델 ─────────────────────────────

class FakeModel:
    """정해 둔 검출을 Ultralytics 의 Boxes 로 돌려준다(tests/test_predictor.py 와 같은 방식). fn(그림 배열) → [[x1,y1,x2,y2,conf,cls]]."""
    names = dict(enumerate(CLASSES))

    def __init__(self, fn):
        self.fn = fn
        self.calls = 0

    def predict(self, source=None, **kw):
        self.calls += 1
        dets = torch.tensor(self.fn(source), dtype=torch.float32).reshape(-1, 6)
        return [SimpleNamespace(boxes=results.Boxes(dets, orig_shape=source.shape[:2]))]


def labeler(cfg, fn) -> ah.Labeler:
    return ah.Labeler(cfg, list(CLASSES), None, "cpu", model=FakeModel(fn))


def test_detect_maps_names_and_drops_the_overlapping_vehicle_class(cfg):
    """자동 라벨링과 같은 차량 겹침 정리 — 같은 차의 car·truck 은 신뢰도 높은 하나만, 사람·탈것은 그대로."""
    dets = [[10, 10, 110, 90, 0.8, CAR], [12, 11, 108, 91, 0.5, TRUCK], [200, 20, 230, 100, 0.7, PERSON]]
    lab = labeler(cfg, lambda img: dets)
    preds = lab.detect(np.zeros((H, W, 3), np.uint8))
    assert [(p[0], round(p[1], 2)) for p in preds] == [("car", 0.8), ("person", 0.7)]
    assert lab.version.endswith("+aihub")


def test_boxes_mark_what_came_from_aihub(cfg):
    lab = labeler(cfg, lambda img: [])
    preds = [RIDER, ("bicycle", 0.6, *SCOOT[2:]), ("car", 0.9, 200.0, 100.0, 300.0, 160.0)]
    q = ah.PM(13, "motorcycle", 20.0, 40.0, 80.0, 140.0)       # 모델이 아무것도 못 찾은 오토바이
    boxes, score, hints = lab.boxes(preds, [P, q], W, H)
    got = sorted((b.cls, b.origin, b.conf) for b in boxes)
    assert got == sorted([(PERSON, "auto", 0.9), (SCOOTER, "aihub", None), (CAR, "auto", 0.9),
                          (MOTO, "aihub", None), (PERSON, "aihub", None)])
    fixed = next(b for b in boxes if b.cls == SCOOTER)          # 클래스만 바뀌고 자리는 모델 박스 그대로
    assert (fixed.cx * W, fixed.cy * H, fixed.w * W, fixed.h * H) == pytest.approx((121.0, 109.0, 30.0, 62.0))
    assert all(0 <= b.cx <= 1 and 0 <= b.cy <= 1 for b in boxes)


def test_missing_project_class_stops(cfg):
    with pytest.raises(SystemExit, match="scooter"):
        ah.Labeler(cfg, [c for c in CLASSES if c != "scooter"], None, "cpu", model=FakeModel(lambda img: []))


# ───────────────────────────── 가짜 데이터 ─────────────────────────────

def frame_bytes(scene: int, n_cars: int, night: bool = False) -> bytes:
    """장면 번호로 정해지는 무늬 + 왼쪽 위 16×16 칸에 '모델이 찾을 차 수' 표식(밝기 = 수 × 10).
    night 면 같은 무늬를 어둡게 하고(왼쪽일수록 더) 불빛 셋을 얹는다 — 밝기 무늬는 달라지고 모서리는 남는다."""
    rng = np.random.default_rng(scene)
    im = Image.new("L", (W, H), int(rng.integers(90, 150)))
    dr = ImageDraw.Draw(im)
    for _ in range(45):                              # 기울기가 제각각인 다각형·선 — 윤곽 방향 무늬가 장면마다 다르게
        cx, cy, r = rng.integers(0, W), rng.integers(0, H), rng.integers(8, 40)
        ang = np.sort(rng.uniform(0, 2 * np.pi, int(rng.integers(3, 6))))
        dr.polygon([(float(cx + r * np.cos(t)), float(cy + r * np.sin(t))) for t in ang], fill=int(rng.integers(20, 236)))
    for _ in range(12):
        x1, y1, x2, y2 = (int(v) for v in (rng.integers(0, W), rng.integers(0, H), rng.integers(0, W), rng.integers(0, H)))
        dr.line((x1, y1, x2, y2), fill=int(rng.integers(180, 256)), width=int(rng.integers(2, 5)))
    a = np.asarray(im, dtype=np.float32)
    if night:
        a = 255.0 * (a / 255.0) ** 2.5 * np.linspace(0.15, 1.0, W, dtype=np.float32)[None, :] ** 2
        yy, xx = np.mgrid[0:H, 0:W]
        for cx, cy in ((60, 50), (200, 120), (280, 40)):
            a += 220.0 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 18.0 ** 2))
    a = np.clip(a, 0, 255).astype(np.uint8)
    a[:16, :16] = n_cars * 10
    buf = io.BytesIO()
    Image.fromarray(a, "L").convert("RGB").save(buf, "JPEG", quality=95)
    return buf.getvalue()


def cars_from_marker(img) -> list:
    """가짜 모델: 표식 칸의 밝기만큼 차 박스를 낸다(PM 박스와 떨어진 오른쪽 아래에)."""
    n = int(round(float(img[:16, :16].mean()) / 10))
    return [[200 + 4 * k, 150, 230 + 4 * k, 180, 0.9 - 0.01 * k, CAR] for k in range(n)]


def label_json(pm: list) -> str:
    return json.dumps({"info": {}, "description": {"frame_id": "0001", "imageWidth": W, "imageHeight": H},
                       "annotations": {"environment": [], "PM": [{"PM_code": c, "shape_type": "bbox", "points": pts}
                                                                 for c, *pts in pm]}})


@pytest.fixture
def aihub(tmp_path):
    """작은 AI Hub 폴더. {영상 id: (장면, 차 수)} 로 프레임을 만든다 — 연출 폴더의 그림은 zip 안에, 실증은 풀린 폴더에."""
    root = tmp_path / "aihub" / "01.데이터"
    labels, images = root / "1.Training" / "라벨링데이터", root / "1.Training" / "원천데이터"
    staged = {"T000001": (1, 2), "T000002": (2, 0), "T000003": (2, 1),      # 2·3 은 같은 화면
              "T000004": (4, 9), "T000005": (5, 3),                         # 4 는 붐빈다
              "T000006": (1, 2, True)}                                      # 1 과 같은 자리의 밤 화면
    real = {"S000001": (11, 1), "S000002": (12, 2)}
    images.mkdir(parents=True)
    with zipfile.ZipFile(images / "TS9.zip", "w") as z:
        for video, spec in staged.items():
            for frame in ("0010", "0020"):
                stem = f"{video}_000_{frame}_C_D_F_1"
                (labels / STAGED).mkdir(parents=True, exist_ok=True)
                (labels / STAGED / f"{stem}.json").write_text(label_json([["30", 100.0, 40.0, 40.0, 100.0]]), "utf-8")
                z.writestr(f"{STAGED}/{stem}.jpg", frame_bytes(*spec))
    for video, (scene, cars) in real.items():
        stem = f"{video}_000_0005_C_D_F_0"
        for d in (labels / REAL, images / REAL):
            d.mkdir(parents=True, exist_ok=True)
        (labels / REAL / f"{stem}.json").write_text(
            label_json([["13", 60.0, 60.0, 60.0, 90.0], ["30", 10.0, 150.0, 10.0, 20.0]]), "utf-8")   # 킥보드는 작다
        (images / REAL / f"{stem}.jpg").write_bytes(frame_bytes(scene, cars))
    # 블랙박스 폴더와 읽지 못할 라벨 — 색인에는 들어가도 고르지 않는다
    (labels / "블랙박스/주간/맑음/연출").mkdir(parents=True)
    (labels / "블랙박스/주간/맑음/연출/T000009_000_0001_B_D_F_1.json").write_text(
        label_json([["30", 100.0, 40.0, 40.0, 100.0]]), "utf-8")
    (labels / STAGED / "T000008_000_0001_C_D_F_1.json").write_text("{깨진 json", "utf-8")
    return SimpleNamespace(root=root, labels=labels, images=images)


# 연출 몫 4 는 고를 수 있는 것(3)보다 많다 — 섞는 순서와 상관없이 여섯 영상을 다 보게 된다
BUCKETS = (ah.Bucket("연출 킥보드", (STAGED,), "scooter", 4), ah.Bucket("실증 오토바이", (REAL,), "motorcycle", 1))


def run(cfg, store, aihub, tmp_path, *, buckets=BUCKETS, seed=7, max_boxes=5):
    idx = tmp_path / "index.jsonl"
    if not idx.exists():
        ah.build_index(aihub.labels, idx)
    cands = ah.candidates(ah.read_index(idx), buckets, min_height=40.0)
    taken, scenes = ah.existing(store)
    lab = labeler(cfg, cars_from_marker)
    frames = ah.Frames(aihub.images)
    try:
        picks, report = ah.choose(buckets, cands, frames, lab, random.Random(seed), taken=taken, scenes=scenes,
                                  max_boxes=max_boxes, log=lambda *_: None)
    finally:
        frames.close()
    for p, key in zip(picks, ah.next_keys(ah.used_keys(cfg, store), len(picks))):
        p.camera = key
    return picks, report, lab


def test_index_skips_broken_labels_and_candidates_follow_the_bucket(aihub, tmp_path):
    idx = tmp_path / "index.jsonl"
    assert ah.build_index(aihub.labels, idx) == 16                       # 연출 12 + 실증 2 + 블랙박스 1 + 깨진 것 1
    rows = list(ah.read_index(idx))
    assert len(rows) == 15 and all(isinstance(r[4], list) for r in rows)
    cands = ah.candidates(iter(rows), BUCKETS, min_height=40.0)
    assert sorted(cands["연출 킥보드"]) == ["T000001", "T000002", "T000003", "T000004", "T000005", "T000006"]
    assert all(len(v) == 2 for v in cands["연출 킥보드"].values())
    assert sorted(cands["실증 오토바이"]) == ["S000001", "S000002"]
    # 실증 프레임의 킥보드는 높이 20 이라, 킥보드를 찾는 묶음이 그 폴더를 봐도 걸리지 않는다
    small = ah.candidates(iter(rows), (ah.Bucket("실증 킥보드", (REAL,), "scooter", 1),), min_height=40.0)
    assert small["실증 킥보드"] == {}


def test_frames_come_from_the_folder_or_the_zip(aihub):
    frames = ah.Frames(aihub.images)
    try:
        assert frames.read(REAL, "S000001_000_0005_C_D_F_0") == frame_bytes(11, 1)
        assert frames.read(STAGED, "T000001_000_0010_C_D_F_1") == frame_bytes(1, 2)       # zip 안
        assert frames.read(STAGED, "T999999_000_0010_C_D_F_1") is None
    finally:
        frames.close()


def test_same_scene_is_told_by_the_thumbnail():
    a, b, c = (ah.signature(frame_bytes(s, n)) for s, n in ((2, 0), (2, 1), (3, 0)))
    assert ah.scene_corr(a, b) > 0.95 and abs(ah.scene_corr(a, c)) < ah.SAME_SCENE


def test_same_camera_is_found_across_day_and_night():
    """밝기 무늬가 달라진 같은 자리(주간·야간) — 축소본 상관으로는 다른 화면이지만 특징점이 한 화면으로 맞는다.
    2026-11-22 에 상관만으로 고른 100장에 같은 자리의 주간·야간 프레임이 다른 카메라로 들어 있었다."""
    day, night, other = (ah.scene_of(frame_bytes(*a)) for a in ((2, 0), (2, 1, True), (3, 0)))
    assert ah.scene_corr(day.sig, night.sig) < ah.SAME_SCENE
    assert float(day.edges @ night.edges) < ah.SAME_EDGES                     # 윤곽 무늬로도 모자라고
    inliers, overlap = ah.orb_match(day, night)                               # 특징점이 한 화면으로 맞는다
    assert inliers >= 50 and overlap > 0.9
    assert ah.same_camera(day, night) and ah.same_camera(night, day)
    assert not ah.same_camera(day, other) and not ah.same_camera(night, other)
    scenes = [ah.scene_of(frame_bytes(s, 0)) for s in range(20, 32)]          # 서로 다른 화면 12장 — 한 쌍도 같다고 하지 않는다
    assert not any(ah.same_camera(a, b) for i, a in enumerate(scenes) for b in scenes[i + 1:])


def test_edge_pattern_alone_can_tell_the_same_place():
    """밝기 상관도 특징점도 못 잡은 같은 자리(11-22 의 시장 앞 횡단보도 주간·야간: 상관 0.17 · 인라이어 6)는 윤곽 방향 무늬로
    잡았다. 여기서는 같은 무늬를 흐리게 해 특징점을 지우고 밝기를 뒤섞어 그 경우를 만든다."""
    cv2 = pytest.importorskip("cv2")
    base = cv2.imdecode(np.frombuffer(frame_bytes(7, 0), np.uint8), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    ramp = np.linspace(0.3, 1.0, base.shape[0], dtype=np.float32)[:, None]    # 위는 어둡게, 아래는 그대로
    dim = np.clip(cv2.GaussianBlur(base, (0, 0), 2.0) * ramp + 20, 0, 255)
    a, b = ah.edge_signature(base), ah.edge_signature(dim)
    c = ah.edge_signature(cv2.imdecode(np.frombuffer(frame_bytes(8, 0), np.uint8), cv2.IMREAD_GRAYSCALE).astype(np.float32))
    assert float(a @ b) >= ah.SAME_EDGES > float(a @ c)
    assert abs(float(a @ a) - 1.0) < 1e-4


def test_overlap_of_a_frame_with_itself_is_whole():
    """같은 화면끼리는 변환이 항등이라 겹치는 몫이 1 이어야 한다 — `cv2.intersectConvexConvex` 는 변이 정확히 겹치면 0.63 을 냈다."""
    s = ah.scene_of(frame_bytes(2, 0))
    inliers, overlap = ah.orb_match(s, s)
    assert inliers > 500 and overlap == pytest.approx(1.0, abs=1e-3)
    assert ah._area(ah._clip([(-10.0, -10.0), (30.0, -10.0), (30.0, 30.0), (-10.0, 30.0)], 20, 20)) == pytest.approx(400.0)
    assert ah._clip([(40.0, 40.0), (50.0, 40.0), (50.0, 50.0)], 20, 20) == []


def test_choose_takes_one_frame_per_video_and_skips_same_scene_and_crowded(cfg, store, aihub, tmp_path):
    picks, report, _ = run(cfg, store, aihub, tmp_path)
    staged = [p for p in picks if p.bucket == "연출 킥보드"]
    videos = [p.video for p in staged]
    assert len(videos) == len(set(videos)) == 3
    assert "T000004" not in videos                                        # 차 9대 — 붐빈다
    assert len({"T000002", "T000003"} & set(videos)) == 1                 # 같은 화면은 한 장만
    assert len({"T000001", "T000006"} & set(videos)) == 1                 # 같은 자리의 낮·밤도 한 장만
    assert dict(report["연출 킥보드"]) == {"후보 영상": 6, "붐빔": 1, "같은 화면": 2, "고름": 3}
    assert [p.video for p in picks if p.bucket == "실증 오토바이"] in (["S000001"], ["S000002"])
    assert [p.camera for p in picks] == ["ah-0001", "ah-0002", "ah-0003", "ah-0004"]
    for p in staged:                                                      # 가짜 모델은 차만 낸다 — 탈것·사람은 후보로
        assert sorted(CLASSES[b.cls] for b in p.boxes if b.origin == "aihub") == ["person", "scooter"]
        assert sum(b.origin == "auto" for b in p.boxes) == len(p.preds)
    again, _, _ = run(cfg, store, aihub, tmp_path)
    assert [(p.video, p.stem) for p in again] == [(p.video, p.stem) for p in picks]       # 같은 씨앗이면 같은 프레임


def test_apply_writes_files_cameras_and_pending_rows(cfg, store, aihub, tmp_path, monkeypatch):
    cfg.cameras.write_text("cameras:\n  hw-01:\n    kind: cctv\n    place: 어딘가\n    split:\n", "utf-8")
    picks, _, lab = run(cfg, store, aihub, tmp_path)
    monkeypatch.setattr(ah, "resolve", lambda p: tmp_path / p)            # 목록을 저장소의 runs/ 가 아니라 임시 폴더에
    n, manifest = ah.apply(cfg, store, picks, lab.version, "20261122_000000")
    assert n == 4 and manifest == tmp_path / "runs" / "aihub_pm" / "import_20261122_000000.tsv"
    cams = load_cameras(cfg.cameras)
    assert sorted(cams) == ["ah-0001", "ah-0002", "ah-0003", "ah-0004", "hw-01"]
    assert cams["ah-0001"].kind == "dataset"
    # AI Hub 그림은 전부 train 이다(2026-11-23 사용자 결정) — 새로 등록하는 카메라에 고정을 적는다. 다른 카메라는 그대로
    assert {k: c.split for k, c in cams.items()} == {**{f"ah-000{i}": "train" for i in range(1, 5)}, "hw-01": None}
    assert cams["ah-0001"].place == f"AI Hub PM 영상 {picks[0].video}"
    assert len(list((cfg.db.parent / "backups").glob("cameras_*.yaml"))) == 1         # 쓰기 전 파일은 한 번만
    with store.connect() as con:
        rows = {r["name"]: r for r in con.execute("SELECT * FROM images")}
    assert len(rows) == 4
    for p in picks:
        r = rows[f"{p.stem}.jpg"]
        assert (r["source"], r["status"], r["camera"], r["domain"], r["captured_at"]) == \
            ("aihub_pm", "pending", p.camera, "cctv", None)
        assert (r["width"], r["height"]) == (W, H) and r["model_version"] == lab.version
        f = cfg.unlabeled / p.camera / f"{p.stem}.jpg"
        assert Path(r["path"]) == f.resolve() and f.read_bytes() == p.data            # AI Hub 바이트 그대로
        saved = sorted((b.cls, b.origin, b.conf) for b in store.get_boxes(r["id"]))
        assert saved == sorted((b.cls, b.origin, b.conf) for b in p.boxes)
    lines = manifest.read_text("utf-8").splitlines()
    assert lines[0].split("\t")[:3] == ["bucket", "camera", "video"] and len(lines) == 5

    # scan 은 이 도구가 넣은 경로를 이미 아는 경로로 건너뛴다(출처·상태·박스 그대로)
    before = {r["name"]: len(store.get_boxes(r["id"])) for r in rows.values()}
    report = ingest(cfg, store)
    assert report.unlabeled_already == 4
    with store.connect() as con:
        after = {r["name"]: r for r in con.execute("SELECT * FROM images")}
    assert {k: (v["source"], v["status"]) for k, v in after.items()} == {k: ("aihub_pm", "pending") for k in rows}
    assert {r["name"]: len(store.get_boxes(r["id"])) for r in after.values()} == before

    # 다시 돌리면 이미 넣은 영상과 DB 에 있는 것과 같은 화면은 빼고, 번호는 이어진다
    more, report2, _ = run(cfg, store, aihub, tmp_path)
    assert [(p.bucket, p.camera) for p in more] == [("실증 오토바이", "ah-0005")]
    assert not {p.video for p in more} & {p.video for p in picks}
    assert dict(report2["연출 킥보드"]) == {"후보 영상": 6, "이미 쓴 영상": 3, "붐빔": 1, "같은 화면": 2, "고름": 0}


def test_failed_apply_puts_files_and_cameras_back(cfg, store, aihub, tmp_path, monkeypatch):
    cfg.cameras.write_text("cameras:\n  hw-01:\n    kind: cctv\n    place: 어딘가\n    split:\n", "utf-8")
    before = cfg.cameras.read_bytes()
    picks, _, lab = run(cfg, store, aihub, tmp_path)

    def boom(items):
        raise RuntimeError("디스크가 찼다")

    monkeypatch.setattr(store, "ingest_batch", boom)
    with pytest.raises(RuntimeError, match="디스크"):
        ah.apply(cfg, store, picks, lab.version, "20261122_000001")
    assert cfg.cameras.read_bytes() == before
    assert not list(cfg.unlabeled.rglob("*.jpg")) and not [d for d in cfg.unlabeled.glob("ah-*")]
    with store.connect() as con:
        assert con.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 0


# ---------------------------------------------------- AI Hub 그림은 전부 train (2026-11-23)

REGISTRY = """\
cameras:
  hw-01:
    kind: cctv
    place: 어딘가
    split:
  ah-0001:
    kind: dataset
    place: AI Hub PM 영상 A
    split:
  ah-0002:
    kind: dataset
    place: AI Hub PM 영상 B
    split: train
  ah-0003:
    kind: dataset
    place: AI Hub PM 영상 C
    split: test
  ah-0004:
    kind: dataset
    place: AI Hub PM 영상 D
    split:
  ah-0005:
    kind: dataset
    place: 손으로 등록해 split 줄이 없는 것
  ah-x:
    kind: dataset
    place: 키 꼴이 다른 것
    split:
"""


@pytest.fixture
def registry(cfg, add_image):
    """고정이 빈 것(ah-0001·0004·0005), train(0002), 사람이 test 로 적은 것(0003). ah-0004 는 이미지가 이미 val 에 있다."""
    cfg.cameras.write_text(REGISTRY, "utf-8")
    add_image("a.jpg", camera="ah-0001")                    # 미배정 — 막지 않는다
    add_image("d.jpg", camera="ah-0004", split="val")       # 이미 평가셋에 든 이미지 — 고정하면 두 분할에 걸친다
    add_image("e.jpg", camera="ah-0005", split="train")     # 이미 train — 고정해도 걸치지 않는다
    add_image("h.jpg", camera="hw-01", split="test")        # 다른 카메라는 보지 않는다
    return cfg.cameras


def pins(path) -> dict:
    return {k: c.split for k, c in load_cameras(path).items()}


def test_pin_plan_takes_only_empty_pins_that_would_not_straddle(cfg, store, registry):
    todo, other, blocked = ah.pin_plan(cfg, store)
    assert todo == ["ah-0001", "ah-0005"]
    assert other == {"ah-0003": "test"}                     # 사람이 적어 둔 고정은 바꾸지 않는다
    assert blocked == {"ah-0004": {"val": 1}}


def test_pin_train_shows_first_and_writes_only_when_told(cfg, store, registry, capsys):
    before = registry.read_bytes()
    assert ah.pin_train(cfg, store, write=False) == 1       # 볼 것(다른 고정·막힘)이 남아 1
    assert registry.read_bytes() == before and not (cfg.db.parent / "backups").exists()
    said = capsys.readouterr().out
    assert "ah-* 카메라 5대: train 고정 1 · 고정할 것 2 · 다른 분할로 고정 1 · 막힘 1" in said
    assert "ah-0003 는 test 로 고정" in said and "ah-0004 의 이미지가 이미 val 1장" in said and "보기만 했다" in said
    assert "ah-NNNN 꼴이 아닌 카메라 1대: ah-x" in said                         # 고정하지 않고 알린다(2026-11-23 리뷰)

    assert ah.pin_train(cfg, store, write=True) == 1
    assert pins(registry) == {"hw-01": None, "ah-0001": "train", "ah-0002": "train", "ah-0003": "test",
                              "ah-0004": None, "ah-0005": "train", "ah-x": None}
    backups = list((cfg.db.parent / "backups").glob("cameras_*.yaml"))
    assert len(backups) == 1 and backups[0].read_bytes() == before           # 쓰기 전 파일 하나

    after = registry.read_bytes()
    assert ah.pin_train(cfg, store, write=True) == 1        # 다시 돌려도 더 고칠 것이 없다
    assert registry.read_bytes() == after and len(list((cfg.db.parent / "backups").glob("cameras_*.yaml"))) == 1


def test_pin_train_is_clean_when_every_camera_can_be_pinned(cfg, store, capsys):
    cfg.cameras.write_text("cameras:\n  ah-0001:\n    kind: dataset\n    place: 가\n    split:\n"
                           "  ah-0002:\n    kind: dataset\n    place: 나\n    split:\n", "utf-8")
    assert ah.pin_train(cfg, store, write=True) == 0
    assert pins(cfg.cameras) == {"ah-0001": "train", "ah-0002": "train"}
    assert ah.pin_train(cfg, store, write=True) == 0 and "고칠 것이 없다" in capsys.readouterr().out


def test_pin_train_runs_without_the_aihub_folder_and_importing_points_at_it(cfg, store, registry, tmp_path, monkeypatch,
                                                                       capsys):
    """`--pin-train` 은 색인·모델·AI Hub 폴더 없이 돈다. 넣으려는 실행은 고정 안 된 카메라가 있으면 먼저 알린다."""
    import yaml

    cfg.path.write_text(yaml.safe_dump(cfg.raw, allow_unicode=True), encoding="utf-8")
    nowhere = tmp_path / "no-aihub"
    monkeypatch.setattr(sys, "argv", ["import_aihub_pm.py", "--config", str(cfg.path), "--root", str(nowhere),
                                      "--pin-train"])
    before = registry.read_bytes()
    assert ah.main() == 1 and registry.read_bytes() == before                # --apply 없이는 보기만

    monkeypatch.setattr(sys, "argv", ["import_aihub_pm.py", "--config", str(cfg.path), "--root", str(nowhere)])
    capsys.readouterr()
    with pytest.raises(SystemExit, match="원천데이터"):
        ah.main()
    assert "train 으로 고정되지 않은 ah-* 카메라 4대" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["import_aihub_pm.py", "--config", str(cfg.path), "--root", str(nowhere),
                                      "--pin-train", "--apply"])
    assert ah.main() == 1 and pins(registry)["ah-0001"] == "train"


# ---------------------------------------------------- 손으로 고른 그림 점검 --check (2026-11-23)

def put(cfg, store, camera: str, name: str, data: bytes, *, source: str = "aihub_pm", status: str = "pending") -> int:
    """그림 파일과 DB 행을 함께 둔다 — 도구가 넣은 것(source aihub_pm)이나 사람이 손으로 넣은 것(collected)처럼."""
    f = cfg.unlabeled / camera / name
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(data)
    store.ingest_batch([{"path": str(f.resolve()), "source": source, "width": W, "height": H, "status": status,
                         "boxes": [], "camera": camera}])
    with store.connect() as con:
        return int(con.execute("SELECT id FROM images WHERE path=?", (str(f.resolve()),)).fetchone()[0])


def aihub_name(video: int, frame: int = 10) -> str:
    return f"T{video:06d}_000_{frame:04d}_C_D_F_1.jpg"


@pytest.fixture
def stock(cfg, store):
    """DB 에 있는 것: 도구가 넣은 둘(장면 1·2), 사람이 `ah-0101` 로 손으로 넣은 하나(장면 22), 직접 모은 카메라의 하나(장면 26)."""
    s = SimpleNamespace()
    s.a = put(cfg, store, "ah-0001", aihub_name(1), frame_bytes(1, 2))
    s.b = put(cfg, store, "ah-0002", aihub_name(2), frame_bytes(2, 0), status="approved")
    s.manual = put(cfg, store, "ah-0101", aihub_name(50), frame_bytes(22, 0), source="collected", status="approved")
    s.own = put(cfg, store, "hw-01", "hw-01_20261109_101500_000.jpg", frame_bytes(26, 1), source="collected")
    return s


@pytest.fixture
def picked(cfg, stock, tmp_path):
    """사용자가 손으로 골라 둔 폴더."""
    d = tmp_path / "picked"
    d.mkdir()
    files = {
        aihub_name(1): frame_bytes(1, 2),               # 이미 넣은 그림 그대로
        aihub_name(1, 20): frame_bytes(1, 3),           # 그 영상의 다른 프레임 — 바이트는 다르다
        aihub_name(50, 33): frame_bytes(23, 0),         # 손으로 넣은 영상의 다른 프레임(화면은 다르게 — 이름으로만 안다)
        aihub_name(77): frame_bytes(2, 1, True),        # 새 영상인데 ah-0002 와 같은 자리의 밤 화면
        aihub_name(80): frame_bytes(24, 0),             # 새 것
        aihub_name(81): frame_bytes(25, 0),             # 새 것
        aihub_name(81, 20): frame_bytes(25, 1),         # 바로 위와 같은 영상
        aihub_name(82): frame_bytes(24, 1),             # 영상은 다르지만 T000080 과 같은 자리
        aihub_name(83): b"jpeg \xff\xd8 not really",    # 읽히지 않는 파일
        aihub_name(84): frame_bytes(26, 0),             # 직접 모은 카메라(hw-01)와 같은 자리 — AI Hub 것끼리만 자리를 견준다
        aihub_name(85): frame_bytes(26, 1),             # 직접 모은 카메라의 그림과 바이트가 같다
        aihub_name(86): frame_bytes(24, 0),             # T000080 과 바이트가 같다(이름만 다르다)
        "zz_photo.jpg": frame_bytes(27, 0),             # AI Hub 이름 꼴이 아닌 새 그림
    }
    for name, data in files.items():
        (d / name).write_bytes(data)
    return d


def by_name(results) -> dict:
    return {c.path.name: c for c in results}


def test_check_tells_same_file_same_video_and_same_place(cfg, store, stock, picked):
    own_path = cfg.unlabeled / "ah-0001" / aihub_name(1)                    # DB 에 이미 있는 경로도 함께 준다
    results = ah.check_files(store, ah.gather([picked, own_path]), log=lambda *_: None)
    got = by_name(c for c in results if c.path.parent == picked.resolve())
    verdicts = {name: c.verdict for name, c in got.items()}
    assert verdicts == {
        aihub_name(1): "같은 그림", aihub_name(1, 20): "같은 영상", aihub_name(50, 33): "같은 영상",
        aihub_name(77): "같은 자리", aihub_name(80): "새 것", aihub_name(81): "새 것",
        aihub_name(81, 20): "준 것끼리 겹침", aihub_name(82): "준 것끼리 겹침", aihub_name(83): "못 읽음",
        aihub_name(84): "새 것", aihub_name(85): "같은 그림", aihub_name(86): "준 것끼리 겹침", "zz_photo.jpg": "새 것"}
    assert [c.verdict for c in results if c.path == own_path.resolve()] == ["이미 등록된 경로"]

    same = got[aihub_name(1)]
    assert same.same_bytes["id"] == stock.a and [k.id for k in same.same_video] == [stock.a]
    frame = got[aihub_name(1, 20)]                                           # 같은 영상의 다른 프레임 — scan 은 못 잡는다
    assert frame.same_bytes is None and [k.id for k in frame.same_video] == [stock.a]
    assert [k.id for k in frame.same_spot] == [stock.a]
    manual = got[aihub_name(50, 33)]                                         # 사람이 손으로 넣은 영상도 안다(카메라 키로)
    assert [(k.id, k.camera) for k in manual.same_video] == [(stock.manual, "ah-0101")] and manual.same_spot == []
    place = got[aihub_name(77)]                                              # 영상은 처음인데 같은 카메라 자리(주간·야간)
    assert place.same_video == [] and [(k.id, k.camera) for k in place.same_spot] == [(stock.b, "ah-0002")]
    assert got[aihub_name(84)].same_spot == []                               # 직접 모은 카메라와는 자리를 견주지 않는다
    assert got[aihub_name(85)].same_bytes["camera"] == "hw-01"               # 바이트는 어느 카메라의 그림과도 견준다
    twin = got[aihub_name(81, 20)]
    assert [p.name for p in twin.twin_video] == [aihub_name(81)] and [p.name for p in twin.twin_spot] == [aihub_name(81)]
    spot = got[aihub_name(82)]
    assert spot.twin_video == [] and [p.name for p in spot.twin_spot] == [aihub_name(80)]
    assert got[aihub_name(86)].twin_bytes.name == aihub_name(80)
    assert got["zz_photo.jpg"].video is None and got[aihub_name(83)].error


def test_check_writes_nothing_but_its_list(cfg, store, stock, picked, monkeypatch, capsys):
    import yaml

    cfg.cameras.write_text("cameras:\n  hw-01:\n    kind: cctv\n    place: 어딘가\n    split:\n", "utf-8")
    cfg.path.write_text(yaml.safe_dump(cfg.raw, allow_unicode=True), encoding="utf-8")
    cameras = cfg.cameras.read_bytes()
    with store.connect() as con:
        before = con.execute("SELECT id, path, status, camera, source, updated_at FROM images ORDER BY id").fetchall()
    files_before = sorted(p for p in cfg.unlabeled.rglob("*") if p.is_file())
    monkeypatch.setattr(sys, "argv", ["import_aihub_pm.py", "--config", str(cfg.path), "--root", str(picked / "none"),
                                      "--check", str(picked)])
    assert ah.main() == 0                                                    # AI Hub 폴더·색인·모델 없이 돈다
    said = capsys.readouterr().out
    assert "준 그림 13장 (AI Hub 이름 꼴 12 · 아닌 것 1)" in said
    assert "새 것 4 · 같은 그림 2 · 같은 영상 2 · 같은 자리 1 · 준 것끼리 겹침 3 · 못 읽음 1" in said
    assert f"[같은 영상] {aihub_name(1, 20)} — 영상 T000001 = ah-0001 id {stock.a} 검수 대기" in said
    assert f"[같은 자리] {aihub_name(77)} — 같은 자리 = ah-0002 id {stock.b} 승인" in said
    assert "다음 번호 ah-0102 부터" in said                                    # 손으로 등록한 ah-0101 다음
    with store.connect() as con:
        assert con.execute("SELECT id, path, status, camera, source, updated_at FROM images ORDER BY id").fetchall() == before
    assert cfg.cameras.read_bytes() == cameras and sorted(p for p in cfg.unlabeled.rglob("*") if p.is_file()) == files_before
    lists = list((cfg.db.parent / "aihub_pm").glob("check_*.tsv"))
    assert len(lists) == 1
    rows = lists[0].read_text("utf-8-sig").splitlines()
    assert rows[0].split("\t")[:3] == ["file", "verdict", "video"] and len(rows) == 14
    row = next(r.split("\t") for r in rows if aihub_name(77) in r)
    assert row[1:6] == ["같은 자리", "T000077", "", "", f"ah-0002:{stock.b}"]


BROKEN = b"jpeg \xff\xd8 not really"


def test_check_byte_copies_take_the_findings_and_unreadable_files_are_not_twins(cfg, store, stock, tmp_path):
    """2026-11-23 Gemini 리뷰 — (가) 함께 준 앞의 파일과 바이트가 같은 파일은 지문을 재지 않고 '준 것끼리 겹침' 으로만 나왔다:
    앞의 파일이 DB 와 '같은 자리'·'같은 영상' 이어도 그랬다(이름을 바꾼 사본이면 영상도 모른다). 앞의 것만 빼고 사본을 넣으면
    같은 자리가 들어간다. (나) 못 읽는 파일도 바이트 색인·영상 목록에 먼저 들어가, 같은 영상의 읽히는 파일이 '준 것끼리 겹침',
    바이트가 같은 다른 깨진 파일이 '못 읽음' 대신 겹침으로 나왔다."""
    d = tmp_path / "picked-twins"
    d.mkdir()
    files = {
        aihub_name(1, 20): frame_bytes(1, 3),           # ah-0001 과 같은 영상(이름)·같은 자리
        aihub_name(77): frame_bytes(2, 1, True),        # ah-0002 와 같은 자리
        aihub_name(83): BROKEN,                         # 못 읽는다
        aihub_name(83, 20): frame_bytes(24, 0),         # 바로 위와 같은 영상인데 읽힌다 — 새 것
        "zz_a.jpg": frame_bytes(2, 1, True),            # T000077 을 이름만 바꾼 사본
        "zz_b.jpg": frame_bytes(1, 3),                  # T000001_000_0020 을 이름만 바꾼 사본
        "zz_c.jpg": BROKEN,                             # 깨진 파일의 사본
    }
    for name, data in files.items():
        (d / name).write_bytes(data)
    got = by_name(ah.check_files(store, ah.gather([d]), log=lambda *_: None))
    assert {name: c.verdict for name, c in got.items()} == {
        aihub_name(1, 20): "같은 영상", aihub_name(77): "같은 자리", aihub_name(83): "못 읽음", aihub_name(83, 20): "새 것",
        "zz_a.jpg": "같은 자리", "zz_b.jpg": "같은 영상", "zz_c.jpg": "못 읽음"}

    copy_a = got["zz_a.jpg"]
    assert copy_a.twin_bytes.name == aihub_name(77) and [(k.id, k.camera) for k in copy_a.same_spot] == [(stock.b, "ah-0002")]
    copy_b = got["zz_b.jpg"]
    assert copy_b.video is None and copy_b.twin_bytes.name == aihub_name(1, 20)
    assert [k.id for k in copy_b.same_video] == [stock.a] and [k.id for k in copy_b.same_spot] == [stock.a]
    assert ah._why(copy_b).startswith(f"영상 T000001 = ah-0001 id {stock.a} 검수 대기")   # 영상 id 는 DB 쪽 이름에서
    assert got[aihub_name(83, 20)].twin_video == []                        # 못 읽는 같은 영상 파일은 세지 않는다
    assert got["zz_c.jpg"].error and got["zz_c.jpg"].twin_bytes is None

    listed = ah.check_report(list(got.values()), tmp_path / "check.tsv", log=lambda *_: None)
    row = next(r.split("\t") for r in listed.read_text("utf-8-sig").splitlines() if r.split("\t")[0].endswith("zz_a.jpg"))
    assert row[1:6] == ["같은 자리", "", "", "", f"ah-0002:{stock.b}"] and row[6] == aihub_name(77)


def test_a_missing_database_stops_instead_of_becoming_an_empty_one(cfg, tmp_path, monkeypatch):
    """2026-11-23 Gemini 리뷰 — `Store(…)` 가 없는 DB 를 새로 만들어서, `--config` 를 잘못 주면 빈 DB 와 견줘 '전부 새 것' 이라고
    했다(점검의 답이 거꾸로 '안전' 이 된다). 이제 어느 실행이든 DB 가 없으면 멈추고 아무것도 만들지 않는다."""
    import yaml

    raw = copy.deepcopy(cfg.raw)
    raw["data"]["db"] = str(tmp_path / "nowhere" / "labeling.db")
    path = tmp_path / "typo.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    picked_dir = tmp_path / "some"
    picked_dir.mkdir()
    (picked_dir / aihub_name(80)).write_bytes(frame_bytes(24, 0))
    for extra in (["--check", str(picked_dir)], ["--pin-train"], ["--pin-train", "--apply"], []):
        monkeypatch.setattr(sys, "argv", ["import_aihub_pm.py", "--config", str(path), "--root", str(tmp_path / "none"),
                                          *extra])
        with pytest.raises(SystemExit, match="DB 가 없습니다"):
            ah.main()
    assert not (tmp_path / "nowhere").exists()


def test_runs_that_only_look_leave_an_old_database_as_it_is(cfg, store, stock, picked, monkeypatch, capsys):
    """2026-11-23 Gemini 리뷰 — 보기만 하는 실행도 `Store(…)` 로 열어 스키마를 맞췄다(옛 사본 DB 에 열을 더했다). 이제 넣기
    (`--apply`) 말고는 읽기 전용으로 연다 — 다각형 열(11-19)이 생기기 전의 DB 를 주어도 그대로다."""
    import yaml

    cfg.path.write_text(yaml.safe_dump(cfg.raw, allow_unicode=True), encoding="utf-8")
    con = sqlite3.connect(cfg.db)
    con.execute("ALTER TABLE masks DROP COLUMN poly")
    con.commit()
    con.close()
    for extra in (["--check", str(picked)], ["--pin-train"], ["--pin-train", "--apply"]):
        monkeypatch.setattr(sys, "argv", ["import_aihub_pm.py", "--config", str(cfg.path), "--root", str(picked / "none"),
                                          *extra])
        assert ah.main() == 0
    monkeypatch.setattr(sys, "argv", ["import_aihub_pm.py", "--config", str(cfg.path), "--root", str(picked / "none")])
    with pytest.raises(SystemExit, match="원천데이터"):                       # 넣는 실행도 보기만일 때는 — 폴더가 없어 멈출 때까지
        ah.main()
    con = sqlite3.connect(cfg.db)
    try:
        assert "poly" not in {r[1] for r in con.execute("PRAGMA table_info(masks)")}
    finally:
        con.close()
    assert "준 그림 13장" in capsys.readouterr().out


def test_images_added_by_hand_under_an_ah_camera_count_as_already_in(cfg, store, aihub, tmp_path):
    """사람이 `ah-NNNN` 카메라로 손으로 넣은 그림(scan 이 출처를 collected 로 적는다)의 영상·자리를 도구가 다시 고르지 않는다.
    2026-11-23 까지는 출처(aihub_pm)만 봐서 몰랐다."""
    put(cfg, store, "ah-0101", "T000005_000_0010_C_D_F_1.jpg", frame_bytes(5, 3), source="collected", status="approved")
    put(cfg, store, "hw-01", "T000001_000_0010_C_D_F_1.jpg", frame_bytes(1, 2), source="collected")   # 다른 카메라는 세지 않는다
    videos, scenes = ah.existing(store)
    assert videos == {"T000005"} and len(scenes) == 1
    picks, report, _ = run(cfg, store, aihub, tmp_path)
    assert "T000005" not in {p.video for p in picks} and report["연출 킥보드"]["이미 쓴 영상"] == 1
    assert [p.camera for p in picks][0] == "ah-0102"                         # 번호도 손으로 등록한 키 다음부터


def test_next_keys_continue_after_the_largest_and_skip_used():
    assert ah.next_keys(set(), 2) == ["ah-0001", "ah-0002"]
    assert ah.next_keys({"ah-0007", "ah-0009", "hw-01", "ah-x"}, 3) == ["ah-0010", "ah-0011", "ah-0012"]
    assert ah.next_keys({"ah-9999"}, 1) == ["ah-10000"]


def test_video_id_from_names():
    assert ah.video_of("S001946_006_0364_C_D_F_0.jpg") == "S001946"
    assert ah.video_of("S001946_006_0364_C_D_F_0__7bfb8759.jpg") == "S001946"         # 정리가 붙인 꼬리표
    assert ah.video_of("hw-01_20261109_101500_000.jpg") is None
    assert ah.view_text("CCTV/야간/우천/연출") == "연출 · 우천 · 야간"
