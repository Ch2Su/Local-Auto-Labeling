"""tools/train_refiner.py·tools/eval_refiner.py — VehicleRefiner 1단계 (2026-11-28, docs/vehicle-refiner.md '4').

본다 — 학습 쪽: 목록 거르기, 주변 사각형이 크롭 안에 남는가(흔들린 박스가 밖으로 나가도 1px), '1280 시야' 배율, 표본 크기·기하,
클래스 가중 평균 1, **torchvision 가중치가 캐시에 없으면 내려받지 않고 멈춘다**, 작은 CNN 으로 CPU 에서 교차 검증을 끝까지
(test fold 는 학습·OOF 에 안 들어가고 final 만 예측), 크롭보다 넓은 --pad 는 멈춤, 있는 결과 폴더는 안 덮음.
평가 쪽: 1:1 짝짓기, 정확 McNemar 값, β 맞추기가 참값 근처, ECE, cross-fit 보정이 그 fold 의 예측을 안 쓰는가, A0 대비 통과
기준(pickup 을 고친 경우 통과·car 재현율을 깎은 경우 미달), `a0` 를 가짜 YOLO 로(크기 다른 그림 건너뛰기·차량 겹침 정리·짝짓기),
`report` 를 끝까지(jsonl 한 줄).

모델·GPU 는 쓰지 않는다(`tinycnn` 은 사전학습 없는 작은 합성곱이다). 그림은 작은 진짜 JPEG.
"""

from __future__ import annotations

import csv
import json
import math
import random
import sqlite3
import sys
import types
from pathlib import Path

import pytest
from PIL import Image

from helpers import ROOT

sys.path.insert(0, str(ROOT / "tools"))
import eval_refiner as ev  # noqa: E402
import train_refiner as tr  # noqa: E402

COLORS = {"car": (200, 40, 40), "pickup": (40, 200, 40), "truck": (40, 40, 200), "bus": (200, 200, 40),
          "responder": (200, 40, 200)}
MANI = ["crop", "image_id", "box_idx", "cls", "camera", "local_only", "group", "fold", "split_ref", "db_split", "domain",
        "captured_at", "origin", "img_w", "img_h", "x1", "y1", "x2", "y2", "long_px", "short_px", "edge",
        "crop_x1", "crop_y1", "crop_x2", "crop_y2"]


def make_crops(root: Path, n_per: int = 6, folds: int = 2, pad: float = 0.15) -> Path:
    """클래스마다 색이 다른 작은 크롭 세트(박스 60×40, 크롭은 15% 넓힘) — fold 0…folds-1 과 test 하나씩."""
    rows = []
    k = 0
    for cls, color in COLORS.items():
        for i in range(n_per + 1):
            fold = "test" if i == n_per else str(i % folds)
            k += 1
            rel = f"crops/{cls}/{k}_0.jpg"
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (78, 52), color).save(root / rel, quality=95)
            x1, y1 = 100.0 + k, 200.0
            rows.append({"crop": rel, "image_id": k, "box_idx": 0, "cls": cls, "camera": f"cam-{k}", "local_only": 0,
                         "group": k, "fold": fold, "split_ref": "new", "db_split": "", "domain": "", "captured_at": "",
                         "origin": "human", "img_w": 2560, "img_h": 1440, "x1": x1, "y1": y1, "x2": x1 + 60, "y2": y1 + 40,
                         "long_px": 60.0, "short_px": 40.0, "edge": 0, "crop_x1": int(x1 - 9), "crop_y1": 194,
                         "crop_x2": int(x1 - 9) + 78, "crop_y2": 246})
    with open(root / "manifest.csv", "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=MANI)
        wr.writeheader()
        wr.writerows(rows)
    (root / "crops_info.json").write_text(json.dumps({"tool": "build_vehicle_crops", "pad": pad, "made_at": "t",
                                                      "source_db": str(root / "db.sqlite")}), encoding="utf-8")
    return root


# ---------------------------------------------------------------- 학습 쪽

def test_manifest_filters_and_geometry(tmp_path):
    crops = make_crops(tmp_path / "c")
    rows = tr.read_manifest(crops, ("car", "pickup"), min_long=48)
    assert {r["cls"] for r in rows} == {"car", "pickup"} and all(r["y"] in (0, 1) for r in rows)
    assert tr.read_manifest(crops, min_long=61) == []
    r = rows[0]
    assert tr.box_in_crop(r) == (9.0, 6.0, 69.0, 46.0)
    g = tr.geom_features(tr.box_in_crop(r), 2560, 1440, (r["crop_x1"], r["crop_y1"]))
    assert g[0] == pytest.approx(math.log(60 / 40)) and g[1] == pytest.approx(60 / 2560) and g[3] == pytest.approx(220 / 1440)


def test_context_rect_stays_inside_and_never_empty():
    assert tr.context_rect((9, 6, 69, 46), 0.15, 78, 52) == (0, 0, 78, 52)
    assert tr.context_rect((20, 10, 40, 30), 0.0, 78, 52) == (20, 10, 40, 30)
    a, t, c, d = tr.context_rect((90, 60, 120, 80), 0.0, 78, 52)          # 흔들려 크롭 밖으로 나간 박스
    assert 0 <= a < c <= 78 and 0 <= t < d <= 52


def test_view_factor_and_sample_shape(tmp_path):
    assert tr.view_factor({"img_w": 2560, "img_h": 1440}, "1280") == 0.5
    assert tr.view_factor({"img_w": 1920, "img_h": 1080}, "1280") == pytest.approx(2 / 3)
    assert tr.view_factor({"img_w": 800, "img_h": 600}, "1280") == 1.0
    assert tr.view_factor({"img_w": 2560, "img_h": 1440}, "native") == 1.0
    crops = make_crops(tmp_path / "c")
    r = tr.read_manifest(crops)[0]
    im = tr.load_images(crops, [r])[r["crop"]]
    a, g1 = tr.make_sample(im, r, imgsz=32, pad=0.15, view="native", train=False, rng=random.Random(0))
    b, g2 = tr.make_sample(im, r, imgsz=32, pad=0.15, view="native", train=False, rng=random.Random(5))
    assert a.size == (32, 32) and a.tobytes() == b.tobytes() and g1 == g2      # 평가는 rng 와 상관없다
    c, _ = tr.make_sample(im, r, imgsz=32, pad=0.15, view="1280", train=True, rng=random.Random(1))
    assert c.size == (32, 32)


def test_class_weights_average_one():
    ys = [0] * 9 + [1] * 4 + [2]
    w = tr.class_weights(ys, 4)
    assert w[3] == 0.0 and w[0] < w[1] < w[2]
    assert sum(w[y] for y in ys) / len(ys) == pytest.approx(1.0)


def test_missing_torchvision_weights_stop_without_downloading(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(tr, "tv_cached_file", lambda name: tmp_path / "nope.pth")
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", lambda *a, **k: pytest.fail("내려받으려 했다"))
    with pytest.raises(SystemExit, match="내려받지 않는다"):
        tr.make_backbone("tv:convnext_tiny")
    with pytest.raises(SystemExit):
        tr.make_backbone("yolo:" + str(tmp_path / "없는.pt"))
    with pytest.raises(SystemExit):
        tr.make_backbone("이상한")


def test_tiny_cross_validation_end_to_end(tmp_path):
    crops = make_crops(tmp_path / "c")
    argv = ["--crops", str(crops), "--name", "t", "--backbone", "tinycnn", "--device", "cpu", "--imgsz", "32",
            "--epochs", "2", "--batch", "8", "--save-final", "--quiet"]
    out = tr.run(tr.parser().parse_args(argv))
    with open(out / "oof.csv", encoding="utf-8", newline="") as f:
        oof = list(csv.DictReader(f))
    assert len(oof) == 30 and {r["fold"] for r in oof} == {"0", "1"}           # 5 클래스 × 6, test 는 빠진다
    assert all(r["pred"] in tr.VEHICLES for r in oof) and "z_responder" in oof[0]
    with open(out / "test_preds.csv", encoding="utf-8", newline="") as f:
        assert {r["fold"] for r in csv.DictReader(f)} == {"test"}
    assert (out / "final.pt").exists()
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert cfg["n_cv"] == 30 and cfg["n_test"] == 5 and cfg["folds_run"] == [0, 1]
    with pytest.raises(SystemExit, match="이미 있습니다"):
        tr.run(tr.parser().parse_args(argv))
    wide = ["--crops", str(crops), "--name", "t2", "--backbone", "tinycnn", "--device", "cpu", "--pad", "0.4"]
    with pytest.raises(SystemExit, match="넓히기"):
        tr.run(tr.parser().parse_args(wide))


def test_geometry_branch_and_freeze_build(tmp_path):
    import torch

    m = tr.build_model("tinycnn", 5, geom=True)
    out = m(torch.zeros(2, 3, 32, 32), torch.zeros(2, 4))
    assert out.shape == (2, 5)
    m2 = tr.build_model("tinycnn", 5, geom=False)
    assert m2(torch.zeros(1, 3, 32, 32), torch.zeros(1, 4)).shape == (1, 5) and m2.geom is None


# ---------------------------------------------------------------- 평가 쪽

def test_match_is_one_to_one():
    gts = [(0, 0, 10, 10), (1, 0, 11, 10)]
    preds = [(0, 0, 10, 10)]
    assert ev.match_one_to_one(gts, preds) == {0: (0, 1.0)}
    assert ev.match_one_to_one([(0, 0, 10, 10)], [(5, 5, 15, 15)]) == {}                # IoU 0.14


def test_mcnemar_exact_values():
    a = [True] * 6 + [False] * 4
    b = [False] * 6 + [False] * 4
    assert ev.mcnemar(a, b) == {"a_only": 6, "b_only": 0, "p": 0.03125}
    assert ev.mcnemar([True, False], [False, True])["p"] == 1.0
    assert ev.mcnemar([True], [True])["p"] == 1.0


def test_fit_beta_recovers_the_true_temperature():
    rng = random.Random(7)
    zs, ys = [], []
    for _ in range(4000):
        z = [rng.gauss(0, 3) for _ in range(5)]
        p = ev.softmax(z, 0.5)
        ys.append(rng.choices(range(5), weights=p)[0])
        zs.append(z)
    assert ev.fit_beta(zs, ys) == pytest.approx(0.5, abs=0.08)


def test_ece_zero_when_calibrated_and_positive_when_overconfident():
    probs = [[0.75, 0.25]] * 4
    assert ev.ece(probs, [0, 0, 0, 1]) == pytest.approx(0.0)
    assert ev.ece([[0.99, 0.01]] * 4, [0, 1, 1, 1]) > 0.7


def oof_rows(spec, folds=("0", "1")):
    """spec: [(정답, 예측, 신뢰 logit)] → OOF 행. 다섯 클래스, fold 는 번갈아."""
    rows = []
    for i, (g, p, s) in enumerate(spec):
        z = [0.0] * 5
        z[tr.VEHICLES.index(p)] = s
        rows.append({"crop": f"c{i}", "cls": g, "fold": folds[i % len(folds)], "group": str(i), "long_px": "100",
                     "z": z, "classes": list(tr.VEHICLES)})
    return rows


def test_cross_fit_does_not_use_the_folds_own_predictions():
    base = oof_rows([("car", "car", 3.0)] * 10 + [("pickup", "car", 3.0)] * 4 + [("pickup", "pickup", 3.0)] * 6)
    m1 = ev.evaluate(base)
    changed = [dict(r, z=[v * 5 for v in r["z"]]) if r["fold"] == "0" else r for r in base]
    m2 = ev.evaluate(changed)
    assert m1["beta_by_fold"]["0"] == m2["beta_by_fold"]["0"]                   # fold 0 의 β 는 fold 1 로만 맞춘다
    assert m1["beta_by_fold"]["1"] != m2["beta_by_fold"]["1"]
    assert m1["accuracy"] == 0.8 and m1["pickup_errors"] == 4 and m1["per_class"]["pickup"]["recall"] == 0.6


def test_versus_a0_criteria():
    spec = [("pickup", "pickup", 4.0)] * 10 + [("car", "car", 4.0)] * 20 + [("truck", "truck", 4.0)] * 5
    rows = oof_rows(spec)
    a0 = {r["crop"]: {"det_cls": "car" if r["cls"] == "pickup" else r["cls"]} for r in rows}
    v = ev.versus_a0(rows, a0, ["car", "truck", "bus", "pickup"])
    assert v["pickup_errors_a0"] == 10 and v["pickup_errors_refiner"] == 0 and v["S_mcnemar"]["p"] < 0.01
    assert v["criteria"]["1_pickup"] and v["criteria"]["2_car_recall"]
    hurt = oof_rows([("pickup", "pickup", 4.0)] * 10 + [("car", "pickup", 4.0)] * 2 + [("car", "car", 4.0)] * 18)
    a0h = {r["crop"]: {"det_cls": "car"} for r in hurt}
    vh = ev.versus_a0(hurt, a0h, ["car", "truck", "bus", "pickup"])
    assert not vh["criteria"]["2_car_recall"] and vh["car_recall_refiner"] == 0.9
    missed = dict(a0h)
    missed["c0"] = {"det_cls": ""}                                              # A0 가 놓친 크롭은 견주지 않는다
    assert ev.versus_a0(hurt, missed, ["car", "pickup"])["n_compared"] == 29


class _T(list):
    def tolist(self):
        return list(self)


class _FakeYOLO:
    """이미지 id 마다 정해 둔 검출을 돌려주는 대역 — `predict(source=경로)` 의 경로 끝 번호로 고른다."""
    names = {0: "person", 1: "car", 2: "truck", 3: "pickup"}
    plan: dict = {}

    def __init__(self, path):
        pass

    def predict(self, source, **kw):
        iid = int(Path(source).stem)
        shape, dets = self.plan[iid]
        boxes = types.SimpleNamespace(xyxy=_T(d[0] for d in dets), conf=_T(d[1] for d in dets),
                                      cls=_T(float(d[2]) for d in dets))
        return [types.SimpleNamespace(orig_shape=shape, boxes=boxes)]


def test_a0_with_a_fake_detector(tmp_path, monkeypatch, cfg):
    """그림 1: GT pickup 자리에 car(0.9)·pickup(0.6)이 겹쳐 나온다 → 차량 겹침 정리로 car 만 남아 det=car.
    그림 2: 검출기가 본 크기가 목록과 달라 건너뛴다. 그림 3: 아무것도 못 찾아 놓침."""
    crops = make_crops(tmp_path / "c", n_per=1, folds=1)
    rows = list(csv.DictReader(open(crops / "manifest.csv", encoding="utf-8", newline="")))
    db = crops / "db.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE images (id INTEGER PRIMARY KEY, path TEXT)")
    for r in rows:
        p = tmp_path / "img" / f"{r['image_id']}.jpg"
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(b"")
        con.execute("INSERT INTO images VALUES (?, ?)", (int(r["image_id"]), str(p)))
    con.commit()
    con.close()
    first = {r["cls"]: r for r in rows if r["fold"] == "0"}
    pk, car, trk = first["pickup"], first["car"], first["truck"]
    box = lambda r: [float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])]  # noqa: E731
    plan = {int(r["image_id"]): ((1440, 2560), []) for r in rows}
    plan[int(pk["image_id"])] = ((1440, 2560), [(box(pk), 0.6, 3), (box(pk), 0.9, 1)])     # 정리를 안 하면 앞의 pickup 과 짝
    plan[int(car["image_id"])] = ((1080, 1920), [(box(car), 0.9, 1)])
    plan[int(trk["image_id"])] = ((1440, 2560), [(box(trk), 0.8, 2)])
    _FakeYOLO.plan = plan
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=_FakeYOLO))
    w = tmp_path / "w.pt"
    w.write_bytes(b"weights")
    cfg.path.write_text(json.dumps(cfg.raw), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["eval_refiner.py", "a0", "--crops", str(crops), "--weights", str(w),
                                      "--config", str(cfg.path), "--device", "cpu"])
    assert ev.main() == 0
    out = crops / "runs" / "a0_w_1280"
    got = {r["crop"]: r for r in csv.DictReader(open(out / "a0.csv", encoding="utf-8", newline=""))}
    assert got[pk["crop"]]["det_cls"] == "car" and float(got[pk["crop"]]["iou"]) == 1.0
    assert got[trk["crop"]]["det_cls"] == "truck"
    assert car["crop"] not in got                                               # 크기가 달라 건너뜀
    assert got[first["bus"]["crop"]]["det_cls"] == ""                            # 놓침
    info = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert info["compared_classes"] == ["car", "pickup", "truck"] and len(info["skipped"]) == 1


def test_a0_only_fold_and_merge(tmp_path, monkeypatch, cfg):
    """A0b — fold 마다 다른 가중치로 그 fold 의 크롭만 짝짓고(`--only-fold`·`--tag`), 합칠 때 같은 크롭이 두 번이면 멈춘다.
    견줄 클래스는 모든 조각이 아는 것(한 조각이 pickup 을 모르면 빠진다)."""
    crops = make_crops(tmp_path / "c", n_per=1, folds=1)
    rows = list(csv.DictReader(open(crops / "manifest.csv", encoding="utf-8", newline="")))
    con = sqlite3.connect(crops / "db.sqlite")
    con.execute("CREATE TABLE images (id INTEGER PRIMARY KEY, path TEXT)")
    for r in rows:
        p = tmp_path / "img" / f"{r['image_id']}.jpg"
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(b"")
        con.execute("INSERT INTO images VALUES (?, ?)", (int(r["image_id"]), str(p)))
    con.commit()
    con.close()
    _FakeYOLO.plan = {int(r["image_id"]): ((1440, 2560), []) for r in rows}
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=_FakeYOLO))
    w = tmp_path / "w.pt"
    w.write_bytes(b"weights")
    cfg.path.write_text(json.dumps(cfg.raw), encoding="utf-8")

    def a0(*extra):
        monkeypatch.setattr(sys, "argv", ["eval_refiner.py", "a0", "--crops", str(crops), "--weights", str(w),
                                          "--config", str(cfg.path), "--device", "cpu", *extra])
        return ev.main()

    assert a0("--only-fold", "0", "--tag", "p0") == 0
    assert a0("--only-fold", "test") == 0
    with pytest.raises(SystemExit, match="크롭이 없습니다"):
        a0("--only-fold", "7")
    p0 = ev.read_a0(crops / "runs" / "p0" / "a0.csv")
    assert {r["crop"] for r in p0.values()} == {r["crop"] for r in rows if r["fold"] == "0"}
    part2 = crops / "runs" / "a0_w_1280_ftest"
    cfg2 = json.loads((part2 / "config.json").read_text(encoding="utf-8"))
    assert cfg2["only_fold"] == "test"
    cfg2["compared_classes"] = ["car", "truck"]
    (part2 / "config.json").write_text(json.dumps(cfg2), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["eval_refiner.py", "merge", "--crops", str(crops), "--parts", "p0", part2.name,
                                      "--tag", "a0b"])
    assert ev.main() == 0
    merged = ev.read_a0(crops / "runs" / "a0b" / "a0.csv")
    assert len(merged) == len(rows)
    assert json.loads((crops / "runs" / "a0b" / "config.json").read_text(encoding="utf-8"))["compared_classes"] == ["car", "truck"]
    monkeypatch.setattr(sys, "argv", ["eval_refiner.py", "merge", "--crops", str(crops), "--parts", "p0", "p0", "--tag", "x"])
    with pytest.raises(SystemExit, match="두 조각"):
        ev.main()


def test_report_end_to_end(tmp_path, monkeypatch, capsys):
    crops = make_crops(tmp_path / "c")
    run_dir = crops / "runs" / "r1"
    run_dir.mkdir(parents=True)
    rows = tr.read_manifest(crops)
    cv = [r for r in rows if r["fold"] != "test"]
    logits = [[4.0 if i == r["y"] else 0.0 for i in range(5)] for r in cv]
    tr.write_preds(run_dir / "oof.csv", cv, logits, tr.VEHICLES)
    (run_dir / "config.json").write_text(json.dumps({"backbone": "tinycnn", "crops_made_at": "t"}), encoding="utf-8")
    a0_dir = crops / "runs" / "a0_x"
    a0_dir.mkdir()
    with open(a0_dir / "a0.csv", "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["crop", "cls", "det_cls", "det_conf", "iou"])
        wr.writeheader()
        for r in cv:
            wr.writerow({"crop": r["crop"], "cls": r["cls"], "det_cls": "car" if r["cls"] == "pickup" else r["cls"],
                         "det_conf": 0.9, "iou": 0.9})
    (a0_dir / "config.json").write_text(json.dumps({"compared_classes": ["car", "pickup", "truck", "bus"]}), encoding="utf-8")
    jsonl = tmp_path / "res.jsonl"
    monkeypatch.setattr(sys, "argv", ["eval_refiner.py", "report", "--crops", str(crops), "--runs", "r1",
                                      "--baseline", "a0_x", "--jsonl", str(jsonl), "--boot", "50"])
    assert ev.main() == 0
    out = capsys.readouterr().out
    assert "정확도 1.0" in out and "기준 2(car 재현율) 통과" in out
    (rec,) = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
    assert rec["run"] == "r1" and rec["n"] == 30 and rec["vs_a0"]["pickup_errors_a0"] == 6
    assert rec["vs_a0"]["pickup_errors_refiner"] == 0 and rec["vs_a0"]["S_mcnemar"]["p"] == 0.03125
