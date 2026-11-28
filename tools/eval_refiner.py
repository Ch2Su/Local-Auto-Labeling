"""VehicleRefiner 1단계 평가 — 기준선(A0)과 지표 (2026-11-28, 사용자 '학습 도구 만들기'. 설계는 docs/vehicle-refiner.md '4.2').

두 하위 명령:

- **`a0`** — 기준선 A0. 검출기(기본 config 의 운영 가중치)를 크롭 세트의 그림마다 통째로 돌려(기본 1280 · conf 0.25 · NMS IoU 0.6 —
  자동 라벨링과 같고, 차량 클래스끼리 겹친 박스도 `tiling.cross_class_keep`(0.6)으로 같은 식으로 정리한다. 헤드는 Ultralytics
  기본 one2many+NMS) 크롭의 GT 박스와 **IoU 0.45 이상, 1:1**(IoU 큰 짝부터)로 짝짓고, 짝 예측의 클래스를 '검출기의 답' 으로
  `<crops>/runs/a0_<가중치>_<입력>/a0.csv` 에 쓴다. 짝이 없으면 놓침(빈칸). 그림 크기가 목록과 다르면 그 그림을 건너뛴다.
  **운영 가중치는 옛 데이터로만 배웠다** — 이 크롭들의 그림을 본 적이 없어 모든 fold 에 공정하다(대신 데이터도 달라 구조의 몫만은
  아니다 — 설계 4.2 의 A0b).
- **`report`** — `train_refiner.py` 의 OOF(`<crops>/runs/<이름>/oof.csv`, `fold=test` 는 쓰지 않는다)로: 정확도·macro-F1·클래스별
  재현율/정밀도·5×5 혼동표·pickup 이 낀 틀림·크기 구간별 정확도·ECE(15칸)·NLL. 확률 보정은 temperature 하나를 **다른 fold 의
  OOF 로 맞춰** 그 fold 에 쓴다(cross-fit — 같은 예측으로 맞추고 재면 낙관적이다). `--baseline a0_…` 이면 A0 가 짝지은 크롭
  (검출기가 찾은 박스 — 분류기가 실제로 손댈 곳) 가운데 A0 모델이 아는 클래스만으로 견주고 설계의 **미리 정한 통과 기준** 셋을
  판정한다. `--compare a:b` 는 두 실행의 같은 크롭에 McNemar 정확 검정과 카메라 그룹째 부트스트랩 신뢰구간. 한 실행마다 지표를
  `--jsonl`(기본 `runs/refiner/results.jsonl` — 수치만, 그림·카메라 키 없음)에 한 줄씩 덧붙인다.

미리 정한 통과 기준(설계 4.2 — 결과를 보고 바꾸지 않는다). S = A0 가 짝지은 크롭 가운데 정답이 car·pickup·truck 인 것:
  1. S 에서 pickup 이 낀 틀림(정답 pickup → 다른 것, 다른 것 → pickup)이 A0 보다 30% 이상 줄고, S 의 맞음/틀림 McNemar p < 0.05
  2. A0 가 짝지은 car 의 재현율이 A0 보다 1%p 넘게 내려가지 않음
  3. 보정한 뒤 ECE ≤ 0.05 (OOF 전체)

  .venv\\Scripts\\python.exe tools\\eval_refiner.py a0 --crops C:\\datasets\\vehicle_crops_20261128 --device 1
  .venv\\Scripts\\python.exe tools\\eval_refiner.py report --crops C:\\datasets\\vehicle_crops_20261128 --runs b1 b2 \\
      --baseline a0_best_y26m_8cls_1280 --compare b1:b2
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import random
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))   # tools/ -> 저장소 루트
sys.path.insert(0, str(ROOT / "tools"))

from autolabel.config import Config, resolve  # noqa: E402  (autolabel 을 먼저 — YOLO_AUTOINSTALL=false 기본값)
from build_vehicle_crops import MARKER, VEHICLES  # noqa: E402

MATCH_IOU = 0.45        # wsl/eval_test.py 의 혼동 행렬 짝짓기와 같다
SIZE_BINS = ((48, 96), (96, 192), (192, float("inf")))
ECE_BINS = 15
CRITERIA = {"pickup_cut": 0.30, "mcnemar_p": 0.05, "car_recall_drop": 0.01, "ece": 0.05}


# ---------------------------------------------------------------- 셈 (모델 없이)

def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_one_to_one(gts: list, preds: list, thr: float = MATCH_IOU) -> dict[int, tuple[int, float]]:
    """{gt 순번: (pred 순번, IoU)} — IoU 가 큰 짝부터, 서로 비어 있을 때만. 같으면 앞 순번 먼저."""
    pairs = sorted(((iou(g, p), gi, pi) for gi, g in enumerate(gts) for pi, p in enumerate(preds)),
                   key=lambda t: (-t[0], t[1], t[2]))
    used_g, used_p, out = set(), set(), {}
    for v, gi, pi in pairs:
        if v < thr:
            break
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        out[gi] = (pi, v)
    return out


def softmax(z: list[float], beta: float = 1.0) -> list[float]:
    m = max(v * beta for v in z)
    e = [math.exp(v * beta - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


def nll(zs: list[list[float]], ys: list[int], beta: float = 1.0) -> float:
    return -sum(math.log(max(softmax(z, beta)[y], 1e-12)) for z, y in zip(zs, ys)) / max(1, len(ys))


def fit_beta(zs: list[list[float]], ys: list[int]) -> float:
    """NLL 을 가장 낮추는 β(= 1/T). softmax(β·z) 의 NLL 은 β 에 대해 볼록하다 — log β 위에서 삼분 탐색."""
    lo, hi = math.log(0.01), math.log(20.0)
    for _ in range(80):
        a, b = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if nll(zs, ys, math.exp(a)) <= nll(zs, ys, math.exp(b)):
            hi = b
        else:
            lo = a
    return math.exp((lo + hi) / 2)


def ece(probs: list[list[float]], ys: list[int], bins: int = ECE_BINS) -> float:
    """기대 보정 오차 — 최댓값 확률을 같은 폭 bins 칸으로 나눠 Σ (칸 비중) × |정확도 − 평균 확률|."""
    tot = [0, ] * bins
    acc = [0.0] * bins
    conf = [0.0] * bins
    for p, y in zip(probs, ys):
        c = max(p)
        k = min(bins - 1, int(c * bins))
        tot[k] += 1
        acc[k] += float(p.index(c) == y)
        conf[k] += c
    n = max(1, len(ys))
    return sum(abs(acc[k] - conf[k]) / n for k in range(bins) if tot[k])


def mcnemar(a_right: list[bool], b_right: list[bool]) -> dict:
    """정확 McNemar — b = A 만 맞힘, c = B 만 맞힘. p = 2 × P(X ≤ min(b, c)), X ~ Bin(b + c, 1/2), 1 로 자름."""
    b = sum(1 for x, y in zip(a_right, b_right) if x and not y)
    c = sum(1 for x, y in zip(a_right, b_right) if y and not x)
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)
    return {"a_only": b, "b_only": c, "p": round(p, 6)}


def confusion(ys: list[int], ps: list[int], n: int) -> list[list[int]]:
    m = [[0] * n for _ in range(n)]
    for y, p in zip(ys, ps):
        m[y][p] += 1
    return m


def class_stats(ys: list[int], ps: list[int], classes) -> dict:
    out = {}
    f1s = []
    for c, name in enumerate(classes):
        tp = sum(1 for y, p in zip(ys, ps) if y == c and p == c)
        n_true = sum(1 for y in ys if y == c)
        n_pred = sum(1 for p in ps if p == c)
        rec = tp / n_true if n_true else None
        prec = tp / n_pred if n_pred else None
        if n_true:
            f1s.append(0.0 if not tp else 2 * prec * rec / (prec + rec))
        out[name] = {"n": n_true, "recall": None if rec is None else round(rec, 4),
                     "precision": None if prec is None else round(prec, 4)}
    return {"per_class": out, "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else None}


def pickup_errors(gt: list[str], pred: list[str]) -> int:
    """정답이 pickup 인데 다른 답, 또는 pickup 이 아닌데 pickup 이라고 한 수."""
    return sum(1 for g, p in zip(gt, pred) if g != p and "pickup" in (g, p))


def group_bootstrap(groups: list, a_right: list[bool], b_right: list[bool], n: int = 2000, seed: int = 0) -> dict:
    """카메라 그룹을 뽑아 다시 세는 부트스트랩 — 정확도 차(A − B)의 2.5·97.5% 점."""
    by_g = collections.defaultdict(list)
    for g, x, y in zip(groups, a_right, b_right):
        by_g[g].append((x, y))
    keys = sorted(by_g)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n):
        sa = sb = tot = 0
        for g in rng.choices(keys, k=len(keys)):
            for x, y in by_g[g]:
                sa += x
                sb += y
                tot += 1
        diffs.append((sa - sb) / tot if tot else 0.0)
    diffs.sort()
    return {"diff": round((sum(a_right) - sum(b_right)) / max(1, len(a_right)), 4),
            "ci95": [round(diffs[int(0.025 * n)], 4), round(diffs[min(n - 1, int(0.975 * n))], 4)], "boot": n}


# ---------------------------------------------------------------- 읽기

def read_oof(path: Path, min_long: float) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["fold"] == "test" or float(r["long_px"]) < min_long:
                continue
            r["z"] = [float(r[k]) for k in r if k.startswith("z_")]
            r["classes"] = [k[2:] for k in r if k.startswith("z_")]
            rows.append(r)
    return rows


def read_a0(path: Path) -> dict[str, dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return {r["crop"]: r for r in csv.DictReader(f)}


# ---------------------------------------------------------------- report

def evaluate(rows: list[dict]) -> dict:
    """OOF 한 실행의 지표. 보정은 fold 마다 다른 fold 로 맞춘 β."""
    classes = rows[0]["classes"]
    ys = [classes.index(r["cls"]) for r in rows]
    zs = [r["z"] for r in rows]
    ps = [z.index(max(z)) for z in zs]
    folds = sorted({r["fold"] for r in rows})
    cal = [None] * len(rows)
    betas = {}
    for f in folds:
        idx_in = [i for i, r in enumerate(rows) if r["fold"] == f]
        idx_out = [i for i, r in enumerate(rows) if r["fold"] != f] or idx_in       # fold 하나뿐이면 제자리(낙관적)
        beta = fit_beta([zs[i] for i in idx_out], [ys[i] for i in idx_out])
        betas[f] = round(beta, 4)
        for i in idx_in:
            cal[i] = softmax(zs[i], beta)
    raw = [softmax(z) for z in zs]
    sizes = {}
    for lo, hi in SIZE_BINS:
        idx = [i for i, r in enumerate(rows) if lo <= float(r["long_px"]) < hi]
        sizes[f"{lo}-{'' if hi == float('inf') else int(hi)}"] = {
            "n": len(idx), "acc": round(sum(ys[i] == ps[i] for i in idx) / len(idx), 4) if idx else None}
    gt = [r["cls"] for r in rows]
    pred = [classes[p] for p in ps]
    return {"n": len(rows), "folds": folds, "classes": classes,
            "accuracy": round(sum(y == p for y, p in zip(ys, ps)) / len(ys), 4), **class_stats(ys, ps, classes),
            "confusion": confusion(ys, ps, len(classes)), "pickup_errors": pickup_errors(gt, pred), "size_bins": sizes,
            "ece_raw": round(ece(raw, ys), 4), "ece_cal": round(ece(cal, ys), 4),
            "nll_raw": round(nll(zs, ys), 4), "nll_cal": round(-sum(math.log(max(c[y], 1e-12)) for c, y in zip(cal, ys)) / len(ys), 4),
            "beta_by_fold": betas}


def versus_a0(rows: list[dict], a0: dict[str, dict], a0_classes: list[str]) -> dict:
    """A0 가 짝지은 크롭 가운데 A0 모델이 아는 클래스만으로 견주고 통과 기준을 판정한다."""
    classes = rows[0]["classes"]
    matched = [r for r in rows if a0.get(r["crop"], {}).get("det_cls") and r["cls"] in a0_classes]
    missing = sum(1 for r in rows if r["crop"] not in a0)
    found = sum(1 for r in rows if a0.get(r["crop"], {}).get("det_cls"))
    ref_pred = [classes[r["z"].index(max(r["z"]))] for r in matched]
    det_pred = [a0[r["crop"]]["det_cls"] for r in matched]
    gt = [r["cls"] for r in matched]
    s_idx = [i for i, g in enumerate(gt) if g in ("car", "pickup", "truck")]
    s_ref = [ref_pred[i] == gt[i] for i in s_idx]
    s_det = [det_pred[i] == gt[i] for i in s_idx]
    pe_ref = pickup_errors([gt[i] for i in s_idx], [ref_pred[i] for i in s_idx])
    pe_det = pickup_errors([gt[i] for i in s_idx], [det_pred[i] for i in s_idx])
    cars = [i for i, g in enumerate(gt) if g == "car"]
    car_ref = sum(ref_pred[i] == "car" for i in cars) / len(cars) if cars else None
    car_det = sum(det_pred[i] == "car" for i in cars) / len(cars) if cars else None
    mc = mcnemar(s_ref, s_det)
    out = {"n_compared": len(matched), "a0_found": found, "a0_missed": len(rows) - found - missing,
           "a0_absent": missing, "a0_classes": a0_classes,
           "acc_refiner": round(sum(a == b for a, b in zip(ref_pred, gt)) / len(gt), 4) if gt else None,
           "acc_a0": round(sum(a == b for a, b in zip(det_pred, gt)) / len(gt), 4) if gt else None,
           "S_n": len(s_idx), "S_acc_refiner": round(sum(s_ref) / len(s_ref), 4) if s_ref else None,
           "S_acc_a0": round(sum(s_det) / len(s_det), 4) if s_det else None, "S_mcnemar": mc,
           "pickup_errors_refiner": pe_ref, "pickup_errors_a0": pe_det,
           "car_recall_refiner": None if car_ref is None else round(car_ref, 4),
           "car_recall_a0": None if car_det is None else round(car_det, 4),
           "a0_confusion": confusion([classes.index(g) for g in gt],
                                     [classes.index(p) if p in classes else len(classes) for p in det_pred],
                                     len(classes) + 1)}
    cut = (pe_det - pe_ref) / pe_det if pe_det else None
    out["criteria"] = {
        "1_pickup": bool(cut is not None and cut >= CRITERIA["pickup_cut"] and mc["p"] < CRITERIA["mcnemar_p"]
                         and mc["b_only"] < mc["a_only"]),
        "2_car_recall": bool(car_ref is not None and car_det is not None and car_det - car_ref <= CRITERIA["car_recall_drop"]),
        "pickup_cut": None if cut is None else round(cut, 4)}
    return out


def compare_runs(a: list[dict], b: list[dict], boot: int) -> dict:
    """같은 크롭에서 두 실행 — McNemar(맞음/틀림)와 그룹 부트스트랩 정확도 차."""
    bmap = {r["crop"]: r for r in b}
    common = [r for r in a if r["crop"] in bmap]
    ar = [r["z"].index(max(r["z"])) == r["classes"].index(r["cls"]) for r in common]
    br = [bmap[r["crop"]]["z"].index(max(bmap[r["crop"]]["z"])) == r["classes"].index(r["cls"]) for r in common]
    return {"n": len(common), "acc_a": round(sum(ar) / len(ar), 4) if ar else None,
            "acc_b": round(sum(br) / len(br), 4) if br else None, "mcnemar": mcnemar(ar, br),
            "bootstrap": group_bootstrap([r["group"] for r in common], ar, br, n=boot)}


def show(name: str, m: dict) -> str:
    cls = m["classes"]
    lines = [f"== {name}: OOF {m['n']}개 · 정확도 {m['accuracy']} · macro-F1 {m['macro_f1']} · pickup 낀 틀림 {m['pickup_errors']} · "
             f"ECE {m['ece_raw']} → {m['ece_cal']}(보정) · NLL {m['nll_raw']} → {m['nll_cal']}"]
    lines.append("   " + " ".join(f"{c:>10s}" for c in cls) + "   ← 예측 / 정답 ↓   재현율  정밀도")
    for i, c in enumerate(cls):
        pc = m["per_class"][c]
        lines.append(f"{c:>10s} " + " ".join(f"{v:10d}" for v in m["confusion"][i]) +
                     f"   {pc['recall'] if pc['recall'] is not None else '-':>6}  {pc['precision'] if pc['precision'] is not None else '-':>6}")
    lines.append("   크기별 정확도: " + ", ".join(f"{k}px {v['acc']}({v['n']})" for k, v in m["size_bins"].items()))
    return "\n".join(lines)


def show_a0(v: dict) -> str:
    c = v["criteria"]
    mc = v["S_mcnemar"]
    return "\n".join([
        f"   A0 대비 (A0 가 찾은 {v['n_compared']}개, A0 놓침 {v['a0_missed']}, A0 클래스 {','.join(v['a0_classes'])}): "
        f"정확도 {v['acc_refiner']} 대 A0 {v['acc_a0']}",
        f"   S(car·pickup·truck {v['S_n']}개): {v['S_acc_refiner']} 대 {v['S_acc_a0']} · pickup 낀 틀림 {v['pickup_errors_refiner']} 대 "
        f"{v['pickup_errors_a0']} (줄어든 몫 {c['pickup_cut']}) · McNemar 분류기만 맞힘 {mc['a_only']} / A0 만 맞힘 {mc['b_only']}, p={mc['p']}",
        f"   car 재현율 {v['car_recall_refiner']} 대 A0 {v['car_recall_a0']}",
        f"   기준 1(pickup) {'통과' if c['1_pickup'] else '미달'} · 기준 2(car 재현율) {'통과' if c['2_car_recall'] else '미달'}"])


def cmd_report(args) -> int:
    crops = Path(args.crops).resolve()
    a0 = a0_classes = None
    if args.baseline:
        bdir = crops / "runs" / args.baseline
        a0 = read_a0(bdir / "a0.csv")
        a0_classes = json.loads((bdir / "config.json").read_text(encoding="utf-8"))["compared_classes"]
    loaded = {}
    jsonl = resolve(args.jsonl)
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    for name in args.runs:
        rdir = crops / "runs" / name
        rows = read_oof(rdir / "oof.csv", args.min_long)
        if not rows:
            print(f"[건너뜀] {name}: 견줄 OOF 가 없습니다")
            continue
        loaded[name] = rows
        m = evaluate(rows)
        cfg = json.loads((rdir / "config.json").read_text(encoding="utf-8"))
        rec = {"at": datetime.now().isoformat(timespec="seconds"), "run": name, "crops": crops.name,
               "crops_made_at": cfg.get("crops_made_at"), "min_long": args.min_long,
               "train": {k: cfg.get(k) for k in ("backbone", "imgsz", "pad", "view", "geom", "freeze", "balance", "epochs",
                                                  "batch", "lr", "head_lr", "wd", "smooth", "seed", "minutes")}, **m}
        print(show(name, m))
        if a0 is not None:
            v = versus_a0(rows, a0, a0_classes)
            v["criteria"]["3_ece"] = m["ece_cal"] <= CRITERIA["ece"]
            rec["vs_a0"] = {"baseline": args.baseline, **v}
            print(show_a0(v) + f" · 기준 3(ECE ≤ {CRITERIA['ece']}) {'통과' if v['criteria']['3_ece'] else '미달'}")
        with open(jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    for pair in args.compare or []:
        a, _, b = pair.partition(":")
        if a not in loaded or b not in loaded:
            print(f"[건너뜀] --compare {pair}: 두 실행을 --runs 에 함께 줄 것")
            continue
        c = compare_runs(loaded[a], loaded[b], args.boot)
        mc, bs = c["mcnemar"], c["bootstrap"]
        print(f"== {a} 대 {b}: 같은 크롭 {c['n']}개 · 정확도 {c['acc_a']} 대 {c['acc_b']} · {a} 만 맞힘 {mc['a_only']} / {b} 만 맞힘 "
              f"{mc['b_only']}, McNemar p={mc['p']} · 차 {bs['diff']} (카메라 그룹 부트스트랩 95% {bs['ci95']})")
        with open(jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps({"at": datetime.now().isoformat(timespec="seconds"), "compare": [a, b], "crops": crops.name,
                                "min_long": args.min_long, **c}, ensure_ascii=False) + "\n")
    print(f"기록: {jsonl}")
    return 0


# ---------------------------------------------------------------- a0

def cmd_a0(args) -> int:
    from ultralytics import YOLO

    from autolabel import tiling

    crops = Path(args.crops).resolve()
    info = json.loads((crops / MARKER).read_text(encoding="utf-8"))
    cfg = Config.load(args.config)
    weights = resolve(args.weights) if args.weights else cfg.weights
    if not weights.exists():
        raise SystemExit(f"[중단] 가중치가 없습니다: {weights}")
    with open(crops / "manifest.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if args.only_fold is not None:
        rows = [r for r in rows if r["fold"] == args.only_fold]
        if not rows:
            raise SystemExit(f"[중단] fold {args.only_fold} 의 크롭이 없습니다")
    by_img = collections.defaultdict(list)
    for r in rows:
        by_img[int(r["image_id"])].append(r)
    db = Path(info["source_db"])
    con = sqlite3.connect("file:" + quote(str(db).replace("\\", "/")) + "?mode=ro", uri=True)
    path_of = dict(con.execute("SELECT id, path FROM images"))
    con.close()
    model = YOLO(str(weights))
    names = model.names
    group = frozenset(tiling.vehicle_ids(names)) if args.class_iou > 0 else frozenset()
    out_rows, skipped = [], []
    for k, (iid, rs) in enumerate(sorted(by_img.items()), 1):
        path = path_of.get(iid)
        if not path or not Path(path).exists():
            skipped.append(f"{iid}: 파일 없음")
            continue
        res = model.predict(source=path, imgsz=args.imgsz, conf=args.conf, iou=args.iou, max_det=300,
                            device=args.device, verbose=False)[0]
        h, w = res.orig_shape
        if (w, h) != (int(rs[0]["img_w"]), int(rs[0]["img_h"])):
            skipped.append(f"{iid}: 검출기가 본 크기 {w}×{h} 가 목록 {rs[0]['img_w']}×{rs[0]['img_h']} 와 다름")
            continue
        xyxy, conf, cls = res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist()
        keep = tiling.cross_class_keep(xyxy, conf, cls, group, args.class_iou)
        preds = [(b, cf, names[int(c)]) for b, cf, c, kp in zip(xyxy, conf, cls, keep) if kp]
        m = match_one_to_one([(float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])) for r in rs],
                             [p[0] for p in preds])
        for gi, r in enumerate(rs):
            pi, v = m.get(gi, (None, 0.0))
            out_rows.append({"crop": r["crop"], "cls": r["cls"], "det_cls": preds[pi][2] if pi is not None else "",
                             "det_conf": round(preds[pi][1], 4) if pi is not None else "", "iou": round(v, 4)})
        if k % 50 == 0:
            print(f"  {k}/{len(by_img)} 장", flush=True)
    tag = args.tag or f"a0_{weights.stem}_{args.imgsz}" + (f"_f{args.only_fold}" if args.only_fold is not None else "")
    out = crops / "runs" / tag
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "a0.csv", "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["crop", "cls", "det_cls", "det_conf", "iou"])
        wr.writeheader()
        wr.writerows(out_rows)
    sha = hashlib.sha256(weights.read_bytes()).hexdigest()[:12]
    model_classes = [names[i] for i in sorted(names)]
    compared = [c for c in VEHICLES if c in model_classes]
    (out / "config.json").write_text(json.dumps({
        "weights": str(weights), "sha256": sha, "only_fold": args.only_fold, "imgsz": args.imgsz, "conf": args.conf,
        "iou": args.iou,
        "class_iou": args.class_iou, "match_iou": MATCH_IOU, "model_classes": model_classes, "compared_classes": compared,
        "crops_made_at": info.get("made_at"), "skipped": skipped, "made_at": datetime.now().isoformat(timespec="seconds")},
        ensure_ascii=False, indent=1), encoding="utf-8")
    found = sum(1 for r in out_rows if r["det_cls"])
    right = sum(1 for r in out_rows if r["det_cls"] == r["cls"])
    print(f"A0 {tag}: 크롭 {len(out_rows)}개 중 짝 {found} · 그중 클래스 맞음 {right} · 놓침 {len(out_rows) - found} "
          f"(가중치 sha256 {sha}…, 견줄 클래스 {','.join(compared)})")
    for s in skipped:
        print(f"[건너뜀] {s}")
    print(f"썼습니다: {out / 'a0.csv'}")
    return 0


def cmd_merge(args) -> int:
    """fold 마다 다른 검출기로 짝지은 a0 결과(A0b)를 하나로 — 같은 크롭이 두 번 나오면 멈춘다. 견줄 클래스는 모든 조각이 아는 것."""
    crops = Path(args.crops).resolve()
    rows, configs, seen = [], [], set()
    for part in args.parts:
        pdir = crops / "runs" / part
        configs.append(json.loads((pdir / "config.json").read_text(encoding="utf-8")))
        for r in read_a0(pdir / "a0.csv").values():
            if r["crop"] in seen:
                raise SystemExit(f"[중단] {r['crop']} 이 두 조각에 있습니다 — fold 가 겹치지 않게 줄 것")
            seen.add(r["crop"])
            rows.append(r)
    compared = [c for c in VEHICLES if all(c in cf["compared_classes"] for cf in configs)]
    out = crops / "runs" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "a0.csv", "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["crop", "cls", "det_cls", "det_conf", "iou"])
        wr.writeheader()
        wr.writerows(rows)
    (out / "config.json").write_text(json.dumps({
        "merged_from": args.parts, "parts": [{k: cf.get(k) for k in ("weights", "sha256", "only_fold", "imgsz")} for cf in configs],
        "compared_classes": compared, "skipped": [s for cf in configs for s in cf.get("skipped", [])],
        "made_at": datetime.now().isoformat(timespec="seconds")}, ensure_ascii=False, indent=1), encoding="utf-8")
    found = sum(1 for r in rows if r["det_cls"])
    print(f"합쳤습니다: {out} (크롭 {len(rows)}개, 짝 {found}, 클래스 맞음 {sum(1 for r in rows if r['det_cls'] == r['cls'])}, "
          f"견줄 클래스 {','.join(compared)})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("a0", help="기준선 — 검출기의 클래스를 GT 박스에 짝짓기")
    a.add_argument("--crops", required=True, type=Path)
    a.add_argument("--config", default=None)
    a.add_argument("--weights", default=None, help="기본: config 의 운영 가중치")
    a.add_argument("--imgsz", type=int, default=1280)
    a.add_argument("--conf", type=float, default=0.25)
    a.add_argument("--iou", type=float, default=0.6)
    a.add_argument("--class-iou", type=float, default=0.6)
    a.add_argument("--device", default="0")
    a.add_argument("--only-fold", default=None, help="이 fold 의 크롭만(A0b — fold 마다 그 fold 를 안 본 검출기로)")
    a.add_argument("--tag", default=None, help="결과 폴더 이름(기본 a0_<가중치>_<입력>[_f<fold>])")
    g = sub.add_parser("merge", help="fold 마다의 a0 결과를 하나로(A0b)")
    g.add_argument("--crops", required=True, type=Path)
    g.add_argument("--parts", nargs="+", required=True)
    g.add_argument("--tag", required=True)
    r = sub.add_parser("report", help="OOF 지표·A0 대비·두 실행 비교")
    r.add_argument("--crops", required=True, type=Path)
    r.add_argument("--runs", nargs="+", required=True)
    r.add_argument("--baseline", default=None, help="a0 결과 폴더 이름(runs/ 아래)")
    r.add_argument("--compare", nargs="*", default=None, metavar="A:B")
    r.add_argument("--min-long", type=float, default=48.0)
    r.add_argument("--boot", type=int, default=2000)
    r.add_argument("--jsonl", default="runs/refiner/results.jsonl")
    args = ap.parse_args()
    return {"a0": cmd_a0, "merge": cmd_merge, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
