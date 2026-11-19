"""옛 고속도로 프레임(검수용 작업 사본)에서 SAM 3 계열의 텍스트 프롬프트 후보를 뽑아, DB 에 남은 OWLv2 후보와 견준다 — 읽기만.

OWLv2 후보 = 보관 DB(검수 전)의 origin 'owl' 박스(responder·cone). 정답 = 작업 사본에서 **승인된** 프레임의 박스(사람이 본 것).
두 후보를 같은 짝짓기(conf 순 탐욕 1:1, IoU 0.5)로 정답과 견주고, 전체 프레임에서는 서로 겹치는지(둘 다·한쪽만)를 센다.

  python -I sam3_hw.py --deps <패키지 폴더> --weights <sam3.1_multiplex.pt> --work <작업 사본 DB> --archive <보관 DB> [--sheet <폴더>]
"""
import argparse
import json
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cmp_common as cc  # noqa: E402
from sam3_propose import NAMES, TO_CLASS, detect, nms_by_class  # noqa: E402

CLS = {8: "responder", 9: "cone"}


def to_xyxy(b, w, h):
    return ((b["cx"] - b["w"] / 2) * w, (b["cy"] - b["h"] / 2) * h, (b["cx"] + b["w"] / 2) * w, (b["cy"] + b["h"] / 2) * h)


def greedy(cands, gts):
    """cands = [(cls, conf, x1, y1, x2, y2)], gts = [(cls, x1, y1, x2, y2)] -> [(후보, 닿은 정답 번호 또는 None)]"""
    gt_xy = np.array([g[1:] for g in gts], dtype=float).reshape(-1, 4)
    used = np.zeros(len(gts), dtype=bool)
    out = []
    for c in sorted(cands, key=lambda t: -t[1]):
        k = None
        if len(gts):
            v = cc.iou_one(np.array(c[2:], dtype=float), gt_xy)
            v[used] = -1
            j = int(v.argmax())
            if v[j] >= cc.MATCH_IOU:
                used[j] = True
                k = j
        out.append((c, k))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deps", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=cc.ROOT / "sam3_out")
    ap.add_argument("--conf", type=float, default=0.1)
    args = ap.parse_args()
    sys.path.insert(0, args.deps)
    import cv2
    import torch
    from ultralytics.models.sam import SAM3SemanticPredictor

    work = sqlite3.connect(f"file:{args.work.as_posix()}?mode=ro", uri=True)
    work.row_factory = sqlite3.Row
    arch = sqlite3.connect(f"file:{args.archive.as_posix()}?mode=ro&immutable=1", uri=True)
    arch.row_factory = sqlite3.Row
    names = [r["name"] for r in work.execute("SELECT name FROM classes ORDER BY idx")]
    imgs = work.execute("SELECT id, path, camera, status, updated_at FROM images ORDER BY id").fetchall()

    pred = SAM3SemanticPredictor(overrides=dict(conf=args.conf, task="segment", mode="predict", model=args.weights, save=False,
                                                verbose=False, device=0, imgsz=1008, quantize=16))
    recs, times = [], []
    for n, im in enumerate(imgs, 1):
        img = cv2.imread(im["path"])
        h, w = img.shape[:2]
        t0 = time.time()
        dets = nms_by_class(detect(pred, img, args.conf))
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        owl = [(CLS[b["cls"]], float(b["conf"] or 0), *to_xyxy(b, w, h))
               for b in arch.execute("SELECT cls, cx, cy, w, h, conf FROM boxes WHERE image_id = ? AND origin = 'owl' AND cls IN (8, 9)", (im["id"],))]
        gt = [(names[b["cls"]], *to_xyxy(b, w, h)) for b in work.execute("SELECT cls, cx, cy, w, h FROM boxes WHERE image_id = ?", (im["id"],))]
        recs.append({"id": im["id"], "file": im["path"], "camera": im["camera"], "status": im["status"], "w": w, "h": h,
                     "sam": [(TO_CLASS[d[0]], d[1], *d[2:], d[0]) for d in dets], "owl": owl, "gt": gt})
        if n % 50 == 0 or n == len(imgs):
            print(f"  {n}/{len(imgs)}장 · 장당 {np.median(times):.2f}초", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "hw_sam31_vs_owl.jsonl", "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    appr = [r for r in recs if r["status"] == "approved"]
    tot = Counter(g[0] for r in appr for g in r["gt"])
    print(f"\n== 승인된 {len(appr)}장(사람이 본 라벨)으로 채점 — 정답 responder {tot['responder']} · cone {tot['cone']} ==")
    print("| 후보 | 문턱 | responder 찾음 | 후보 수 | cone 찾음 | 후보 수 |")
    print("|---|---|---|---|---|---|")
    for label, key, confs in (("OWLv2(DB 기록)", "owl", (0.0, 0.35, 0.5)), ("SAM 3.1", "sam", (0.25, 0.35, 0.5, 0.7))):
        for conf in confs:
            hit, n = Counter(), Counter()
            for r in appr:
                for c, k in greedy([x[:6] for x in r[key] if x[1] >= conf], r["gt"]):
                    n[c[0]] += 1
                    hit[c[0]] += k is not None and r["gt"][k][0] == c[0]
            print(f"| {label} | {conf if conf else '전부'} | {hit['responder']}/{tot['responder']} | {n['responder']} | {hit['cone']}/{tot['cone']} | {n['cone']} |")

    print(f"\n== 전체 {len(recs)}장에서 두 후보가 겹치는가 (SAM 3.1 은 conf 0.5 이상, OWLv2 는 DB 에 남은 것 전부, IoU 0.5) ==")
    both, only_s, only_o = Counter(), Counter(), Counter()
    by_name = Counter()
    for r in recs:
        sam = [x for x in r["sam"] if x[1] >= 0.5]
        for cls in ("responder", "cone"):
            s = [x for x in sam if x[0] == cls]
            o = [x for x in r["owl"] if x[0] == cls]
            m = greedy([x[:6] for x in s], [(x[0], *x[2:6]) for x in o])
            nb = sum(k is not None for _, k in m)
            both[cls] += nb
            only_s[cls] += len(s) - nb
            only_o[cls] += len(o) - nb
        for x in sam:
            by_name[x[6]] += 1
    for cls in ("responder", "cone"):
        print(f"  {cls}: 둘 다 {both[cls]} · SAM 3.1 만 {only_s[cls]} · OWLv2 만 {only_o[cls]}")
    print("  SAM 3.1 후보의 이름별 수(conf 0.5 이상):", dict(by_name.most_common()))
    cam = Counter()
    for r in recs:
        cam[(r["camera"], "sam")] += sum(1 for x in r["sam"] if x[1] >= 0.5 and x[0] == "responder")
        cam[(r["camera"], "owl")] += sum(1 for x in r["owl"] if x[0] == "responder")
    print("  카메라별 responder 후보 (SAM 3.1 / OWLv2):", {c: f"{cam[(c, 'sam')]}/{cam[(c, 'owl')]}" for c in sorted({r['camera'] for r in recs})})
    print(f"장당 중앙값 {np.median(times):.2f}초 · {args.out / 'hw_sam31_vs_owl.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
