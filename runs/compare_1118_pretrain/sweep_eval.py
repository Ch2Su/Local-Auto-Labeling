"""갈래마다 conf 문턱을 바꿔 가며 찾음·클래스까지·헛 박스를 세고, 정답 크기별로 찾은 수를 본다(추론만).

한 번 낮은 conf(0.05)로 예측해 두고 문턱마다 걸러서 짝짓는다 — 짝짓기는 cmp_common.count 와 같은 식(conf 순 탐욕 1:1, IoU 0.5).
  python -I sweep_eval.py --device cpu --sets o365_zero,coco_zero,A,P2S
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cmp_common as cc  # noqa: E402

CONFS = (0.25, 0.35, 0.5)


def predict_all(model, files, imgsz, device, batch):
    out = []
    for i in range(0, len(files), batch):
        chunk = files[i:i + batch]
        res = model.predict([str(f) for f in chunk], imgsz=imgsz, conf=0.05, iou=0.6, max_det=300, device=device, verbose=False)
        for f, r in zip(chunk, res):
            h, w = r.orig_shape
            gts = cc.load_gt(Path(f), w, h)
            scale = imgsz / max(w, h)
            bins = [cc.size_bin(min(g[3] - g[1], g[4] - g[2]) * scale) for g in gts]
            b = r.boxes
            out.append((gts, bins, b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), [model.names[int(c)] for c in b.cls.cpu().numpy()]))
    return out


def tally(preds, ok, conf):
    t = {"total": 0, "found": 0, "klass": 0, "false": 0, "size_total": Counter(), "size_found": Counter(),
         "cls_total": Counter(), "cls_found": Counter(), "cls_klass": Counter()}
    for gts, bins, xy, cf, names in preds:
        t["total"] += len(gts)
        for g, sb in zip(gts, bins):
            t["size_total"][sb] += 1
            t["cls_total"][g[0]] += 1
        gt_xy = np.array([g[1:] for g in gts], dtype=float).reshape(-1, 4)
        used = np.zeros(len(gts), dtype=bool)
        for j in np.argsort(-cf):
            if cf[j] < conf or names[j] not in ok:
                continue
            if len(gts):
                v = cc.iou_one(xy[j], gt_xy)
                v[used] = -1
                k = int(v.argmax())
                if v[k] >= cc.MATCH_IOU:
                    used[k] = True
                    t["found"] += 1
                    t["size_found"][bins[k]] += 1
                    t["cls_found"][gts[k][0]] += 1
                    if gts[k][0] in ok[names[j]]:
                        t["klass"] += 1
                        t["cls_klass"][gts[k][0]] += 1
                    continue
            t["false"] += 1
    return t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True)
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    if args.threads:
        torch.set_num_threads(args.threads)
    device = "cpu" if args.device == "cpu" else int(args.device)
    fixed = {"coco_zero": (cc.REPO / "weights" / "yolo26m.pt", cc.COCO_OK),
             "o365_zero": (cc.ROOT / "dl_o365" / "yolo26m-objv1-150.pt", cc.O365_OK),
             "operating": (cc.REPO / "models" / "best_y26m_8cls.pt", cc.SELF_OK)}
    for name in args.sets.split(","):
        preds = []
        for fold in (0, 1, 2):
            if name in fixed:
                path, ok = fixed[name]
            else:                                   # 학습한 갈래: runs/<갈래>_f<fold>_e<epoch>[_s<seed>]/weights/last.pt
                arm, _, seed = name.partition("@")  # 이름은 '갈래[:epoch][@seed]' — epoch 을 안 적으면 100 (예: FZ23:25, P2S@1)
                arm, _, ep = arm.partition(":")
                path = cc.ROOT / "runs" / (f"{arm}_f{fold}_e{ep or 100}" + (f"_s{seed}" if seed else "")) / "weights" / "last.pt"
                ok = cc.SELF_OK
            model = YOLO(str(path))
            preds += predict_all(model, cc.fold_files(fold, "val"), args.imgsz, device, args.batch)
        for conf in CONFS:
            t = tally(preds, ok, conf)
            rec = {"set": name, "imgsz": args.imgsz, "conf": conf, "total": t["total"], "found": t["found"], "klass": t["klass"],
                   "false": t["false"], "size": {b: [t["size_found"][b], t["size_total"][b]] for b in cc.SIZE_BINS},
                   "cls": {c: [t["cls_found"][c], t["cls_klass"][c], t["cls_total"][c]] for c in cc.NAMES if t["cls_total"][c]}}
            with open(cc.ROOT / "sweep.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            line = (f"{name} imgsz {args.imgsz} conf {conf}: 찾음 {t['found']}/{t['total']} ({t['found'] / t['total']:.1%}) · "
                    f"클래스까지 {t['klass']} · 헛 박스 {t['false']}")
            if conf == CONFS[0]:
                line += " · 크기별 " + ", ".join(f"{b} {t['size_found'][b]}/{t['size_total'][b]}" for b in cc.SIZE_BINS)
                line += " · 클래스별(찾음/클래스까지/정답) " + ", ".join(f"{c} {v[0]}/{v[1]}/{v[2]}" for c, v in rec["cls"].items())
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
