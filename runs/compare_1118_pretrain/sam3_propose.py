"""SAM 3 계열(3·3.1)의 텍스트 프롬프트로 출동 차량·라바콘 후보를 찾아, 같은 평가 세트(175장)의 정답과 견준다 — 추론만.

이름은 OWLv2 후보 도구(tools/ovd_propose.py)와 같다: fire truck·ambulance·police car·tow truck → responder, traffic cone → cone.
같은 물체에 이름이 여럿 걸리면 우리 클래스 기준으로 점수 높은 것 하나만 남긴다(IoU 0.5). 짝짓기는 다른 측정과 같다(conf 순 탐욕
1:1, IoU 0.5) — 정답 전체와 짝지어, 후보가 닿은 곳이 ① 그 클래스의 정답 ② 다른 클래스의 정답(일반 차를 출동 차량으로 본 것 등)
③ 정답이 없는 자리 중 어디인지 센다.

SAM 3 는 그림을 정사각형으로 **늘려** 넣는다(가로 2560 → 1008 이면 0.39배). --tiles N 이면 통째에 더해 N×N 조각(겹침 20%)으로도
묻고 합친다. 패키지는 프로젝트 .venv 밖의 별도 폴더(--deps)에서 읽는다 — YOLO_AUTOINSTALL=false 로 돌릴 것.

  python -I sam3_propose.py --deps <패키지 폴더> --weights <sam3.1_multiplex.pt> [--tiles 2] [--tag 이름] [--limit N] [--dry]
결과: <--out>/<tag>.jsonl(이미지마다 후보 — 파일 이름이 있어 로컬에만), 요약은 화면과 <--out>/<tag>_summary.json.
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cmp_common as cc  # noqa: E402

NAMES = ["fire truck", "ambulance", "police car", "tow truck", "traffic cone"]
TO_CLASS = {"fire truck": "responder", "ambulance": "responder", "police car": "responder", "tow truck": "responder",
            "traffic cone": "cone"}
CONFS = (0.25, 0.35, 0.5)


def nms_by_class(dets, iou=0.5):
    """우리 클래스가 같은 후보끼리 겹치면 점수 높은 것만. dets = [(name, conf, x1, y1, x2, y2)]"""
    keep = []
    for d in sorted(dets, key=lambda t: -t[1]):
        box = np.array(d[2:], dtype=float)
        same = [k for k in keep if TO_CLASS[k[0]] == TO_CLASS[d[0]]]
        if same and cc.iou_one(box, np.array([k[2:] for k in same], dtype=float)).max() >= iou:
            continue
        keep.append(d)
    return keep


def tiles_of(w, h, n, overlap=0.2):
    tw, th = w / (n - (n - 1) * overlap), h / (n - (n - 1) * overlap)
    out = []
    for j in range(n):
        for i in range(n):
            x1, y1 = int(round(i * tw * (1 - overlap))), int(round(j * th * (1 - overlap)))
            out.append((x1, y1, min(w, int(round(x1 + tw))), min(h, int(round(y1 + th)))))
    return out


def detect(pred, img, conf):
    pred.reset_image()
    pred.set_image(img)
    res = pred(text=NAMES)[0]
    b = res.boxes
    names = res.names
    xy, cf, cl = b.xyxy.float().cpu().numpy(), b.conf.float().cpu().numpy(), b.cls.cpu().numpy().astype(int)
    return [(names[c], float(s), *map(float, box)) for box, s, c in zip(xy, cf, cl) if s >= conf]


def score(recs, conf):
    """(클래스별 찾은 정답 수, 후보가 닿은 곳의 수)"""
    total, hit, where = Counter(), Counter(), Counter()
    for r in recs:
        gts = r["gt"]
        for g in gts:
            total[g[0]] += 1
        gt_xy = np.array([g[1:] for g in gts], dtype=float).reshape(-1, 4)
        used = np.zeros(len(gts), dtype=bool)
        for name, s, *box in sorted((d for d in r["dets"] if d[1] >= conf), key=lambda t: -t[1]):
            cls = TO_CLASS[name]
            if len(gts):
                v = cc.iou_one(np.array(box, dtype=float), gt_xy)
                v[used] = -1
                k = int(v.argmax())
                if v[k] >= cc.MATCH_IOU:
                    used[k] = True
                    if gts[k][0] == cls:
                        hit[cls] += 1
                        where[(cls, "맞음")] += 1
                    else:
                        where[(cls, f"다른 클래스({gts[k][0]})")] += 1
                    continue
            where[(cls, "정답 없는 자리")] += 1
    return total, hit, where


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deps", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=1008)
    ap.add_argument("--tiles", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.1, help="이 점수 이상 후보를 남긴다(요약은 0.25·0.35·0.5 로 다시 거른다)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", type=Path, default=cc.ROOT / "sam3_out")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry", action="store_true", help="체크포인트를 읽지 않고(임의 가중치) 흐름만 본다")
    args = ap.parse_args()
    sys.path.insert(0, args.deps)

    import cv2
    import torch

    if args.dry:
        import ultralytics.models.sam.build_sam3 as b3

        b3._load_checkpoint = lambda model, checkpoint, interactive=False: model
    from ultralytics.models.sam import SAM3SemanticPredictor

    pred = SAM3SemanticPredictor(overrides=dict(conf=0.999 if args.dry else args.conf, task="segment", mode="predict", model=args.weights,
                                                save=False, verbose=False, device=0, imgsz=args.imgsz, quantize=16))
    files = sorted({f for fold in (0, 1, 2) for f in cc.fold_files(fold, "val")})
    if args.limit:
        files = files[: args.limit]
    tag = args.tag or (Path(args.weights).stem + (f"_t{args.tiles}" if args.tiles else ""))
    args.out.mkdir(parents=True, exist_ok=True)
    recs, times = [], []
    torch.cuda.reset_peak_memory_stats()
    for n, f in enumerate(files, 1):
        img = cv2.imread(str(f))
        h, w = img.shape[:2]
        t0 = time.time()
        dets = detect(pred, img, args.conf)
        if args.tiles and max(w, h) > args.imgsz:
            for x1, y1, x2, y2 in tiles_of(w, h, args.tiles):
                for name, s, a, b, c, d in detect(pred, np.ascontiguousarray(img[y1:y2, x1:x2]), args.conf):
                    dets.append((name, s, a + x1, b + y1, c + x1, d + y1))
        dets = nms_by_class(dets)
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        recs.append({"file": str(f), "w": w, "h": h, "gt": cc.load_gt(Path(f), w, h), "dets": dets})
        if n % 25 == 0 or n == len(files):
            print(f"  {n}/{len(files)}장 · 장당 {np.median(times):.2f}초", flush=True)
    with open(args.out / f"{tag}.jsonl", "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"tag": tag, "weights": Path(args.weights).name, "imgsz": args.imgsz, "tiles": args.tiles, "images": len(recs), "dry": args.dry,
               "sec_per_image_median": round(float(np.median(times)), 3), "vram_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
               "by_conf": {}}
    for conf in CONFS:
        total, hit, where = score(recs, conf)
        summary["by_conf"][str(conf)] = {"total": {c: total[c] for c in ("responder", "cone")}, "hit": {c: hit[c] for c in ("responder", "cone")},
                                         "where": {f"{c}:{k}": v for (c, k), v in sorted(where.items())}}
        cand = Counter()
        for (c, _), v in where.items():
            cand[c] += v
        print(f"conf {conf}: responder 정답 {total['responder']}개 중 {hit['responder']} 찾음(후보 {cand['responder']}개) · "
              f"cone {total['cone']}개 중 {hit['cone']} 찾음(후보 {cand['cone']}개)")
        print("    후보가 닿은 곳:", {f"{c}:{k}": v for (c, k), v in sorted(where.items())})
    (args.out / f"{tag}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"장당 중앙값 {summary['sec_per_image_median']}초 · VRAM 최대 {summary['vram_peak_gib']} GiB · {args.out / (tag + '.jsonl')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
