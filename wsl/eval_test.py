#!/usr/bin/env python3
"""가중치를 test 세트에 재서 한 줄씩(JSONL) 남긴다 — pod·WSL 에서 도는 평가 (2026-11-12).

`train_wsl.py` 처럼 **프로젝트 코드를 import 하지 않는다**(Ultralytics 만 쓴다). 로컬의 `tools/test_baseline.py` 는 지금 DB 의
test 분할을 재는 도구라, 실험용으로 따로 만든 세트(`tools/build_subset_dataset.py`)와 pod 에서는 이것을 쓴다.

가중치 × 세트 × 입력 크기마다 한 줄:
- `map`(mAP50-95)·`map50`·`precision`·`recall`, 클래스별 `map`·`map50`·`p`·`r`·`n`(정답 박스 수)
- **`at_conf`** — 라벨링에 쓰는 문턱(기본 0.25)에서 센 것: `hit`(맞게 찾은 정답), `missed`(놓친 정답), `false`(정답이
  없는 자리의 예측 — 헛 박스), `wrong_class`(자리는 맞는데 클래스가 다른 것). 자동 라벨링을 고치는 일의 양은 mAP 보다
  이 수에 가깝다. Ultralytics 의 혼동 행렬(IoU 0.45 로 짝짓는다)에서 읽는다 — 그래서 `plots=True` 로 돈다.

  python3 eval_test.py --weights a.pt b.pt --data set1=/root/datasets/x/dataset/data.yaml --imgsz 640 1280 --out out.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def sha256_head(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def summarize(res, names: dict, conf: float) -> dict:
    b = res.box
    nt = getattr(res, "nt_per_class", None)
    per = {}
    for k, c in enumerate(res.ap_class_index):
        p, r, ap50, ap = b.class_result(k)
        per[names[int(c)]] = {"map": round(float(ap), 4), "map50": round(float(ap50), 4), "p": round(float(p), 4),
                              "r": round(float(r), 4), "n": int(nt[int(c)]) if nt is not None else None}
    out = {"map": round(float(b.map), 4), "map50": round(float(b.map50), 4), "precision": round(float(b.mp), 4),
           "recall": round(float(b.mr), 4), "classes": per}
    cm = getattr(res, "confusion_matrix", None)
    if cm is not None:
        m = cm.matrix                       # m[예측 클래스, 정답 클래스], 마지막 줄·칸 = 배경
        nc = m.shape[0] - 1
        hit = float(m[:nc, :nc].diagonal().sum())
        out["at_conf"] = {"conf": conf, "hit": int(hit), "missed": int(m[nc, :nc].sum()), "false": int(m[:nc, nc].sum()),
                          "wrong_class": int(m[:nc, :nc].sum() - hit),
                          "per_class": {names[c]: {"hit": int(m[c, c]), "missed": int(m[nc, c]), "false": int(m[c, nc])}
                                        for c in range(nc)}}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--data", nargs="+", required=True, metavar="이름=data.yaml", help="잴 세트 — 이름은 기록에 남는다")
    ap.add_argument("--imgsz", nargs="+", type=int, required=True)
    ap.add_argument("--out", required=True, help="결과를 덧붙일 JSONL")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.25, help="at_conf 를 세는 문턱 (mAP 는 늘 0.001 부터 센다)")
    ap.add_argument("--iou", type=float, default=0.6, help="NMS IoU (config.yaml 의 predict.iou 와 같게)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="/tmp/eval_test", help="Ultralytics 가 그림을 쓰는 곳")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    from ultralytics import YOLO

    sets = []
    for item in args.data:
        name, sep, path = item.partition("=")
        if not sep:
            ap.error(f"--data 는 '이름=data.yaml' 꼴로: {item}")
        sets.append((name, path))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for w in args.weights:
        wp = Path(w)
        if not wp.exists():
            print(f"[건너뜀] 가중치가 없습니다: {w}", flush=True)
            continue
        model = YOLO(w)
        digest = sha256_head(wp)
        for name, data in sets:
            for sz in args.imgsz:
                t0 = time.time()
                res = model.val(data=data, split=args.split, imgsz=sz, batch=args.batch, device=args.device,
                                iou=args.iou, plots=True, verbose=False, project=args.project,
                                name=f"{wp.stem}_{name}_{sz}", exist_ok=True)
                rec = {"weights": wp.name, "sha256": digest, "set": name, "split": args.split, "imgsz": sz,
                       "seconds": round(time.time() - t0, 1), "note": args.note,
                       **summarize(res, model.names, args.conf)}
                with open(out, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                a = rec.get("at_conf", {})
                print(f"RESULT {wp.name} {name} imgsz={sz} map={rec['map']} map50={rec['map50']} "
                      f"hit={a.get('hit')} missed={a.get('missed')} false={a.get('false')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
