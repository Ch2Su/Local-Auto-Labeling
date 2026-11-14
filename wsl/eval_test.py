#!/usr/bin/env python3
"""가중치를 test 세트에 재서 한 줄씩(JSONL) 남긴다 — pod·WSL 에서 도는 평가 (2026-11-12).

`train_wsl.py` 처럼 **프로젝트 코드를 import 하지 않는다**(Ultralytics 만 쓴다). 로컬의 `tools/test_baseline.py` 는 지금 DB 의
test 분할을 재는 도구라, 실험용으로 따로 만든 세트(`tools/build_subset_dataset.py`)와 pod 에서는 이것을 쓴다.

가중치 × 세트 × 입력 크기마다 한 줄:
- `map`(mAP50-95)·`map50`·`precision`·`recall`, 클래스별 `map`·`map50`·`p`·`r`·`n`(정답 박스 수)
- **`at_conf`** — 라벨링에 쓰는 문턱(기본 0.25)에서 센 것: `hit`(맞게 찾은 정답), `missed`(놓친 정답), `false`(정답이
  없는 자리의 예측 — 헛 박스), `wrong_class`(자리는 맞는데 클래스가 다른 것). 자동 라벨링을 고치는 일의 양은 mAP 보다
  이 수에 가깝다. Ultralytics 의 혼동 행렬(IoU 0.45 로 짝짓는다)에서 읽는다 — 그래서 `plots=True` 로 돈다.
  **혼동 행렬의 문턱은 `pin_confusion_thresholds` 가 고정한다(2026-11-15)** — Ultralytics 버전마다 기본이 달라, 그냥 두면
  에러 없이 다른 문턱에서 센 수가 `conf: 0.25` 라는 이름으로 남는다. 기록마다 `ultralytics` 버전도 적는다.

  python3 eval_test.py --weights a.pt b.pt --data set1=/root/datasets/x/dataset/data.yaml --imgsz 640 1280 --out out.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
import time
from pathlib import Path

MATCH_IOU = 0.45        # 혼동 행렬에서 예측과 정답을 짝짓는 IoU — 11-12 기록(Ultralytics 8.4.142)이 이 값으로 센 것이다


def pin_confusion_thresholds(conf: float, match_iou: float = MATCH_IOU) -> None:
    """Ultralytics 의 혼동 행렬이 버전과 상관없이 `conf`·`match_iou` 로 세게 한다.

    8.4.212 의 검증기는 혼동 행렬에 val 의 conf 를 그대로 넘기는데(`process_batch(predn, pbatch, conf=self.args.conf)`) val 의
    기본 conf 는 0.001 이다 — 그 0.001 과 IoU 0.5 로 센다. 8.4.142 는 val 에 conf 를 안 주면 0.25 를 넘기고
    (`confusion_matrix_conf = 0.25 if conf is None else conf`) `process_batch` 의 기본 `iou_thres` 가 0.45 라, 그 버전으로 잰
    11-12 기록은 conf 0.25·IoU 0.45 로 센 수다(11-15 에 소스로 확인 — 8.4.142 에서는 이 고정이 값을 바꾸지 않는다). 같은 가중치·
    같은 119장(newtest8, 640)에서 8.4.212 는 헛 박스를 20 이 아니라 988 로 냈고, 여기서 두 값을 고정하니 놓침·헛 박스가 세
    해상도 모두 11-12 와 같아졌다(2026-11-15, docs/experiments.md '새 풀 프레임 데이터').

    - val 에 `conf=0.25` 를 넘기는 것으로 고치지 말 것 — 그러면 mAP 도 0.25 위에서만 센다(PR 곡선이 잘린다).
    - 짝짓는 순서까지 맞추지는 않는다: 8.4.212 는 conf 가 높은 예측부터 정답을 하나씩 차지하고, 8.4.142 는 IoU 가 큰 짝부터
      남긴다. 그래서 `hit` 와 `wrong_class` 는 8.4.142 의 수와 1~2개 오갈 수 있다(둘의 합, `missed`, `false` 는 같았다).
    - 두 번 불러도 겹쳐 감싸지 않는다(마지막 값이 남는다).
    """
    from ultralytics.utils import metrics

    orig = metrics.ConfusionMatrix.process_batch
    orig = getattr(orig, "_unpinned", orig)
    missing = {"conf", "iou_thres"} - set(inspect.signature(orig).parameters)
    if missing:                 # 조용히 다른 문턱으로 세느니 멈춘다
        raise SystemExit(f"[중단] 이 Ultralytics 의 ConfusionMatrix.process_batch 에 {sorted(missing)} 인자가 없습니다 — "
                         "at_conf 의 문턱을 고정할 수 없습니다(wsl/eval_test.py 의 pin_confusion_thresholds).")

    def process_batch(self, detections, batch, *_args, **_kwargs):
        return orig(self, detections, batch, conf=conf, iou_thres=match_iou)

    process_batch._unpinned = orig
    metrics.ConfusionMatrix.process_batch = process_batch


def sha256_head(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def summarize(res, names: dict, conf: float, match_iou: float = MATCH_IOU) -> dict:
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
        out["at_conf"] = {"conf": conf, "match_iou": match_iou, "hit": int(hit), "missed": int(m[nc, :nc].sum()),
                          "false": int(m[:nc, nc].sum()),
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
    ap.add_argument("--match-iou", type=float, default=MATCH_IOU,
                    help="at_conf 에서 예측과 정답을 짝짓는 IoU (옛 기록과 견주려면 그대로 둘 것)")
    ap.add_argument("--iou", type=float, default=0.6, help="NMS IoU (config.yaml 의 predict.iou 와 같게)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="/tmp/eval_test", help="Ultralytics 가 그림을 쓰는 곳")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    import ultralytics
    from ultralytics import YOLO

    pin_confusion_thresholds(args.conf, args.match_iou)
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
                       "ultralytics": ultralytics.__version__,
                       **summarize(res, model.names, args.conf, args.match_iou)}
                with open(out, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                a = rec.get("at_conf", {})
                print(f"RESULT {wp.name} {name} imgsz={sz} map={rec['map']} map50={rec['map50']} "
                      f"hit={a.get('hit')} missed={a.get('missed')} false={a.get('false')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
