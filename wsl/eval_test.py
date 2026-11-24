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
  에러 없이 다른 문턱에서 센 수가 `conf: 0.25` 라는 이름으로 남는다. 기록마다 `ultralytics`·`torch` 버전과 `batch` 도 적는다
  (batch 4 와 8 은 mAP 넷째 자리가 다르고 어느 쪽이 어느 값인지는 PC 마다 반대였다 — PC 가 다른 기록끼리는 넷째 자리를
  견주지 말 것. `at_conf` 는 batch 와 상관없이 같았다. 11-12 기록은 batch 4, docs/experiments.md '새 풀 프레임 데이터').

**채점에서 뺄 자리 (2026-11-24 사용자 결정 — `src/autolabel/masks.py` 머리말 'test 의 영역').** test 이미지의 가릴 영역은
칠하지 않는다(평가는 실제 화면 그대로). 대신 세트를 만드는 쪽이 반 넘게 덮인 정답 박스를 라벨에서 빼고, 영역이 칠할 픽셀을
`ignore/<분할>/<이름>.png`(라벨 파일 자리에 `labels` 대신 `ignore`) 흰색으로 남긴다. 여기서는 그 그림이 있는 이미지의 예측 가운데
**박스 안 픽셀의 반 이상이 흰색인 것**을 셈에서 뺀다(`IgnoreValidator` — mAP·`at_conf` 둘 다). 그 자리의 애매한 물체를 모델이
잡아도 헛 박스로 세지 않는다는 뜻이다(COCO 의 iscrowd·VOC 의 difficult 와 같은 생각). 프로젝트 코드를 import 하지 않으려고
영역 좌표 대신 그림을 읽는다 — 그래서 경계에서는 픽셀 한 칸만큼 정답 쪽 판정(`masks.covered`, 정확한 넓이)과 다를 수 있다.
뺀 수는 기록의 `ignore`(그림이 있던 장수·`at_conf` 문턱 위에서 뺀 예측 수)로 남긴다 — 덮어서 점수를 올리는 일이 숫자로 보이게.
`tools/test_baseline.py`·`test_tiles.py`(로컬 test 측정)도 이 파일의 `ignore_validator`·`drop_ignored` 를 쓴다 — 한 곳에 둔다.

  python3 eval_test.py --weights a.pt b.pt --data set1=/root/datasets/x/dataset/data.yaml --imgsz 640 1280 --out out.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import sys
import time
from pathlib import Path

MATCH_IOU = 0.45        # 혼동 행렬에서 예측과 정답을 짝짓는 IoU — 11-12 기록(Ultralytics 8.4.142)이 이 값으로 센 것이다
IGNORE_DIR = "ignore"   # 채점에서 뺄 자리 그림의 폴더 — src/autolabel/masks.py 의 IGNORE_DIR 와 같아야 한다(고치면 둘 다)
IGNORE_COVER = 0.5      # 예측 박스의 이만큼 이상이 그 자리면 뺀다 — masks.DROP(정답 박스를 라벨에서 빼는 문턱)과 같다


def ignore_path(im_file) -> Path | None:
    """그림 경로 → 그 그림의 '채점에서 뺄 자리' 그림 경로(없을 수 있다). Ultralytics 가 라벨을 찾는 규칙(경로의 마지막
    `images` 를 `labels` 로)과 같은 자리에 `ignore`, 확장자 .png. 경로에 `images` 폴더가 없으면 None."""
    parts = list(Path(im_file).parts)
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == "images":
            parts[i] = IGNORE_DIR
            return Path(*parts).with_suffix(".png")
    return None


def load_ignore(im_file):
    """그 그림의 채점에서 뺄 자리 — bool 배열(줄, 칸), 없으면 None."""
    p = ignore_path(im_file)
    if p is None or not p.is_file():
        return None
    import numpy as np
    from PIL import Image

    with Image.open(p) as im:
        return np.asarray(im.convert("1"), dtype=bool)


def covered_px(bitmap, box, shape) -> float:
    """원본 픽셀 xyxy 박스 안에서 뺄 자리 픽셀의 몫 — 픽셀 중심이 박스 안인 칸으로 센다(masks.poly_pixels 와 같은 뜻).
    shape 는 원본 (높이, 폭) — 그림 크기가 달라도 비율로 맞춘다. 중심을 하나도 안 품는 아주 작은 박스는 가운데 칸으로 본다."""
    H, W = bitmap.shape
    sx, sy = W / float(shape[1]), H / float(shape[0])
    x1, y1, x2, y2 = box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy
    c0, c1 = max(0, math.ceil(x1 - 0.5)), min(W, math.ceil(x2 - 0.5))
    r0, r1 = max(0, math.ceil(y1 - 0.5)), min(H, math.ceil(y2 - 0.5))
    if c1 <= c0 or r1 <= r0:
        cx = min(W - 1, max(0, int((x1 + x2) / 2)))
        cy = min(H - 1, max(0, int((y1 + y2) / 2)))
        return float(bitmap[cy, cx])
    return float(bitmap[r0:r1, c0:c1].mean())


def drop_ignored(boxes, bitmap, shape) -> list[bool]:
    """박스(원본 픽셀 xyxy 의 목록)마다 남길지 — 뺄 자리가 IGNORE_COVER 이상 덮으면 False. bitmap 이 None 이면 전부 남긴다."""
    if bitmap is None:
        return [True] * len(boxes)
    return [covered_px(bitmap, b, shape) < IGNORE_COVER for b in boxes]


def ignore_validator():
    """`model.val(validator=…)` 에 줄 검증기 클래스 — 채점에서 뺄 자리에 반 넘게 든 예측을 mAP·혼동 행렬 전에 뺀다(머리말).

    Ultralytics 의 `DetectionValidator.update_metrics` 앞에서 예측(입력 크기 좌표)을 원본 좌표로 되돌려(`ops.scale_boxes` —
    정답 박스를 원본으로 돌릴 때와 같은 함수) 거른다. 예측의 꼴(이미지마다 bboxes·conf·cls 사전)이 다른 버전이면 조용히
    다르게 세지 않고 멈춘다. 마지막으로 만든 검증기는 `cls.last` — 뺀 수(`ignore_images`·`ignored_conf`)를 거기서 읽는다."""
    import torch
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.utils import ops

    class IgnoreValidator(DetectionValidator):
        last = None

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.ignore_maps: dict[str, object] = {}
            self.ignore_images = 0          # 뺄 자리 그림이 있던 장수
            self.ignored_conf: list[float] = []     # 뺀 예측의 신뢰도 — 문턱 위의 수는 부르는 쪽이 센다
            type(self).last = self

        def _map(self, im_file):
            if im_file not in self.ignore_maps:
                bm = self.ignore_maps[im_file] = load_ignore(im_file)
                self.ignore_images += bm is not None
            return self.ignore_maps[im_file]

        def update_metrics(self, preds, batch):
            for si, pred in enumerate(preds):
                if not isinstance(pred, dict) or not {"bboxes", "conf", "cls"} <= set(pred):
                    raise SystemExit("[중단] 이 Ultralytics 의 예측 꼴(이미지마다 bboxes·conf·cls 사전)이 달라 채점에서 뺄 "
                                     "자리를 적용할 수 없습니다 — wsl/eval_test.py 의 ignore_validator")
                bm = self._map(batch["im_file"][si])
                if bm is None or not len(pred["bboxes"]):
                    continue
                shape = batch["ori_shape"][si]
                boxes = ops.scale_boxes(batch["img"].shape[2:], pred["bboxes"].clone(), shape, batch["ratio_pad"][si])
                keep = torch.tensor(drop_ignored(boxes.tolist(), bm, shape), dtype=torch.bool, device=pred["bboxes"].device)
                if bool(keep.all()):
                    continue
                self.ignored_conf.extend(pred["conf"][~keep].tolist())
                preds[si] = {k: (v[keep] if hasattr(v, "shape") and len(v) == len(keep) else v) for k, v in pred.items()}
            super().update_metrics(preds, batch)

    return IgnoreValidator


def ignore_summary(validator_cls, conf: float) -> dict:
    """기록에 남길 '채점에서 뺀 것' — 뺄 자리 그림이 있던 장수와, conf 이상에서 뺀 예측 수."""
    v = getattr(validator_cls, "last", None)
    if v is None:
        return {}
    return {"images": v.ignore_images, "preds_at_conf": sum(1 for c in v.ignored_conf if c >= conf)}


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
      클래스별 `false` 도 달라진다(합은 같다) — 11-15 에 같은 PC·torch 에서 버전만 바꿔 재 보니 달라진 것은 이것들뿐이었다.
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

    import torch
    import ultralytics
    from ultralytics import YOLO

    pin_confusion_thresholds(args.conf, args.match_iou)
    validator = ignore_validator()
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
                                name=f"{wp.stem}_{name}_{sz}", exist_ok=True, validator=validator)
                rec = {"weights": wp.name, "sha256": digest, "set": name, "split": args.split, "imgsz": sz,
                       "seconds": round(time.time() - t0, 1), "note": args.note, "batch": args.batch,
                       "ultralytics": ultralytics.__version__, "torch": torch.__version__,
                       **summarize(res, model.names, args.conf, args.match_iou),
                       "ignore": ignore_summary(validator, args.conf)}
                with open(out, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                a = rec.get("at_conf", {})
                print(f"RESULT {wp.name} {name} imgsz={sz} map={rec['map']} map50={rec['map50']} "
                      f"hit={a.get('hit')} missed={a.get('missed')} false={a.get('false')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
