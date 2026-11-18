"""사전학습 비교 실험의 공용 함수 — 세트 경로, 정답 읽기, conf 0.25 에서 찾은·틀린 박스 세기."""
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = Path(r"C:\local_auto_labeling")
DS = ROOT / "set" / "dataset"
NAMES = ["bicycle", "bus", "car", "motorcycle", "person", "scooter", "truck", "pickup", "responder", "cone"]
SELF_OK = {n: {n} for n in NAMES}
# COCO 로만 학습한 모델의 클래스 -> 우리 클래스(맞다고 칠 것). responder·cone·scooter 는 COCO 에 없다.
COCO_OK = {"person": {"person"}, "bicycle": {"bicycle"}, "car": {"car"}, "motorcycle": {"motorcycle"},
           "bus": {"bus"}, "truck": {"truck", "pickup"}}
MATCH_IOU = 0.5
CONF = 0.25


def iou_one(box, others):
    x1 = np.maximum(box[0], others[:, 0]); y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2]); y2 = np.minimum(box[3], others[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    return inter / (a + b - inter + 1e-9)


def load_gt(img: Path, w: int, h: int):
    p = DS / "labels" / "test" / f"{img.stem}.txt"
    out = []
    if p.exists():
        for line in p.read_text().split("\n"):
            t = line.split()
            if len(t) >= 5:
                c, cx, cy, bw, bh = int(t[0]), *map(float, t[1:5])
                out.append((NAMES[c], (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def count(model, files, ok, imgsz, batch=8):
    """conf 순 탐욕 1:1 짝짓기(IoU 0.5). found = 자리만 맞음, klass = 클래스도 맞음, false = 정답 없는 예측.

    ok = {모델 클래스 이름: 그 예측이 맞다고 칠 우리 클래스들}. ok 에 없는 클래스의 예측은 세지 않는다(other).
    """
    total, found, klass, false = Counter(), Counter(), Counter(), Counter()
    other = 0
    files = [Path(f) for f in files]
    for i in range(0, len(files), batch):
        chunk = files[i:i + batch]
        res = model.predict([str(f) for f in chunk], imgsz=imgsz, conf=CONF, iou=0.6, max_det=300, device=0, verbose=False)
        for f, r in zip(chunk, res):
            h, w = r.orig_shape
            gts = load_gt(f, w, h)
            for g in gts:
                total[g[0]] += 1
            gt_xy = np.array([g[1:] for g in gts], dtype=float).reshape(-1, 4)
            used = np.zeros(len(gts), dtype=bool)
            b = r.boxes
            conf = b.conf.cpu().numpy(); xy = b.xyxy.cpu().numpy(); cls = b.cls.cpu().numpy().astype(int)
            for j in np.argsort(-conf):
                name = model.names[int(cls[j])]
                if name not in ok:
                    other += 1
                    continue
                if len(gts):
                    v = iou_one(xy[j], gt_xy)
                    v[used] = -1
                    k = int(v.argmax())
                    if v[k] >= MATCH_IOU:
                        used[k] = True
                        found[gts[k][0]] += 1
                        if gts[k][0] in ok[name]:
                            klass[gts[k][0]] += 1
                        continue
                false[name] += 1
    return {"total": dict(total), "found": dict(found), "klass": dict(klass), "false": dict(false), "other": other}


def brief(c: dict) -> str:
    n = sum(c["total"].values()) or 1
    f, k, x = sum(c["found"].values()), sum(c["klass"].values()), sum(c["false"].values())
    return f"정답 {n} · 찾음 {f} ({f / n:.1%}) · 클래스까지 {k} ({k / n:.1%}) · 헛 박스 {x}"


def fold_files(fold: int, part: str) -> list[Path]:
    return [Path(x) for x in (ROOT / "folds" / f"f{fold}_{part}.txt").read_text(encoding="utf-8").split("\n") if x.strip()]
