"""갈래마다 conf 0.25 에서 '찾았는데 클래스가 틀린 것' 이 어느 클래스로 갔는지와 헛 박스의 예측 클래스를 센다(추론만).

짝짓기는 cmp_common.count 와 같다(conf 순 탐욕 1:1, IoU 0.5). 세트 이름은 sweep_eval.py 와 같은 '갈래[:epoch][@seed]' 꼴이고
'@' 앞이 같으면 seed 평균도 찍는다. 출력에 파일 이름·카메라 키는 없다.
  python -I confusion_eval.py --sets FZ11R:25,FZ11R:25@1,FZ11U:25
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cmp_common as cc  # noqa: E402


def weights(name: str, fold: int) -> Path:
    if name == "operating":
        return cc.REPO / "models" / "best_y26m_8cls.pt"
    arm, _, seed = name.partition("@")
    arm, _, ep = arm.partition(":")
    return cc.ROOT / "runs" / (f"{arm}_f{fold}_e{ep or 100}" + (f"_s{seed}" if seed else "")) / "weights" / "last.pt"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()
    from ultralytics import YOLO

    res = {}
    for name in args.sets.split(","):
        mix, false = Counter(), Counter()
        for fold in (0, 1, 2):
            model = YOLO(str(weights(name, fold)))
            files = cc.fold_files(fold, "val")
            for i in range(0, len(files), 8):
                chunk = files[i:i + 8]
                out = model.predict([str(f) for f in chunk], imgsz=args.imgsz, conf=cc.CONF, iou=0.6, max_det=300, device=0, verbose=False)
                for f, r in zip(chunk, out):
                    h, w = r.orig_shape
                    gts = cc.load_gt(f, w, h)
                    gt_xy = np.array([g[1:] for g in gts], dtype=float).reshape(-1, 4)
                    used = np.zeros(len(gts), dtype=bool)
                    b = r.boxes
                    conf = b.conf.cpu().numpy(); xy = b.xyxy.cpu().numpy(); cls = b.cls.cpu().numpy().astype(int)
                    for j in np.argsort(-conf):
                        pred = model.names[int(cls[j])]
                        if len(gts):
                            v = cc.iou_one(xy[j], gt_xy)
                            v[used] = -1
                            k = int(v.argmax())
                            if v[k] >= cc.MATCH_IOU:
                                used[k] = True
                                if gts[k][0] != pred:
                                    mix[(gts[k][0], pred)] += 1
                                continue
                        false[pred] += 1
        res[name] = (mix, false)
        print(f"{name}: 클래스 틀림 {sum(mix.values())} — " + ", ".join(f"{a}→{b} {n}" for (a, b), n in mix.most_common(10)) +
              f" · 헛 박스 {sum(false.values())} — " + ", ".join(f"{k} {n}" for k, n in false.most_common(6)), flush=True)
    arms = defaultdict(list)
    for name in res:
        arms[name.partition("@")[0]].append(name)
    print("\n## seed 평균 — 정답 클래스 → 예측 클래스(틀린 것), responder 가 낀 것과 많은 것")
    keys = Counter()
    for mix, _ in res.values():
        keys.update(mix)
    show = [k for k, _ in keys.most_common() if "responder" in k][:8] + [k for k, _ in keys.most_common(8) if "responder" not in k][:6]
    print("| 갈래 | seed 수 | " + " | ".join(f"{a}→{b}" for a, b in show) + " | 헛 박스 중 responder | 헛 박스 전체 |")
    print("|---|---|" + "---|" * (len(show) + 2))
    for arm, names in arms.items():
        n = len(names)
        cells = [sum(res[x][0][k] for x in names) / n for k in show]
        fr = sum(res[x][1]["responder"] for x in names) / n
        fa = sum(sum(res[x][1].values()) for x in names) / n
        print(f"| {arm} | {n} | " + " | ".join(f"{c:.1f}" for c in cells) + f" | {fr:.1f} | {fa:.1f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
