"""freeze=N 이 실제로 무엇을 고정하는지 CPU 에서 아주 짧게(1 epoch·몇 배치) 돌려 확인한다 — GPU 는 다른 학습이 쓰는 중이다.

보는 것: 학습되는 파라미터 수, 얼린 층의 가중치·BatchNorm 통계가 출발 가중치와 같은지, 안 얼린 층은 바뀌었는지.
  python -I -B smoke_freeze.py --freeze 23
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cmp_common as cc  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", type=int, required=True)
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    os.chdir(cc.ROOT / "work")
    src = cc.REPO / "models" / "best_y26m_8cls.pt"
    out = cc.ROOT / "runs_smoke"
    name = f"FZ{args.freeze}"
    start = YOLO(str(src)).model.state_dict()
    model = YOLO(str(src))
    model.train(data=str(cc.ROOT / "folds" / "f0.yaml"), epochs=1, batch=4, imgsz=320, device="cpu", workers=0, seed=0,
                fraction=0.1, project=str(out), name=name, exist_ok=True, plots=False, pretrained=True, amp=False,
                cos_lr=True, cache=False, box=7.5, cls=1.5, dfl=1.5, close_mosaic=10, save_period=-1,
                freeze=args.freeze)
    tm = model.trainer.model
    total = sum(p.numel() for p in tm.parameters())
    train = sum(p.numel() for p in tm.parameters() if p.requires_grad)
    print(f"SMOKE freeze={args.freeze}: 파라미터 {total:,} 중 학습되는 것 {train:,} ({train / total:.1%})")
    end = tm.state_dict()                      # EMA 가 아니라 학습된 모델 그 자체
    same = diff = 0
    first_changed = None
    bn_frozen_same = bn_frozen_diff = 0
    for k, v in end.items():
        if k not in start or start[k].shape != v.shape or not v.dtype.is_floating_point:
            continue
        layer = int(k.split(".")[1])
        equal = torch.equal(start[k].float().cpu(), v.float().cpu())
        if layer < args.freeze:
            same += equal
            diff += (not equal)
            if "running_" in k:
                bn_frozen_same += equal
                bn_frozen_diff += (not equal)
        elif not equal and first_changed is None:
            first_changed = k
    print(f"SMOKE 얼린 층(0..{args.freeze - 1})의 텐서: 그대로 {same} · 바뀜 {diff}  (그중 BatchNorm 통계: 그대로 {bn_frozen_same} · 바뀜 {bn_frozen_diff})")
    print(f"SMOKE 안 얼린 쪽에서 처음 바뀐 텐서: {first_changed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
