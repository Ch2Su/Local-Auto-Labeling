"""Objects365 v1 사전학습 가중치(dl_o365/yolo26m-objv1-150.pt)의 클래스 이름을 우리 이름에 맞춘 사본을 만든다.

Ultralytics 8.4.212 는 미세조정을 시작할 때 **이름이 같은 클래스의 분류 출력 줄**을 옮긴다(_remap_cls_by_names). 그대로 두면
person·bicycle·car·motorcycle·bus·truck 6줄만 옮겨지고, 이름이 다른 pickup truck·traffic cone·fire truck·ambulance 는 버려진다.
여기서는 가중치 값은 그대로 두고 이름만 바꾼다:
  pickup truck -> pickup, traffic cone -> cone, ambulance -> responder
responder 는 한 줄만 받을 수 있어, ambulance 줄을 ambulance·fire truck 두 줄의 평균으로 바꿔 둔다(분류 마지막 conv 의 가중치·bias).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cmp_common as cc  # noqa: E402

RENAME = {"pickup truck": "pickup", "traffic cone": "cone", "ambulance": "responder"}
MERGE = ("ambulance", "fire truck")          # 앞의 줄 := 두 줄의 평균


def main() -> int:
    src = cc.ROOT / "dl_o365" / "yolo26m-objv1-150.pt"
    out = cc.ROOT / "o365_mapped.pt"
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    done = 0
    for key in ("ema", "model"):
        model = ckpt.get(key)
        if not isinstance(model, torch.nn.Module):
            continue
        idx = {str(v).strip().lower(): k for k, v in model.names.items()}
        a, b = idx[MERGE[0]], idx[MERGE[1]]
        head = model.model[-1]
        n = 0
        for attr in ("cv3", "one2one_cv3"):
            for seq in getattr(head, attr):
                conv = seq[-1]
                with torch.no_grad():
                    conv.weight[a] = (conv.weight[a].float() + conv.weight[b].float()).div(2).to(conv.weight.dtype)
                    conv.bias[a] = (conv.bias[a].float() + conv.bias[b].float()).div(2).to(conv.bias.dtype)
                n += 1
        names = dict(model.names)
        for old, new in RENAME.items():
            names[idx[old]] = new
        model.names = names
        done += 1
        print(f"{key}: 평균으로 바꾼 분류 conv {n}개 · 이름 바꿈 {RENAME}")
    if not done:
        raise SystemExit("체크포인트에 모델이 없다")
    torch.save(ckpt, out)
    back = torch.load(out, map_location="cpu", weights_only=False)
    names = (back.get("ema") or back["model"]).names
    ours = {n: [k for k, v in names.items() if str(v).strip().lower() == n] for n in cc.NAMES}
    print("우리 클래스 이름과 같은 줄:", ours)
    print(f"쓴 파일: {out.name} · 맞는 클래스 {sum(1 for v in ours.values() if v)}/10")
    return 0


if __name__ == "__main__":
    sys.exit(main())
