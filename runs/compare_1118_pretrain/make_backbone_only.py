"""yolo26m.pt 에서 backbone(0–10층)만 사전학습 값으로 남기고 neck·Detect(11층부터)는 새로 초기화한 체크포인트를 만든다.

'backbone 만 사전학습' 갈래가 '전부 사전학습' 갈래와 같은 길(.pt 에서 얹기)로 학습되게 하려는 것이다.
"""
import sys

import torch

import cmp_common as cc

LAST_BACKBONE = 10


def main() -> int:
    from ultralytics.nn.tasks import DetectionModel

    src = cc.REPO / "weights" / "yolo26m.pt"
    out = cc.ROOT / "backbone_only.pt"
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    key = "ema" if ckpt.get("ema") is not None else "model"
    model = ckpt[key]
    pre = {k: v.clone() for k, v in model.state_dict().items()}
    torch.manual_seed(0)
    fresh = DetectionModel(model.yaml, ch=3, nc=model.yaml["nc"], verbose=False)
    fsd = fresh.state_dict()
    if set(fsd) != set(pre):
        raise SystemExit(f"새로 만든 망과 체크포인트의 텐서 이름이 다르다: {len(set(fsd) ^ set(pre))}개")
    new, kept, reset = {}, 0, 0
    for k, v in pre.items():
        if int(k.split(".")[1]) <= LAST_BACKBONE:
            new[k] = v
            kept += v.numel()
        else:
            if fsd[k].shape != v.shape:
                raise SystemExit(f"모양이 다르다: {k}")
            new[k] = fsd[k].to(v.dtype)
            reset += v.numel()
    model.load_state_dict(new)
    torch.save(ckpt, out)

    # 확인: 다시 읽어 backbone 은 같고 그 뒤는 다른가
    again = torch.load(out, map_location="cpu", weights_only=False)[key].state_dict()
    same_bb = all(torch.equal(again[k], pre[k]) for k in pre if int(k.split(".")[1]) <= LAST_BACKBONE)
    w_keys = [k for k in pre if int(k.split(".")[1]) > LAST_BACKBONE and k.endswith("conv.weight")]
    diff_rest = sum(not torch.equal(again[k], pre[k]) for k in w_keys)
    print(f"쓴 파일: {out.name} (키 '{key}') · 남긴 값 {kept / 1e6:.2f}M · 새로 초기화 {reset / 1e6:.2f}M")
    print(f"확인: backbone 텐서가 원본과 같음 = {same_bb} · 11층부터의 conv 가중치 {len(w_keys)}개 중 원본과 다른 것 {diff_rest}개")
    return 0 if same_bb and diff_rest == len(w_keys) else 1


if __name__ == "__main__":
    sys.exit(main())
