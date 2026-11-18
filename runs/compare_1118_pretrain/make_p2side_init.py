"""P2 곁가지 구조(p2/yolo26-p2side.yaml)에 yolo26m.pt 의 사전학습 가중치를 **역할대로** 얹은 시작 체크포인트를 만든다.

- 0–22층: 이름 그대로(번호·모양이 같다).
- Detect(새 26층 ← 옛 23층): 옛 가지 0·1·2(P3·P4·P5)를 새 가지 1·2·3 으로 — box·cls·one2one 전부, 마지막 conv 까지.
- 새 가지 0(P2): 안쪽 conv 는 옛 P3 가지를 복사해 시작하고 box 마지막 conv 도 복사한다. **cls 마지막 conv 는 새로 초기화한
  값 그대로**(bias 가 아주 낮아 처음에는 P2 에서 아무것도 내지 않는다).
- 23–25층(P2 특징): 새로 초기화.

그 뒤 사전학습 모델과 예측이 같은지 확인한다(P3–P5 는 같은 계산이고 P2 는 조용해야 한다).
"""
import re
import sys

import torch

import cmp_common as cc

HEADS = ("cv2", "cv3", "one2one_cv2", "one2one_cv3")
OLD_DET, NEW_DET = 23, 26


def main() -> int:
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    src = cc.REPO / "weights" / "yolo26m.pt"
    out = cc.ROOT / "p2" / "p2side_init.pt"
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    key = "ema" if ckpt.get("ema") is not None else "model"
    base = ckpt[key].float()
    bsd = base.state_dict()

    torch.manual_seed(0)
    new = DetectionModel(str(cc.ROOT / "p2" / "yolo26m-p2side.yaml"), ch=3, nc=base.yaml["nc"], verbose=False)
    nsd = new.state_dict()
    fresh = {k: v.clone() for k, v in nsd.items()}
    pat = re.compile(rf"^model\.{OLD_DET}\.({'|'.join(HEADS)})\.(\d+)\.(.+)$")
    put, stats = {}, {"same": 0, "shift": 0, "p2copy": 0}
    for k, v in bsd.items():
        idx = int(k.split(".")[1])
        if idx < OLD_DET:
            put[k] = v
            stats["same"] += 1
            continue
        m = pat.match(k)
        if not m:
            raise SystemExit(f"Detect 에서 모르는 텐서: {k}")
        attr, lvl, rest = m.group(1), int(m.group(2)), m.group(3)
        put[f"model.{NEW_DET}.{attr}.{lvl + 1}.{rest}"] = v
        stats["shift"] += 1
        if lvl == 0:
            last_cls = attr.endswith("cv3") and rest.split(".")[0] == "2"      # cls 마지막 conv 는 복사하지 않는다
            if not last_cls:
                put[f"model.{NEW_DET}.{attr}.0.{rest}"] = v
                stats["p2copy"] += 1
    bad = [k for k, v in put.items() if k not in nsd or nsd[k].shape != v.shape]
    if bad:
        raise SystemExit(f"모양이 안 맞는 텐서 {len(bad)}개 (예: {bad[0]})")
    nsd.update(put)
    new.load_state_dict(nsd)
    new.names = dict(base.names)
    untouched = [k for k in nsd if k not in put]
    n_new = sum(fresh[k].numel() for k in untouched if fresh[k].dtype.is_floating_point)
    n_all = sum(v.numel() for v in nsd.values() if v.dtype.is_floating_point)
    ckpt["model"], ckpt["ema"] = new.half(), None
    torch.save(ckpt, out)
    print(f"쓴 파일: {out.name} · 텐서: 그대로 {stats['same']} · Detect 가지 옮김 {stats['shift']} · P2 가지에 복사 {stats['p2copy']} · "
          f"새로 초기화 {len(untouched)}개 ({n_new / 1e6:.2f}M / {n_all / 1e6:.2f}M)")
    print("새로 초기화한 묶음:", sorted({'.'.join(k.split('.')[:2]) if int(k.split('.')[1]) != NEW_DET
                                   else '.'.join(k.split('.')[:5]) for k in untouched}))

    # 확인: 사전학습 모델과 같은 예측인가 (평가 묶음 0 의 앞 16장, 640, conf 0.25)
    files = [str(f) for f in cc.fold_files(0, "val")[:16]]
    a = YOLO(str(src)).predict(files, imgsz=640, conf=0.25, device=0, verbose=False)
    m2 = YOLO(str(out))
    b = m2.predict(files, imgsz=640, conf=0.25, device=0, verbose=False)
    n_a = sum(len(r.boxes) for r in a); n_b = sum(len(r.boxes) for r in b)
    worst = 0.0
    same_n = True
    for ra, rb in zip(a, b):
        if len(ra.boxes) != len(rb.boxes):
            same_n = False
            continue
        if len(ra.boxes):
            d = (ra.boxes.data.cpu() - rb.boxes.data.cpu()).abs().max().item()
            worst = max(worst, d)
    print(f"확인: 박스 수 {n_a} 대 {n_b} · 이미지마다 수가 같음 = {same_n} · 좌표·conf·클래스의 가장 큰 차이 {worst:.6f}")
    print("stride:", [int(s) for s in m2.model.stride], "· 파라미터:", f"{sum(p.numel() for p in m2.model.parameters()) / 1e6:.2f}M")
    return 0 if same_n else 1


if __name__ == "__main__":
    sys.exit(main())
