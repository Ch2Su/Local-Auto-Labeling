"""가중치 없이 SAM 3 텍스트 프롬프트 모델을 조립해 본다 — 의존 패키지가 다 있는지, 한 장에 시간·VRAM 이 얼마나 드는지.

체크포인트 읽기만 건너뛴다(임의 가중치) — 결과 박스는 뜻이 없고 계산량만 본다. 프로젝트 .venv 에는 아무것도 설치하지 않는다
(YOLO_AUTOINSTALL=false 로 돌릴 것 — 빠진 패키지가 있으면 설치하지 않고 멈춘다).
  python -I sam3_dry.py <별도 패키지 폴더> <이미지> [imgsz] [16]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, sys.argv[1])

import torch  # noqa: E402

import ultralytics.models.sam.build_sam3 as b3  # noqa: E402

b3._load_checkpoint = lambda model, checkpoint, interactive=False: model      # 체크포인트 없이 조립만

from ultralytics.models.sam import SAM3SemanticPredictor  # noqa: E402

NAMES = ["fire truck", "ambulance", "police car", "tow truck", "traffic cone"]


def main() -> int:
    img = sys.argv[2]
    imgsz = int(sys.argv[3]) if len(sys.argv) > 3 else 1008
    half = len(sys.argv) > 4 and sys.argv[4] == "16"
    import clip
    import ftfy
    import regex
    import timm
    print("패키지:", "timm", timm.__version__, "· ftfy", ftfy.__version__, "· regex", regex.__version__, "· clip", Path(clip.__file__).parent)
    ov = dict(conf=0.999, task="segment", mode="predict", model="sam3_dry.pt", save=False, verbose=False, device=0, imgsz=imgsz)
    if half:
        ov["quantize"] = 16
    pred = SAM3SemanticPredictor(overrides=ov)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    pred.set_image(img)
    torch.cuda.synchronize()
    t1 = time.time()
    n_par = sum(p.numel() for p in pred.model.parameters())
    print(f"모델 파라미터 {n_par / 1e6:.1f}M · 입력 크기 {pred.imgsz} · half {half} · dtype {next(pred.model.parameters()).dtype}")
    print(f"첫 장(모델 조립·CUDA 데우기 포함) {t1 - t0:.1f}초")
    res = pred(text=NAMES)
    torch.cuda.synchronize()
    t2 = time.time()
    print(f"프롬프트 {len(NAMES)}개 추론 {t2 - t1:.2f}초 · 결과 박스 {len(res[0].boxes)} · names {res[0].names}")
    # 둘째 장부터의 시간(같은 그림을 다시 — 인코딩 + 프롬프트)
    for k in range(2):
        t = time.time()
        pred.reset_image()
        pred.set_image(img)
        torch.cuda.synchronize()
        ta = time.time()
        pred(text=NAMES)
        torch.cuda.synchronize()
        tb = time.time()
        print(f"  다시 {k + 1}: 인코딩 {ta - t:.2f}초 · 프롬프트 {tb - ta:.2f}초")
    print(f"VRAM 최대 {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB (torch 가 잡은 양 {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
