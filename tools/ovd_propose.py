"""OWLv2(이름만 주면 찾는 검출기)로 출동 차량·라바콘 후보를 찾아 JSONL 로 쓴다 — `tools/import_proposals.py` 가 읽어 검수 대기로 넣는다.

**왜 따로 도나 (2026-11-09).** OWLv2 는 transformers 가 필요해 이 저장소의 .venv(Ultralytics)와 가상 환경을 나눈다. 그래서 이
스크립트는 `autolabel` 을 import 하지 않고 DB 도 열지 않는다 — `import_proposals.py list` 가 쓴 이미지 목록(TSV: id·경로)을 받아
후보만 쓴다. 이 PC 의 가상 환경은 `E:\\OpenVocab\\.venv`(python 3.12 · torch 2.10 cu128 · transformers 5.5.3 · pillow),
모델 캐시는 `HF_HOME=E:\\OpenVocab\\hf`.

    $env:HF_HOME='E:\\OpenVocab\\hf'; $env:HF_HUB_OFFLINE='1'
    E:\\OpenVocab\\.venv\\Scripts\\python.exe tools\\ovd_propose.py runs\\ovd\\<묶음>\\images.tsv runs\\ovd\\<묶음>\\owl.jsonl

**모델.** `google/owlv2-base-patch16-ensemble`(Apache-2.0). 아래 이름 9개를 함께 물어 상자마다 가장 높은 이름 하나만 남긴다 —
일반 car·truck·bus·person 을 같이 물어야 '그냥 트럭'이 견인차로 남는 일이 준다. 점수 `--threshold`(기본 0.15) 이상 상자 가운데
새 클래스로 갈 이름(소방차·구급차·경찰차·견인차·라바콘)만 쓴다. 우리 클래스로의 대응과 문턱은 import_proposals.py 가 정한다.

**시험 (11-09, DetectView 쪽 세션).** 고속도로 사고 영상 6개(11-07 녹화, 1080×720, 154장): 소방차는 잘 찾고(hw-06·hw-08 거의 매
프레임), 분명한 라바콘은 0.5~0.87. 견인차 이름으로 노란 화살표판 차도 잡는다(어차피 출동 차량). 경찰차는 흰 SUV·택시를 0.4~0.57 로
잘못 잡고 구급차는 거의 못 찾는다. 학습 데이터 백업 6,535장: 라바콘 후보는 대부분 진짜, 경찰차 후보는 대부분 S.RIDE 택시 —
**검수 없이 쓰면 안 된다.** RTX 5060 에서 한 장 0.25초 안팎(첫 장은 CUDA 데우기로 15초).

**좌표.** EXIF 회전을 적용한 화면 기준 픽셀 xyxy 와 그 화면의 폭·높이를 쓴다 — DB 라벨과 같은 기준(CLAUDE.md 'EXIF 회전 이미지').
OWLv2 는 긴 변에 맞춘 정사각형으로 채워(오른쪽·아래) 보므로 결과를 (긴 변, 긴 변) 크기로 되돌리면 원본 픽셀이 된다.

이미 쓴 id 는 건너뛴다(다시 돌리면 이어서). 읽지 못한 이미지는 error 칸을 남기고 넘어간다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

MODEL = "google/owlv2-base-patch16-ensemble"
QUERIES = ["fire truck", "ambulance", "police car", "tow truck", "traffic cone", "person", "car", "truck", "bus"]
KEEP = ("fire truck", "ambulance", "police car", "tow truck", "traffic cone")


def read_list(path: Path) -> list[tuple[int, str]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        image_id, p = line.split("\t")[:2]
        out.append((int(image_id), p))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("images", type=Path, help="import_proposals.py list 가 쓴 TSV(id·경로)")
    ap.add_argument("out", type=Path, help="쓸 JSONL — 있으면 이어서")
    ap.add_argument("--threshold", type=float, default=0.15, help="이 점수 이상만 쓴다(기본 0.15)")
    ap.add_argument("--device", default="cuda", help="cuda 또는 cpu (학습이 돌고 있으면 GPU 를 쓰지 말 것)")
    a = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")

    import torch
    from PIL import Image, ImageOps
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    todo = read_list(a.images)
    done = set()
    if a.out.exists():
        for line in a.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    todo = [(i, p) for i, p in todo if i not in done]
    print(f"이미지 {len(todo) + len(done):,}장 · 이미 한 것 {len(done):,} · 할 것 {len(todo):,}", flush=True)
    if not todo:
        return 0
    t0 = time.time()
    proc = Owlv2Processor.from_pretrained(MODEL)
    model = Owlv2ForObjectDetection.from_pretrained(MODEL).to(a.device).eval()
    print(f"모델 올림 {time.time() - t0:.1f}초 ({a.device})", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    t1 = time.time()
    with a.out.open("a", encoding="utf-8") as f:
        for n, (image_id, p) in enumerate(todo, 1):
            rec = {"id": image_id, "path": p, "model": MODEL}
            try:
                with Image.open(p) as im:
                    img = ImageOps.exif_transpose(im).convert("RGB")
                w, h = img.size
                s = max(w, h)
                with torch.no_grad():
                    inp = proc(text=[QUERIES], images=img, return_tensors="pt").to(a.device)
                    r = proc.post_process_grounded_object_detection(
                        outputs=model(**inp), threshold=a.threshold,
                        target_sizes=torch.tensor([[s, s]]), text_labels=[QUERIES])[0]
                rec.update(w=w, h=h, boxes=[
                    {"label": lab, "score": round(float(sc), 4), "xyxy": [round(float(v), 1) for v in b]}
                    for b, sc, lab in zip(r["boxes"].tolist(), r["scores"].tolist(), r["text_labels"]) if lab in KEEP])
            except Exception as e:   # 한 장이 깨져도 나머지는 돈다
                rec["error"] = f"{type(e).__name__}: {e}"[:300]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if n % 200 == 0:
                f.flush()
                print(f"{n:,}/{len(todo):,}장 · {time.time() - t1:.0f}초", flush=True)
    print(f"끝 — {len(todo):,}장 · {time.time() - t1:.0f}초 → {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
