"""검수 서버가 쓰는 `Segmenter.segment` 를 bench.py 와 같은 그림·프롬프트로 불러, 측정의 thybrid(64) 결과와 같은지 본다.

segment 는 정규화 박스를 돌려주므로 원본 px 로 되돌려 IoU 를 센다(측정은 소수 넷째 자리로 적었다 — 그 안에서 같아야 한다).
segment 는 잘라 넣은 뒤 다음 클릭에서 통째 그림을 다시 인코딩한다(측정은 통째 묻기를 먼저 몰아서 했다) — 결과는 같아야 한다.

  .venv\\Scripts\\python.exe -I runs\\sam_click_bench\\check_segmenter.py [--limit N]
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from bench import PROMPTS, iou, load_set, prompts_for  # noqa: E402

from autolabel.config import Config  # noqa: E402
from autolabel.segmenter import SegmenterUnavailable, Segmenter  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ref", default=str(HERE / "out" / "thybrid_s64.jsonl.gz"))
    args = ap.parse_args()
    ref = {(r["id"], r["k"]): r for r in map(json.loads, gzip.open(args.ref, "rt", encoding="utf-8"))}
    cfg = Config.load(ROOT / "config.yaml")
    seg = Segmenter(cfg)
    print("crop_below", seg.crop_below, "crop_size", seg.crop_size, "model", seg.model_name, flush=True)
    data, _ = load_set(ROOT / "data" / "visible" / "labeling.db", args.limit)
    n = same = 0
    worst = []
    t0 = time.perf_counter()
    for i, d in enumerate(data, 1):
        W, H = d["W"], d["H"]
        for b in d["boxes"]:
            p = prompts_for(d["id"], b["k"], b["gt"], W, H)
            r = ref[(d["id"], b["k"])]
            for kind in PROMPTS:
                point = p[kind] if kind != "rough" else None
                box = p[kind] if kind == "rough" else None
                try:
                    out, _ = seg.segment(Path(d["path"]), point=point, box=box, cls=b["cls"])
                    xyxy = ((out.cx - out.w / 2) * W, (out.cy - out.h / 2) * H, (out.cx + out.w / 2) * W, (out.cy + out.h / 2) * H)
                    got = round(iou(xyxy, b["gt"]), 4)
                except SegmenterUnavailable:
                    got = None
                want = r[kind]
                n += 1
                if got == want or (got is not None and want is not None and abs(got - want) <= 1e-4):
                    same += 1
                else:
                    worst.append((d["id"], b["k"], kind, want, got))
        if i % 40 == 0 or i == len(data):
            print(f"  {i}/{len(data)}장 · 같음 {same}/{n} · {time.perf_counter() - t0:.0f}초", flush=True)
    print(json.dumps({"prompts": n, "same": same, "differ": len(worst), "examples": worst[:10]}, ensure_ascii=False))
    return 0 if not worst else 1


if __name__ == "__main__":
    sys.exit(main())
