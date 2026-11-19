"""옛 이미지와 묶음별 검증 이미지의 dHash 거리를 본다 — 같은 카메라 화면이 섞였는지 가늠하려고(읽기만, 로컬 출력).

  python -I near_check.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cmp_common as cc  # noqa: E402
from make_extra import dhash  # noqa: E402


def main() -> int:
    index = [json.loads(x) for x in (cc.ROOT / "extra" / "index.jsonl").read_text(encoding="utf-8").split("\n") if x.strip()]
    for fold in (0, 1, 2):
        val = cc.fold_files(fold, "val")
        vh = [(v.name.split("_")[0], dhash(v)) for v in val]
        hist = Counter()
        pairs = Counter()
        for x in index:
            cam, d = min(((c, bin(x["dhash"] ^ h).count("1")) for c, h in vh), key=lambda t: t[1])
            hist["≤6" if d <= 6 else "7-10" if d <= 10 else "11-14" if d <= 14 else "15+"] += 1
            if d <= 14:
                pairs[(x["camera"] or ("크롭:" + "/".join(x["tags"][:2]) if x["tags"] else "크롭:(태그 없음)"), cam, "≤10" if d <= 10 else "11-14")] += 1
        print(f"묶음 {fold}: 가장 가까운 검증 이미지와의 거리 분포 {dict(hist)}")
        for k, n in pairs.most_common(14):
            print("    ", n, k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
