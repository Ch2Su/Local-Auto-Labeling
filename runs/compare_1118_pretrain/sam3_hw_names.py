"""승인된 고속도로 프레임에서 SAM 3.1 후보를 이름별·문턱별로 센다(후보 / 맞음) — 읽기만.

  python -I sam3_hw_names.py <hw_sam31_vs_owl.jsonl>
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sam3_hw import greedy  # noqa: E402

CONFS = (0.5, 0.6, 0.7, 0.8, 0.9)
NAMES = ("fire truck", "ambulance", "police car", "tow truck", "traffic cone")


def main() -> int:
    recs = [json.loads(x) for x in Path(sys.argv[1]).read_text(encoding="utf-8").split("\n") if x.strip()]
    appr = [r for r in recs if r["status"] == "approved"]
    print(f"승인 {len(appr)}장 · 칸 = 후보 / 맞음")
    print("| 이름 | " + " | ".join(f"conf {c}" for c in CONFS) + " |")
    print("|---|" + "---|" * len(CONFS))
    rows = defaultdict(list)
    for conf in CONFS:
        c = defaultdict(Counter)
        for r in appr:
            name_of = {tuple(x[:6]): x[6] for x in r["sam"]}
            for cand, k in greedy([tuple(x[:6]) for x in r["sam"] if x[1] >= conf], [tuple(g) for g in r["gt"]]):
                nm = name_of[cand]
                c[nm]["n"] += 1
                c[nm]["hit"] += k is not None and r["gt"][k][0] == cand[0]
        for nm in NAMES:
            rows[nm].append(f"{c[nm]['n']} / {c[nm]['hit']}")
    for nm in NAMES:
        print(f"| {nm} | " + " | ".join(rows[nm]) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
