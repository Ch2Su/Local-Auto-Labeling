"""이름마다 문턱을 다르게 줬을 때의 후보 수·맞은 수 — 두 결과 파일(175장 세트, 고속도로 프레임)을 읽기만 한다.

  python -I sam3_mixed.py <sam31_full.jsonl> <hw_sam31_vs_owl.jsonl>
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sam3_hw import greedy  # noqa: E402
from sam3_propose import TO_CLASS  # noqa: E402

SETS = {"전부 0.5": dict.fromkeys(TO_CLASS, 0.5),
        "소방·구급·라바콘 0.5 / 경찰·견인 0.7": {"fire truck": 0.5, "ambulance": 0.5, "traffic cone": 0.5, "police car": 0.7, "tow truck": 0.7},
        "소방·구급·라바콘 0.5 / 경찰·견인 0.8": {"fire truck": 0.5, "ambulance": 0.5, "traffic cone": 0.5, "police car": 0.8, "tow truck": 0.8},
        "소방·구급 0.5 / 라바콘 0.6 / 경찰·견인 0.8": {"fire truck": 0.5, "ambulance": 0.5, "traffic cone": 0.6, "police car": 0.8, "tow truck": 0.8}}


def run(title, items):
    """items = [(후보 [(cls, conf, x1, y1, x2, y2, name)], 정답 [(cls, x1, y1, x2, y2)])]"""
    tot = Counter(g[0] for _, gts in items for g in gts)
    print(f"== {title} — 정답 responder {tot['responder']} · cone {tot['cone']} ==")
    print("| 문턱 | responder 찾음 | 후보 | 틀린 후보 | cone 찾음 | 후보 | 틀린 후보 |")
    print("|---|---|---|---|---|---|---|")
    for label, th in SETS.items():
        hit, n = Counter(), Counter()
        for cands, gts in items:
            keep = [tuple(c[:6]) for c in cands if c[1] >= th[c[6]]]
            for c, k in greedy(keep, gts):
                n[c[0]] += 1
                hit[c[0]] += k is not None and gts[k][0] == c[0]
        print(f"| {label} | {hit['responder']}/{tot['responder']} | {n['responder']} | {n['responder'] - hit['responder']} | "
              f"{hit['cone']}/{tot['cone']} | {n['cone']} | {n['cone'] - hit['cone']} |")


def main() -> int:
    a = [json.loads(x) for x in Path(sys.argv[1]).read_text(encoding="utf-8").split("\n") if x.strip()]
    run("지금 데이터 175장", [([(TO_CLASS[d[0]], d[1], *d[2:], d[0]) for d in r["dets"]], [tuple(g) for g in r["gt"]]) for r in a])
    b = [json.loads(x) for x in Path(sys.argv[2]).read_text(encoding="utf-8").split("\n") if x.strip()]
    run("옛 고속도로 프레임 중 승인된 것", [([tuple(x) for x in r["sam"]], [tuple(g) for g in r["gt"]]) for r in b if r["status"] == "approved"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
