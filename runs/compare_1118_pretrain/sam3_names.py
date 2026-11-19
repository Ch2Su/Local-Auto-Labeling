"""이름(프롬프트)마다 문턱을 올려 가며 후보 수·맞은 수를 본다 — sam3_propose.py 의 jsonl 을 읽기만.

  python -I sam3_names.py <후보.jsonl>
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sam3_analyze import match  # noqa: E402
from sam3_propose import TO_CLASS  # noqa: E402

CONFS = (0.25, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9)


def main() -> int:
    recs = [json.loads(x) for x in Path(sys.argv[1]).read_text(encoding="utf-8").split("\n") if x.strip()]
    print("| 이름 | " + " | ".join(f"conf {c}" for c in CONFS) + " |   (칸 = 후보 수 / 그중 맞은 수)")
    print("|---|" + "---|" * len(CONFS))
    rows = defaultdict(list)
    for conf in CONFS:
        c = defaultdict(Counter)
        for r in recs:
            for d, k in match(r, conf):
                c[d[0]]["n"] += 1
                c[d[0]]["hit"] += k is not None and r["gt"][k][0] == TO_CLASS[d[0]]
        for name in TO_CLASS:
            rows[name].append(f"{c[name]['n']} / {c[name]['hit']}")
    for name in TO_CLASS:
        print(f"| {name} | " + " | ".join(rows[name]) + " |")
    # 이름 둘(fire truck·ambulance)만 썼다면 — 같은 물체에 다른 이름이 먼저 붙어 가려진 것은 없는지 따로 센다
    for keep in (("fire truck", "ambulance"), ("fire truck", "ambulance", "tow truck"), ("fire truck", "ambulance", "police car")):
        line = []
        for conf in (0.25, 0.5):
            sub = [{"gt": r["gt"], "dets": [d for d in r["dets"] if d[0] in keep]} for r in recs]
            n = hit = 0
            for r in sub:
                for d, k in match(r, conf):
                    n += 1
                    hit += k is not None and r["gt"][k][0] == "responder"
            line.append(f"conf {conf}: 후보 {n} · 맞음 {hit}/36")
        print(f"  {' + '.join(keep)} 만: " + " · ".join(line))
    return 0


if __name__ == "__main__":
    sys.exit(main())
