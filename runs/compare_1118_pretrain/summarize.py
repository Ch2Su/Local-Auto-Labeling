"""results.jsonl·sweep.jsonl·실행별 results.csv 를 읽어 갈래별 표를 찍는다(3묶음 합계) — docs/experiments.md 에 옮겨 적을 수치.

이 파일 옆에 results.jsonl·sweep.jsonl 이, 곡선용 csv 는 runs/<이름>/results.csv 또는 <이름>_results.csv 로 있으면 된다.
  python -I summarize.py
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAMES = ["car", "truck", "pickup", "bus", "person", "responder", "cone", "scooter", "bicycle", "motorcycle"]
ARMS = [("A", 0), ("A", 1), ("B", 0), ("C", 0), ("O", 0), ("OM", 0), ("P2N", 0), ("P2S", 0), ("P2S", 1),
        ("F", 0), ("FZ11", 0), ("FZ23", 0)]
BINS = ("<8", "8-16", "16-32", "32+")


def load(name):
    p = HERE / name
    return [json.loads(x) for x in p.read_text(encoding="utf-8").split("\n") if x.strip()] if p.exists() else []


def total(cs):
    t, f, k, x = Counter(), Counter(), Counter(), 0
    for c in cs:
        t.update(c["total"]); f.update(c["found"]); k.update(c["klass"]); x += sum(c["false"].values())
    return t, f, k, x


def curve(name):
    for p in (HERE / "runs" / name / "results.csv", HERE / f"{name}_results.csv"):
        if p.exists():
            rows = list(csv.DictReader(open(p, newline="", encoding="utf-8")))
            key = next(k for k in rows[0] if k.strip() == "metrics/mAP50-95(B)")
            return [float(r[key]) for r in rows]
    return []


def main() -> int:
    res = [r for r in load("results.jsonl") if r["epochs"] == 100]
    print("## 3묶음 합계 (마지막 가중치 last.pt, 640, conf 0.25)")
    print("| 갈래 | seed | mAP50-95 묶음별 | 평균 | 찾음 | 클래스까지 | 헛 박스 | best.pt 평균 mAP | 실행당 초(중앙값) |")
    print("|---|---|---|---|---|---|---|---|---|")
    per_class, curves = {}, {}
    for arm, seed in ARMS:
        rs = sorted([r for r in res if r["arm"] == arm and r.get("seed", 0) == seed], key=lambda r: r["fold"])
        if len(rs) != 3:
            print(f"| {arm} | {seed} | (묶음 {len(rs)}개뿐) |")
            continue
        t, f, k, x = total([r["last"] for r in rs])
        maps = [r["last"]["map"] for r in rs]
        bmaps = [r["best"]["map"] for r in rs]
        secs = sorted(r["train_seconds"] for r in rs)
        print(f"| {arm} | {seed} | {' · '.join(f'{m:.3f}' for m in maps)} | {sum(maps) / 3:.3f} | {sum(f.values())} | "
              f"{sum(k.values())} | {x} | {sum(bmaps) / 3:.3f} | {secs[1]:.0f} |")
        per_class[(arm, seed)] = (t, f, k)
        cv = [curve(r["name"]) for r in rs]
        if all(len(c) >= 100 for c in cv):
            curves[(arm, seed)] = [sum(c[e - 1] for c in cv) / 3 for e in (1, 10, 25, 50, 100)] + \
                                  [sum(max(c) for c in cv) / 3, [c.index(max(c)) + 1 for c in cv]]
    print("\n## 클래스별 찾음/클래스까지 (정답 수는 머리글)")
    t0 = next(iter(per_class.values()))[0]
    print("| 갈래 | seed | " + " | ".join(f"{n} ({t0.get(n, 0)})" for n in NAMES) + " |")
    print("|---|---|" + "---|" * len(NAMES))
    for (arm, seed), (t, f, k) in per_class.items():
        print(f"| {arm} | {seed} | " + " | ".join(f"{f.get(n, 0)}/{k.get(n, 0)}" for n in NAMES) + " |")
    print("\n## 검증 mAP50-95 곡선 (3묶음 평균)")
    print("| 갈래 | seed | 1 epoch | 10 | 25 | 50 | 100 | 묶음별 최댓값의 평균 | 최댓값이 나온 epoch |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (arm, seed), v in curves.items():
        print(f"| {arm} | {seed} | " + " | ".join(f"{x:.3f}" for x in v[:6]) + f" | {v[6]} |")
    sw = load("sweep.jsonl")
    if sw:
        print("\n## conf 문턱별 찾음/헛 박스, conf 0.25 에서 정답 크기별 찾은 수 (640)")
        sets = []
        for r in sw:
            if r["set"] not in sets:
                sets.append(r["set"])
        tot = next(r for r in sw if r["conf"] == 0.25)["size"]
        print("| 세트 | 0.25 | 0.35 | 0.5 | " + " | ".join(f"{b}px ({tot[b][1]})" for b in BINS) + " |")
        print("|---|---|---|---|" + "---|" * len(BINS))
        for s in sets:
            row = {r["conf"]: r for r in sw if r["set"] == s and r["imgsz"] == 640}       # 같은 세트를 두 번 쟀으면 뒤의 것
            if len(row) < 3:
                continue
            print(f"| {s} | " + " | ".join(f"{row[c]['found']} / {row[c]['false']}" for c in (0.25, 0.35, 0.5)) + " | " +
                  " | ".join(str(row[0.25]["size"][b][0]) for b in BINS) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
