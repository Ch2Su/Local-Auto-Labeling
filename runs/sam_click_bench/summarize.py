"""bench.py 결과 요약 — 갈래·프롬프트별 IoU, 크기·출처별, t 와의 짝 차이(그림 단위 부트스트랩 95%).

  .venv\\Scripts\\python.exe -I runs\\sam_click_bench\\summarize.py [갈래 …]   → 표를 찍고 out/summary.json 에 쓴다
"""
from __future__ import annotations

import gzip
import json
import random
import sys
from pathlib import Path
from statistics import mean, median

OUT = Path(__file__).resolve().parent / "out"
PROMPTS = ("center", "inner", "rough")
SIZE_EDGES = (16, 32, 64, 128)                   # 원본 픽셀의 짧은 변
SIZE_NAMES = ("<16", "16-32", "32-64", "64-128", ">=128")
CLASSES = ["bicycle", "bus", "car", "motorcycle", "person", "scooter", "truck", "pickup", "responder", "cone"]


def size_bin(r) -> str:
    s = min(r["bw"], r["bh"])
    for e, n in zip(SIZE_EDGES, SIZE_NAMES):
        if s < e:
            return n
    return SIZE_NAMES[-1]


def val(r, p) -> float:
    v = r.get(p)
    return 0.0 if v is None else v           # 마스크가 안 나온 것은 IoU 0 으로 센다


def stats(rows, p) -> dict:
    if not rows:
        return {"n": 0}
    v = [val(r, p) for r in rows]
    return {"n": len(v), "mean": round(mean(v), 4), "median": round(median(v), 4),
            "fail": round(sum(x < 0.5 for x in v) / len(v), 4), "ge90": round(sum(x >= 0.9 for x in v) / len(v), 4)}


def boot_diff(a_rows, b_rows, p, reps=2000, seed=0):
    """같은 박스끼리 b − a 의 평균과, 그림을 다시 뽑는 부트스트랩 95% 구간."""
    key = lambda r: (r["id"], r["k"])  # noqa: E731
    A = {key(r): val(r, p) for r in a_rows}
    by_img: dict[int, list[float]] = {}
    for r in b_rows:
        if key(r) in A:
            by_img.setdefault(r["id"], []).append(val(r, p) - A[key(r)])
    imgs = list(by_img)
    flat = [d for i in imgs for d in by_img[i]]
    rng = random.Random(seed)
    ms = []
    for _ in range(reps):
        pick = [d for _ in imgs for d in by_img[rng.choice(imgs)]]
        ms.append(mean(pick))
    ms.sort()
    return {"n": len(flat), "mean": round(mean(flat), 4), "lo": round(ms[int(0.025 * reps)], 4),
            "hi": round(ms[int(0.975 * reps)], 4), "better": round(sum(d > 0.02 for d in flat) / len(flat), 3),
            "worse": round(sum(d < -0.02 for d in flat) / len(flat), 3)}


def main() -> int:
    names = sys.argv[1:] or ["t", "s", "sam31", "tcrop"]
    data, meta = {}, {}
    for n in names:
        f = OUT / f"{n}.jsonl.gz"
        if f.exists():
            data[n] = [json.loads(l) for l in gzip.open(f, "rt", encoding="utf-8")]
            meta[n] = json.loads((OUT / f"{n}.meta.json").read_text(encoding="utf-8"))
    if not data:
        raise SystemExit("결과가 없다")
    base = data.get("t")
    summary = {"meta": meta, "overall": {}, "by_size": {}, "by_origin": {}, "by_class": {}, "vs_t": {}}
    any_rows = next(iter(data.values()))
    summary["rough_input"] = stats(any_rows, "rough_in")
    sizes = {s: sum(size_bin(r) == s for r in any_rows) for s in SIZE_NAMES}
    summary["size_counts"] = sizes

    print(f"박스 {len(any_rows)}개 · 크기(원본 짧은 변) {sizes} · 거친 박스 자체 IoU {summary['rough_input']['mean']}")
    print("\n| 갈래 | 프롬프트 | 평균 IoU | 중앙값 | 실패(<0.5) | 0.9 이상 |")
    print("|---|---|---|---|---|---|")
    for n, rows in data.items():
        summary["overall"][n] = {}
        for p in PROMPTS:
            s = stats(rows, p)
            summary["overall"][n][p] = s
            print(f"| {n} | {p} | {s['mean']:.3f} | {s['median']:.3f} | {s['fail']:.1%} | {s['ge90']:.1%} |")

    print("\n크기별 평균 IoU (center / inner / rough)")
    print("| 갈래 | " + " | ".join(f"{s} ({sizes[s]})" for s in SIZE_NAMES) + " |")
    print("|---|" + "---|" * len(SIZE_NAMES))
    for n, rows in data.items():
        summary["by_size"][n] = {}
        cells = []
        for s in SIZE_NAMES:
            sub = [r for r in rows if size_bin(r) == s]
            summary["by_size"][n][s] = {p: stats(sub, p) for p in PROMPTS}
            cells.append(" / ".join(f"{summary['by_size'][n][s][p].get('mean', float('nan')):.2f}" for p in PROMPTS))
        print(f"| {n} | " + " | ".join(cells) + " |")

    print("\n출처별 평균 IoU (center / inner / rough) — human 에는 SAM t 로 만든 박스가 섞였을 수 있다")
    for n, rows in data.items():
        summary["by_origin"][n] = {}
        line = []
        for o in sorted({r["origin"] for r in rows}):
            sub = [r for r in rows if r["origin"] == o]
            summary["by_origin"][n][o] = {p: stats(sub, p) for p in PROMPTS}
            line.append(f"{o}({len(sub)}) " + " / ".join(f"{summary['by_origin'][n][o][p]['mean']:.3f}" for p in PROMPTS))
        print(f"  {n}: " + " · ".join(line))

    print("\n클래스별 inner 평균 IoU")
    for n, rows in data.items():
        summary["by_class"][n] = {}
        line = []
        for c in sorted({r["cls"] for r in rows}):
            sub = [r for r in rows if r["cls"] == c]
            summary["by_class"][n][CLASSES[c]] = {p: stats(sub, p) for p in PROMPTS}
            line.append(f"{CLASSES[c]}({len(sub)}) {summary['by_class'][n][CLASSES[c]]['inner']['mean']:.2f}")
        print(f"  {n}: " + " · ".join(line))

    if base:
        print("\nt 와의 차 (같은 박스, 그림 단위 부트스트랩 95%) — 나음/나쁨은 IoU 0.02 넘게")
        for n, rows in data.items():
            if n == "t":
                continue
            summary["vs_t"][n] = {p: boot_diff(base, rows, p) for p in PROMPTS}
            for p in PROMPTS:
                d = summary["vs_t"][n][p]
                print(f"  {n} {p}: {d['mean']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] · 나음 {d['better']:.1%} 나쁨 {d['worse']:.1%}")

    print("\n| 갈래 | 불러오기 | 인코딩(장) | 프롬프트 | VRAM 최대(할당) | 마스크 없음 | 통째로 다시 |")
    print("|---|---|---|---|---|---|---|")
    for n, m in meta.items():
        print(f"| {n} | {m['load_s']}초 | {m['encode_ms_mean']} ms | {m['ask_ms_mean']} ms | {m['peak_alloc_gib']} GiB | "
              f"{m['no_mask']} | {m['fallbacks']} |")
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
