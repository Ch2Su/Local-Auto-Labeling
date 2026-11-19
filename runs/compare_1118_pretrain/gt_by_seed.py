"""gt_detail.jsonl(gt_detail.py 가 쓴 것)을 갈래별로 seed 를 모아 본다 — 정답 박스 하나하나를 seed 몇 개에서 클래스까지 맞혔나.

세트 이름은 gt_detail.py 와 같은 '갈래[:epoch][@seed]' 꼴이고, '@' 앞이 같으면 한 갈래로 묶는다.
출력에 파일 이름·카메라 키는 없다(번호·묶음·640 입력에서의 짧은 변만).
  python -I gt_by_seed.py [gt_detail.jsonl]
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "gt_detail.jsonl"
    recs = [json.loads(x) for x in path.read_text(encoding="utf-8").split("\n") if x.strip()]
    sets = list(dict.fromkeys(n for r in recs for n in r["pred"]))
    arms = defaultdict(list)
    for n in sets:
        arms[n.partition("@")[0]].append(n)
    for cls in ("responder", "cone"):
        rs = [r for r in recs if r["cls"] == cls]
        print(f"== {cls} 정답 {len(rs)}개 — 갈래마다 seed 별 클래스까지 맞힌 수 · 찾았는데 다른 클래스로 낸 것 ==")
        for arm, names in arms.items():
            hits = [sum(1 for r in rs if (r["pred"].get(n) or ("",))[0] == cls) for n in names]
            other = Counter((r["pred"].get(n) or ("(못 찾음)",))[0] for n in names for r in rs
                            if (r["pred"].get(n) or ("",))[0] != cls)
            print(f"  {arm:12s} seed {len(names)}개: 맞힘 {hits} (평균 {sum(hits) / len(hits):.1f}) · 나머지 {dict(other.most_common())}")
        print(f"  정답별 — 갈래마다 맞힌 seed 수 (seed 수가 같은 갈래끼리 견줄 것)")
        dist = {arm: Counter() for arm in arms}
        for i, r in enumerate(rs, 1):
            row = {arm: sum(1 for n in names if (r["pred"].get(n) or ("",))[0] == cls) for arm, names in arms.items()}
            for arm, k in row.items():
                dist[arm][k] += 1
            if any(row.values()):
                print(f"   {i:2d} f{r['fold']} {r['short640']:6.1f}px  " + "  ".join(f"{a}={k}/{len(arms[a])}" for a, k in row.items()))
        print("  맞힌 seed 수의 분포(정답 개수):", {a: dict(sorted(d.items())) for a, d in dist.items()})
        # 카메라별(이름 대신 정답이 많은 순서의 번호) — 효과가 한 카메라에 몰렸는지
        cams = Counter(r["camera"] for r in rs)
        order = {c: f"카메라{i}" for i, (c, _) in enumerate(cams.most_common(), 1)}
        print("  카메라별 — 정답 수 · 갈래마다 seed 평균 맞힌 수 · 정답이 든 이미지 수")
        for c, n in cams.most_common():
            sub = [r for r in rs if r["camera"] == c]
            row = {a: sum(1 for nm in names for r in sub if (r["pred"].get(nm) or ("",))[0] == cls) / len(names) for a, names in arms.items()}
            print(f"   {order[c]}: 정답 {n} · 이미지 {len({r['file'] for r in sub})} · " + "  ".join(f"{a}={v:.1f}" for a, v in row.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
