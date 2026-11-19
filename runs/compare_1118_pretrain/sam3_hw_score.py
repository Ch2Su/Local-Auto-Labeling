"""sam3_hw.py 의 결과를 다시 채점한다 — OWLv2 후보를 '넣은 직후'의 것으로 바로잡아서(읽기만).

DB 의 지금 박스에서 origin 'owl' 만 세면, 사람이 고친 후보는 origin 이 바뀌어 빠진다(승인된 프레임에서 OWLv2 가 불리해진다).
그래서 보관 DB 의 변경 이력에서 OWLv2 묶음('OWLv2 후보 …')이 넣은 직후의 박스(after 의 origin 'owl', responder·cone)를 후보로 쓴다.
승인 프레임을 '오늘 새로 검수한 것'과 '전부터 승인이던 것'으로 나눠서도 본다.

  python -I sam3_hw_score.py <hw_sam31_vs_owl.jsonl> <보관 DB> [--since 2026-11-19]
"""
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sam3_hw import CLS, greedy  # noqa: E402


def main() -> int:
    recs = [json.loads(x) for x in Path(sys.argv[1]).read_text(encoding="utf-8").split("\n") if x.strip()]
    arch = sqlite3.connect(f"file:{Path(sys.argv[2]).as_posix()}?mode=ro&immutable=1", uri=True)
    arch.row_factory = sqlite3.Row
    owl = defaultdict(list)
    seen = set()
    for r in arch.execute("SELECT image_id, after FROM box_history WHERE batch LIKE 'OWLv2%' ORDER BY at"):
        if r["image_id"] in seen or not r["after"]:
            continue                                   # 이미지마다 묶음 하나(첫 줄)만
        seen.add(r["image_id"])
        for b in json.loads(r["after"]):
            if len(b) >= 7 and b[6] == "owl" and int(b[0]) in CLS:
                owl[r["image_id"]].append((CLS[int(b[0])], float(b[5] or 0), b[1], b[2], b[3], b[4]))
    was_approved = {r[0] for r in arch.execute("SELECT id FROM images WHERE status = 'approved'")}
    cropped = 0
    for r in recs:
        w, h = r["w"], r["h"]
        # OWLv2 를 넣은 뒤(11-09 오후) hw-04 줌 화면 36장은 오른쪽을 잘랐다(1081×720 → 824×720, 이름 끝 _c0-0-824-720 —
        # docs/provenance.md). 이력의 좌표는 자르기 전 그림 기준이라 그 폭으로 픽셀을 구하고, 잘려 나간 자리의 후보는 뺀다.
        ow = 1081 if "_c0-0-824-720" in r["file"] else w
        cropped += ow != w
        out = []
        for c, s, cx, cy, bw, bh in owl.get(r["id"], []):
            x1, y1, x2, y2 = (cx - bw / 2) * ow, (cy - bh / 2) * h, (cx + bw / 2) * ow, (cy + bh / 2) * h
            if (x1 + x2) / 2 >= w:
                continue
            out.append((c, s, x1, y1, min(x2, w), y2))
        r["owl0"] = out
    print("자르기 보정한 프레임", cropped)
    print("OWLv2 가 넣은 직후의 후보(이력에서):", sum(len(r["owl0"]) for r in recs), "개 ·", dict(Counter(x[0] for r in recs for x in r["owl0"])),
          "· 그 묶음이 닿은 프레임", sum(1 for r in recs if r["id"] in seen))
    groups = {"승인 전부": [r for r in recs if r["status"] == "approved"],
              "오늘 새로 검수": [r for r in recs if r["status"] == "approved" and r["id"] not in was_approved],
              "전부터 승인": [r for r in recs if r["status"] == "approved" and r["id"] in was_approved]}
    for gname, appr in groups.items():
        tot = Counter(g[0] for r in appr for g in r["gt"])
        cams = Counter(r["camera"] for r in appr)
        print(f"\n== {gname} {len(appr)}장 {dict(cams)} — 정답 responder {tot['responder']} · cone {tot['cone']} ==")
        print("| 후보 | 문턱 | responder 찾음 | 후보 | 다른 클래스·빈 자리 | cone 찾음 | 후보 |")
        print("|---|---|---|---|---|---|---|")
        for label, key, confs in (("OWLv2(넣은 직후)", "owl0", (0.0, 0.5)), ("OWLv2(보관 DB 에 남은 것)", "owl", (0.0,)),
                                  ("SAM 3.1", "sam", (0.25, 0.5, 0.7))):
            for conf in confs:
                hit, n = Counter(), Counter()
                for r in appr:
                    for c, k in greedy([tuple(x[:6]) for x in r[key] if x[1] >= conf], [tuple(g) for g in r["gt"]]):
                        n[c[0]] += 1
                        hit[c[0]] += k is not None and r["gt"][k][0] == c[0]
                print(f"| {label} | {conf if conf else '전부'} | {hit['responder']}/{tot['responder']} | {n['responder']} | {n['responder'] - hit['responder']} | "
                      f"{hit['cone']}/{tot['cone']} | {n['cone']} |")
        # SAM 3.1 이름별 (conf 0.5)
        c = defaultdict(Counter)
        for r in appr:
            pairs = greedy([tuple(x[:6]) for x in r["sam"] if x[1] >= 0.5], [tuple(g) for g in r["gt"]])
            name_of = {tuple(x[:6]): x[6] for x in r["sam"]}
            for cand, k in pairs:
                nm = name_of[cand]
                c[nm]["n"] += 1
                c[nm]["hit"] += k is not None and r["gt"][k][0] == cand[0]
        print("  SAM 3.1 이름별(conf 0.5, 후보/맞음):", {k: f"{v['n']}/{v['hit']}" for k, v in c.items()})
    # 전체에서 겹침 (OWLv2 는 넣은 직후의 것)
    both, only_s, only_o = Counter(), Counter(), Counter()
    for r in recs:
        for cls in ("responder", "cone"):
            s = [tuple(x[:6]) for x in r["sam"] if x[1] >= 0.5 and x[0] == cls]
            o = [x for x in r["owl0"] if x[0] == cls]
            nb = sum(k is not None for _, k in greedy(s, [(x[0], *x[2:6]) for x in o]))
            both[cls] += nb
            only_s[cls] += len(s) - nb
            only_o[cls] += len(o) - nb
    print(f"\n== 전체 {len(recs)}장 — SAM 3.1(conf 0.5 이상)과 OWLv2(넣은 직후)가 겹치는가 ==")
    for cls in ("responder", "cone"):
        print(f"  {cls}: 둘 다 {both[cls]} · SAM 3.1 만 {only_s[cls]} · OWLv2 만 {only_o[cls]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
