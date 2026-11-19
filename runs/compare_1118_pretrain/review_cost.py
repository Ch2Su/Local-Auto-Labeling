"""검수 사본 스냅숏으로 '그만두면 남는 것 · 남은 양 · 난이도' 를 센다 (읽기 전용 — 스냅숏만 연다).

  python stop_stats.py <스냅숏 DB> <처음 상태 DB> [앞선 스냅숏 DB]
"""
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

REPO = Path(r"C:\local_auto_labeling")
sys.path.insert(0, str(REPO / "src"))

from autolabel import grouping as G  # noqa: E402
from autolabel import masks as M  # noqa: E402
from autolabel.store import Store  # noqa: E402

GAP = 300  # 이력 두 줄 사이가 이보다 길면 쉰 것으로 본다(초)


def q(vals, p):
    vals = sorted(vals)
    if not vals:
        return 0
    k = min(len(vals) - 1, max(0, round(p * (len(vals) - 1))))
    return vals[k]


def main() -> int:
    snap, init_path = Path(sys.argv[1]), Path(sys.argv[2])
    prev_path = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    store = Store(snap)
    con = sqlite3.connect(f"file:{snap.as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    init = sqlite3.connect(f"file:{init_path.as_posix()}?mode=ro&immutable=1", uri=True)
    st0 = dict(init.execute("SELECT id, status FROM images"))
    h0 = init.execute("SELECT MAX(id) FROM box_history").fetchone()[0] or 0
    names = {r["idx"]: r["name"] for r in con.execute("SELECT idx, name FROM classes")}
    imgs = {r["id"]: dict(r) for r in con.execute("SELECT id, status, camera, width, height, path FROM images")}
    boxes = defaultdict(list)
    for r in con.execute("SELECT image_id, cls, cx, cy, w, h, conf, origin FROM boxes ORDER BY id"):
        boxes[r["image_id"]].append(dict(r))

    rows = [r for r in con.execute("SELECT id, image_id, at, source, batch FROM box_history WHERE id > ? ORDER BY at, id", (h0,))
            if not (r["batch"] or "").startswith("SAM 3.1")]
    touched = {r["image_id"] for r in rows}
    A = sorted(i for i, r in imgs.items() if r["status"] == "approved" and st0.get(i) != "approved")
    A0 = sorted(i for i, r in imgs.items() if r["status"] == "approved" and st0.get(i) == "approved")
    W = sorted(i for i in touched if imgs[i]["status"] == "pending")
    P = sorted(i for i, r in imgs.items() if r["status"] == "pending" and i not in touched)
    print(f"프레임 — 사용자 승인 {len(A)} · 처음부터 승인 {len(A0)} · 손댔는데 검수 대기 {len(W)} · 손 안 댄 검수 대기 {len(P)} · 합 {len(imgs)}")

    print("\n이미지 크기:", dict(Counter(f"{r['width']}x{r['height']}" for r in imgs.values()).most_common()))

    # ── 시간 (편집 이력 사이 간격 GAP 이하만 더함 — 보기만 한 시간·쉰 시간은 빠진다)
    t = [datetime.fromisoformat(r["at"]) for r in rows]
    per = Counter()
    active, sessions = 0.0, 1 if rows else 0
    for k in range(1, len(rows)):
        g = (t[k] - t[k - 1]).total_seconds()
        if 0 <= g <= GAP:
            per[rows[k]["image_id"]] += g
            active += g
        else:
            sessions += 1
    print(f"\n편집 이력 {len(rows)}줄 · {t[0].isoformat()} ~ {t[-1].isoformat()} · 이어진 편집 시간 {active / 3600:.1f}시간 · 끊긴 구간 {sessions}")
    tA = [per[i] / 60 for i in A]
    print(f"  사용자 승인 {len(A)}장에 든 시간 {sum(tA) / 60:.1f}시간 — 장당 평균 {statistics.mean(tA):.1f}분 · 중앙값 {statistics.median(tA):.1f}분 · 최대 {max(tA):.0f}분")
    print(f"  검수 대기로 남은 손댄 {len(W)}장에 든 시간 {sum(per[i] for i in W) / 3600:.1f}시간")
    nbA = sum(len(boxes[i]) for i in A)
    print(f"  승인 프레임 박스 {nbA}개 → 박스당 {sum(tA) * 60 / nbA:.0f}초")
    if prev_path and prev_path.exists():
        prev = sqlite3.connect(f"file:{prev_path.as_posix()}?mode=ro&immutable=1", uri=True)
        stp = dict(prev.execute("SELECT id, status FROM images"))
        newA = [i for i in A if stp.get(i) != "approved"]
        tn = [per[i] / 60 for i in newA]
        if newA:
            print(f"  앞선 스냅숏 뒤에 승인한 {len(newA)}장: {sum(tn) / 60:.1f}시간 — 장당 평균 {statistics.mean(tn):.1f}분 · 박스 수 중앙값 {statistics.median(len(boxes[i]) for i in newA)}")
            print("   ", [(i, imgs[i]["camera"], len(boxes[i]), round(per[i] / 60)) for i in newA])

    # ── 그룹별 박스
    def summarize(label, ids, use_masks):
        nb = [len(boxes[i]) for i in ids]
        print(f"\n[{label}] {len(ids)}장 · 박스 {sum(nb)}개 — 장당 중앙값 {statistics.median(nb) if nb else 0} · 90% {q(nb, 0.9)} · 최대 {max(nb) if nb else 0}"
              f" · 40개 넘는 프레임 {sum(1 for n in nb if n > 40)}장")
        print("  카메라별 장수:", dict(sorted(Counter(imgs[i]["camera"] for i in ids).items())))
        print("  카메라별 장당 박스 중앙값:", {c: statistics.median(len(boxes[i]) for i in ids if imgs[i]["camera"] == c)
                                     for c in sorted({imgs[i]["camera"] for i in ids})})
        cls_all, cls_keep, short = Counter(), Counter(), defaultdict(list)
        n_drop = 0
        for i in ids:
            mk = store.get_masks(i) if use_masks else []
            Wd, Hd = imgs[i]["width"], imgs[i]["height"]
            for b in boxes[i]:
                nm = names[b["cls"]]
                cls_all[nm] += 1
                if mk and M.covered(SimpleNamespace(**b), mk) >= M.DROP:
                    n_drop += 1
                    continue
                cls_keep[nm] += 1
                short[nm].append(min(b["w"] * Wd, b["h"] * Hd))
        print("  박스(전부):", dict(cls_all.most_common()))
        if use_masks:
            print(f"  가려져 빠지는 박스 {n_drop}개({n_drop / max(1, sum(nb)):.0%}) · 학습에 남는 박스:", dict(cls_keep.most_common()))
        allshort = [v for vs in short.values() for v in vs]
        if allshort:
            print(f"  {'남는 ' if use_masks else ''}박스 짧은 변(px) — 중앙값 {statistics.median(allshort):.0f} · 16px 미만 {sum(v < 16 for v in allshort) / len(allshort):.0%}"
                  f" · 32px 미만 {sum(v < 32 for v in allshort) / len(allshort):.0%}")
        for nm in ("responder", "cone"):
            if short[nm]:
                v = short[nm]
                print(f"  {nm} {len(v)}개 짧은 변 — 중앙값 {statistics.median(v):.0f}px · 25% {q(v, 0.25):.0f} · 75% {q(v, 0.75):.0f} · 최소 {min(v):.0f} · 최대 {max(v):.0f}"
                      f" · 32px 미만 {sum(x < 32 for x in v)}개")
        return cls_keep

    summarize("사용자 승인", A, True)
    summarize("처음부터 승인", A0, True)
    summarize("손댔는데 검수 대기", W, True)
    summarize("손 안 댄 검수 대기", P, False)
    org = Counter(b["origin"] for i in P + W for b in boxes[i] if b["cls"] == 8)
    print("\n검수 대기 프레임의 responder 박스 출처:", dict(org))

    # ── 같은 장면 (dHash 해밍 거리)
    hs = {}
    for i in A:
        h = G.dhash(Path(imgs[i]["path"]))
        if h:
            hs[i] = int(h, 16)
    for dist in (G.DHASH_MAX, 6, 10):
        par = {i: i for i in hs}

        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        ids = sorted(hs)
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                if imgs[ids[a]]["camera"] == imgs[ids[b]]["camera"] and bin(hs[ids[a]] ^ hs[ids[b]]).count("1") <= dist:
                    par[find(ids[a])] = find(ids[b])
        groups = Counter(find(i) for i in ids)
        print(f"사용자 승인 {len(ids)}장의 장면 묶음(같은 카메라·dHash 거리 ≤ {dist}): {len(groups)}묶음 · 큰 묶음 {sorted(groups.values(), reverse=True)[:6]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
