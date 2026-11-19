"""검수용 작업 사본에서 사용자가 검수한 것을 센다 — 서버가 쓰는 DB 는 읽기 전용으로만 연다.

  python review_progress.py <작업 사본 DB> <처음 상태 DB(서버 첫 백업)> <스냅숏 폴더>

작업 사본 DB 를 sqlite backup 으로 스냅숏에 떠서, Store 는 그 스냅숏에만 연다(이력 목록·다각형 영역을 Store 로 읽으려고).
기준 상태: SAM 3.1 후보가 들어간 프레임은 그 묶음 직후, 아니면 처음 상태(서버 첫 백업) — 둘 다 이력으로 되짚는다.
"""
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

REPO = Path(r"C:\local_auto_labeling")
sys.path.insert(0, str(REPO / "src"))

from autolabel import masks as M  # noqa: E402
from autolabel.store import Store  # noqa: E402


def xyxy(b):
    return b["cx"] - b["w"] / 2, b["cy"] - b["h"] / 2, b["cx"] + b["w"] / 2, b["cy"] + b["h"] / 2


def iou(a, b):
    ax1, ay1, ax2, ay2 = xyxy(a)
    bx1, by1, bx2, by2 = xyxy(b)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    u = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / u if u > 0 else 0.0


def match(src, dst, thr=0.5):
    """탐욕 1:1 짝짓기(클래스 무관). ([(i, j, iou)], 짝 없는 src 번호, 짝 없는 dst 번호)."""
    pairs = sorted(((iou(a, b), i, j) for i, a in enumerate(src) for j, b in enumerate(dst)), reverse=True)
    ui, uj, out = set(), set(), []
    for v, i, j in pairs:
        if v < thr:
            break
        if i in ui or j in uj:
            continue
        ui.add(i)
        uj.add(j)
        out.append((i, j, v))
    return out, [i for i in range(len(src)) if i not in ui], [j for j in range(len(dst)) if j not in uj]


def main() -> int:
    live, init_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    snap = out / "labeling.db"
    if snap.exists():
        snap.unlink()
    src = sqlite3.connect(f"file:{live.as_posix()}?mode=ro", uri=True)
    dst = sqlite3.connect(snap)
    src.backup(dst)
    dst.close()
    src.close()

    init = sqlite3.connect(f"file:{init_path.as_posix()}?mode=ro&immutable=1", uri=True)
    init.row_factory = sqlite3.Row
    store = Store(snap)
    con = sqlite3.connect(f"file:{snap.as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row

    names = {r["idx"]: r["name"] for r in con.execute("SELECT idx, name FROM classes ORDER BY idx")}
    nm = lambda c: names.get(c, str(c))  # noqa: E731
    print("클래스:", names)
    now = {r["id"]: dict(r) for r in con.execute(
        "SELECT id, name, status, camera, split, width, height, updated_at FROM images")}
    st0 = {r["id"]: r["status"] for r in init.execute("SELECT id, status FROM images")}
    h0 = init.execute("SELECT MAX(id) FROM box_history").fetchone()[0] or 0
    print("이미지 상태 — 처음:", dict(Counter(st0.values())), "· 지금:", dict(Counter(r["status"] for r in now.values())))
    trans = Counter((st0.get(i), r["status"]) for i, r in now.items() if st0.get(i) != r["status"])
    print("상태가 바뀐 것:", {f"{a}→{b}": n for (a, b), n in trans.items()})

    rows = con.execute("SELECT id, image_id, at, source, batch FROM box_history WHERE id > ? ORDER BY id", (h0,)).fetchall()
    is_sam = lambda r: (r["batch"] or "").startswith("SAM 3.1")  # noqa: E731
    sam_rows = [r for r in rows if is_sam(r)]
    sam_last = max((r["id"] for r in sam_rows), default=h0)
    user_rows = [r for r in rows if not is_sam(r)]
    print(f"\n처음 뒤 이력 {len(rows)}줄 — SAM 3.1 묶음 {len(sam_rows)}줄, 그 밖 {len(user_rows)}줄")
    by_hour = Counter((r["at"][:13], r["source"]) for r in user_rows)
    for (hour, source), n in sorted(by_hour.items()):
        print(f"  {hour}시 {source}: {n}줄")
    batches = Counter((r["batch"], r["source"]) for r in user_rows if r["batch"])
    if batches:
        print("  사용자 쪽 묶음:", dict(batches))
    touched = {r["image_id"] for r in user_rows}
    sam_imgs = {r["image_id"] for r in sam_rows}
    before_sam_on_sam_imgs = sum(1 for r in user_rows if r["id"] < sam_last and r["image_id"] in sam_imgs)
    print(f"  손댄 이미지 {len(touched)}장 · SAM 후보 프레임에 SAM 전 손댄 줄 {before_sam_on_sam_imgs}")

    hist_cache: dict[int, list] = {}

    def history(i):
        if i not in hist_cache:
            hist_cache[i] = list(reversed(store.box_history(i, limit=1_000_000)))
        return hist_cache[i]

    def boxes_now(i):
        return [dict(r) for r in con.execute(
            "SELECT cls, cx, cy, w, h, conf, origin FROM boxes WHERE image_id=? ORDER BY id", (i,))]

    def state_at(i, hid):
        last, nxt = None, None
        for e in history(i):
            if e["id"] <= hid:
                last = e
            else:
                nxt = e
                break
        if last is not None:
            return last["after"]
        return nxt["before"] if nxt is not None else boxes_now(i)

    def masks_init(i):
        return init.execute("SELECT COUNT(*) FROM masks WHERE image_id=?", (i,)).fetchone()[0]

    approved_new = sorted(i for i, r in now.items() if r["status"] == "approved" and st0.get(i) != "approved")
    other_new = sorted(i for i, r in now.items() if r["status"] != "approved" and st0.get(i) != r["status"])
    wip = sorted(i for i in touched if now[i]["status"] == "pending")
    print(f"\n사용자가 새로 승인한 프레임 {len(approved_new)}장 · 다른 상태로 바꾼 것 {len(other_new)}장 · 손댔는데 아직 검수 대기 {len(wip)}장")

    tot = defaultdict(Counter)
    sam_out = Counter()
    issues = []
    detail = []
    for i in approved_new + wip:
        ref_hid = sam_last if i in sam_imgs else h0
        ref = state_at(i, ref_hid)
        cur = boxes_now(i)
        pairs, gone, new = match(ref, cur)
        kept = sum(1 for a, b, v in pairs if ref[a]["cls"] == cur[b]["cls"] and v >= 0.9)
        moved = [(ref[a]["cls"]) for a, b, v in pairs if ref[a]["cls"] == cur[b]["cls"] and v < 0.9]
        retag = [(ref[a]["cls"], cur[b]["cls"]) for a, b, v in pairs if ref[a]["cls"] != cur[b]["cls"]]
        dele = [ref[a] for a in gone]
        add = [cur[b] for b in new]
        grp = "approved" if i in approved_new else "wip"
        t = tot[grp]
        t["frames"] += 1
        t["ref"] += len(ref)
        t["now"] += len(cur)
        t["kept"] += kept
        t["moved"] += len(moved)
        for a, b in retag:
            t[f"retag {nm(a)}→{nm(b)}"] += 1
        for b in dele:
            t[f"del {nm(b['cls'])} ({b['origin']})"] += 1
        for b in add:
            t[f"add {nm(b['cls'])}"] += 1
        for c in moved:
            t[f"moved {nm(c)}"] += 1

        # SAM 3.1 후보가 어떻게 됐나 (승인 프레임만 — 판정이 끝난 것)
        if grp == "approved" and i in sam_imgs:
            first_sam = next(e for e in history(i) if e["id"] > h0 and (e["batch"] or "").startswith("SAM 3.1"))
            pre = first_sam["before"]
            fate = {a: ("kept" if ref[a]["cls"] == cur[b]["cls"] else f"→{nm(cur[b]['cls'])}") for a, b, v in pairs}
            for a in gone:
                fate[a] = "deleted"
            for a, b in enumerate(ref):
                if b["origin"] != "sam3":
                    continue
                kind = "retag" if any(iou(b, p) >= 0.9 for p in pre) else "add"
                if kind == "retag":
                    was = max(pre, key=lambda p: iou(b, p))["cls"]
                    kind = f"retag({nm(was)})" if was != b["cls"] else "already"
                sam_out[(nm(b["cls"]), kind, fate[a])] += 1

        # 영역·점검
        mk = store.get_masks(i)
        npoly = sum(1 for m in mk if M.is_poly(m))
        bad_poly = [m for m in mk if M.is_poly(m) and (len(m) < 3 or M.poly_area(m) <= 0)]
        W, H = now[i]["width"], now[i]["height"]
        dup = [(nm(cur[a]["cls"]), nm(cur[b]["cls"]), round(iou(cur[a], cur[b]), 2))
               for a in range(len(cur)) for b in range(a + 1, len(cur)) if iou(cur[a], cur[b]) >= 0.5]
        near = [(nm(cur[a]["cls"]), nm(cur[b]["cls"]), round(iou(cur[a], cur[b]), 2))
                for a in range(len(cur)) for b in range(a + 1, len(cur))
                if 0.3 <= iou(cur[a], cur[b]) < 0.5 and 8 in (cur[a]["cls"], cur[b]["cls"])]
        cov = [(nm(b["cls"]), round(M.covered(SimpleNamespace(**b), mk), 2)) for b in cur] if mk else []
        dropped = [c for c, v in cov if v >= M.DROP]
        partial = [(c, v) for c, v in cov if 0.2 <= v < M.DROP]
        tiny = [nm(b["cls"]) for b in cur if b["w"] * W < 8 or b["h"] * H < 8]
        t["masks_now"] += len(mk)
        t["masks_init"] += masks_init(i)
        t["polys"] += npoly
        t["dropped_boxes"] += len(dropped)
        cnt = Counter(nm(b["cls"]) for b in cur)
        line = (f"  #{i} {now[i]['camera']} {now[i]['split']} {grp} 박스 {len(ref)}→{len(cur)}"
                f" (+{len(add)} -{len(dele)} 클래스바꿈 {len(retag)} 옮김 {len(moved)})"
                f" responder {cnt.get('responder', 0)} cone {cnt.get('cone', 0)}"
                f" · 영역 {masks_init(i)}→{len(mk)}(다각형 {npoly}) · 가려져 빠질 박스 {len(dropped)}")
        detail.append({"id": i, "group": grp, "camera": now[i]["camera"], "split": now[i]["split"],
                       "ref": len(ref), "now": len(cur), "add": [nm(b["cls"]) for b in add],
                       "del": [(nm(b["cls"]), b["origin"]) for b in dele],
                       "retag": [(nm(a), nm(b)) for a, b in retag], "masks": len(mk), "polys": npoly,
                       "dup": dup, "near_responder": near, "dropped": dropped, "partial": partial, "tiny": tiny,
                       "bad_poly": len(bad_poly)})
        print(line)
        for k, v in (("겹침 IoU≥0.5", dup), ("responder 와 0.3~0.5 겹침", near), ("반쯤 가려짐", partial),
                     ("8px 미만", tiny), ("이상한 다각형", bad_poly)):
            if v:
                print(f"      {k}: {v}")
                issues.append((i, k, v))

    for grp, t in tot.items():
        print(f"\n[{grp}] 합계:")
        for k in sorted(t, key=lambda k: (not k.startswith(("frames", "ref", "now", "kept", "moved", "masks", "polys", "dropped")), k)):
            print(f"  {k}: {t[k]}")
    if sam_out:
        print("\nSAM 3.1 후보의 결과 (승인한 프레임):")
        for (c, kind, fate), n in sorted(sam_out.items()):
            print(f"  {c:10s} {kind:22s} {fate:12s} {n}")
    (out / "review_progress.json").write_text(json.dumps(
        {"detail": detail, "sam": [[*k, n] for k, n in sam_out.items()],
         "totals": {g: dict(t) for g, t in tot.items()}}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n문제 후보:", len(issues), "건 · 기록:", out / "review_progress.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
