"""사용자가 승인한 프레임의 responder·cone 박스가 어디서 왔는지(후보 그대로·고침·새로 그림)와 가려져 빠지는지를 센다.

  python resp_breakdown.py <스냅숏 DB> <처음 상태 DB>
"""
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from review_progress import M, Store, match  # noqa: E402


def main() -> int:
    snap, init_path = Path(sys.argv[1]), Path(sys.argv[2])
    store = Store(snap)
    con = sqlite3.connect(f"file:{snap.as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    init = sqlite3.connect(f"file:{init_path.as_posix()}?mode=ro&immutable=1", uri=True)
    st0 = dict(init.execute("SELECT id, status FROM images").fetchall())
    h0 = init.execute("SELECT MAX(id) FROM box_history").fetchone()[0] or 0
    ids = [r[0] for r in con.execute("SELECT id FROM images WHERE status='approved' ORDER BY id") if st0.get(r[0]) != "approved"]
    out = Counter()
    for i in ids:
        hist = list(reversed(store.box_history(i, limit=1_000_000)))
        hist = [e for e in hist if e["id"] > h0]
        # 사람이 손대기 전 = SAM 3.1 묶음 직후(있으면), 없으면 처음 상태
        sam = [e for e in hist if (e["batch"] or "").startswith("SAM 3.1")]
        if sam:
            ref = sam[-1]["after"]
        else:
            ref = hist[0]["before"] if hist else None
        cur = [dict(r) for r in con.execute("SELECT cls, cx, cy, w, h, conf, origin FROM boxes WHERE image_id=? ORDER BY id", (i,))]
        if ref is None:
            ref = cur
        mk = store.get_masks(i)
        pairs, gone, new = match(ref, cur)
        src = {}
        for a, b, v in pairs:
            if ref[a]["cls"] == cur[b]["cls"]:
                src[b] = f"후보·이전 라벨 그대로({ref[a]['origin']})" + ("" if v >= 0.9 else " 크기·자리 고침")
            else:
                src[b] = "다른 클래스에서 바꿈"
        for b in new:
            src[b] = "새로 그림"
        for b, box in enumerate(cur):
            if box["cls"] not in (8, 9):
                continue
            masked = bool(mk) and M.covered(SimpleNamespace(**box), mk) >= M.DROP
            name = "responder" if box["cls"] == 8 else "cone"
            out[(name, src[b], "가려져 빠짐" if masked else "학습에 들어감")] += 1
    for k, n in sorted(out.items()):
        print(" · ".join(k), n)
    for name in ("responder", "cone"):
        print(name, "합계", sum(n for k, n in out.items() if k[0] == name),
              "— 학습에 들어감", sum(n for k, n in out.items() if k[0] == name and k[2] == "학습에 들어감"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
