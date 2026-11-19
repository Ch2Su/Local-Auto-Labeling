"""사용자가 승인한 프레임의 라벨을 정답으로 놓고 SAM 3.1 원래 검출(프롬프트별·점수별)을 다시 잰다.

police car 를 빼고 넣은 후보가 사용자가 새로 찾은 responder 를 얼마나 놓쳤는지, police car 를 넣었으면 얼마나 맞았을지.
가려져 빠질 박스(영역이 반 넘게 덮음)는 정답에서 빼고, 영역이 반 넘게 덮는 검출은 맞음·틀림 어디에도 세지 않는다.

  python sam_vs_review.py <스냅숏 DB> <처음 상태 DB> <hw_sam31_vs_owl.jsonl>
"""
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from review_progress import M, Store, iou, match  # noqa: E402

THRS = (0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def as_box(cls, x1, y1, x2, y2, W, H):
    return {"cls": cls, "cx": (x1 + x2) / 2 / W, "cy": (y1 + y2) / 2 / H, "w": (x2 - x1) / W, "h": (y2 - y1) / H}


def nms(dets, thr=0.5):
    keep = []
    for d in sorted(dets, key=lambda d: -d["score"]):
        if all(iou(d, k) < thr for k in keep):
            keep.append(d)
    return keep


def main() -> int:
    snap, init_path, jl = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    store = Store(snap)
    con = sqlite3.connect(f"file:{snap.as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    init = sqlite3.connect(f"file:{init_path.as_posix()}?mode=ro&immutable=1", uri=True)
    st0 = dict(init.execute("SELECT id, status FROM images").fetchall())
    h0 = init.execute("SELECT MAX(id) FROM box_history").fetchone()[0] or 0
    sam = {}
    for line in jl.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        sam[r["id"]] = r
    ids = [r[0] for r in con.execute("SELECT id FROM images WHERE status='approved' ORDER BY id")
           if st0.get(r[0]) != "approved" and r[0] in sam]
    print("대상 프레임", len(ids), "장")

    gt_all = {}        # id -> [(box, user_found)]
    dets_all = {}      # id -> [det]
    for i in ids:
        W, H = con.execute("SELECT width, height FROM images WHERE id=?", (i,)).fetchone()
        assert (W, H) == (sam[i]["w"], sam[i]["h"]), (i, W, H, sam[i]["w"], sam[i]["h"])
        hist = [e for e in reversed(store.box_history(i, limit=1_000_000)) if e["id"] > h0]
        s = [e for e in hist if (e["batch"] or "").startswith("SAM 3.1")]
        ref = s[-1]["after"] if s else (hist[0]["before"] if hist else None)
        cur = [dict(r) for r in con.execute("SELECT cls, cx, cy, w, h, origin FROM boxes WHERE image_id=? ORDER BY id", (i,))]
        ref = cur if ref is None else ref
        pairs, gone, new = match(ref, cur)
        found = {b for b in new} | {b for a, b, v in pairs if ref[a]["cls"] != cur[b]["cls"]}
        mk = store.get_masks(i)
        hidden = lambda b: bool(mk) and M.covered(SimpleNamespace(**b), mk) >= M.DROP  # noqa: E731
        gt_all[i] = [(b, k in found) for k, b in enumerate(cur) if b["cls"] in (8, 9) and not hidden(b)]
        others = [b for b in cur if b["cls"] not in (8, 9) and not hidden(b)]
        dets = []
        for cls, score, x1, y1, x2, y2, prompt in sam[i]["sam"]:
            d = as_box(8 if cls == "responder" else 9, x1, y1, x2, y2, W, H)
            if hidden(d):
                continue
            d.update(score=score, prompt=prompt, on_other=any(iou(d, o) >= 0.5 for o in others))
            dets.append(d)
        dets_all[i] = dets

    n_gt = Counter()
    for i in ids:
        for b, f in gt_all[i]:
            n_gt[(b["cls"], f)] += 1
    print("정답(학습에 들어가는 것): responder", n_gt[(8, False)] + n_gt[(8, True)], "(그중 사용자가 새로 찾음", n_gt[(8, True)], ")",
          "· cone", n_gt[(9, False)] + n_gt[(9, True)])

    def score_set(pick):
        """pick(det) 이 참인 검출만 NMS 로 묶어 정답과 짝짓는다 → (맞음, 틀림, 그중 다른 차에 얹힘, 찾은 정답, 찾은 '새로 찾은' 정답)."""
        tp = fp = fp_other = 0
        hit, hit_new = Counter(), Counter()
        for i in ids:
            for cls in (8, 9):
                d = nms([x for x in dets_all[i] if x["cls"] == cls and pick(x)])
                g = [(b, f) for b, f in gt_all[i] if b["cls"] == cls]
                pairs, dl, gl = match(d, [b for b, f in g])
                tp += len(pairs)
                fp += len(dl)
                fp_other += sum(1 for k in dl if d[k]["on_other"])
                for a, b, v in pairs:
                    hit[cls] += 1
                    hit_new[cls] += g[b][1]
        return tp, fp, fp_other, hit, hit_new

    prompts = sorted({d["prompt"] for v in dets_all.values() for d in v})
    print("\n프롬프트별 (NMS 0.5 뒤, 정답과 IoU≥0.5 짝): 점수≥ | 검출 | 맞음 | 틀림(그중 다른 차 위) | responder 정답 찾음(새로 찾은 것 중)")
    for p in prompts:
        for t in THRS:
            tp, fp, fo, hit, hn = score_set(lambda d, p=p, t=t: d["prompt"] == p and d["score"] >= t)
            if tp + fp == 0:
                continue
            print(f"  {p:14s} {t:.2f} | {tp + fp:3d} | {tp:3d} | {fp:3d}({fo:3d}) | resp {hit[8]:3d}({hn[8]:2d}) cone {hit[9]:3d}")
    print("\n조합:")
    combos = {
        "넣은 것과 비슷: police car 뺀 전부 ≥0.5": lambda d: d["prompt"] != "police car" and d["score"] >= 0.5,
        "+ police car ≥0.9": lambda d: (d["prompt"] != "police car" and d["score"] >= 0.5) or (d["prompt"] == "police car" and d["score"] >= 0.9),
        "+ police car ≥0.8": lambda d: (d["prompt"] != "police car" and d["score"] >= 0.5) or (d["prompt"] == "police car" and d["score"] >= 0.8),
        "+ police car ≥0.7": lambda d: (d["prompt"] != "police car" and d["score"] >= 0.5) or (d["prompt"] == "police car" and d["score"] >= 0.7),
        "+ police car ≥0.5": lambda d: d["score"] >= 0.5,
        "전부 ≥0.25": lambda d: d["score"] >= 0.25,
    }
    for name, f in combos.items():
        tp, fp, fo, hit, hn = score_set(f)
        print(f"  {name:38s} 검출 {tp + fp:3d} 맞음 {tp:3d} 틀림 {fp:3d}(다른 차 위 {fo:3d}) · responder {hit[8]}/{n_gt[(8, False)] + n_gt[(8, True)]}"
              f" (새로 찾은 것 {hn[8]}/{n_gt[(8, True)]}) · cone {hit[9]}/{n_gt[(9, False)] + n_gt[(9, True)]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
