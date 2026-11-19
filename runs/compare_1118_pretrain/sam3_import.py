"""SAM 3.1 후보(sam3_hw.py 의 jsonl)를 **검수용 작업 사본**의 검수 대기 프레임에 넣는다 — 실제 프로젝트 DB 가 아니다.

박스 만드는 법은 OWLv2 후보 도구와 같다(tools/import_proposals.py 의 plan 을 그대로 부른다): responder 후보가 이미 있는 차량 박스와
IoU 0.5 이상이면 그 박스의 클래스만 바꾸고, 이미 responder 면 건너뛰고, 겹치는 차량이 없으면 새로 넣는다. cone 후보는 이미 있는
cone 과 IoU 0.3 이상이면 버린다. 다른 점: ① 문턱을 **이름마다** 준다(--min 이름=값) ② 바꾸거나 넣은 박스의 origin 은 'sam3'
③ 승인된 프레임은 건드리지 않는다(검수 대기만) ④ 묶음 이름을 따로 준다 — 검수 화면 '변경 기록' 에서 묶음째 되돌릴 수 있다.
기본은 보기만. --apply 면 DB 를 먼저 떠 두고(<db>.bak_sam3_<시각>) 쓴다.

11-20 추가: --max 이름=값 은 그 값 **미만**만 쓴다(이미 넣은 띠를 다시 넣지 않게 — 예: tow truck 0.6~0.7 만), --skip-edited-since
<UTC ISO 시각> 은 그 뒤에 사람이 고친(ui·mask·restore·revert) 검수 대기 프레임을 건너뛴다(사람이 되돌린 후보를 다시 넣지 않게).

  python -I sam3_import.py <hw_sam31_vs_owl.jsonl> --config <작업 사본 config.yaml> --batch "이름" --min "fire truck=0.5" … [--apply]
"""
import argparse
import collections
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(r"C:\local_auto_labeling")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from autolabel.config import Config  # noqa: E402
from autolabel.scoring import image_uncertainty  # noqa: E402
from autolabel.store import Store  # noqa: E402
from autolabel.yolo_io import Box  # noqa: E402
from import_proposals import box_key, plan  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch", required=True, help="묶음 이름(뒤에 시각이 붙는다)")
    ap.add_argument("--min", action="append", default=[], help="이름=문턱 (준 이름만 쓴다)")
    ap.add_argument("--max", action="append", default=[], help="이름=위 문턱 (이 값 미만만 쓴다)")
    ap.add_argument("--skip-edited-since", default=None, help="이 UTC ISO 시각 뒤에 사람이 고친 프레임은 건너뛴다")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    mins = {k.strip(): float(v) for k, v in (m.split("=") for m in a.min)}
    maxs = {k.strip(): float(v) for k, v in (m.split("=") for m in a.max)}
    cfg = Config.load(a.config)
    if REPO in cfg.db.parents:
        raise SystemExit(f"[멈춤] {cfg.db} 는 저장소 안의 DB 다 — 이 스크립트는 작업 사본에만 쓴다")
    store = Store(cfg.db)
    names = store.get_classes()
    idx = {n: i for i, n in enumerate(names)}
    recs = [json.loads(x) for x in a.jsonl.read_text(encoding="utf-8").split("\n") if x.strip()]
    edited = set()
    if a.skip_edited_since:
        ro = sqlite3.connect(f"file:{cfg.db.as_posix()}?mode=ro", uri=True)
        edited = {r[0] for r in ro.execute("SELECT DISTINCT image_id FROM box_history WHERE at >= ? AND source IN "
                                           "('ui', 'mask', 'restore', 'revert')", (a.skip_edited_since,))}
        ro.close()
    todo, tally, skipped, by_name, by_cam = [], collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    for r in recs:
        img = store.get_image(r["id"])
        if img is None:
            skipped["DB 에 없음"] += 1
            continue
        if img["status"] != "pending":
            skipped[f"상태 {img['status']}"] += 1
            continue
        if r["id"] in edited:
            skipped["사람이 고친 프레임"] += 1
            continue
        cands = [{"label": x[6], "score": x[1], "xyxy": x[2:6]} for x in r["sam"]
                 if x[6] in mins and mins[x[6]] <= x[1] < maxs.get(x[6], float("inf"))]
        if not cands:
            continue
        before = store.get_boxes(r["id"])
        after, did = plan(before, {"w": r["w"], "h": r["h"], "boxes": cands}, idx, {"responder": 0.0, "cone": 0.0})
        if not did:
            continue
        old = collections.Counter(box_key(b) for b in before)
        fixed = []
        for b in after:                              # 이번에 바꾸거나 넣은 박스만 origin 을 sam3 로
            k = box_key(b)
            if old[k] > 0:
                old[k] -= 1
                fixed.append(b)
            else:
                fixed.append(Box(b.cls, b.cx, b.cy, b.w, b.h, b.conf, "sam3"))
        todo.append((img, before, fixed, did))
        tally.update(did)
        by_cam[img["camera"]] += did["responder_retag"] + did["responder_add"] + did["cone_add"]
        for c in cands:
            by_name[c["label"]] += 1
    print(f"후보 파일 {len(recs)}장 · 바뀔 이미지 {len(todo)}장 · 건너뜀 {dict(skipped)}")
    print("  문턱:", mins, "· 위 문턱(미만):", maxs)
    print("  문턱을 넘은 후보(이름별):", dict(by_name))
    print("  박스:", ", ".join(f"{k} {v}" for k, v in sorted(tally.items())))
    print("  카메라별 바뀌거나 더해질 박스:", dict(sorted(by_cam.items())))
    retag_from = collections.Counter()
    for img, before, after, did in todo:
        for b0, b1 in zip(before, after):
            if b0.cls != b1.cls:
                retag_from[names[b0.cls]] += 1
    print("  클래스가 바뀌는 박스의 원래 클래스:", dict(retag_from))
    if not a.apply:
        print("(보기만 했다 — 넣으려면 --apply)")
        return 0
    if not todo:
        return 0
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = cfg.db.with_name(f"{cfg.db.name}.bak_sam3_{stamp}")
    with sqlite3.connect(cfg.db) as src, sqlite3.connect(bak) as dst:
        src.backup(dst)
    print("DB 를 떠 두었다:", bak)
    batch = f"{a.batch} {datetime.now():%m-%d %H:%M}"
    low, high, empty = float(cfg.get("review.low_band", 0.30)), float(cfg.get("review.high_band", 0.80)), float(cfg.get("review.empty_score", 0.60))
    log = a.jsonl.with_name(f"applied_sam3_{stamp}.tsv")
    done = moved = 0
    with log.open("w", encoding="utf-8") as f:
        f.write("id\tn_before\tn_after\tdid\tbefore\n")
        for img, before, after, did in todo:
            now = store.get_image(img["id"])
            if now is None or now["status"] != "pending" or [box_key(b) for b in store.get_boxes(img["id"])] != [box_key(b) for b in before]:
                moved += 1                               # 고른 뒤 사람이 고쳤거나 승인했다 — 덮지 않는다
                continue
            score = image_uncertainty([b.conf for b in after], low=low, high=high, empty=empty)
            store.set_boxes(img["id"], after, uncertainty=score, source="bulk", batch=batch)
            f.write(f"{img['id']}\t{len(before)}\t{len(after)}\t{json.dumps(dict(did))}\t{json.dumps([b.to_dict() for b in before])}\n")
            done += 1
    print(f"넣었다: {done}장 · 그 사이 바뀌어 건너뜀 {moved} · 묶음 '{batch}' · 목록 {log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
