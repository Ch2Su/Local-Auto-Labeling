"""OWLv2 후보(`tools/ovd_propose.py` 의 JSONL)로 responder(출동 차량)·cone(라바콘) 박스를 만들어 **검수 대기** 로 넣는다 (2026-11-09).

두 클래스는 11-09 사용자 결정으로 붙였다(config.yaml 8·9번, 판정은 docs/label-rules.md '출동 차량·라바콘'). 운영 모델은 둘을 몰라
자동 라벨링이 못 하므로, 이름만 주면 찾는 검출기(OWLv2)의 후보를 사람이 검수하게 넣는다. 세 단계:

    # 1) 이미지 목록(TSV: id·경로) — 조건을 겹쳐 주면 모두 맞는 것만
    .venv\\Scripts\\python.exe tools\\import_proposals.py list runs\\ovd\\<묶음>\\images.tsv --status approved,pending
    .venv\\Scripts\\python.exe tools\\import_proposals.py list runs\\ovd\\<묶음>\\images.tsv --camera <카메라키>
    .venv\\Scripts\\python.exe tools\\import_proposals.py list runs\\ovd\\<묶음>\\images.tsv --ids <id 한 줄에 하나 파일>
    # 2) OWLv2 — 다른 가상 환경(ovd_propose.py 머리말)
    # 3) 보기만(기본) → 넣기
    .venv\\Scripts\\python.exe tools\\import_proposals.py apply runs\\ovd\\<묶음>\\owl.jsonl
    .venv\\Scripts\\python.exe tools\\import_proposals.py apply runs\\ovd\\<묶음>\\owl.jsonl --apply
    # 묶음을 검수 UI '변경 내역' 에서 되돌린 뒤, 승인이었다가 검수 대기로 간 이미지를 승인으로 돌린다
    .venv\\Scripts\\python.exe tools\\import_proposals.py restore runs\\ovd\\<묶음>\\applied_<시각>.tsv [--apply]

**박스 만드는 법 (11-09 초안 — 검수해 보며 고친다).**
- OWLv2 이름 → 우리 클래스: fire truck·ambulance·police car·tow truck → responder, traffic cone → cone. 문턱은 responder 0.30,
  cone 0.40(`--min-responder`·`--min-cone`) — 사고 영상 시험에서 소방차는 0.3 위가 거의 진짜, 학습 데이터에서 라바콘은 0.4 위가
  대부분 진짜였다. 경찰차 이름은 택시·흰 SUV 를 0.4~0.57 로 잡는다 — **검수에서 지울 것**.
- 같은 클래스 후보끼리 IoU 0.5 이상이면 점수 높은 것 하나만(한 차를 소방차·견인차로 둘 다 부르기도 한다).
- responder 후보가 이미 있는 차량 박스(car·truck·pickup·bus)와 IoU 0.5 이상이면 **그 박스의 클래스만** responder 로 바꾼다 —
  좌표는 기존 것(사람·모델 박스가 OWLv2 보다 꼭 맞다). 이미 responder 박스와 겹치면 건너뛰고, 겹치는 차량 박스가 없으면
  OWLv2 상자를 새로 넣는다.
- cone 후보는 이미 있는 cone 박스와 IoU 0.3 이상이면 버리고, 아니면 넣는다.
- 바꾸거나 넣은 박스는 origin 'owl'·신뢰도 = OWLv2 점수(검수 화면에서 자동 박스처럼 신뢰도가 붙는다).

**쓰기.** `--apply` 면 DB 를 `labeling.db.bak_owl_<시각>` 으로 떠 두고, 이미지마다 박스를 `store.set_boxes(source='bulk',
batch='OWLv2 후보 <월-일 시:분>')` 로 바꾼다 — 그래서 검수 UI '변경 내역' 의 일괄 수정 목록에 뜨고 묶음째 되돌릴 수 있다(박스만).
**승인 이미지가 바뀌면 검수 대기로 돌린다** — 자동 승인은 없다(사람이 본 것만 학습에 든다). 박스를 정한 뒤 그 이미지의 박스가
바뀌었으면(그 사이 사람이 고쳤으면) 건너뛴다. 바꾼 목록·원래 상태·원래 박스는 `<JSONL 폴더>/applied_<시각>.tsv` 에 남는다.
바뀌기 전 박스가 없던 이미지는 이력이 남지 않아(store.set_boxes) 되돌리기로 지워지지 않는다 — 드물다(셈에 따로 보인다).

**주의.** `labeler.py label --all` 은 검수 대기 이미지를 운영 모델 예측으로 덮어써 여기서 넣은 박스를 지운다 — 쓰지 말 것.
"""

from __future__ import annotations

import argparse
import collections
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # tools/ -> 저장소 루트

from autolabel.config import Config  # noqa: E402
from autolabel.scoring import image_uncertainty  # noqa: E402
from autolabel.store import Store  # noqa: E402
from autolabel.yolo_io import Box  # noqa: E402

TO_CLASS = {"fire truck": "responder", "ambulance": "responder", "police car": "responder", "tow truck": "responder",
            "traffic cone": "cone"}
NEW_CLASSES = ("responder", "cone")
VEHICLES = ("car", "truck", "pickup", "bus")
SAME_IOU = 0.5      # 같은 클래스 후보끼리 한 물체로 보는 IoU
RETAG_IOU = 0.5     # responder 후보가 기존 차량 박스와 같은 차라고 보는 IoU
CONE_IOU = 0.3      # cone 후보가 기존 cone 박스와 같은 콘이라고 보는 IoU


def xyxy(b: Box) -> tuple[float, float, float, float]:
    return b.cx - b.w / 2, b.cy - b.h / 2, b.cx + b.w / 2, b.cy + b.h / 2


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def box_key(b: Box) -> tuple:
    return (b.cls, round(b.cx, 6), round(b.cy, 6), round(b.w, 6), round(b.h, 6), b.conf, b.origin)


def plan(boxes: list[Box], rec: dict, idx: dict[str, int], mins: dict[str, float]) -> tuple[list[Box], collections.Counter]:
    """이미지 한 장의 새 박스 목록과 무엇을 했는지 셈. 바뀐 것이 없으면 셈이 비어 있다."""
    w, h = float(rec["w"]), float(rec["h"])
    cands = []
    for p in rec.get("boxes", []):
        c = TO_CLASS.get(p["label"])
        if c is None or p["score"] < mins[c]:
            continue
        x1, y1, x2, y2 = p["xyxy"]
        n = (max(0.0, x1 / w), max(0.0, y1 / h), min(1.0, x2 / w), min(1.0, y2 / h))
        if n[2] > n[0] and n[3] > n[1]:
            cands.append((p["score"], c, n))
    kept: list[tuple[float, str, tuple]] = []
    for sc, c, n in sorted(cands, key=lambda t: -t[0]):
        if all(c != kc or iou(n, kn) < SAME_IOU for _, kc, kn in kept):
            kept.append((sc, c, n))
    out = list(boxes)
    did = collections.Counter()
    veh = {idx[v] for v in VEHICLES if v in idx}
    for sc, c, n in kept:
        if c == "responder":
            if any(b.cls == idx["responder"] and iou(xyxy(b), n) >= RETAG_IOU for b in out):
                did["responder_already"] += 1
                continue
            best, best_iou = None, RETAG_IOU
            for i, b in enumerate(out):
                if b.cls in veh:
                    v = iou(xyxy(b), n)
                    if v >= best_iou:
                        best, best_iou = i, v
            if best is not None:
                b = out[best]
                out[best] = Box(idx["responder"], b.cx, b.cy, b.w, b.h, round(sc, 4), "owl")
                did["responder_retag"] += 1
                continue
        elif any(b.cls == idx["cone"] and iou(xyxy(b), n) >= CONE_IOU for b in out):
            did["cone_already"] += 1
            continue
        nb = Box(idx[c], (n[0] + n[2]) / 2, (n[1] + n[3]) / 2, n[2] - n[0], n[3] - n[1], round(sc, 4), "owl").clamped()
        if nb.is_valid():
            out.append(nb)
            did[f"{c}_add"] += 1
    changed = [box_key(b) for b in out] != [box_key(b) for b in boxes]
    return out, (did if changed else collections.Counter())


def cmd_list(cfg: Config, store: Store, a) -> int:
    where, args = [], []
    if a.status:
        st = [s.strip() for s in a.status.split(",") if s.strip()]
        where.append(f"status IN ({','.join('?' * len(st))})")
        args += st
    if a.camera:
        where.append("camera=?")
        args.append(a.camera)
    ids = None
    if a.ids:
        ids = {int(x) for x in Path(a.ids).read_text(encoding="utf-8").split() if x.strip().isdigit()}
    if not where and ids is None:
        print("고를 조건이 없다 — --status·--camera·--ids 중 하나는 줄 것")
        return 2
    with store.connect() as con:
        rows = con.execute(f"SELECT id, path, status FROM images{' WHERE ' + ' AND '.join(where) if where else ''}"
                           " ORDER BY id", args).fetchall()
    rows = [r for r in rows if ids is None or r["id"] in ids]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(f"{r['id']}\t{r['path']}\n" for r in rows), encoding="utf-8")
    print(f"{len(rows):,}장 → {a.out}  (상태별 {dict(collections.Counter(r['status'] for r in rows))})")
    return 0


def class_index(cfg: Config, store: Store, write: bool) -> dict[str, int] | None:
    """DB 클래스 → 번호. 새 두 클래스가 DB 에 없으면 --apply 때 config.yaml 순서대로 붙인다(번호가 config 와 같아야 한다)."""
    names = store.get_classes()
    want = list(cfg.get("classes") or [])
    missing = [c for c in NEW_CLASSES if c not in names]
    if missing:
        if not write:
            print(f"DB 클래스에 아직 없음: {', '.join(missing)} — --apply 때 붙인다")
            names = names + missing
        else:
            for c in missing:
                store.add_class(c)
            names = store.get_classes()
            print(f"DB 클래스에 붙였다: {', '.join(missing)}")
    for c in NEW_CLASSES:
        if c in want and want.index(c) != names.index(c):
            print(f"[멈춤] '{c}' 의 번호가 config.yaml({want.index(c)})과 DB({names.index(c)})에서 다르다 — DB classes 표를 먼저 볼 것")
            return None
    return {n: i for i, n in enumerate(names)}


def backup(cfg: Config, stamp: str) -> Path:
    bak = cfg.db.with_name(f"{cfg.db.name}.bak_owl_{stamp}")
    with sqlite3.connect(cfg.db) as src, sqlite3.connect(bak) as dst:
        src.backup(dst)
    return bak


def cmd_apply(cfg: Config, store: Store, a) -> int:
    idx = class_index(cfg, store, write=False)
    if idx is None:
        return 2
    mins = {"responder": a.min_responder, "cone": a.min_cone}
    recs = [json.loads(line) for line in a.proposals.read_text(encoding="utf-8").splitlines() if line.strip()]
    todo, tally, skipped = [], collections.Counter(), collections.Counter()
    by_status = collections.Counter()
    for rec in recs:
        if rec.get("error"):
            skipped["읽기 오류"] += 1
            continue
        img = store.get_image(rec["id"])
        if img is None:
            skipped["DB 에 없음"] += 1
            continue
        if img["status"] not in ("approved", "pending"):
            skipped[f"상태 {img['status']}"] += 1
            continue
        before = store.get_boxes(rec["id"])
        after, did = plan(before, rec, idx, mins)
        if not did:
            continue
        todo.append((img, before, after, did))
        tally.update(did)
        by_status[img["status"]] += 1
    print(f"후보 파일 {len(recs):,}장 · 바뀔 이미지 {len(todo):,}장 (지금 상태 {dict(by_status)}) · 건너뜀 {dict(skipped)}")
    print("  박스:", ", ".join(f"{k} {v:,}" for k, v in sorted(tally.items())))
    print(f"  문턱: responder ≥ {a.min_responder} · cone ≥ {a.min_cone}")
    for img, before, after, did in todo[: a.show]:
        print(f"  id {img['id']:>6} {img['status']:8} {len(before)}→{len(after)}개 {dict(did)}  {Path(img['path']).name}")
    if not a.apply:
        print("\n(보기만 했다 — 넣으려면 --apply. 승인 이미지는 검수 대기로 간다)")
        return 0
    if not todo:
        return 0
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"DB 를 떠 두었다: {backup(cfg, stamp)}")
    idx = class_index(cfg, store, write=True)
    if idx is None:
        return 2
    batch = f"OWLv2 후보 {datetime.now():%m-%d %H:%M}"
    low = float(cfg.get("review.low_band", 0.30))
    high = float(cfg.get("review.high_band", 0.80))
    empty = float(cfg.get("review.empty_score", 0.60))
    log = a.proposals.with_name(f"applied_{stamp}.tsv")
    done = changed_status = moved = 0
    with log.open("w", encoding="utf-8") as f:
        f.write("id\told_status\tn_before\tn_after\tdid\tbefore\n")
        for img, before, after, did in todo:
            if [box_key(b) for b in store.get_boxes(img["id"])] != [box_key(b) for b in before]:
                moved += 1   # 고른 뒤 사람이 고쳤다 — 덮지 않는다
                continue
            score = image_uncertainty([b.conf for b in after], low=low, high=high, empty=empty)
            store.set_boxes(img["id"], after, uncertainty=score, source="bulk", batch=batch)
            if img["status"] == "approved":
                store.set_status(img["id"], "pending")
                changed_status += 1
            f.write(f"{img['id']}\t{img['status']}\t{len(before)}\t{len(after)}\t{json.dumps(dict(did))}\t"
                    f"{json.dumps([b.to_dict() for b in before])}\n")
            done += 1
    print(f"넣었다: {done:,}장(그중 승인 → 검수 대기 {changed_status:,}장) · 그 사이 바뀌어 건너뜀 {moved} · 묶음 '{batch}' · 목록 {log}")
    return 0


def cmd_restore(cfg: Config, store: Store, a) -> int:
    """묶음을 되돌린 뒤 — 박스가 원래대로 돌아왔고 지금 검수 대기인 옛 승인 이미지를 승인으로."""
    n_ok, n_skip = [], collections.Counter()
    for line in a.log.read_text(encoding="utf-8").splitlines()[1:]:
        image_id, old_status, _nb, _na, _did, before = line.split("\t")
        if old_status != "approved":
            continue
        img = store.get_image(int(image_id))
        if img is None or img["status"] != "pending":
            n_skip["지금 검수 대기가 아님"] += 1
            continue
        want = [box_key(Box(**d)) for d in json.loads(before)]
        if [box_key(b) for b in store.get_boxes(int(image_id))] != want:
            n_skip["박스가 원래와 다름(되돌리지 않았거나 그 뒤 고침)"] += 1
            continue
        n_ok.append(int(image_id))
    print(f"승인으로 돌릴 이미지 {len(n_ok):,}장 · 건너뜀 {dict(n_skip)}")
    if a.apply:
        for image_id in n_ok:
            store.set_status(image_id, "approved")
        print("돌렸다")
    else:
        print("(보기만 했다 — 돌리려면 --apply)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="OWLv2 후보 → responder·cone 검수 대기 (머리말 참고)")
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list", help="이미지 목록(TSV: id·경로)을 쓴다")
    p.add_argument("out", type=Path)
    p.add_argument("--status", help="쉼표로 — 예: approved,pending")
    p.add_argument("--camera")
    p.add_argument("--ids", help="id 를 담은 파일(공백·줄 구분)")
    p = sub.add_parser("apply", help="후보 JSONL 을 박스로(기본은 보기만)")
    p.add_argument("proposals", type=Path)
    p.add_argument("--min-responder", type=float, default=0.30)
    p.add_argument("--min-cone", type=float, default=0.40)
    p.add_argument("--show", type=int, default=15, help="예시로 보일 이미지 수")
    p.add_argument("--apply", action="store_true")
    p = sub.add_parser("restore", help="되돌린 묶음의 옛 승인 이미지를 승인으로")
    p.add_argument("log", type=Path)
    p.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    cfg = Config.load(a.config)
    store = Store(cfg.db)
    return {"list": cmd_list, "apply": cmd_apply, "restore": cmd_restore}[a.cmd](cfg, store, a)


if __name__ == "__main__":
    raise SystemExit(main())
