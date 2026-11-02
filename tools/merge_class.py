"""맨 끝 클래스를 다른 클래스에 합친다 — 박스를 옮기고 그 클래스를 목록에서 뺀다 (2026-11-02, van → car).

**맨 끝 클래스만 합친다.** 중간 번호를 빼면 뒤 번호가 모두 밀려 박스·변경 이력·학습한 가중치가 다른
클래스를 가리키게 된다(CLAUDE.md '클래스 번호는 순서가 전부다').

**config.yaml 의 classes 에서 먼저 뺄 것.** 남아 있으면 다음 scan 이 그 클래스를 DB 에 되살린다
(`ingest.discover_classes`). 그래서 config 의 목록이 'DB 목록에서 합칠 클래스만 뺀 것' 이 아니면 멈춘다.

**박스는 클래스만 바꾼다** — 신뢰도·origin 은 그대로 둔다. 검수 UI 의 일괄 수정(`set_class_bulk`)은 사람이
고른 것으로 보고 origin='human' 으로 바꾸는데, 검수 대기인 O365 박스까지 사람이 본 것처럼 되면 안 된다.
이미지마다 box_history 에 전후를 남긴다(source='bulk', batch='클래스 합침: <원래> → <합칠 곳> <시각>').

**되돌리기.** 떠 둔 DB(`labeling.db.bak_merge_<시각>`)로 바꾸는 것이 가장 확실하다. 그 뒤에 고친 것을
살려야 하면, 검수 UI '클래스 추가' 로 그 이름을 맨 끝에 다시 넣고 '변경 기록' 에서 그 묶음을 되돌린다
(그 뒤 다시 고친 이미지는 건너뛴다).

검수 서버는 끄고 돌릴 것 — 열려 있던 화면은 새로 고쳐야 새 클래스 목록을 받는다.

    python tools/merge_class.py van car            # 무엇이 바뀌는지 보여만 준다 (DB 를 건드리지 않음)
    python tools/merge_class.py van car --apply    # DB 를 떠 두고 합친다
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # tools/ -> 저장소 루트

from autolabel.config import Config  # noqa: E402
from autolabel.store import Store, _iou_xywh, _now, _write_lock  # noqa: E402
from autolabel.yolo_io import Box  # noqa: E402

OVERLAP = 0.85   # 합친 뒤 같은 자리에 겹치는 박스 — label_qa 의 duplicate 와 같은 기준


def approved_counts(store: Store, names: list[str]) -> dict[str, int]:
    with store.connect() as con:
        rows = con.execute("SELECT b.cls, COUNT(*) FROM boxes b JOIN images i ON i.id = b.image_id"
                           " WHERE i.status = 'approved' GROUP BY b.cls ORDER BY b.cls").fetchall()
    return {(names[c] if c < len(names) else f"cls_{c}"): n for c, n in rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="합쳐서 없앨 클래스 (목록 맨 끝이어야 한다)")
    ap.add_argument("dst", help="합칠 곳")
    ap.add_argument("--config", help="설정 파일 (기본: 저장소의 config.yaml)")
    ap.add_argument("--apply", action="store_true", help="실제로 DB 를 바꾼다 (먼저 DB 를 떠 둔다)")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    store = Store(cfg.db)
    names = store.get_classes()
    for n in (args.src, args.dst):
        if n not in names:
            sys.exit(f"DB 에 '{n}' 클래스가 없습니다: {names}")
    si, di = names.index(args.src), names.index(args.dst)
    if si == di:
        sys.exit("같은 클래스입니다.")
    if si != len(names) - 1:
        sys.exit(f"'{args.src}' 는 {si}번이라 맨 끝({len(names) - 1}번)이 아닙니다 — 빼면 뒤 번호가 밀립니다.")
    keep = names[:-1]
    if list(cfg.get("classes") or []) != keep:
        sys.exit(f"config.yaml 의 classes 를 먼저 {keep} 로 고치세요 — 지금은 {cfg.get('classes')}."
                 " 남겨 두면 다음 scan 이 되살립니다.")

    with store.connect() as con:
        rows = con.execute(
            "SELECT b.image_id, b.cls, b.cx, b.cy, b.w, b.h, b.origin, i.status FROM boxes b"
            " JOIN images i ON i.id = b.image_id WHERE b.cls IN (?, ?)", (si, di)).fetchall()
    moving = [r for r in rows if r["cls"] == si]
    images = sorted({r["image_id"] for r in moving})
    # 합친 뒤 같은 자리에 dst 가 둘이 되는 곳 — 원래 src 박스와 dst 박스가 크게 겹친 것
    staying: dict[int, list] = {}
    for r in rows:
        if r["cls"] == di:
            staying.setdefault(r["image_id"], []).append(r)
    dups = []
    for a in moving:
        for b in staying.get(a["image_id"], []):
            v = _iou_xywh((a["cx"], a["cy"], a["w"], a["h"]), (b["cx"], b["cy"], b["w"], b["h"]))
            if v >= OVERLAP:
                dups.append((a["image_id"], a["status"], round(v, 3)))
    print(f"{args.src}({si}번) → {args.dst}({di}번): 박스 {len(moving):,}개 / 이미지 {len(images):,}장")
    print("  이미지 상태별 박스:", dict(Counter(r["status"] for r in moving)))
    print("  origin 별 박스:", dict(Counter(r["origin"] for r in moving)))
    print(f"  합치면 같은 자리에 {args.dst} 가 둘이 되는 곳 (IoU ≥ {OVERLAP}): {len(dups)}곳", sorted(dups))
    print("  승인 박스 수:", approved_counts(store, names))
    if not args.apply:
        print("\n보기만 했습니다. 바꾸려면 --apply")
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = cfg.db.with_name(f"{cfg.db.name}.bak_merge_{stamp}")
    src_con, bak_con = sqlite3.connect(cfg.db), sqlite3.connect(bak)
    try:
        src_con.backup(bak_con)
    finally:
        bak_con.close()
        src_con.close()
    print(f"\nDB 를 떠 두었습니다: {bak}")

    batch = f"클래스 합침: {args.src} → {args.dst} {stamp}"
    now = _now()
    with _write_lock, store.connect() as con:
        for image_id in images:
            old = con.execute("SELECT cls, cx, cy, w, h, conf, origin FROM boxes WHERE image_id=? ORDER BY id",
                              (image_id,)).fetchall()
            new = [Box(di if r["cls"] == si else r["cls"], r["cx"], r["cy"], r["w"], r["h"], r["conf"], r["origin"])
                   for r in old]
            Store._replace_boxes(con, image_id, new, source="bulk", batch=batch)
            con.execute("UPDATE images SET updated_at=? WHERE id=?", (now, image_id))
        left = con.execute("SELECT COUNT(*) FROM boxes WHERE cls=?", (si,)).fetchone()[0]
        if left:
            raise RuntimeError(f"{args.src} 박스가 {left}개 남아 되돌렸습니다.")
        con.execute("DELETE FROM classes WHERE idx=?", (si,))
    # DB 옆 classes.txt 는 이름을 잃었을 때 되살릴 기록이다 (server.py '클래스 추가' 와 같은 꼴)
    cfg.classes_txt.write_text("\n".join(keep) + "\n", encoding="utf-8")
    print(f"합쳤습니다 — 이미지 {len(images):,}장, 변경 기록 묶음 '{batch}'")
    print("  클래스:", store.get_classes())
    print("  승인 박스 수:", approved_counts(store, keep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
