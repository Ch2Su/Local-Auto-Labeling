"""DB 이미지에 **가릴 영역**(학습 세트를 만들 때만 회색 114 로 칠할 사각형 — `src/autolabel/masks.py`)을 넣는다. 원본 그림·
박스·상태는 그대로다.

검수 화면의 M(가릴 영역) 모드와 같은 일을 명령으로 한다 — 고정 카메라의 연속 프레임처럼 같은 자리를 여러 장에 넣을 때
쓴다. **처음 판(2026-11-09 오후)은 그림 파일에 직접 칠했다**(이름 끝 `_m`, hw-04 사다리 뒤 대원 무리 11장). 원본을
바꾸지 않으려고 같은 날 영역을 DB 에 두고 학습 세트를 만들 때만 칠하게 바꿨고, 그 11장은 칠하기 전 그림으로 되돌렸다
(docs/provenance.md).

**규칙.**
- 영역은 지금 이미지(EXIF 회전을 적용한 화면) 픽셀 `id:x1,y1,x2,y2` — `--mask` 를 여러 번, 같은 id 를 되풀이하면 그 장에
  여러 개. 여러 장에 같은 영역은 `--ids 8017-8029 --rect x1,y1,x2,y2`.
- 이미 있는 영역 뒤에 더한다. `--replace` 면 고른 장의 영역을 이것으로 바꾼다(`--replace` 에 영역을 안 주면 그 장들의
  영역을 지운다 — `--ids` 만).
- **test 이미지에는 넣지 않는다**(분할이 test 이거나 cameras.yaml 에서 test 로 고정한 카메라면 멈춘다) — 평가는 실제
  화면 그대로여야 한다.
- 영역들의 합집합이 박스 넓이의 반 이상을 덮으면 그 박스는 **학습 라벨에서만** 빠진다(DB 박스는 그대로). 반 미만으로
  덮이는 박스는 그만큼 칠해진 채로 배운다 — 조금은 괜찮고, 크게 덮이면 영역을 좁히거나 그림 가장자리면 자른다
  (`tools/crop_images.py`, 판정은 docs/label-rules.md) — 보기에서 '일부 가림'.
- DB 는 `store.set_masks(..., source='mask')` — 영역 전후가 변경 이력('가릴 영역')에 남고 updated_at 이 바뀐다.

**순서.** 기본은 보기만 한다. `--preview <jpg>` 는 영역 둘레를 칠하기 전·후로 나란히 그린 대조 그림(박스: 학습에서 빠짐
빨강, 일부 가림 노랑). `--apply` 면 DB 를 `labeling.db.bak_mask_<시각>` 으로 떠 두고 쓴다. 도는 사이 검수 화면에서 바뀐
이미지는 건너뛴다. **쓴 뒤에는 그 이미지를 열어 둔 검수 화면을 새로 고칠 것**(새로 고치기 전에는 영역이 안 보인다).

    python tools/mask_images.py --mask 8030:480,236,529,328 --preview mask_check.jpg
    python tools/mask_images.py --ids 8017-8029 --rect 480,240,527,346 --apply
"""

from __future__ import annotations

import argparse
import collections
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # tools/ -> 저장소 루트

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from autolabel.config import Config  # noqa: E402
from autolabel.masks import DROP, MAX_MASKS, clean, covered, pixel_rect  # noqa: E402
from autolabel.provenance import load_cameras  # noqa: E402
from autolabel.store import Store  # noqa: E402

GRAY = (114, 114, 114)

Rect = tuple[int, int, int, int]


def parse_rect(text: str, what: str) -> Rect:
    try:
        x1, y1, x2, y2 = (int(v) for v in text.split(","))
    except ValueError:
        raise SystemExit(f"{what} 는 x1,y1,x2,y2 (지금 이미지 픽셀) — 지금 {text!r}") from None
    if min(x1, y1) < 0 or x1 >= x2 or y1 >= y2:
        raise SystemExit(f"{what} 영역이 틀렸다(x1<x2·y1<y2, 0 이상): {text!r}")
    return x1, y1, x2, y2


def parse_mask(text: str) -> tuple[int, Rect]:
    iid, sep, rest = text.partition(":")
    if not sep or not iid.strip().isdigit():
        raise SystemExit(f"--mask 는 id:x1,y1,x2,y2 — 지금 {text!r}")
    return int(iid), parse_rect(rest, "--mask")


def parse_ids(text: str) -> list[int]:
    """'8017-8029,8030' → [8017, …, 8029, 8030]."""
    out: list[int] = []
    for part in (p.strip() for p in text.split(",")):
        if not part:
            continue
        a, sep, b = part.partition("-")
        lo, hi = int(a), int(b) if sep else int(a)
        if lo > hi:
            raise SystemExit(f"--ids 의 범위가 거꾸로다: {part}")
        out.extend(range(lo, hi + 1))
    return sorted(set(out))


def preview(out: Path, items: list) -> None:
    """영역 둘레(여백 40px)를 칠하기 전·후로 나란히 — 박스는 학습에서 빠짐 빨강, 일부 가림 노랑, 나머지는 그리지 않는다."""
    tiles = []
    for iid, img, rects_px, boxes, fracs in items:
        h, w = img.shape[:2]
        painted = img.copy()
        for x1, y1, x2, y2 in rects_px:
            painted[y1:y2, x1:x2] = GRAY
        if rects_px:
            x1 = max(0, min(q[0] for q in rects_px) - 40)
            y1 = max(0, min(q[1] for q in rects_px) - 40)
            x2 = min(w, max(q[2] for q in rects_px) + 40)
            y2 = min(h, max(q[3] for q in rects_px) + 40)
        else:
            x1, y1, x2, y2 = 0, 0, w, h
        pair = []
        for src in (img, painted):
            roi = cv2.resize(src[y1:y2, x1:x2], None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            for b, f in zip(boxes, fracs):
                if f < 0.01:
                    continue
                a, t = (b.cx - b.w / 2) * w, (b.cy - b.h / 2) * h
                c, d = (b.cx + b.w / 2) * w, (b.cy + b.h / 2) * h
                col = (40, 40, 255) if f >= DROP else (0, 220, 255)
                cv2.rectangle(roi, (int((a - x1) * 3), int((t - y1) * 3)), (int((c - x1) * 3), int((d - y1) * 3)), col, 2)
            pair.append(roi)
        tile = np.hstack([pair[0], np.full((pair[0].shape[0], 6, 3), 255, np.uint8), pair[1]])
        label = f"{iid}  out {sum(f >= DROP for f in fracs)}  part {sum(0.01 <= f < DROP for f in fracs)}"
        cv2.putText(tile, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(tile, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        tiles.append(tile)
    hmax = max(t.shape[0] for t in tiles)
    tiles = [np.vstack([t, np.full((hmax - t.shape[0], t.shape[1], 3), 255, np.uint8)]) for t in tiles]
    rows = []
    for i in range(0, len(tiles), 3):
        row = [x for t in tiles[i:i + 3] for x in (t, np.full((hmax, 10, 3), 255, np.uint8))][:-1]
        rows.append(np.hstack(row))
    wmax = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.full((r.shape[0], wmax - r.shape[1], 3), 255, np.uint8)]) for r in rows]
    sheet = np.vstack([x for r in rows for x in (r, np.full((10, wmax, 3), 255, np.uint8))][:-1])
    cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(str(out))
    print(f"대조 그림: {out} ({sheet.shape[1]}×{sheet.shape[0]})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--mask", action="append", default=[], help="id:x1,y1,x2,y2 — 여러 번")
    ap.add_argument("--ids", default=None, help="--rect 를 넣을 이미지 id — 8017-8029,8030 처럼")
    ap.add_argument("--rect", action="append", default=[], help="--ids 의 장마다 넣을 영역 x1,y1,x2,y2 — 여러 번")
    ap.add_argument("--replace", action="store_true", help="이미 있는 영역 뒤에 더하지 않고 이것으로 바꾼다")
    ap.add_argument("--preview", type=Path, default=None, help="칠하기 전·후 대조 그림(JPEG)을 여기에 쓴다")
    ap.add_argument("--apply", action="store_true", help="실제로 넣는다(아니면 보기만)")
    a = ap.parse_args()
    if bool(a.ids) != bool(a.rect) and not (a.ids and a.replace):
        raise SystemExit("--ids 와 --rect 는 함께 준다(--replace 로 지울 때만 --ids 혼자)")

    wanted: dict[int, list[Rect]] = collections.defaultdict(list)
    for m in a.mask:
        iid, rect = parse_mask(m)
        wanted[iid].append(rect)
    if a.ids:
        rects = [parse_rect(t, "--rect") for t in a.rect]
        for iid in parse_ids(a.ids):
            wanted[iid].extend(rects)
    if not wanted:
        raise SystemExit("--mask 나 --ids(+--rect)로 이미지를 하나 이상 고르세요")

    cfg = Config.load(a.config)
    store = Store(cfg.db)
    cams = load_cameras(cfg.cameras)
    ids = sorted(wanted)
    with store.connect() as con:
        q = ",".join("?" * len(ids))
        rows = con.execute(f"SELECT id, path, status, updated_at, camera, split FROM images WHERE id IN ({q}) ORDER BY id",
                           ids).fetchall()
    missing = sorted(set(ids) - {r["id"] for r in rows})
    tests = [r["id"] for r in rows
             if r["split"] == "test" or (r["camera"] in cams and cams[r["camera"]].split == "test")]
    if missing or tests:
        raise SystemExit(f"멈춘다 — DB 에 없는 id {missing} · test 이미지 {tests} (평가는 실제 화면 그대로)")

    plan, shown = [], []
    tally: collections.Counter = collections.Counter()
    for r in rows:
        img = cv2.imdecode(np.fromfile(r["path"], np.uint8), cv2.IMREAD_COLOR)   # EXIF 회전을 적용해 연다(화면과 같다)
        if img is None:
            raise SystemExit(f"{r['id']}: 파일을 못 읽는다 — {r['path']}")
        h, w = img.shape[:2]
        bad = [q for q in wanted[r["id"]] if q[2] > w or q[3] > h]
        if bad:
            raise SystemExit(f"{r['id']}: 영역 {bad} 이 이미지({w}×{h}) 밖으로 나간다")
        add = [(x1 / w, y1 / h, x2 / w, y2 / h) for x1, y1, x2, y2 in wanted[r["id"]]]
        old = store.get_masks(r["id"])
        new = clean(add if a.replace else [*old, *add])
        if len(new) > MAX_MASKS:
            raise SystemExit(f"{r['id']}: 가릴 영역이 {len(new)}개 — 한 장에 {MAX_MASKS}개까지")
        boxes = store.get_boxes(r["id"])
        fracs = [covered(b, new) for b in boxes]
        out = collections.Counter(b.cls for b, f in zip(boxes, fracs) if f >= DROP)
        part = [f"{b.cls}:{f:.0%}" for b, f in zip(boxes, fracs) if 0.01 <= f < DROP]   # 1% 미만은 경계 반올림
        tally["out"] += sum(out.values())
        tally["part"] += len(part)
        plan.append((r, new))
        shown.append((r["id"], img, [pixel_rect(m, w, h) for m in new], boxes, fracs))
        print(f"  {r['id']} {r['status']:8s} 영역 {len(old)} → {len(new)}: 박스 {len(boxes)}"
              + (f" · 학습에서 빠짐 {dict(out)}" if out else "")
              + (f" · 일부 가림(클래스:몫) {part}" if part else ""))
    print(f"\n{len(rows)}장 — 학습 라벨에서 빠지는 박스 {tally['out']} · 일부 가려지는 박스 {tally['part']}"
          " (DB 박스는 그대로)")
    if a.preview:
        preview(a.preview, shown)
    if not a.apply:
        print("\n(보기만 했다 — 넣으려면 --apply. DB 를 떠 둔다)")
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = cfg.db.with_name(f"{cfg.db.name}.bak_mask_{stamp}")
    with sqlite3.connect(cfg.db) as src, sqlite3.connect(bak) as dst:
        src.backup(dst)
    print(f"\nDB 를 떠 두었다: {bak}")
    done, changed = 0, []
    for r, new in plan:
        if store.set_masks(r["id"], new, expect=(r["status"], r["updated_at"]), source="mask"):
            done += 1
        else:
            changed.append(r["id"])
    print(f"넣음 {done}장" + (f" · 도는 사이 바뀌어 건너뛴 것 {changed}" if changed else "")
          + " — 그 이미지를 열어 둔 검수 화면은 새로 고칠 것")
    return 0


if __name__ == "__main__":
    sys.exit(main())
