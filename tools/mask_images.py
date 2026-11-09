"""DB 이미지의 사각형 영역을 **회색(114)으로 칠한다** — 사람이 가를 수 없는 무리를 학습에서 빼는 '무시 영역'. 반 넘게
덮인 박스는 지우고, 상태(승인·검수 대기·건너뜀·휴지통)는 그대로 둔다.

**왜(2026-11-09 사용자 결정).** YOLO(Ultralytics) 에는 '이 자리는 보지 마' 하는 무시 영역이 없다. 사람이 한 명씩 가를 수
없는 무리를 그리지 않고 두면 모델은 그 사람들을 '사람 아님'으로 배우고, 장을 통째로 빼면 나머지 라벨까지 잃는다. 그 자리만
학습 때 여백 색(Ultralytics 레터박스·모자이크 바탕 114)으로 칠하면 그 자리에서는 아무것도 배우지 않는다. 처음 쓴 곳은
hw-04 줌 화면의 소방차와 트레일러 사이 틈(사다리 뒤 대원 무리 — 8030 과 건너뛴 10장).
- 물체 박스 **안**을 칠하면 그 물체에 회색 조각이 붙어 배운다 — 그럴 땐 칠하지 말고 자른다(`tools/crop_images.py`).
  영역은 라벨 박스를 덮지 않게 좁게 잡는다(반 미만으로 덮인 박스는 그대로 두고 보기에 '일부 덮임' 으로 알린다).
- **test 카메라에는 칠하지 않는다**(멈춘다) — 평가는 실제 화면 그대로여야 한다.

**규칙.**
- 영역은 지금 이미지 픽셀 `id:x1,y1,x2,y2`(`--mask` 를 여러 번, 같은 id 를 되풀이하면 그 장에 여러 개). 영역들의 합집합이
  박스 넓이의 반 이상을 덮으면 그 박스를 지운다.
- 새 파일은 같은 폴더에 이름 끝 `_m`(이미 `_m` 이면 그대로), JPEG 95. 옛 파일은 지우지 않고 `<DB 폴더>/backups/mask_<시각>/`
  로 옮긴다(목록 `manifest.tsv` 도 거기 — git 밖).
- DB 는 `store.replace_file(..., keep_status=True, source='mask')` — 크기는 그대로라 박스 좌표는 바뀌지 않는다. 지운 박스는
  변경 이력('회색 칠하기')에 남고, dhash 는 비운다(다음 분할 때 다시 잰다). 불확실도는 남은 박스로 다시 매긴다.

**순서.** 기본은 보기만 한다(장마다 지울·일부 덮이는 박스). `--preview <jpg>` 는 영역 둘레를 칠하기 전·후로 나란히 그린
대조 그림을 쓴다(파일·DB 는 그대로). `--apply` 면 DB 를 `labeling.db.bak_mask_<시각>` 으로 떠 두고 쓴다. 도는 사이 검수
화면에서 바뀐 이미지는 덮지 않고 건너뛴다. **돌린 뒤 검수 화면을 새로 고칠 것**(열려 있던 장은 옛 그림이 보인다).

    python tools/mask_images.py --camera hw-04 --mask 8030:480,236,529,328 --preview runs_preview.jpg
    python tools/mask_images.py --camera hw-04 --mask 8030:480,236,529,328 --apply
"""

from __future__ import annotations

import argparse
import collections
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # tools/ -> 저장소 루트

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from autolabel.config import Config  # noqa: E402
from autolabel.provenance import load_cameras  # noqa: E402
from autolabel.scoring import image_uncertainty  # noqa: E402
from autolabel.store import Store  # noqa: E402

GRAY = (114, 114, 114)          # Ultralytics 레터박스·모자이크 바탕과 같은 색
DROP = 0.5                      # 박스 넓이의 이만큼 이상이 칠해지면 지운다

Rect = tuple[int, int, int, int]


def parse_mask(text: str) -> tuple[int, Rect]:
    try:
        iid, rest = text.split(":", 1)
        x1, y1, x2, y2 = (int(v) for v in rest.split(","))
    except ValueError:
        raise SystemExit(f"--mask 는 id:x1,y1,x2,y2 (지금 이미지 픽셀) — 지금 {text!r}") from None
    if min(x1, y1) < 0 or x1 >= x2 or y1 >= y2:
        raise SystemExit(f"--mask 영역이 틀렸다(x1<x2·y1<y2, 0 이상): {text!r}")
    return int(iid), (x1, y1, x2, y2)


def masked_name(name: str) -> str:
    stem = Path(name).stem
    return f"{stem if stem.endswith('_m') else stem + '_m'}.jpg"


def covered(box, img_w: int, img_h: int, cover: np.ndarray) -> float:
    """박스 넓이 가운데 칠해지는 몫(픽셀로 센다 — 영역이 여러 개여도 겹친 곳을 두 번 세지 않는다)."""
    a = max(0, int(round((box.cx - box.w / 2) * img_w)))
    t = max(0, int(round((box.cy - box.h / 2) * img_h)))
    c = min(img_w, int(round((box.cx + box.w / 2) * img_w)))
    d = min(img_h, int(round((box.cy + box.h / 2) * img_h)))
    if c <= a or d <= t:
        return 0.0
    return float(cover[t:d, a:c].mean())


def preview(out: Path, items: list, names: list[str]) -> None:
    """영역 둘레(여백 40px)를 칠하기 전·후로 나란히 — 박스는 남는 것 노랑, 지울 것 빨강."""
    tiles = []
    for r, img, painted, boxes, fracs, rects in items:
        h, w = img.shape[:2]
        x1 = max(0, min(q[0] for q in rects) - 40)
        y1 = max(0, min(q[1] for q in rects) - 40)
        x2 = min(w, max(q[2] for q in rects) + 40)
        y2 = min(h, max(q[3] for q in rects) + 40)
        pair = []
        for src in (img, painted):
            roi = cv2.resize(src[y1:y2, x1:x2], None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            for b, f in zip(boxes, fracs):
                a, t = (b.cx - b.w / 2) * w, (b.cy - b.h / 2) * h
                c, d = (b.cx + b.w / 2) * w, (b.cy + b.h / 2) * h
                col = (40, 40, 255) if f >= DROP else (0, 220, 255)
                cv2.rectangle(roi, (int((a - x1) * 3), int((t - y1) * 3)), (int((c - x1) * 3), int((d - y1) * 3)), col, 2)
            pair.append(roi)
        tile = np.hstack([pair[0], np.full((pair[0].shape[0], 6, 3), 255, np.uint8), pair[1]])
        label = f"{r['id']} {r['status']}  drop {sum(f >= DROP for f in fracs)}"
        cv2.putText(tile, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(tile, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        tiles.append(tile)
    hmax = max(t.shape[0] for t in tiles)
    tiles = [np.vstack([t, np.full((hmax - t.shape[0], t.shape[1], 3), 255, np.uint8)]) for t in tiles]
    rows = []
    for i in range(0, len(tiles), 3):
        row = tiles[i:i + 3]
        row = [x for t in row for x in (t, np.full((hmax, 10, 3), 255, np.uint8))][:-1]
        rows.append(np.hstack(row))
    wmax = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.full((r.shape[0], wmax - r.shape[1], 3), 255, np.uint8)]) for r in rows]
    sheet = np.vstack([x for r in rows for x in (r, np.full((10, wmax, 3), 255, np.uint8))][:-1])
    cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(str(out))
    print(f"대조 그림: {out} ({sheet.shape[1]}×{sheet.shape[0]})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--camera", required=True, help="고른 이미지가 모두 이 카메라여야 한다(test 카메라면 멈춘다)")
    ap.add_argument("--mask", action="append", default=[], help="id:x1,y1,x2,y2 — 여러 번")
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--preview", type=Path, default=None, help="칠하기 전·후 대조 그림(JPEG)을 여기에 쓴다")
    ap.add_argument("--apply", action="store_true", help="실제로 칠한다(아니면 보기만)")
    a = ap.parse_args()
    if not a.mask:
        raise SystemExit("--mask 를 하나 이상 주세요")

    cfg = Config.load(a.config)
    cam = load_cameras(cfg.cameras).get(a.camera)
    if cam is None:
        raise SystemExit(f"{cfg.cameras} 에 {a.camera} 가 없다")
    if cam.split == "test":
        raise SystemExit(f"{a.camera} 는 test 카메라다 — 평가 그림은 칠하지 않는다")
    store = Store(cfg.db)
    names = store.get_classes()
    masks: dict[int, list[Rect]] = collections.defaultdict(list)
    for m in a.mask:
        iid, rect = parse_mask(m)
        masks[iid].append(rect)
    ids = sorted(masks)
    with store.connect() as con:
        q = ",".join("?" * len(ids))
        rows = con.execute(f"SELECT id, path, name, status, updated_at, width, height, camera, model_version"
                           f" FROM images WHERE id IN ({q}) ORDER BY id", ids).fetchall()
    missing = sorted(set(ids) - {r["id"] for r in rows})
    other = [r["id"] for r in rows if r["camera"] != a.camera]
    if missing or other:
        raise SystemExit(f"멈춘다 — DB 에 없는 id {missing} · 다른 카메라 {other}")

    low = float(cfg.get("review.low_band", 0.30))
    high = float(cfg.get("review.high_band", 0.80))
    empty = float(cfg.get("review.empty_score", 0.60))
    plan, shown = [], []
    tally: collections.Counter = collections.Counter()
    for r in rows:
        w, h = r["width"], r["height"]
        bad = [q for q in masks[r["id"]] if q[2] > w or q[3] > h]
        if bad:
            raise SystemExit(f"{r['id']}: 영역 {bad} 이 이미지({w}×{h}) 밖으로 나간다")
        img = cv2.imdecode(np.fromfile(r["path"], np.uint8), cv2.IMREAD_COLOR)
        if img is None or img.shape[:2] != (h, w):
            raise SystemExit(f"{r['id']}: 파일을 못 읽거나 크기가 DB 와 다르다 — {r['path']}")
        cover = np.zeros((h, w), np.float32)
        painted = img.copy()
        for x1, y1, x2, y2 in masks[r["id"]]:
            cover[y1:y2, x1:x2] = 1.0
            painted[y1:y2, x1:x2] = GRAY
        boxes = store.get_boxes(r["id"])
        fracs = [covered(b, w, h, cover) for b in boxes]
        after = [b for b, f in zip(boxes, fracs) if f < DROP]
        drop = [names[b.cls] for b, f in zip(boxes, fracs) if f >= DROP]
        part = [f"{names[b.cls]} {f:.0%}" for b, f in zip(boxes, fracs) if 0.01 <= f < DROP]   # 1% 미만은 경계 반올림
        tally[r["status"], "drop"] += len(drop)
        tally[r["status"], "part"] += len(part)
        plan.append((r, painted, boxes, after))
        shown.append((r, img, painted, boxes, fracs, masks[r["id"]]))
        print(f"  {r['id']} {r['status']:8s} 영역 {masks[r['id']]}: 박스 {len(boxes)} → {len(after)}"
              + (f" · 지움 {dict(collections.Counter(drop))}" if drop else "")
              + (f" · 일부 덮임 {part}" if part else ""))
    print(f"\n{len(rows)}장 — " + " · ".join(f"{st}: 지움 {tally[st, 'drop']}·일부 덮임 {tally[st, 'part']}"
                                       for st in sorted({k[0] for k in tally})))
    if a.preview:
        preview(a.preview, shown, names)
    if not a.apply:
        print("\n(보기만 했다 — 칠하려면 --apply. DB 를 떠 두고, 옛 파일은 backups/mask_<시각>/ 로 옮긴다)")
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = cfg.db.with_name(f"{cfg.db.name}.bak_mask_{stamp}")
    with sqlite3.connect(cfg.db) as src, sqlite3.connect(bak) as dst:
        src.backup(dst)
    print(f"\nDB 를 떠 두었다: {bak}")
    keep_dir = cfg.db.parent / "backups" / f"mask_{stamp}"
    keep_dir.mkdir(parents=True, exist_ok=False)
    done, changed, stuck = 0, [], []
    lines = ["id\tstatus\told_name\tnew_name\tmasks\tn_before\tn_after"]
    for r, painted, boxes, after in plan:
        old = Path(r["path"])
        new = old.with_name(masked_name(r["name"]))
        ok, buf = cv2.imencode(".jpg", painted, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
        if not ok:
            stuck.append(f"{r['id']}(JPEG)")
            continue
        if new != old and new.exists():
            stuck.append(f"{r['id']}(새 이름이 이미 있음)")
            continue
        # 옛 파일을 먼저 백업 폴더로 옮긴다 — 이름이 같아도(_m 을 다시 칠할 때) 덮어쓰기 전에 남는다
        try:
            shutil.move(str(old), str(keep_dir / old.name))
        except OSError as exc:               # 검수 서버가 그 순간 읽고 있으면 Windows 가 막을 수 있다 — 그 장은 그대로
            stuck.append(f"{r['id']}(옛 파일 옮기기: {exc})")
            continue
        new.write_bytes(buf.tobytes())
        score = image_uncertainty([b.conf for b in after], low=low, high=high, empty=empty)
        if not store.replace_file(r["id"], str(new.resolve()), r["width"], r["height"], after,
                                  expect=(r["status"], r["updated_at"]), uncertainty=score,
                                  model_version=r["model_version"] or "", keep_status=True, source="mask"):
            new.unlink(missing_ok=True)      # 도는 사이 검수 화면에서 바뀌었다 — 옛 파일을 제자리로
            shutil.move(str(keep_dir / old.name), str(old))
            changed.append(r["id"])
            continue
        lines.append(f"{r['id']}\t{r['status']}\t{old.name}\t{new.name}\t{masks[r['id']]}\t{len(boxes)}\t{len(after)}")
        done += 1
    (keep_dir / "manifest.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"칠함 {done}장 · 옛 파일 {keep_dir}"
          + (f" · 도는 사이 바뀌어 그대로 둔 것 {changed}" if changed else "")
          + (f" · [경고] {stuck}" if stuck else ""))
    return 0 if not stuck else 1


if __name__ == "__main__":
    sys.exit(main())
