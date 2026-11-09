"""DB 에 들어 있는 이미지를 사각형 영역으로 잘라 바꾼다 — 박스는 그 영역 좌표로 옮기고 **상태(승인·검수 대기·휴지통)는
그대로** 둔다.

**왜(2026-11-09 사용자 결정).** hw-04 를 줌을 당겨 찍은 화면(11:36~11:40 녹화 36장)에서, 오른쪽
소방차 뒤에 사람 한두 명이 화면에 찍힌 지명 글자와 소방차 사다리 사이에 끼어 있다. 키워 봐도 사람인지, 몇 명인지 가를 수
없어 그리든 안 그리든 짐작이다. VIRAT 0500·0101 처럼 그 자리를 잘라 낸다(x 824 오른쪽 — 사용자가 그은 선). 그 자리만
회색으로 덮는 방법은 그 자리가 소방차 박스 안이라 소방차에 회색 조각이 붙어 고르지 않았다. 앞서 찍은 넓은 화면(같은
카메라, 줌이 다르다)은 그 자리가 없어 자르지 않는다 — 그래서 카메라 단위(`cameras.yaml` 의 crop)가 아니라 id 로 고른다.

**규칙.**
- 영역은 지금 이미지 픽셀 `[x1, y1, x2, y2]`. 고른 이미지는 모두 크기가 같아야 한다(다르면 멈춘다 — 다른 화면이 섞였다).
- 박스는 영역으로 자르고, 원래 넓이의 30% 미만만 남거나 1px 이하면 버린다(`import_virat.crop_gt` 와 같은 규칙).
  사람이 그린 박스도 좌표만 옮긴다(다시 라벨링하지 않는다). 출처·신뢰도는 그대로다.
- 새 파일은 같은 폴더에 이름 끝 `_c<x1>-<y1>-<x2>-<y2>`(처음 통째 화면 기준 — 이미 잘린 이미지를 또 자르면 꼬리표를
  합친다), JPEG 95. 옛 파일은 지우지 않고 `<DB 폴더>/backups/crop_<시각>/` 로 옮긴다(목록 `manifest.tsv` 도 거기 —
  이미지 이름이 든 목록이라 git 밖에 둔다).
- DB 는 `store.replace_file(..., keep_status=True, source='crop')`: id·태그·도메인·카메라·상태 그대로, 이전 박스는 변경
  이력('자르기')에 남고, dhash 는 비운다(다음 분할 때 다시 잰다). 불확실도는 남은 박스로 다시 매긴다.
- 자르기 전 이력으로는 되돌릴 수 없다 — 박스가 통째 화면 좌표라 서버가 막는다(409). 일괄 수정 묶음 되돌리기도 이
  이미지들을 건너뛴다(지금 박스가 묶음 직후와 다르다).

**순서.** 기본은 보기만 한다(장마다 남는·줄어드는·잃는 박스). `--apply` 면 DB 를 `labeling.db.bak_crop_<시각>` 으로 떠
두고 쓴다. 도는 사이 검수 화면에서 바뀐 이미지는 덮지 않고 건너뛴다.

    python tools/crop_images.py --camera hw-04 --ids 7996-8031 --box 0,0,824,720            # 보기만
    python tools/crop_images.py --camera hw-04 --ids 7996-8031 --box 0,0,824,720 --apply
"""

from __future__ import annotations

import argparse
import collections
import re
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # tools/ -> 저장소 루트

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from autolabel.config import Config  # noqa: E402
from autolabel.scoring import image_uncertainty  # noqa: E402
from autolabel.store import Store  # noqa: E402
from autolabel.yolo_io import Box  # noqa: E402

KEEP = 0.3                                   # 영역에 원래 넓이의 이만큼도 안 남는 박스는 버린다
TAIL_RE = re.compile(r"_c(\d+)-(\d+)-(\d+)-(\d+)$")

Rect = tuple[int, int, int, int]


def parse_ids(text: str) -> list[int]:
    """'7996-8031,8040' → [7996, …, 8031, 8040]."""
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = (int(v) for v in part.split("-", 1))
            if a > b:
                raise SystemExit(f"--ids 의 범위가 거꾸로다: {part}")
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def parse_rect(text: str) -> Rect:
    try:
        x1, y1, x2, y2 = (int(v) for v in text.split(","))
    except ValueError:
        raise SystemExit(f"--box 는 x1,y1,x2,y2 (지금 이미지 픽셀) — 지금 {text!r}") from None
    if min(x1, y1) < 0 or x1 >= x2 or y1 >= y2:
        raise SystemExit(f"--box 가 틀렸다(x1<x2·y1<y2, 0 이상): {text!r}")
    return x1, y1, x2, y2


def crop_name(name: str, rect: Rect) -> str:
    """새 파일 이름. 꼬리표는 처음 통째 화면 기준 영역이라, 이미 잘린 이름이면 그 왼쪽 위만큼 더한다."""
    stem = Path(name).stem
    painted = stem.endswith("_m")            # tools/mask_images.py 가 칠한 그림 — 표시는 끝에 그대로 둔다
    if painted:
        stem = stem[:-2]
    ox = oy = 0
    m = TAIL_RE.search(stem)
    if m:
        ox, oy = int(m.group(1)), int(m.group(2))
        stem = stem[:m.start()]
    x1, y1, x2, y2 = rect
    return f"{stem}_c{ox + x1}-{oy + y1}-{ox + x2}-{oy + y2}{'_m' if painted else ''}.jpg"


def crop_box(b: Box, img_w: int, img_h: int, rect: Rect) -> tuple[Box | None, str]:
    """박스 하나를 영역 좌표로 — (새 박스 또는 None, 'kept' | 'clipped' | 'lost')."""
    x1, y1, x2, y2 = rect
    a, t, c, d = (b.cx - b.w / 2) * img_w, (b.cy - b.h / 2) * img_h, (b.cx + b.w / 2) * img_w, (b.cy + b.h / 2) * img_h
    ca, ct, cc, cd = max(a, x1), max(t, y1), min(c, x2), min(d, y2)
    if cc - ca <= 1 or cd - ct <= 1 or (cc - ca) * (cd - ct) < KEEP * (c - a) * (d - t):
        return None, "lost"
    cw, ch = x2 - x1, y2 - y1
    nb = Box(b.cls, ((ca + cc) / 2 - x1) / cw, ((ct + cd) / 2 - y1) / ch, (cc - ca) / cw, (cd - ct) / ch,
             b.conf, b.origin).clamped()
    if not nb.is_valid():
        return None, "lost"
    clipped = a < x1 - 0.5 or t < y1 - 0.5 or c > x2 + 0.5 or d > y2 + 0.5     # 셈에만 쓴다(반 픽셀은 봐준다)
    return nb, ("clipped" if clipped else "kept")


def backup_db(db: Path, stamp: str) -> Path:
    bak = db.with_name(f"{db.name}.bak_crop_{stamp}")
    with sqlite3.connect(db) as src, sqlite3.connect(bak) as dst:
        src.backup(dst)
    return bak


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--camera", required=True, help="고른 이미지가 모두 이 카메라여야 한다(다른 게 섞이면 멈춘다)")
    ap.add_argument("--ids", required=True, help="이미지 id — 7996-8031,8040 처럼")
    ap.add_argument("--box", required=True, help="남길 영역 x1,y1,x2,y2 (지금 이미지 픽셀)")
    ap.add_argument("--quality", type=int, default=95, help="새 JPEG 품질 (import_roadplus 와 같은 95)")
    ap.add_argument("--apply", action="store_true", help="실제로 바꾼다(아니면 보기만)")
    a = ap.parse_args()

    cfg = Config.load(a.config)
    store = Store(cfg.db)
    ids, rect = parse_ids(a.ids), parse_rect(a.box)
    names = store.get_classes()
    with store.connect() as con:
        q = ",".join("?" * len(ids))
        rows = con.execute(f"SELECT id, path, name, status, updated_at, width, height, camera, model_version"
                           f" FROM images WHERE id IN ({q}) ORDER BY id", ids).fetchall()
    missing = sorted(set(ids) - {r["id"] for r in rows})
    other = [r["id"] for r in rows if r["camera"] != a.camera]
    sizes = collections.Counter((r["width"], r["height"]) for r in rows)
    if missing or other or len(sizes) != 1:
        raise SystemExit(f"멈춘다 — DB 에 없는 id {missing[:10]} · 다른 카메라 {other[:10]} · 크기 {dict(sizes)}")
    (img_w, img_h), = sizes
    if rect[2] > img_w or rect[3] > img_h:
        raise SystemExit(f"영역 {rect} 이 이미지({img_w}×{img_h}) 밖으로 나간다")
    gone = [r["id"] for r in rows if not Path(r["path"]).exists()]
    if gone:
        raise SystemExit(f"파일이 없는 이미지 {gone[:10]}")

    low = float(cfg.get("review.low_band", 0.30))
    high = float(cfg.get("review.high_band", 0.80))
    empty = float(cfg.get("review.empty_score", 0.60))
    plan = []
    tally: collections.Counter = collections.Counter()
    lost_cls: collections.Counter = collections.Counter()
    for r in rows:
        before = store.get_boxes(r["id"])
        after, what = [], collections.Counter()
        for b in before:
            nb, kind = crop_box(b, img_w, img_h, rect)
            what[kind] += 1
            if nb is None:
                lost_cls[r["status"], names[b.cls] if b.cls < len(names) else str(b.cls)] += 1
            else:
                after.append(nb)
        for kind, n in what.items():
            tally[r["status"], kind] += n
        plan.append((r, before, after))
        print(f"  {r['id']} {r['status']:8s} {r['name']}: 박스 {len(before)} → {len(after)}"
              f" (줄어듦 {what['clipped']} · 잃음 {what['lost']})")
    print(f"\n{len(rows)}장 · {img_w}×{img_h} → {rect[2] - rect[0]}×{rect[3] - rect[1]} (영역 {list(rect)})")
    for st in sorted({k[0] for k in tally}):
        print(f"  {st}: 그대로 {tally[st, 'kept']} · 줄어듦 {tally[st, 'clipped']} · 잃음 {tally[st, 'lost']}"
              f" — 잃는 것 {dict(sorted((c, n) for (s, c), n in lost_cls.items() if s == st))}")
    if not a.apply:
        print("\n(보기만 했다 — 바꾸려면 --apply. DB 를 떠 두고, 옛 파일은 backups/crop_<시각>/ 로 옮긴다)")
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\nDB 를 떠 두었다: {backup_db(cfg.db, stamp)}")
    keep_dir = cfg.db.parent / "backups" / f"crop_{stamp}"
    keep_dir.mkdir(parents=True, exist_ok=False)
    done, changed, stuck = 0, [], []
    lines = ["id\tstatus\told_name\tnew_name\tsize\tn_before\tn_after"]
    for r, before, after in plan:
        old = Path(r["path"])
        img = cv2.imdecode(np.fromfile(str(old), np.uint8), cv2.IMREAD_COLOR)
        if img is None or img.shape[:2] != (img_h, img_w):
            stuck.append(f"{r['id']}(읽기·크기)")
            continue
        part = np.ascontiguousarray(img[rect[1]:rect[3], rect[0]:rect[2]])
        ok, buf = cv2.imencode(".jpg", part, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
        if not ok:
            stuck.append(f"{r['id']}(JPEG)")
            continue
        new = old.with_name(crop_name(r["name"], rect))
        if new.exists():
            stuck.append(f"{r['id']}(새 이름이 이미 있음)")
            continue
        new.write_bytes(buf.tobytes())
        score = image_uncertainty([b.conf for b in after], low=low, high=high, empty=empty)
        if not store.replace_file(r["id"], str(new.resolve()), part.shape[1], part.shape[0], after,
                                  expect=(r["status"], r["updated_at"]), uncertainty=score,
                                  model_version=r["model_version"] or "", keep_status=True, source="crop"):
            new.unlink(missing_ok=True)      # 도는 사이 검수 화면에서 바뀌었다 — 옛 파일·행 그대로
            changed.append(r["id"])
            continue
        try:
            shutil.move(str(old), str(keep_dir / old.name))
        except OSError as exc:               # DB 는 이미 새 파일 — 옛 파일은 그 자리에 남는다
            stuck.append(f"{r['id']}(옛 파일 옮기기: {exc})")
        lines.append(f"{r['id']}\t{r['status']}\t{old.name}\t{new.name}\t{part.shape[1]}x{part.shape[0]}\t"
                     f"{len(before)}\t{len(after)}")
        done += 1
    (keep_dir / "manifest.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"잘라 바꿈 {done}장 · 옛 파일 {keep_dir}"
          + (f" · 도는 사이 바뀌어 그대로 둔 것 {changed}" if changed else "")
          + (f" · [경고] {stuck}" if stuck else ""))
    return 0 if not stuck else 1


if __name__ == "__main__":
    sys.exit(main())
