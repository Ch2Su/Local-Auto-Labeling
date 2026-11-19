"""옛 데이터(보관 폴더)에서 responder·cone 박스가 든 이미지를 **복사해** 학습 쪽에만 더하는 묶음 목록을 만든다.

보관 폴더는 읽기만 한다(DB 는 immutable 로 연다). 그림은 반드시 복사한다 — Ultralytics 는 깨진 JPEG 을 그 자리에 다시
저장하므로(CLAUDE.md 함정 3) 보관 폴더의 파일을 학습 목록에 직접 넣으면 안 된다. 가릴 영역은 지금 세트와 같이 칠하고
반 넘게 가려진 박스를 뺀다(autolabel.masks). 라벨은 DB 의 박스 그대로다(검수 전 OWLv2 후보 포함 — 고치지 않는다).

갈래: X1 = 두 클래스 박스가 든 옛 이미지 전부, X2 = 그중 카메라가 적힌 프레임만(크롭 빼고).
묶음마다 빼는 것: ① 그 묶음의 검증 카메라와 같은 장소로 보이는 태그의 이미지(--exclude 의 json — 장소 이름이라 git 에 넣지
않는다) ② 그 묶음 검증 이미지와 dHash 가 --near 비트 안쪽인 이미지.

  python -I make_extra.py <…/20261110> --exclude extra/exclude_by_fold.json
"""
import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cmp_common as cc  # noqa: E402

sys.path.insert(0, str(cc.REPO / "src"))
from autolabel.masks import split_boxes, write_masked  # noqa: E402
from autolabel.yolo_io import Box, write_yolo_txt  # noqa: E402

MARK = "\\data\\visible\\"
PLAIN = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}     # 그대로 복사해도 Ultralytics 가 읽는 것
VARIANTS = {"X1": ("crop", "cam"), "X2": ("cam",)}


def dhash(path, size=8):
    """src/autolabel/grouping.py 의 dhash 와 같은 식."""
    with Image.open(path) as im:
        orient = im.getexif().get(0x0112, 1)
        im.draft("L", (size * 8, size * 8))
        src = ImageOps.exif_transpose(im) if orient in (2, 3, 4, 5, 6, 7, 8) else im
        small = src.convert("L").resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(small, dtype=np.int16)
    v = 0
    for b in (a[:, 1:] > a[:, :-1]).flatten():
        v = (v << 1) | int(b)
    return v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("archive", type=Path)
    ap.add_argument("--exclude", type=Path, default=None, help="{묶음 번호: [태그 이름, …]} json (로컬에만 둔다)")
    ap.add_argument("--near", type=int, default=10, help="검증 이미지와 dHash 가 이 비트 안쪽이면 그 묶음에서 뺀다")
    args = ap.parse_args()

    vis = args.archive / "data" / "visible"
    con = sqlite3.connect(f"file:{(vis / 'labeling.db').as_posix()}?mode=ro&immutable=1", uri=True)
    con.row_factory = sqlite3.Row
    names = [r["name"] for r in con.execute("SELECT name FROM classes ORDER BY idx")]
    assert names == cc.NAMES, names
    r_idx, c_idx = names.index("responder"), names.index("cone")
    ids = [r[0] for r in con.execute("SELECT DISTINCT b.image_id FROM boxes b JOIN images i ON i.id = b.image_id "
                                     "WHERE b.cls IN (?, ?) AND i.status IN ('approved', 'pending') ORDER BY 1", (r_idx, c_idx))]
    masks = {}
    for r in con.execute("SELECT image_id, x1, y1, x2, y2 FROM masks ORDER BY image_id, id"):
        masks.setdefault(r["image_id"], []).append((r["x1"], r["y1"], r["x2"], r["y2"]))
    tags = {}
    for r in con.execute("SELECT it.image_id, t.name FROM image_tags it JOIN tags t ON t.id = it.tag_id"):
        tags.setdefault(r[0], []).append(r[1])

    out = cc.ROOT / "extra" / "dataset"
    index = []
    stats = Counter()
    for iid in ids:
        row = con.execute("SELECT * FROM images WHERE id = ?", (iid,)).fetchone()
        p = row["path"]
        src = Path(str(vis) + "\\" + p[p.lower().find(MARK) + len(MARK):])
        boxes = [Box(cls=b["cls"], cx=b["cx"], cy=b["cy"], w=b["w"], h=b["h"], conf=b["conf"], origin=b["origin"] or "auto")
                 for b in con.execute("SELECT cls, cx, cy, w, h, conf, origin FROM boxes WHERE image_id = ? ORDER BY id", (iid,))]
        group = "cam" if row["camera"] else "crop"
        stem = f"{src.stem}__{hashlib.sha1(str(src).encode('utf-8')).hexdigest()[:8]}"
        rects = masks.get(iid)
        dropped = []
        if rects:
            boxes, dropped = split_boxes(boxes, rects)
            dst = out / "images" / group / f"{stem}.jpg"
            if not dst.exists():
                write_masked(src, dst, rects)
            stats["가릴 영역을 칠한 이미지"] += 1
        elif src.suffix.lower() in PLAIN:
            dst = out / "images" / group / f"{stem}{src.suffix}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(src, dst)
        else:                                   # avif 등 — 회전을 적용해 JPEG 으로 다시 쓴다
            dst = out / "images" / group / f"{stem}.jpg"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                with Image.open(src) as im:
                    ImageOps.exif_transpose(im).convert("RGB").save(dst, "JPEG", quality=95)
            stats["JPEG 으로 다시 쓴 이미지"] += 1
        write_yolo_txt(out / "labels" / group / f"{stem}.txt", boxes)
        stats["가려져 뺀 박스"] += len(dropped)
        index.append({"id": iid, "file": str(dst), "group": group, "camera": row["camera"], "status": row["status"],
                      "domain": row["domain"], "tags": sorted(tags.get(iid, [])), "boxes": len(boxes),
                      "responder": sum(b.cls == r_idx for b in boxes), "cone": sum(b.cls == c_idx for b in boxes),
                      "dhash": dhash(dst)})
    (cc.ROOT / "extra").mkdir(exist_ok=True)
    with open(cc.ROOT / "extra" / "index.jsonl", "w", encoding="utf-8") as f:      # 로컬에만 — 카메라 키·파일 이름·태그가 있다
        for rec in index:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("복사한 옛 이미지", len(index), "장 —", dict(Counter(x["group"] for x in index)), dict(stats))
    print("  박스", sum(x["boxes"] for x in index), "· responder", sum(x["responder"] for x in index), "· cone", sum(x["cone"] for x in index))

    exclude = json.loads(args.exclude.read_text(encoding="utf-8")) if args.exclude else {}
    for fold in (0, 1, 2):
        val = cc.fold_files(fold, "val")
        vh = [dhash(v) for v in val]
        bad_tags = set(exclude.get(str(fold), []))
        base = [x for x in (cc.ROOT / "folds" / f"f{fold}_train.txt").read_text(encoding="utf-8").split("\n") if x.strip()]
        base_yaml = (cc.ROOT / "folds" / f"f{fold}.yaml").read_text(encoding="utf-8")
        for var, groups in VARIANTS.items():
            keep, why = [], Counter()
            for x in index:
                if x["group"] not in groups:
                    continue
                if bad_tags & set(x["tags"]):
                    why["검증 카메라와 같은 장소(태그)"] += 1
                    continue
                if min(bin(x["dhash"] ^ h).count("1") for h in vh) <= args.near:
                    why[f"검증 이미지와 dHash {args.near}비트 안쪽"] += 1
                    continue
                keep.append(x)
            keep.sort(key=lambda x: (x["group"] != groups[0], x["file"]))        # 갈래마다 첫 파일의 폴더가 달라 라벨 캐시가 안 겹친다
            lst = cc.ROOT / "folds" / f"f{fold}_train_{var}.txt"
            lst.write_text("\n".join([x["file"] for x in keep] + base) + "\n", encoding="utf-8")
            yml = cc.ROOT / "folds" / f"f{fold}_{var}.yaml"
            yml.write_text(base_yaml.replace(f"f{fold}_train.txt", f"f{fold}_train_{var}.txt"), encoding="utf-8")
            assert f"f{fold}_train_{var}.txt" in yml.read_text(encoding="utf-8")
            print(f"묶음 {fold} {var}: 원래 학습 {len(base)}장 + 옛 {len(keep)}장 "
                  f"(responder {sum(x['responder'] for x in keep)} · cone {sum(x['cone'] for x in keep)} · 박스 {sum(x['boxes'] for x in keep)}) "
                  f"· 뺀 것 {dict(why)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
