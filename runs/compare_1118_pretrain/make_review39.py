"""검수 사본에서 사용자가 승인한 프레임을 학습 쪽에만 더하는 두 갈래 목록을 만든다 — 같은 프레임·같은 그림이고 라벨만 다르다.

  R39 = 사용자가 검수한 라벨(검수 사본 DB). 가릴 영역(다각형 포함 — Store.masks_by_image)을 칠하고 반 넘게 덮인 박스를 뺀다
        (autolabel.masks.split_boxes — 학습 세트와 같은 판정).
  U39 = 같은 프레임의 검수 전 라벨(보관 DB). 사본의 처음 상태가 아니라 보관 DB 를 쓴다 — 사본의 검수 대기 프레임에는 SAM 3.1
        후보가 섞였고, 보관 DB 가 798장 측정(make_extra.py)의 라벨과 같은 출처다.

사용자가 승인한 프레임 = 지금 승인이고 처음 상태(검수 서버의 첫 하루 백업)에서는 승인이 아니던 것. 처음부터 승인이던 프레임은
넣지 않는다. 묶음마다 빼는 규칙은 make_extra.py 와 같다 — ① 그 묶음의 검증 카메라와 같은 장소로 보이는 태그(보관 DB 의 태그,
extra/exclude_by_fold.json) ② 그 묶음 검증 이미지와 dHash 가 --near(10)비트 안쪽(보관 폴더의 원본 그림으로 잰다). 승인 프레임
가운데 일부는 검수 전에 responder·cone 박스가 없어 798장(make_extra.py)에 없었다 — 798장에 든 프레임은 X2 목록의 판정과
같은지 견주어 알린다.

검수 사본 DB 는 sqlite backup 으로 떠서 그 사본만 Store 로 연다(서버가 쓰는 DB 는 읽기 전용으로만). 보관 DB·처음 상태 DB 는
immutable 로 읽기만 한다. 그림은 복사본만 쓴다(Ultralytics 가 깨진 JPEG 을 제자리에 다시 저장한다 — CLAUDE.md 함정 3).
갈래·묶음마다 폴더를 따로 둔다 — Ultralytics 는 정렬한 첫 파일의 라벨 폴더에 캐시를 두므로 목록이 다른 실행이 한 캐시를
덮어쓰지 않게. 파일 이름은 보관 id(hw<id>)로만 짓는다.

  python -I make_review39.py <검수 사본 DB> <처음 상태 DB> <…/20261110> [--force]
"""
import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cmp_common as cc  # noqa: E402

from make_extra import dhash  # noqa: E402  (src/autolabel/grouping.py 의 dhash 와 같은 식)

sys.path.insert(0, str(cc.REPO / "src"))
from autolabel.masks import from_row, split_boxes, write_masked  # noqa: E402
from autolabel.store import Store  # noqa: E402
from autolabel.yolo_io import Box, write_yolo_txt  # noqa: E402

MARK = "\\data\\visible\\"
PLAIN = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}     # 그대로 복사해도 Ultralytics 가 읽는 것
ARMS = ("R39", "U39")
BOX_SQL = "SELECT cls, cx, cy, w, h, conf, origin FROM boxes WHERE image_id = ? ORDER BY id"


def sha1(p: Path) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_boxes(con, iid) -> list[Box]:
    return [Box(cls=b[0], cx=b[1], cy=b[2], w=b[3], h=b[4], conf=b[5], origin=b[6] or "auto") for b in con.execute(BOX_SQL, (iid,))]


def read_masks(con) -> dict[int, list]:
    """옛 DB(다각형 열 없음)와 새 DB 를 함께 읽는다 — 열이 있으면 꼭짓점까지(masks.from_row)."""
    cols = {r[1] for r in con.execute("PRAGMA table_info(masks)")}
    sel = "image_id, x1, y1, x2, y2" + (", poly" if "poly" in cols else "")
    out: dict[int, list] = {}
    for r in con.execute(f"SELECT {sel} FROM masks ORDER BY image_id, id"):
        out.setdefault(int(r[0]), []).append(from_row(*tuple(r)[1:]))
    return out


def place(src: Path, dst_dir: Path, stem: str, rects) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    if rects:
        dst = dst_dir / f"{stem}.jpg"
        write_masked(src, dst, rects)
    elif src.suffix.lower() in PLAIN:
        dst = dst_dir / f"{stem}{src.suffix.lower()}"
        shutil.copy2(src, dst)
    else:                                   # avif 등 — 회전을 적용해 JPEG 으로 다시 쓴다
        dst = dst_dir / f"{stem}.jpg"
        with Image.open(src) as im:
            ImageOps.exif_transpose(im).convert("RGB").save(dst, "JPEG", quality=95)
    return dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("live", type=Path, help="검수 사본 DB (서버가 쓰는 것 — 읽기 전용으로 떠 온다)")
    ap.add_argument("init", type=Path, help="처음 상태 DB (검수 서버의 첫 하루 백업)")
    ap.add_argument("archive", type=Path, help="보관 폴더 …/20261110")
    ap.add_argument("--near", type=int, default=10, help="검증 이미지와 dHash 가 이 비트 안쪽이면 그 묶음에서 뺀다(make_extra.py 와 같게)")
    ap.add_argument("--force", action="store_true", help="이미 만든 review39 폴더를 지우고 다시 만든다")
    args = ap.parse_args()

    out = cc.ROOT / "review39"
    if out.exists():
        if not args.force:
            raise SystemExit(f"{out} 가 이미 있다 — 다시 만들려면 --force")
        shutil.rmtree(out)
    out.mkdir()
    snap = out / "labeling_snapshot.db"
    src = sqlite3.connect(f"file:{args.live.as_posix()}?mode=ro", uri=True)
    dst = sqlite3.connect(snap)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()

    store = Store(snap)
    con = sqlite3.connect(f"file:{snap.as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    init = sqlite3.connect(f"file:{args.init.as_posix()}?mode=ro&immutable=1", uri=True)
    vis = args.archive / "data" / "visible"
    arc = sqlite3.connect(f"file:{(vis / 'labeling.db').as_posix()}?mode=ro&immutable=1", uri=True)
    arc.row_factory = sqlite3.Row
    for c in (con, arc):
        names = [r[0] for r in c.execute("SELECT name FROM classes ORDER BY idx")]
        assert names == cc.NAMES, names

    st0 = dict(init.execute("SELECT id, status FROM images").fetchall())
    ids = [r[0] for r in con.execute("SELECT id FROM images WHERE status = 'approved' ORDER BY id") if st0.get(r[0]) != "approved"]
    n0 = sum(1 for r in con.execute("SELECT id FROM images WHERE status = 'approved'") if st0.get(r[0]) == "approved")
    print(f"사용자가 승인한 프레임 {len(ids)}장 (처음부터 승인이던 {n0}장은 넣지 않음)")
    masks_r = store.masks_by_image()
    masks_u = read_masks(arc)

    # 묶음마다 남길 프레임 — make_extra.py 와 같은 규칙(장소 태그·검증 이미지와 dHash)
    tags = {}
    for r in arc.execute("SELECT it.image_id, t.name FROM image_tags it JOIN tags t ON t.id = it.tag_id"):
        tags.setdefault(r[0], set()).add(r[1])
    exclude = json.loads((cc.ROOT / "extra" / "exclude_by_fold.json").read_text(encoding="utf-8"))
    arc_src = {}
    for iid in ids:
        p = arc.execute("SELECT path FROM images WHERE id = ?", (iid,)).fetchone()[0]
        arc_src[iid] = Path(str(vis) + "\\" + p[p.lower().find(MARK) + len(MARK):])
    hashes = {iid: dhash(arc_src[iid]) for iid in ids}
    allowed, why = {}, {}
    for fold in (0, 1, 2):
        vh = [dhash(v) for v in cc.fold_files(fold, "val")]
        bad = set(exclude.get(str(fold), []))
        allowed[fold] = set()
        for iid in ids:
            if bad & tags.get(iid, set()):
                why[(fold, iid)] = "장소 태그"
            elif min(bin(hashes[iid] ^ h).count("1") for h in vh) <= args.near:
                why[(fold, iid)] = f"dHash {args.near}비트 안쪽"
            else:
                allowed[fold].add(iid)
    # 798장에 든 프레임은 X2 목록의 판정과 같아야 한다
    file2id = {}
    for line in (cc.ROOT / "extra" / "index.jsonl").read_text(encoding="utf-8").split("\n"):
        if line.strip():
            rec = json.loads(line)
            file2id[rec["file"]] = rec["id"]
    in798 = set(file2id.values()) & set(ids)
    differ = []
    for fold in (0, 1, 2):
        lst = [x for x in (cc.ROOT / "folds" / f"f{fold}_train_X2.txt").read_text(encoding="utf-8").split("\n") if x.strip()]
        x2 = {file2id[x] for x in lst if x in file2id}
        differ += [(fold, i) for i in in798 if (i in x2) != (i in allowed[fold])]
    print(f"그중 798장(X1·X2)에 든 것 {len(in798)}장 · 묶음별 판정이 X2 목록과 다른 것 {len(differ)}건 {differ[:6]}")
    for fold in (0, 1, 2):
        print(f"  묶음 {fold}: 남김 {len(allowed[fold])}장 · 뺌 {dict(Counter(v for (f, _), v in why.items() if f == fold))}")

    recs = []
    tot = {a: Counter() for a in ARMS}
    dropped = {a: Counter() for a in ARMS}
    for iid in ids:
        r = con.execute("SELECT path, camera, width, height FROM images WHERE id = ?", (iid,)).fetchone()
        a = arc.execute("SELECT path, camera, width, height FROM images WHERE id = ?", (iid,)).fetchone()
        assert a is not None and a["camera"] == r["camera"] and (a["width"], a["height"]) == (r["width"], r["height"]), iid
        src_r, src_u = Path(r["path"]), arc_src[iid]
        same = sha1(src_r) == sha1(src_u)
        keep, gone = {}, {}
        keep["R39"], gone["R39"] = split_boxes(read_boxes(con, iid), masks_r.get(iid, []))
        keep["U39"], gone["U39"] = split_boxes(read_boxes(arc, iid), masks_u.get(iid, []))
        srcs = {"R39": (src_r, masks_r.get(iid, [])), "U39": (src_u, masks_u.get(iid, []))}
        folds = [f for f in (0, 1, 2) if iid in allowed[f]]
        rec = {"id": iid, "camera": r["camera"], "same_image": same, "folds": folds,
               "masks": {k: len(v[1]) for k, v in srcs.items()}}
        for arm in ARMS:
            for b in keep[arm]:
                tot[arm][cc.NAMES[b.cls]] += 1
            for b in gone[arm]:
                dropped[arm][cc.NAMES[b.cls]] += 1
            rec[arm] = dict(Counter(cc.NAMES[b.cls] for b in keep[arm]))
            for fold in folds:
                d = out / f"{arm}_f{fold}"
                img = place(srcs[arm][0], d / "images", f"hw{iid}", srcs[arm][1])
                write_yolo_txt(d / "labels" / f"hw{iid}.txt", keep[arm])
                rec.setdefault("files", {})[f"{arm}_f{fold}"] = str(img)
        recs.append(rec)
    with open(out / "index.jsonl", "w", encoding="utf-8") as f:      # 로컬에만 — 카메라 키가 있다
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("그림이 사본과 보관 폴더에서 같은 프레임:", sum(x["same_image"] for x in recs), "/", len(recs))
    print("가릴 영역이 있는 프레임 — R39:", sum(1 for x in recs if x["masks"]["R39"]), "· U39:", sum(1 for x in recs if x["masks"]["U39"]))
    for arm in ARMS:
        print(f"{arm} 라벨(가려져 빠진 것 뺌) 합 {sum(tot[arm].values())}:", dict(tot[arm].most_common()),
              f"· 가려져 빠진 박스 {sum(dropped[arm].values())}")

    for fold in (0, 1, 2):
        base = [x for x in (cc.ROOT / "folds" / f"f{fold}_train.txt").read_text(encoding="utf-8").split("\n") if x.strip()]
        base_yaml = (cc.ROOT / "folds" / f"f{fold}.yaml").read_text(encoding="utf-8")
        sel = [x for x in recs if fold in x["folds"]]
        for arm in ARMS:
            files = [x["files"][f"{arm}_f{fold}"] for x in sel]
            lst = cc.ROOT / "folds" / f"f{fold}_train_{arm}.txt"
            lst.write_text("\n".join(files + base) + "\n", encoding="utf-8")
            yml = cc.ROOT / "folds" / f"f{fold}_{arm}.yaml"
            yml.write_text(base_yaml.replace(f"f{fold}_train.txt", f"f{fold}_train_{arm}.txt"), encoding="utf-8")
            assert f"f{fold}_train_{arm}.txt" in yml.read_text(encoding="utf-8")
            c = Counter()
            for x in sel:
                c.update(x[arm])
            print(f"묶음 {fold} {arm}: 원래 학습 {len(base)}장 + {len(sel)}장 (responder {c['responder']} · cone {c['cone']} · 박스 {sum(c.values())})"
                  f" · 이 묶음에서 뺀 프레임 {len(recs) - len(sel)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
