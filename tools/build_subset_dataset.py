"""조건에 맞는 승인 이미지만 골라 **실험용** 학습 세트를 만든다 (2026-11-12, 사용자 요청 — 11-10 에 보관한 옛 데이터의
CCTV 이미지로 큰 모델·큰 해상도 학습을 재 보려고).

**프로젝트의 DB·데이터와 원본 DB 는 건드리지 않는다.** 원본 DB(보관 폴더의 옛 DB 든 지금 DB 든)를 읽기 전용으로 열어
`<out>/labeling.db` 로 사본을 뜨고, 그 사본에서 고른 이미지만 남긴 뒤 평소 학습과 같은 길(`trainer.build_dataset` — 분할은
DB 에 적힌 그대로, train·val 의 가릴 영역은 회색으로 칠하고 test 는 실제 화면 그대로)로 `<out>/dataset/` 을 쓴다.

**고르는 기준** (전부 AND, 승인 이미지만):
- `--domain`(기본 cctv — `images.domain`, `any` 면 보지 않는다)
- `--drop-classes`(기본 responder cone)의 박스가 **하나라도 있는 이미지는 통째로 뺀다.** 박스만 지우면 그 물체가 라벨 없는
  배경이 된다. 빼는 클래스는 목록의 맨 끝이어야 한다(번호가 순서라 중간을 빼면 남은 번호가 밀린다) — 남은 클래스만
  `data.yaml` 에 들어간다. 안 빼려면 `--drop-classes` 만 적는다.
- `--tag 이름`(하나라도 붙은 이미지만)·`--not-tag 이름`(붙은 이미지는 뺀다) — 여러 번 줄 수 있다

**분할은 원본 DB 의 것을 그대로 쓴다.** 배정 안 된 이미지가 있으면 멈춘다 — 여기서 새로 나누면 원본에서 잰 점수와 견줄
수 없다. `--force-split test` 는 고른 이미지를 전부 test 로 둔 **평가 전용** 세트를 만든다(모델이 본 적 없는 새 데이터를
통째로 재 볼 때. data.yaml 의 train·val 도 같은 폴더를 가리키니 **이 세트로 학습하지 말 것**). 기본은 평소 test 처럼 가릴
영역을 칠하지 않는다 — `--paint-masks` 면 칠하고 반 넘게 가려진 박스를 뺀다(가릴 영역이 '라벨하지 않기로 한 물체' 를 덮은
새 데이터는 칠하지 않으면 그 물체를 찾은 예측이 전부 헛 박스로 세어진다. 칠한 세트의 점수는 평소 test 점수와 견주지 말 것).

**경로.** 옛 DB 는 이미지를 옮기기 전의 절대경로로 들고 있다 — `--path-map "옛 뿌리=새 뿌리"` 로 사본의 경로를 바꾼다.

기본은 보기만 한다. `--apply` 때 `<out>/`(없거나 비어 있어야 한다)에 `labeling.db`·`dataset/`·`subset_info.json` 을 쓰고,
`--tar` 면 `dataset.tar`(Runpod 로 올릴 묶음, 압축 없음)도 만든다.

  .venv\\Scripts\\python.exe tools\\build_subset_dataset.py --source-db <DB> --out <폴더> [--path-map A=B] [--apply] [--tar]
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import shutil
import sqlite3
import sys
import tarfile
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))   # tools/ -> 저장소 루트

from autolabel.config import Config  # noqa: E402
from autolabel.masks import split_boxes, write_masked  # noqa: E402
from autolabel.store import Store  # noqa: E402
from autolabel.trainer import build_dataset  # noqa: E402
from autolabel.yolo_io import write_yolo_txt  # noqa: E402

SPLITS = ("train", "val", "test")


def open_readonly(db: Path) -> sqlite3.Connection:
    con = sqlite3.connect("file:" + quote(str(db).replace("\\", "/")) + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def class_names(con: sqlite3.Connection) -> list[str]:
    return [r["name"] for r in con.execute("SELECT idx, name FROM classes ORDER BY idx")]


def split_classes(names: list[str], drop: list[str]) -> tuple[list[str], list[int]]:
    """(남길 클래스 이름, 뺄 클래스 번호). 빼는 클래스는 목록의 맨 끝이어야 한다."""
    missing = [n for n in drop if n not in names]
    if missing:
        raise ValueError(f"클래스 목록에 없는 이름: {', '.join(missing)} (목록: {', '.join(names)})")
    idx = sorted(names.index(n) for n in dict.fromkeys(drop))
    keep = len(names) - len(idx)
    if idx != list(range(keep, len(names))):
        raise ValueError("빼는 클래스는 목록의 맨 끝이어야 합니다 — 중간을 빼면 남은 클래스 번호가 밀립니다 "
                         f"(목록: {', '.join(names)} / 빼려는 것: {', '.join(drop)})")
    return names[:keep], idx


def select_ids(con: sqlite3.Connection, *, domain: str = "cctv", drop_cls: list[int] = (),
               tags: list[str] = (), not_tags: list[str] = ()) -> list[int]:
    """기준에 맞는 승인 이미지 id (id 순)."""
    sql = "SELECT i.id FROM images i WHERE i.status='approved'"
    args: list = []
    if domain != "any":
        sql += " AND i.domain=?"
        args.append(domain)
    if drop_cls:
        sql += (" AND NOT EXISTS (SELECT 1 FROM boxes b WHERE b.image_id=i.id"
                f" AND b.cls IN ({','.join('?' * len(drop_cls))}))")
        args += list(drop_cls)
    tagged = ("EXISTS (SELECT 1 FROM image_tags it JOIN tags t ON t.id=it.tag_id"
              " WHERE it.image_id=i.id AND t.name IN ({}))")
    if tags:
        sql += " AND " + tagged.format(",".join("?" * len(tags)))
        args += list(tags)
    if not_tags:
        sql += " AND NOT " + tagged.format(",".join("?" * len(not_tags)))
        args += list(not_tags)
    return [int(r[0]) for r in con.execute(sql + " ORDER BY i.id", args)]


def map_path(path: str, path_map: list[tuple[str, str]]) -> str:
    for old, new in path_map:
        if path.startswith(old):
            return new + path[len(old):]
    return path


def survey(con: sqlite3.Connection, ids: list[int], names: list[str], path_map: list[tuple[str, str]]) -> dict:
    """고른 이미지의 분할·클래스별 박스·가릴 영역·없는 파일을 센다(읽기만)."""
    out = {"images": len(ids), "splits": collections.Counter(), "boxes": collections.Counter(),
           "masked": 0, "missing": [], "no_group": 0}
    chosen = set(ids)
    for r in con.execute("SELECT id, path, split, grp FROM images"):
        if r["id"] in chosen:
            out["splits"][r["split"] or "(미배정)"] += 1
            out["no_group"] += r["grp"] is None
            if not Path(map_path(r["path"], path_map)).exists():
                out["missing"].append(r["path"])
    for r in con.execute("SELECT image_id, cls FROM boxes"):
        if r["image_id"] in chosen:
            out["boxes"][names[r["cls"]] if r["cls"] < len(names) else f"cls_{r['cls']}"] += 1
    out["masked"] = len({r[0] for r in con.execute("SELECT DISTINCT image_id FROM masks")} & chosen)
    return out


def describe(info: dict) -> str:
    lines = [f"고른 이미지 {info['images']:,}장 — "
             + ", ".join(f"{s} {info['splits'][s]:,}" for s in (*SPLITS, "(미배정)") if info["splits"].get(s)),
             f"박스 {sum(info['boxes'].values()):,}개 — "
             + ", ".join(f"{k} {v:,}" for k, v in sorted(info["boxes"].items(), key=lambda t: -t[1])),
             f"가릴 영역이 있는 이미지 {info['masked']}장"]
    if info["missing"]:
        lines.append(f"[경고] 파일이 없는 이미지 {len(info['missing'])}장 (예: {info['missing'][0]}) — --path-map 을 볼 것")
    return "\n".join(lines)


def write_test_only(store: Store, root: Path, *, paint_masks: bool = False) -> dict:
    """고른 이미지를 전부 test 로. 기본은 가릴 영역을 칠하지 않고 박스도 빼지 않는다(평가는 실제 화면 그대로).
    paint_masks 면 학습 세트처럼 칠하고 반 넘게 가려진 박스를 뺀다 — 가릴 영역이 '라벨하지 않기로 한 물체' 를 덮은
    데이터에서는, 칠하지 않으면 그 물체를 맞게 찾은 예측이 전부 헛 박스로 세어진다.
    이름은 `trainer.build_dataset` 과 같은 꼴(`<이름>__<경로 sha1 8자>`)."""
    names = store.get_classes()
    masks = store.masks_by_image() if paint_masks else {}
    n = boxes_n = masked = dropped_n = 0
    for row, boxes in store.approved_with_boxes():
        src = Path(row["path"])
        stem = f"{src.stem}__{hashlib.sha1(str(src).encode('utf-8')).hexdigest()[:8]}"
        rects = masks.get(row["id"])
        if rects:
            boxes, dropped = split_boxes(boxes, rects)
            write_masked(src, root / "images" / "test" / f"{stem}.jpg", rects)
            masked += 1
            dropped_n += len(dropped)
        else:
            dst = root / "images" / "test" / f"{stem}{src.suffix}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        write_yolo_txt(root / "labels" / "test" / f"{stem}.txt", boxes)
        n += 1
        boxes_n += len(boxes)
    # Ultralytics 는 data.yaml 에 train·val 이 없으면 받지 않는다 — 같은 폴더를 가리키게 둔다(평가 전용)
    spec = {"path": str(root), "train": "images/test", "val": "images/test", "test": "images/test",
            "names": {i: name for i, name in enumerate(names)}}
    (root / "data.yaml").write_text(yaml.safe_dump(spec, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return {"test": n, "boxes": boxes_n, "masked": masked, "mask_dropped": dropped_n}


def build(source_db: Path, out: Path, *, domain: str = "cctv", drop_classes: list[str] = ("responder", "cone"),
          tags: list[str] = (), not_tags: list[str] = (), path_map: list[tuple[str, str]] = (),
          force_split: str | None = None, paint_masks: bool = False, tar: bool = False,
          config: Path | None = None) -> dict:
    """사본을 떠 고르고 `<out>/dataset` 을 쓴다. 쓴 것의 요약(dict)을 돌려준다 — `<out>/subset_info.json` 과 같다."""
    source_db, out = Path(source_db).resolve(), Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"{out} 에 이미 파일이 있습니다 — 없는 폴더나 빈 폴더를 줄 것(덮어쓰지 않는다)")
    if out == source_db.parent or source_db.parent in out.parents:
        raise RuntimeError("원본 DB 가 있는 폴더 안에는 쓰지 않습니다 — 다른 자리를 줄 것")
    out.mkdir(parents=True, exist_ok=True)

    src = open_readonly(source_db)
    dst = sqlite3.connect(out / "labeling.db")
    try:
        src.backup(dst)                                 # 원본은 읽기만 — 사본에서만 고친다
    finally:
        dst.close()
        src.close()

    store = Store(out / "labeling.db")
    keep_names, drop_idx = split_classes(store.get_classes(), list(drop_classes))
    with store.connect() as con:
        ids = select_ids(con, domain=domain, drop_cls=drop_idx, tags=list(tags), not_tags=list(not_tags))
        everything = [int(r[0]) for r in con.execute("SELECT id FROM images")]
    if not ids:
        raise RuntimeError("기준에 맞는 승인 이미지가 없습니다")
    chosen = set(ids)
    store.delete_images([i for i in everything if i not in chosen])
    with store.connect() as con:
        for old, new in path_map:
            con.execute("UPDATE images SET path = ? || substr(path, ?) WHERE substr(path, 1, ?) = ?",
                        (new, len(old) + 1, len(old), old))
        rows = con.execute("SELECT id, path, split, grp FROM images").fetchall()
    missing = [r["path"] for r in rows if not Path(r["path"]).exists()]
    if missing:
        raise RuntimeError(f"파일이 없는 이미지 {len(missing)}장 (예: {missing[0]}) — --path-map 을 볼 것")
    store.set_classes(keep_names, drop_extra=True)

    if paint_masks and not force_split:
        raise ValueError("--paint-masks 는 --force-split test 와 함께만 씁니다(평소 세트는 train·val 만 칠한다)")
    if force_split:
        if force_split != "test":
            raise ValueError("--force-split 은 test 만 됩니다(평가 전용 세트)")
        store.set_splits([(i, "test") for i in ids])
        counts = {"train": 0, "val": 0, **write_test_only(store, out / "dataset", paint_masks=paint_masks)}
        extra = {}
    else:
        loose = [r["id"] for r in rows if r["split"] not in SPLITS or r["grp"] is None]
        if loose:
            raise RuntimeError(f"분할이나 묶음이 정해지지 않은 이미지 {len(loose)}장 (예: id {loose[0]}) — 여기서 새로 나누면 "
                               "원본과 어긋납니다. 원본에서 먼저 나누거나 --force-split test 로 평가 전용 세트를 만들 것")
        raw = yaml.safe_load((config or ROOT / "config.yaml").read_text(encoding="utf-8"))
        cameras = source_db.parent / "cameras.yaml"     # 카메라가 적힌 이미지가 있으면 분할이 등록 파일을 읽는다
        raw["data"].update({"db": str(out / "labeling.db"), "dataset": str(out / "dataset"),
                            "cameras": str(cameras if cameras.exists() else out / "cameras.yaml")})
        raw.setdefault("train", {})["hardlink_images"] = False
        raw["classes"] = keep_names
        before = {r["id"]: r["split"] for r in rows}
        report = build_dataset(Config(raw, ROOT / "config.yaml"), store)
        with store.connect() as con:
            after = {int(r["id"]): r["split"] for r in con.execute("SELECT id, split FROM images")}
        if after != before or report.missing:
            raise RuntimeError("세트를 만드는 사이 분할이 바뀌었거나 빠진 이미지가 있습니다 — 결과를 쓰지 말 것")
        counts = {"train": report.train, "val": report.val, "test": report.test, "boxes": report.boxes}
        extra = {"masked": report.masked, "mask_dropped": report.mask_dropped, "masked_test": report.masked_test,
                 "per_class": dict(sorted(report.per_class.items()))}

    info = {"made_at": datetime.now().isoformat(timespec="seconds"), "source_db": str(source_db), "out": str(out),
            "criteria": {"status": "approved", "domain": domain, "drop_classes": list(drop_classes), "tags": list(tags),
                         "not_tags": list(not_tags), "force_split": force_split, "paint_masks": paint_masks},
            "classes": keep_names, **counts, **extra}
    if tar:
        with tarfile.open(out / "dataset.tar", "w") as tf:
            tf.add(out / "dataset", arcname="dataset")
        info["tar_bytes"] = (out / "dataset.tar").stat().st_size
    (out / "subset_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source-db", required=True, type=Path, help="원본 DB (읽기 전용으로 연다)")
    ap.add_argument("--out", required=True, type=Path, help="쓸 폴더 — 없거나 비어 있어야 한다")
    ap.add_argument("--domain", default="cctv", help="images.domain (기본 cctv, any 면 보지 않는다)")
    ap.add_argument("--drop-classes", nargs="*", default=["responder", "cone"],
                    help="이 클래스의 박스가 있는 이미지는 빼고 클래스 목록에서도 뺀다 (기본 responder cone)")
    ap.add_argument("--tag", action="append", default=[], help="이 태그가 하나라도 붙은 이미지만 (여러 번)")
    ap.add_argument("--not-tag", action="append", default=[], help="이 태그가 붙은 이미지는 뺀다 (여러 번)")
    ap.add_argument("--path-map", action="append", default=[], metavar="옛=새",
                    help="DB 에 적힌 경로의 앞머리를 바꾼다 (옮겨 둔 옛 데이터)")
    ap.add_argument("--force-split", choices=["test"], default=None, help="고른 이미지를 전부 test 로 (평가 전용 세트)")
    ap.add_argument("--paint-masks", action="store_true",
                    help="--force-split test 와 함께: 평가 세트에도 가릴 영역을 칠하고 반 넘게 가려진 박스를 뺀다")
    ap.add_argument("--tar", action="store_true", help="dataset.tar 도 만든다")
    ap.add_argument("--apply", action="store_true", help="실제로 만든다 (아니면 보기만)")
    args = ap.parse_args()

    path_map = []
    for item in args.path_map:
        old, sep, new = item.partition("=")
        if not sep or not old or not new:
            ap.error(f"--path-map 은 '옛=새' 꼴로: {item}")
        path_map.append((old, new))

    con = open_readonly(args.source_db)
    try:
        names = class_names(con)
        keep_names, drop_idx = split_classes(names, args.drop_classes)
        everything = select_ids(con, domain=args.domain)
        ids = select_ids(con, domain=args.domain, drop_cls=drop_idx, tags=args.tag, not_tags=args.not_tag)
        info = survey(con, ids, names, path_map)
        only_class = len(everything) - len(select_ids(con, domain=args.domain, drop_cls=drop_idx))
    finally:
        con.close()
    print(f"원본 DB: {args.source_db}")
    print(f"기준: 승인 · domain {args.domain} · 뺄 클래스 {', '.join(args.drop_classes) or '(없음)'}"
          + (f" · 태그 {args.tag}" if args.tag else "") + (f" · 뺄 태그 {args.not_tag}" if args.not_tag else ""))
    print(f"남는 클래스 {len(keep_names)}개: {', '.join(keep_names)}")
    print(f"뺄 클래스의 박스가 있어 빠지는 이미지 {only_class}장")
    print(describe(info))
    if not args.apply:
        print("\n보기만 했습니다 — 만들려면 --apply")
        return 0
    if info["missing"]:
        print("\n파일이 없는 이미지가 있어 만들지 않았습니다.")
        return 1
    made = build(args.source_db, args.out, domain=args.domain, drop_classes=args.drop_classes, tags=args.tag,
                 not_tags=args.not_tag, path_map=path_map, force_split=args.force_split,
                 paint_masks=args.paint_masks, tar=args.tar)
    print(f"\n만들었습니다: {args.out}")
    print(f"  train {made['train']:,} / val {made['val']:,} / test {made['test']:,} (박스 {made['boxes']:,}개)"
          + (f" · 가릴 영역을 칠한 이미지 {made['masked']}장, 반 넘게 가려져 뺀 박스 {made['mask_dropped']}개"
             if made.get("masked") else ""))
    if made.get("tar_bytes"):
        print(f"  dataset.tar {made['tar_bytes'] / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
