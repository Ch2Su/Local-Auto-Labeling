"""승인 이미지의 차량 박스를 원본 해상도로 잘라 VehicleRefiner 실험용 크롭 세트를 만든다 (2026-11-28, 사용자 '크롭 도구 만들기').

설계는 docs/vehicle-refiner.md '2. 데이터'. **DB·원본 그림은 건드리지 않는다** — DB 는 읽기 전용으로 열어 임시 사본을 뜨고
(`Store` 가 열 때 표·열을 더할 수 있어서), 그 사본에서 읽은 뒤 지운다. 기본은 보기만 하고 `--apply` 때 `<out>/` 에 쓴다:
`crops/<클래스>/<image_id>_<k>.jpg`(k = 그 이미지 박스 목록의 순번), `manifest.csv`, `crops_info.json`(이 도구가 쓴 폴더라는
표식 — 표식이 있는 폴더만 비우고 다시 쓴다), `--sheet N` 이면 `sheets/<클래스>.jpg`(사람이 눈으로 보는 대조 시트).

**고르는 것.** 승인 이미지의 `--classes`(기본 차량 다섯) 박스. **가릴 영역이 반 넘게 덮은 박스는 뺀다**(`masks.drops` — 학습
세트·데이터 점검과 같은 판정). 크기 문턱(설계의 48px)은 여기서 걸지 않는다 — 목록의 `long_px` 로 학습 쪽이 거른다.

**좌표.** 라벨과 같은 **EXIF 회전을 적용한 화면 기준**이다(`plain_image_open` + `ImageOps.exif_transpose`). 화면 크기가 DB 의
원시 크기와도, 그 가로·세로를 바꾼 것과도 다르면 그 이미지는 건너뛰고 알린다(`labeler.py check`). 박스를 사방으로 박스 크기의
`--pad`(기본 15%)만큼 넓혀 그림 안으로 자르고, **원본 해상도 그대로** JPEG(`--quality`)로 쓴다 — 줄이는 것은 학습 때 한다.

**fold.** 분할이 통째로 옮기는 묶음(`health.unit_ids` — 같은 카메라, dHash 로 이어진 `grp`)을 한 그룹으로, `--same-place`
(짝 json, `runs/aihub_split_1127/ah_pairs.json` 꼴 `[{"a": [카메라, …], "b": [카메라, …]}]`)의 두 카메라도 한 그룹으로 묶어
`--folds` 개로 나눈다(`assign_folds` — 클래스 비중을 맞추는 욕심쟁이 배정, 드문 클래스를 많이 가진 그룹부터. sklearn 이
이 `.venv` 에 없어 직접 짰다). **`--ref-dataset`**(학습 세트 폴더 — `images/<분할>/<원래 이름>__<해시>.jpg`, 예: WSL 의
`~/datasets/n1280` 은 Windows 에서 `\\\\wsl.localhost\\Ubuntu\\home\\<사용자>\\datasets\\n1280`)를 주면 그 세트의 `--ref-exclude`
분할(기본 test)에 든 이미지가 속한 그룹을 통째로 fold 에서 빼고 `fold=test` 로 적는다(2단 전체를 그 test 로 재려고 —
설계 4.4). 목록의 `split_ref` 는 그 세트에서의 분할(없으면 `new`).

**AI Hub(카메라 `ah-*`·출처 `aihub_pm`)** 크롭은 `local_only=1` 로 적는다(`label_qa_claude.local_only` 와 같은 규칙). 이 도구는
아무것도 밖으로 보내지 않는다 — 이 폴더를 Runpod·git·Roboflow 로 옮기지 말 것(루트 CLAUDE.md '먼저 물을 것'). 저장소 안에
쓰면 `.gitignore`(`*`)를 둔다.

  .venv\\Scripts\\python.exe tools\\build_vehicle_crops.py --out C:\\datasets\\vehicle_crops_20261128 \\
      --ref-dataset \\\\wsl.localhost\\Ubuntu\\home\\<사용자>\\datasets\\n1280 \\
      --same-place runs\\aihub_split_1127\\ah_pairs.json [--sheet 60] [--apply]
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import random
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))   # tools/ -> 저장소 루트
sys.path.insert(0, str(ROOT / "tools"))

from autolabel.config import Config  # noqa: E402
from autolabel.health import unit_ids  # noqa: E402
from autolabel.masks import drops  # noqa: E402
from autolabel.store import Store  # noqa: E402
from autolabel.yolo_io import plain_image_open  # noqa: E402
from label_qa_claude import local_only  # noqa: E402

VEHICLES = ("car", "pickup", "truck", "bus", "responder")
MARKER = "crops_info.json"
GATE = 48               # 설계의 크기 문턱(원본 긴 변 px) — 요약에만 쓴다(거르지는 않는다)
SHEET_TILE = 160
SHEET_COLS = 10
FIELDS = ["crop", "image_id", "box_idx", "cls", "camera", "local_only", "group", "fold", "split_ref", "db_split", "domain",
          "captured_at", "origin", "img_w", "img_h", "x1", "y1", "x2", "y2", "long_px", "short_px", "edge",
          "crop_x1", "crop_y1", "crop_x2", "crop_y2"]


# ---------------------------------------------------------------- 읽기

def copy_db(source_db: Path, dst: Path) -> None:
    """원본을 읽기 전용으로 열어 dst 로 뜬다(원본에는 아무것도 쓰지 않는다)."""
    src = sqlite3.connect("file:" + quote(str(source_db).replace("\\", "/")) + "?mode=ro", uri=True)
    out = sqlite3.connect(dst)
    try:
        src.backup(out)
    finally:
        out.close()
        src.close()


def load_items(store: Store, classes: list[str]) -> tuple[list[dict], dict]:
    """승인 이미지마다 {id, path, camera, source, grp, …, boxes: [(순번, Box)], masked: n}. 고른 클래스만, 가려진 박스는 뺀다."""
    names = store.get_classes()
    unknown = [c for c in classes if c not in names]
    if unknown:
        raise SystemExit(f"[중단] DB 에 없는 클래스: {', '.join(unknown)} (DB 클래스: {', '.join(names)})")
    want = {names.index(c) for c in classes}
    masks = store.masks_by_image()
    with store.connect() as con:
        con.row_factory = sqlite3.Row
        rows = [dict(r) for r in con.execute(
            "SELECT id, path, name, width, height, camera, source, grp, split, domain, captured_at FROM images "
            "WHERE status = 'approved' ORDER BY id")]
    masked = collections.Counter()
    items = []
    for r in rows:
        rects = masks.get(r["id"], [])
        keep = []
        for k, b in enumerate(store.get_boxes(r["id"])):
            if b.cls not in want:
                continue
            if drops(b, rects):
                masked[names[b.cls]] += 1
                continue
            keep.append((k, b))
        r["boxes"] = keep
        items.append(r)
    return items, {"names": names, "masked": dict(masked)}


def ref_splits(ref: Path) -> dict[str, str]:
    """학습 세트 폴더의 {원래 파일 이름 stem: 분할}. 이름은 trainer.build_dataset 의 `<stem>__<해시>` 꼴."""
    out: dict[str, str] = {}
    for split_dir in sorted((ref / "images").iterdir()):
        if not split_dir.is_dir():
            continue
        for f in split_dir.iterdir():
            out[f.stem.rsplit("__", 1)[0]] = split_dir.name
    if not out:
        raise SystemExit(f"[중단] {ref}/images/<분할>/ 에 그림이 없습니다")
    return out


def load_pairs(path: Path | None) -> list[tuple[str, str]]:
    if not path:
        return []
    pairs = []
    for p in json.loads(Path(path).read_text(encoding="utf-8")):
        pairs.append((str(p["a"][0]), str(p["b"][0])))
    return pairs


# ---------------------------------------------------------------- 묶음·fold

def groups_of(items: list[dict], pairs: list[tuple[str, str]]) -> dict[int, int]:
    """{이미지 id: 그룹 번호} — unit_ids(카메라·grp)에 같은 자리 짝을 더 묶는다. 번호는 그룹에서 가장 작은 id."""
    unit = unit_ids([{"id": r["id"], "camera": r["camera"], "grp": r["grp"]} for r in items])
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    by_cam: dict[str, int] = {}
    for r in items:
        if r["camera"]:
            by_cam.setdefault(r["camera"], unit[r["id"]])
    for a, b in pairs:
        if a in by_cam and b in by_cam:
            ra, rb = find(by_cam[a]), find(by_cam[b])
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    roots = {r["id"]: find(unit[r["id"]]) for r in items}
    lowest: dict[int, int] = {}
    for i, g in roots.items():
        lowest[g] = min(lowest.get(g, i), i)
    return {i: lowest[g] for i, g in roots.items()}


def assign_folds(counts: dict[int, collections.Counter], k: int, seed: int = 0) -> dict[int, int]:
    """그룹마다 클래스 수(Counter)를 받아 {그룹: fold}. 그룹은 쪼개지 않는다.

    드문 클래스를 많이 가진 그룹부터(무게 = Σ 그 그룹의 클래스 수 / 그 클래스 전체 수) 차례로, 넣었을 때 fold 마다의 클래스
    비중이 1/k 에서 덜 벗어나는 fold 에 넣는다(크롭 수 비중도 한 클래스처럼 센다). 같으면 크롭이 적은 fold, 그다음 번호 순.
    같은 무게의 순서만 seed 로 섞는다 — 같은 입력·seed 면 같은 결과."""
    if k < 2:
        raise ValueError("fold 는 2개 이상")
    total = collections.Counter()
    for c in counts.values():
        total.update(c)
    n_all = sum(total.values()) or 1
    rng = random.Random(seed)
    order = list(counts)
    rng.shuffle(order)
    order.sort(key=lambda g: -sum(n / total[c] for c, n in counts[g].items() if total[c]))
    folds = [collections.Counter() for _ in range(k)]
    sizes = [0] * k
    out: dict[int, int] = {}

    def dev(x: float) -> float:
        return (x - 1.0 / k) ** 2

    for g in order:
        u = counts[g]
        n_u = sum(u.values())
        best = None
        for f in range(k):
            cost = sum(dev((folds[f][c] + n) / total[c]) - dev(folds[f][c] / total[c]) for c, n in u.items() if total[c])
            cost += dev((sizes[f] + n_u) / n_all) - dev(sizes[f] / n_all)
            key = (round(cost, 12), sizes[f], f)
            if best is None or key < best:
                best = key
        f = best[2]
        out[g] = f
        folds[f].update(u)
        sizes[f] += n_u
    return out


# ---------------------------------------------------------------- 자르기

def crop_rect(box, w: int, h: int, pad: float) -> tuple[tuple[float, float, float, float], tuple[int, int, int, int]]:
    """(박스 px 좌표, 넓혀 그림 안으로 자른 정수 사각형). 박스는 정규화 cx·cy·w·h(화면 기준)."""
    bw, bh = box.w * w, box.h * h
    x1, y1 = box.cx * w - bw / 2, box.cy * h - bh / 2
    x2, y2 = x1 + bw, y1 + bh
    cx1 = max(0, math.floor(x1 - pad * bw))
    cy1 = max(0, math.floor(y1 - pad * bh))
    cx2 = min(w, math.ceil(x2 + pad * bw))
    cy2 = min(h, math.ceil(y2 + pad * bh))
    if cx2 <= cx1:
        cx2 = min(w, cx1 + 1)
    if cy2 <= cy1:
        cy2 = min(h, cy1 + 1)
    return (x1, y1, x2, y2), (cx1, cy1, cx2, cy2)


def open_display(path: Path):
    """화면 기준 그림(EXIF 회전 적용, RGB)."""
    from PIL import ImageOps

    with plain_image_open()(path) as im:
        im = ImageOps.exif_transpose(im)
        return im.convert("RGB")


def size_ok(disp: tuple[int, int], raw: tuple[int, int]) -> bool:
    return disp == raw or disp == (raw[1], raw[0])


# ---------------------------------------------------------------- 만들기

def plan(items: list[dict], classes: list[str], names: list[str], *, ref: dict[str, str] | None, ref_exclude: str,
         pairs: list[tuple[str, str]], k: int, seed: int) -> dict:
    """크롭마다 그룹·fold·split_ref 를 정한다(그림은 열지 않는다)."""
    group = groups_of(items, pairs)
    split_ref = {}
    unmatched_ref = 0
    if ref is not None:
        stems = {Path(r["path"]).stem: r["id"] for r in items}
        unmatched_ref = sum(1 for s in ref if s not in stems)
        for r in items:
            split_ref[r["id"]] = ref.get(Path(r["path"]).stem, "new")
    excluded = {group[r["id"]] for r in items if split_ref.get(r["id"]) == ref_exclude}
    counts: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in items:
        g = group[r["id"]]
        if g in excluded:
            continue
        for _, b in r["boxes"]:
            counts[g][names[b.cls]] += 1
    fold_of = assign_folds(dict(counts), k, seed) if counts else {}
    return {"group": group, "split_ref": split_ref, "excluded": excluded, "fold_of": fold_of,
            "unmatched_ref": unmatched_ref}


def build(items: list[dict], meta: dict, classes: list[str], p: dict, *, out: Path | None, pad: float, quality: int,
          apply: bool) -> tuple[list[dict], dict]:
    """크롭 목록(행)을 만들고, apply 면 그림을 쓴다. 그림 크기가 DB 와 안 맞는 이미지는 건너뛴다."""
    names = meta["names"]
    rows: list[dict] = []
    skipped: list[str] = []
    for r in items:
        if not r["boxes"]:
            continue
        path = Path(r["path"])
        if not path.exists():
            skipped.append(f"{r['id']}: 파일 없음")
            continue
        try:
            im = open_display(path)
        except Exception as exc:                        # noqa: BLE001 — 열리지 않는 그림은 알리고 넘어간다
            skipped.append(f"{r['id']}: 열리지 않음 ({exc})")
            continue
        w, h = im.size
        if not size_ok((w, h), (int(r["width"]), int(r["height"]))):
            skipped.append(f"{r['id']}: 그림 크기 {w}×{h} 가 DB {r['width']}×{r['height']} 와 다름 (labeler.py check)")
            continue
        g = p["group"][r["id"]]
        fold = "test" if g in p["excluded"] else p["fold_of"].get(g)
        for k, b in r["boxes"]:
            cls = names[b.cls]
            (x1, y1, x2, y2), (cx1, cy1, cx2, cy2) = crop_rect(b, w, h, pad)
            rel = f"crops/{cls}/{r['id']}_{k}.jpg"
            if apply:
                dst = out / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                im.crop((cx1, cy1, cx2, cy2)).save(dst, quality=quality)
            bw, bh = x2 - x1, y2 - y1
            rows.append({
                "crop": rel, "image_id": r["id"], "box_idx": k, "cls": cls, "camera": r["camera"] or "",
                "local_only": int(local_only(r["source"], r["camera"])), "group": g, "fold": fold,
                "split_ref": p["split_ref"].get(r["id"], ""), "db_split": r["split"] or "", "domain": r["domain"] or "",
                "captured_at": r["captured_at"] or "", "origin": b.origin, "img_w": w, "img_h": h,
                "x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1),
                "long_px": round(max(bw, bh), 1), "short_px": round(min(bw, bh), 1),
                "edge": int(x1 <= 1 or y1 <= 1 or x2 >= w - 1 or y2 >= h - 1),
                "crop_x1": cx1, "crop_y1": cy1, "crop_x2": cx2, "crop_y2": cy2})
        im.close()
    return rows, {"skipped": skipped}


def write_sheets(out: Path, rows: list[dict], n: int, seed: int) -> list[str]:
    """클래스마다 크롭 n 개(무작위)를 SHEET_TILE 칸에 가로세로비를 지켜 넣은 대조 시트. 칸 아래에 id·fold."""
    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    by_cls: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_cls[r["cls"]].append(r)
    written = []
    (out / "sheets").mkdir(exist_ok=True)
    for cls, rs in sorted(by_cls.items()):
        pick = rng.sample(rs, min(n, len(rs)))
        cols = min(SHEET_COLS, len(pick))
        lines = math.ceil(len(pick) / cols)
        cell = SHEET_TILE + 16
        sheet = Image.new("RGB", (cols * SHEET_TILE, lines * cell), (40, 40, 40))
        draw = ImageDraw.Draw(sheet)
        for i, r in enumerate(pick):
            with plain_image_open()(out / r["crop"]) as c:
                c = c.convert("RGB")
                c.thumbnail((SHEET_TILE - 4, SHEET_TILE - 4))
                x, y = (i % cols) * SHEET_TILE, (i // cols) * cell
                sheet.paste(c, (x + (SHEET_TILE - c.width) // 2, y + (SHEET_TILE - c.height) // 2))
            draw.text((x + 3, y + SHEET_TILE), f"{r['image_id']}_{r['box_idx']} f{r['fold']}", fill=(230, 230, 230))
        dst = out / "sheets" / f"{cls}.jpg"
        sheet.save(dst, quality=90)
        written.append(str(dst))
    return written


def prepare_out(out: Path, source_db: Path) -> None:
    """쓸 폴더 확인 — 없거나 비었거나 이 도구의 표식이 있어야 한다. 표식이 있으면 지난 결과를 지운다."""
    out = out.resolve()
    if out == source_db.parent or source_db.parent in out.parents:
        raise SystemExit("[중단] 원본 DB 가 있는 폴더 안에는 쓰지 않습니다 — 다른 자리를 줄 것")
    if out.exists() and any(out.iterdir()):
        if not (out / MARKER).exists():
            raise SystemExit(f"[중단] {out} 에 이 도구가 쓰지 않은 파일이 있습니다 — 없는 폴더나 빈 폴더를 줄 것")
        for name in ("crops", "sheets"):
            shutil.rmtree(out / name, ignore_errors=True)
        for name in ("manifest.csv", MARKER):
            (out / name).unlink(missing_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    if out == ROOT or ROOT in out.parents:
        (out / ".gitignore").write_text("*\n", encoding="utf-8")


def summary(rows: list[dict], classes: list[str], k: int, meta: dict, p: dict) -> dict:
    folds = [str(f) for f in range(k)] + ["test"]
    table = {}
    for cls in classes:
        rs = [r for r in rows if r["cls"] == cls]
        table[cls] = {
            "crops": len(rs), f"long>={GATE}": sum(1 for r in rs if r["long_px"] >= GATE),
            "masked_out": meta["masked"].get(cls, 0),
            "local_only": sum(r["local_only"] for r in rs),
            "folds": {f: sum(1 for r in rs if str(r["fold"]) == f) for f in folds},
            "groups": {f: len({r["group"] for r in rs if str(r["fold"]) == f}) for f in folds},
        }
    return {"classes": table, "crops": len(rows), "excluded_groups": len(p["excluded"]),
            "unmatched_ref": p["unmatched_ref"]}


def describe(s: dict, classes: list[str], k: int) -> str:
    folds = [str(f) for f in range(k)] + ["test"]
    out = [f"{'클래스':10s} {'크롭':>5s} {'>=' + str(GATE):>6s} {'가려짐':>6s} {'AI Hub':>6s}  " +
           " ".join(f"{('f' + f if f != 'test' else 'test'):>9s}" for f in folds)]
    for cls in classes:
        t = s["classes"][cls]
        cells = " ".join(f"{t['folds'][f]:>4d}/{t['groups'][f]:<4d}" for f in folds)
        out.append(f"{cls:10s} {t['crops']:5d} {t[f'long>={GATE}']:6d} {t['masked_out']:6d} {t['local_only']:6d}  {cells}")
    out.append(f"(fold 칸은 크롭/그룹 수. 합 {s['crops']}개, test 로 뺀 그룹 {s['excluded_groups']}개)")
    if s["unmatched_ref"]:
        out.append(f"[알림] 기준 세트의 그림 {s['unmatched_ref']}장은 지금 DB 의 승인 이미지에 없습니다 — 그새 바뀐 이미지일 수 있습니다")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--source-db", type=Path, default=None, help="기본: config 의 DB (읽기 전용으로 연다)")
    ap.add_argument("--out", type=Path, default=None, help="--apply 때 쓸 폴더(저장소 밖 권장)")
    ap.add_argument("--classes", nargs="+", default=list(VEHICLES))
    ap.add_argument("--pad", type=float, default=0.15, help="박스 크기 대비 사방으로 넓힐 몫")
    ap.add_argument("--ref-dataset", type=Path, default=None, help="분할을 되짚을 학습 세트 폴더(images/<분할>/)")
    ap.add_argument("--ref-exclude", default="test", help="그 세트의 이 분할에 든 그룹은 fold 에서 빼고 test 로 적는다")
    ap.add_argument("--same-place", type=Path, default=None, help="같은 자리 카메라 짝 json — 한 그룹으로 묶는다")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--sheet", type=int, default=0, help="클래스마다 이만큼 뽑은 대조 시트(sheets/)")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    source_db = Path(args.source_db or cfg.db).resolve()
    if args.apply and not args.out:
        ap.error("--apply 에는 --out 이 필요하다")
    ref = ref_splits(args.ref_dataset) if args.ref_dataset else None
    pairs = load_pairs(args.same_place)

    with tempfile.TemporaryDirectory(prefix="vcrops_") as tmp:
        db = Path(tmp) / "labeling.db"
        copy_db(source_db, db)
        store = Store(db)
        items, meta = load_items(store, args.classes)
        del store
    p = plan(items, args.classes, meta["names"], ref=ref, ref_exclude=args.ref_exclude, pairs=pairs, k=args.folds,
             seed=args.seed)
    if args.apply:
        prepare_out(args.out, source_db)
    rows, info = build(items, meta, args.classes, p, out=args.out.resolve() if args.apply else None, pad=args.pad,
                       quality=args.quality, apply=args.apply)
    s = summary(rows, args.classes, args.folds, meta, p)
    print(f"원본 DB: {source_db} (읽기 전용 사본에서 읽음)")
    print(describe(s, args.classes, args.folds))
    for line in info["skipped"]:
        print(f"[건너뜀] {line}")
    if not args.apply:
        print("보기만 했습니다 — 쓰려면 --out <폴더> --apply")
        return 0
    out = args.out.resolve()
    with open(out / "manifest.csv", "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(rows)
    sheets = write_sheets(out, rows, args.sheet, args.seed) if args.sheet > 0 else []
    info_out = {"tool": "build_vehicle_crops", "made_at": datetime.now().isoformat(timespec="seconds"),
                "source_db": str(source_db), "classes": args.classes, "pad": args.pad, "folds": args.folds,
                "seed": args.seed, "quality": args.quality,
                "ref_dataset": str(args.ref_dataset) if args.ref_dataset else None, "ref_exclude": args.ref_exclude,
                "same_place": str(args.same_place) if args.same_place else None, "skipped": info["skipped"], **s}
    (out / MARKER).write_text(json.dumps(info_out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"썼습니다: {out} (크롭 {len(rows)}개, manifest.csv, {MARKER})")
    for sh in sheets:
        print(f"대조 시트: {sh}")
    if any(r["local_only"] for r in rows):
        print("[주의] AI Hub 크롭(local_only=1)이 들어 있습니다 — 이 폴더를 PC 밖(Runpod·git·Roboflow)으로 옮기지 말 것")
    return 0


if __name__ == "__main__":
    sys.exit(main())
