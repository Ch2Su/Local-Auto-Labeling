"""Objects365 의 van·pickup 사진을 골라 **검수 대기(pending)** 로 넣는다 — 라벨은 우리 9클래스로 바꿔서.

**왜 scan 으로 안 넣나.** 시드(COCO) 경로는 라벨을 검수 없이 곧바로 승인하고, 이름이 우리 클래스와
다른 카테고리(`SUV`·`Pickup Truck` 등)는 버린다. 미라벨 폴더로 넣으면 라벨을 아예 읽지 않는다.
그런데 O365 의 `Van` 은 우리 판정표와 다르다 — 무작위 크롭 48장에 SUV·시에나·캐노피 픽업·버스 뒷면이
섞여 있었다(2026-10-03). 그래서 라벨을 살리되 사람이 한 장씩 보게 검수 대기로 넣는다.

**고르는 기준.** 이미지에 '깨끗한' Van 또는 Pickup Truck 박스(iscrowd·isfake·isreflected 가 아니고,
긴 변을 640 으로 줄였을 때 짧은 변 32px 이상)가 있어야 한다. 반은 van 이 든 사진, 반은 pickup 이 든
사진에서 뽑는다. 아래 사진은 통째로 뺀다:
  - 우리 클래스에 iscrowd 박스(무리를 박스 하나로 묶은 것)가 있는 사진 — 우리는 하나씩 그린다.
  - Ambulance 가 있는 사진 — 모양에 따라 van·truck 이 갈려 대응을 정하지 않았다.
  - 파일이 없거나, 실제 크기가 json 과 다르거나, EXIF 회전이 걸린 사진 — 좌표 기준이 애매하다.

**클래스 대응** (2026-10-03 사용자 결정): Scooter(오토바이형)는 motorcycle, 포스터·장난감(isfake)과
반사(isreflected) 박스는 버린다. 대응표에 없는 카테고리(굴착기·삼륜차·휠체어 등)는 우리 클래스가 아니라
넣지 않는다. **O365 에는 전동 킥보드 클래스가 없다** — 사진에 있으면 검수 때 그려야 한다.

**넣는 곳.** 이미지를 `<unlabeled>/o365/` 로 복사하고(원본은 그대로) DB 에 status='pending',
source='o365', 박스 origin='o365' 로 넣는다. 승인하면 다른 사진처럼 tidy·분할을 탄다.
`--apply` 전에 DB 를 `labeling.db.bak_o365_<시각>` 으로 떠 두고, 넣은 목록을 `runs/o365_import/` 에 남긴다.

**주의.** `labeler.py label --all` 은 검수 대기 이미지도 모델 예측으로 덮어쓴다 — 넣은 뒤에는 쓰지 말 것.

    python tools/import_o365.py                    # 무엇을 넣을지 보여만 준다 (파일·DB 를 건드리지 않음)
    python tools/import_o365.py --apply            # 500장을 넣는다
    python tools/import_o365.py --limit 1000 --seed 1 --apply
"""

from __future__ import annotations

import argparse
import collections
import json
import pickle
import random
import re
import shutil
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # tools/ -> 저장소 루트

from PIL import Image  # noqa: E402

from autolabel.config import Config, resolve  # noqa: E402
from autolabel.store import Store  # noqa: E402
from autolabel.yolo_io import Box  # noqa: E402

O365_ROOT = Path(r"E:\datasets\objects365")

# O365 카테고리 이름 -> 우리 클래스 이름
MAP = {
    "Person": "person",
    "Car": "car", "SUV": "car", "Sports Car": "car",
    "Van": "van",
    "Pickup Truck": "pickup",
    "Truck": "truck", "Heavy Truck": "truck", "Fire Truck": "truck",
    "Bus": "bus",
    "Motorcycle": "motorcycle", "Scooter": "motorcycle",
    "Bicycle": "bicycle",
}
EXCLUDE_IMAGE_WITH = {"Ambulance"}
TARGETS = ("Van", "Pickup Truck")
MIN_SHORT_640 = 32

IMG_RE = re.compile(rb'\{"height": (\d+), "id": (\d+), "license": \d+, "width": (\d+), '
                    rb'"file_name": "images/v\d/patch(\d+)/([^"]+)"[^{}]*\}')
# 뒤쪽 주석은 필드 순서가 다르다(image_id 가 앞에 온다) — 객체째 잘라 json 으로 읽는다.
ANN_RE = re.compile(rb'\{[^{}]*"category_id"[^{}]*\}')
CAT_RE = re.compile(rb'"category_id": (\d+)')


def build_index(root: Path) -> dict:
    """json(5GB)을 흘려 읽어 우리와 관련된 박스만 이미지별로 모은다. json.load 는 수십 GB 를 먹는다."""
    src = root / "zhiyuan_objv2_train.json"
    cache = root / "o365_index.pkl"
    stamp = (src.stat().st_size, src.stat().st_mtime_ns)
    if cache.exists():
        with cache.open("rb") as f:
            d = pickle.load(f)
        if d.get("stamp") == stamp:
            return d
    t0 = time.time()
    print("색인을 만든다 (1~2분) ...", flush=True)
    with src.open("rb") as f:
        f.seek(max(0, stamp[0] - 20000))
        tail = f.read()
    t = tail[tail.find(b'"categories"'):]
    cats = {c["id"]: c["name"] for c in json.loads(t[t.find(b"["): t.find(b"]") + 1])}
    keep = {k for k, v in cats.items() if v in MAP or v in EXCLUDE_IMAGE_WITH}
    images: dict[int, tuple] = {}
    boxes: dict[int, list] = collections.defaultdict(list)
    n_tok = n_ann = 0
    carry = b""
    with src.open("rb") as f:
        while True:
            b = f.read(64 << 20)
            buf = carry + b
            end = buf.rfind(b"}") + 1 if b else len(buf)   # 경계에 걸친 객체를 다음 조각으로 넘긴다
            work, carry = buf[:end], buf[end:]
            n_tok += work.count(b'"category_id"')
            for m in IMG_RE.finditer(work):
                h, iid, w, p, name = m.groups()
                images[int(iid)] = (int(p), name.decode(), int(w), int(h))
            for m in ANN_RE.finditer(work):
                n_ann += 1
                cat = int(CAT_RE.search(m.group()).group(1))
                if cat in keep:
                    a = json.loads(m.group())
                    x, y, bw, bh = a["bbox"]
                    boxes[a["image_id"]].append((cat, x, y, bw, bh, a.get("iscrowd", 0),
                                                 a.get("isfake", 0), a.get("isreflected", 0)))
            if not b:
                break
    if n_ann != n_tok:
        raise SystemExit(f"주석 {n_tok}개 중 {n_ann}개만 읽혔다 — json 형식이 예상과 다르다")
    d = {"stamp": stamp, "cats": cats, "images": images, "boxes": dict(boxes)}
    with cache.open("wb") as f:
        pickle.dump(d, f, protocol=5)
    print(f"색인 완료: 이미지 {len(images):,} / 주석 {n_ann:,} ({time.time() - t0:.0f}s)")
    return d


def is_clean(b: tuple) -> bool:
    return not (b[5] or b[6] or b[7])


def candidates(d: dict, root: Path) -> tuple[dict[str, list[int]], collections.Counter]:
    """{'Van': [이미지 id...], 'Pickup Truck': [...]} 와 뺀 이유별 수."""
    cats, images, boxes = d["cats"], d["images"], d["boxes"]
    out: dict[str, list[int]] = {t: [] for t in TARGETS}
    why: collections.Counter = collections.Counter()
    for iid, bl in boxes.items():
        p, name, w, h = images[iid]
        s = 640 / max(w, h)
        hit = {cats[b[0]] for b in bl
               if cats[b[0]] in TARGETS and is_clean(b) and min(b[3], b[4]) * s >= MIN_SHORT_640}
        if not hit:
            continue
        names = {cats[b[0]] for b in bl}
        if names & EXCLUDE_IMAGE_WITH:
            why["Ambulance 가 있음"] += 1
            continue
        if any(b[5] and cats[b[0]] in MAP for b in bl):
            why["우리 클래스에 iscrowd 박스"] += 1
            continue
        if not (root / "train" / f"patch{p}" / name).exists():
            why["파일 없음(받지 않은 묶음 등)"] += 1
            continue
        for t in hit:
            out[t].append(iid)
    return out, why


def convert(d: dict, iid: int, class_idx: dict[str, int], dropped: collections.Counter) -> list[Box]:
    cats, images, boxes = d["cats"], d["images"], d["boxes"]
    _, _, w, h = images[iid]
    out = []
    for b in boxes[iid]:
        name = cats[b[0]]
        if name not in MAP:
            dropped[f"대응 없음: {name}"] += 1
            continue
        if b[6] or b[7]:
            dropped["포스터·장난감·반사"] += 1
            continue
        x, y, bw, bh = b[1:5]
        box = Box(class_idx[MAP[name]], (x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h,
                  None, "o365").clamped()
        if box.is_valid():
            out.append(box)
        else:
            dropped["크기 0"] += 1
    return out


def check_file(path: Path, w: int, h: int) -> str | None:
    """넣어도 되면 None, 아니면 이유."""
    try:
        with Image.open(path) as im:
            if im.size != (w, h):
                return "실제 크기가 json 과 다름"
            if im.getexif().get(0x0112, 1) not in (1, None):
                return "EXIF 회전"
    except Exception as exc:  # 깨진 파일
        return f"열리지 않음 ({type(exc).__name__})"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--root", type=Path, default=O365_ROOT, help="Objects365 를 푼 폴더")
    ap.add_argument("--limit", type=int, default=500, help="넣을 장수 (van 반, pickup 반)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--apply", action="store_true", help="실제로 복사하고 DB 에 넣는다")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    store = Store(cfg.db)
    classes = store.get_classes()
    class_idx = {n: i for i, n in enumerate(classes)}
    missing = sorted(set(MAP.values()) - set(class_idx))
    if missing:
        raise SystemExit(f"DB 에 없는 클래스: {missing} (지금 {classes})")

    d = build_index(args.root)
    pools, why = candidates(d, args.root)
    print(f"후보: van 이 든 사진 {len(pools['Van']):,} / pickup 이 든 사진 {len(pools['Pickup Truck']):,}")
    for k, v in why.most_common():
        print(f"  뺌 — {k}: {v:,}")

    dest_dir = cfg.unlabeled / "o365"
    known = set(store.known_statuses())
    known_names = {Path(p).name for p in known}
    rng = random.Random(args.seed)
    for pool in pools.values():
        rng.shuffle(pool)

    chosen: list[int] = []
    seen: set[int] = set()
    skipped: collections.Counter = collections.Counter()
    order = [pools["Van"], pools["Pickup Truck"]]
    pos = [0, 0]
    turn = 0
    while len(chosen) < args.limit and any(pos[k] < len(order[k]) for k in (0, 1)):
        k = turn % 2
        turn += 1
        if pos[k] >= len(order[k]):
            continue
        iid = order[k][pos[k]]
        pos[k] += 1
        if iid in seen:
            continue
        seen.add(iid)
        p, name, w, h = d["images"][iid]
        if name in known_names:
            skipped["이미 DB 에 있음"] += 1
            continue
        reason = check_file(args.root / "train" / f"patch{p}" / name, w, h)
        if reason:
            skipped[reason] += 1
            continue
        chosen.append(iid)

    dropped: collections.Counter = collections.Counter()
    per_image = {iid: convert(d, iid, class_idx, dropped) for iid in chosen}
    n_cls = collections.Counter(classes[b.cls] for bl in per_image.values() for b in bl)
    n_img = collections.Counter(classes[c] for bl in per_image.values() for c in {b.cls for b in bl})
    print(f"\n고른 사진 {len(chosen)}장 (seed {args.seed}), 박스 {sum(n_cls.values()):,}개")
    for c in classes:
        if n_cls[c]:
            print(f"  {c:<11} 박스 {n_cls[c]:>5}  사진 {n_img[c]:>4}")
    for k, v in skipped.most_common():
        print(f"  건너뜀 — {k}: {v}")
    for k, v in dropped.most_common():
        print(f"  버린 박스 — {k}: {v}")

    if not args.apply:
        print(f"\n(보기만 했다. 넣으려면 --apply — {dest_dir} 로 복사하고 DB 에 검수 대기로 넣는다)")
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = cfg.db.with_name(f"{cfg.db.name}.bak_o365_{stamp}")
    with sqlite3.connect(cfg.db) as src, sqlite3.connect(bak) as dst:
        src.backup(dst)
    print(f"\nDB 를 떠 두었다: {bak}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    items, rows = [], []
    for iid in chosen:
        p, name, w, h = d["images"][iid]
        dst = dest_dir / name
        if dst.exists():
            skipped["복사할 자리에 이미 파일이 있음"] += 1
            continue
        shutil.copy2(args.root / "train" / f"patch{p}" / name, dst)
        items.append({
            "path": str(dst.resolve()), "source": "o365", "width": w, "height": h,
            "status": "pending", "boxes": per_image[iid], "uncertainty": 0.0, "model_version": "o365",
        })
        rows.append(f"{iid}\tpatch{p}/{name}\t{dst.resolve()}\t{len(per_image[iid])}")
    store.ingest_batch(items)

    log_dir = resolve("runs") / "o365_import"
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest = log_dir / f"{stamp}.tsv"
    manifest.write_text("o365_id\tsource\tpath\tboxes\n" + "\n".join(rows) + "\n", encoding="utf-8")
    print(f"검수 대기로 {len(items)}장을 넣었다. 목록: {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
