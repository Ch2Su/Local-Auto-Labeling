"""데이터 점검 (2026-11-15 사용자 요청) — 지금 데이터가 어떻게 생겼는지 한 화면으로 (검수 화면의 '데이터 점검' 보기, health.js).

Roboflow 의 'Dataset Health Check' 에서 가져온 것: 이미지·박스 수, 박스 없는 이미지, 클래스 균형, 이미지당 객체 수, 박스 위치
히트맵. 이 프로젝트에 맞춰 더한 것: 클래스마다 **카메라 수와 분할별 몫**(분할이 카메라째라 카메라가 세 곳 미만인 클래스는
train·val·test 에 다 들어갈 수 없다), **학습 입력에서의 박스 크기**(긴 변을 imgsz 에 맞췄을 때 짧은 변 px — 해상도를 올릴지 볼 때),
카메라별 표, 가릴 영역이 학습 라벨에서 빼는 박스 수.

**DB 를 읽기만 한다.** `store.health_snapshot` 이 한 시점의 줄을 주고, 여기서는 그것을 세기만 한다(파일은 열지 않는다) —
그래서 열 때마다 새로 센다(캐시 없음).

알아 둘 것:
- 박스 크기는 DB 의 `width`·`height`(회전 전 원시 크기)로 센다. EXIF 로 돌려 보는 그림(루트 CLAUDE.md 'EXIF 회전 이미지')은
  가로·세로가 뒤바뀌어 크기가 조금 어긋난다 — 파일을 열어야 알 수 있어서 그대로 뒀다(영상 캡처로 넣은 그림에는 없다).
- 히트맵은 박스 **중심**이 놓인 칸을 센다(정규화 좌표 48×27 칸). 큰 박스도 한 번이다.
- '묶음 수'(units)는 분할이 통째로 옮기는 단위다 — 카메라가 있으면 카메라, 없으면(옛 legacy) 장면 묶음 `grp`, 그것도 없으면 그 한 장.
- **가릴 영역이 반 이상 덮은 박스는 기본으로 세지 않는다**(`masked_out` — 학습 세트와 같은 식: `masks.covered` 가 `masks.DROP`
  이상, trainer.build_dataset). DB 에는 남아 있지만 학습 라벨에는 없는 박스라, 넣고 세면 클래스 균형·박스 크기가 학습이 보는
  것과 달라진다. 화면의 '가려진 박스' 로 넣고 셀 수도 있다(`drop_masked=False`).
  **test 분할 이미지의 박스는 가려졌어도 뺀 것으로 치지 않는다** — 학습 세트와 내보내기가 test 는 칠하지도, 박스를 빼지도
  않는다(평가는 실제 화면 그대로: trainer.build_dataset·export_yolo 의 `split == "test"`). 영역은 검수하며 먼저 덮고 분할은
  나중에 정해지므로 영역이 있는 이미지가 test 로 갈 수 있다 — 그런 이미지 수는 `masks.test_images`(2026-11-15 리뷰에서 고침:
  전에는 분할을 안 보고 빼서 test 의 박스 수·크기가 학습 세트의 labels/test 보다 적게 나왔다).
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter, namedtuple
from statistics import median

from .masks import DROP, covered
from .store import NO_CAMERA

SCOPES = ("approved", "all")            # 승인만(학습에 들어가는 것) · 휴지통만 뺀 전부
SIZE_EDGES = (8, 16, 32, 64)            # 짧은 변 px 의 칸 경계 — [0,8) [8,16) [16,32) [32,64) [64,∞)
# 이미지당 객체 수의 칸 — 0~5 는 하나씩(가려진 박스를 빼면 중앙값이 2 다: 1~5 를 한 칸에 두니 141장 중 115장이 거기 몰렸다)
COUNT_BINS: tuple[tuple[int, int | None], ...] = (
    (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 10), (11, 20), (21, 30), (31, None))
GRID = (48, 27)                         # 히트맵 칸 수(가로, 세로) — 16:9 화면에서 칸이 정사각형
ID_CAP = 1000                           # 눌러서 열 이미지 id 는 이만큼만 싣는다(주소 길이 — 목록은 GET ?ids= 로 받는다)
MIN_UNITS = 3                           # 이보다 적은 묶음에만 있는 클래스는 train·val·test 에 다 들어갈 수 없다

_SPLITS = ("train", "val", "test")
_Rect = namedtuple("_Rect", "cx cy w h")   # masks.covered 가 읽는 꼴


def short_side(w: float, h: float, width: int, height: int, imgsz: int) -> float:
    """박스(정규화 w·h)의 짧은 변이 학습 입력에서 몇 px 인가 — 그림의 긴 변을 imgsz 에 맞춘다(Ultralytics 레터박스)."""
    longest = max(width, height)
    if longest <= 0:
        return 0.0
    return min(w * width, h * height) * imgsz / longest


def size_bin(px: float) -> int:
    return bisect_right(SIZE_EDGES, px)


def count_bin(n: int) -> int:
    for k, (lo, hi) in enumerate(COUNT_BINS):
        if n >= lo and (hi is None or n <= hi):
            return k
    return len(COUNT_BINS) - 1


def _unit(im: dict) -> str:
    if im["camera"]:
        return "c:" + im["camera"]
    if im["grp"]:
        return "g:" + im["grp"]
    return f"i:{im['id']}"


def _split(im: dict) -> str:
    return im["split"] if im["split"] in _SPLITS else "none"


def _capped(ids) -> dict:
    ids = sorted(ids)
    return {"ids": ids[:ID_CAP], "more": max(0, len(ids) - ID_CAP)}


def _zero_splits() -> dict[str, int]:
    return {"train": 0, "val": 0, "test": 0, "none": 0}


def _names(rows: list[dict], extra=None) -> str:
    return ", ".join(f"{r['name']}({extra(r)})" if extra else r["name"] for r in rows)


def _rects(snap: dict) -> dict[int, list]:
    out: dict[int, list] = {}
    for image_id, x1, y1, x2, y2 in snap.get("masks", ()):
        out.setdefault(image_id, []).append((x1, y1, x2, y2))
    return out


def masked_out(snap: dict) -> list[bool]:
    """snap["boxes"] 와 같은 순서로 — 가릴 영역이 반 이상 덮어 **학습 라벨에서 빠지는** 박스인가(DB 에는 그대로 있다).

    학습 세트를 만들 때와 같은 판정이다(`masks.split_boxes`). 화면 글자·먼 배경을 영역으로 덮으면 그 아래의 자동 라벨 박스가
    DB 에 남는다 — 2026-11-15 의 DB 는 승인 박스 1,277개 가운데 619개가 그랬다. 그래서 수치는 기본으로 이 박스를 빼고 센다.
    **test 분할 이미지의 박스는 빠지지 않는다**(학습 세트가 test 는 칠하지도 빼지도 않는다 — 머리말) — 늘 False.
    """
    rects = _rects(snap)
    whole = _test_ids(snap)
    return [image_id in rects and image_id not in whole and covered(_Rect(cx, cy, w, h), rects[image_id]) >= DROP
            for image_id, _c, cx, cy, w, h in snap["boxes"]]


def _test_ids(snap: dict) -> set[int]:
    """test 분할 이미지 — 가릴 영역이 있어도 학습 세트·내보내기가 원본 그대로 쓰는 이미지."""
    return {im["id"] for im in snap["images"] if im.get("split") == "test"}


def report(snap: dict, classes: list[str], cameras: dict, *, scope: str, imgsz: int,
           val_ratio: float, test_ratio: float, counts: dict | None = None, drop_masked: bool = True) -> dict:
    """화면에 그릴 수치 전부. snap 은 `store.health_snapshot`, cameras 는 `provenance.load_cameras`(못 읽었으면 {}).

    drop_masked(기본) 면 가릴 영역이 빼는 박스는 **어느 수치에도 넣지 않는다**(학습이 보는 것과 같게) — 몇 개인지만
    클래스·카메라마다 `masked_out` 으로 따로 준다. 끄면 DB 의 박스를 전부 센다.
    """
    images = snap["images"]
    by_id = {im["id"]: im for im in images}
    rects = _rects(snap)
    gone = masked_out(snap)

    n_cls = max([len(classes), *(c + 1 for _i, c, *_ in snap["boxes"])])
    cls = [{"boxes": 0, "images": set(), "cams": Counter(), "units": set(), "split": _zero_splits(),
            "sizes": [0] * (len(SIZE_EDGES) + 1), "px": [], "masked_out": 0} for _ in range(n_cls)]
    cam: dict[str | None, dict] = {}
    per_image: Counter = Counter()
    split_boxes = _zero_splits()
    size_all = [0] * (len(SIZE_EDGES) + 1)
    n_boxes = n_gone = 0

    def cam_row(key: str | None) -> dict:
        row = cam.get(key)
        if row is None:
            row = cam[key] = {"images": 0, "boxes": 0, "splits": _zero_splits(), "masked": 0, "masked_out": 0,
                              "classes": set(), "first": None, "last": None}
        return row

    for im in images:
        row = cam_row(im["camera"])
        row["images"] += 1
        row["splits"][_split(im)] += 1
        if im["id"] in rects:
            row["masked"] += 1
        day = (im["captured_at"] or "")[:10]
        if day:
            row["first"] = day if row["first"] is None else min(row["first"], day)
            row["last"] = day if row["last"] is None else max(row["last"], day)

    for (image_id, c, _cx, _cy, w, h), out in zip(snap["boxes"], gone):
        im = by_id.get(image_id)
        if im is None:
            continue
        st = cls[c]
        row = cam[im["camera"]]
        if out:
            st["masked_out"] += 1
            row["masked_out"] += 1
            n_gone += 1
            if drop_masked:
                continue
        n_boxes += 1
        per_image[image_id] += 1
        sp = _split(im)
        split_boxes[sp] += 1
        px = short_side(w, h, im["width"], im["height"], imgsz)
        k = size_bin(px)
        size_all[k] += 1
        st["boxes"] += 1
        st["images"].add(image_id)
        st["cams"][im["camera"]] += 1
        st["units"].add(_unit(im))
        st["split"][sp] += 1
        st["sizes"][k] += 1
        st["px"].append(px)
        row["boxes"] += 1
        row["classes"].add(c)

    class_rows = []
    for i, st in enumerate(cls):
        top = None
        if st["boxes"]:
            key, n = max(st["cams"].items(), key=lambda kv: (kv[1], kv[0] or ""))
            top = {"camera": key, "share": round(n / st["boxes"], 4)}
        class_rows.append({
            "idx": i, "name": classes[i] if i < len(classes) else f"cls_{i}",
            "boxes": st["boxes"], "images": len(st["images"]),
            "cameras": sum(1 for k in st["cams"] if k), "no_camera_boxes": st["cams"].get(None, 0),
            "units": len(st["units"]), "top": top, "split": st["split"], "sizes": st["sizes"],
            "median_px": round(median(st["px"]), 1) if st["px"] else None, "masked_out": st["masked_out"],
        })

    bins = [{"lo": lo, "hi": hi, "ids": []} for lo, hi in COUNT_BINS]
    values = []
    for im in images:
        n = per_image.get(im["id"], 0)
        values.append(n)
        bins[count_bin(n)]["ids"].append(im["id"])
    count_bins = [{"lo": b["lo"], "hi": b["hi"], "n": len(b["ids"]), **_capped(b["ids"])} for b in bins]
    busiest = max(images, key=lambda im: (per_image.get(im["id"], 0), -im["id"]), default=None)

    split_images = _zero_splits()
    for im in images:
        split_images[_split(im)] += 1

    cam_rows = []
    for key, row in cam.items():
        reg = cameras.get(key) if key else None
        cam_rows.append({
            "key": key, "registered": reg is not None, "place": getattr(reg, "place", None),
            "kind": getattr(reg, "kind", None), "pinned": getattr(reg, "split", None) or None,
            "images": row["images"], "boxes": row["boxes"], "splits": row["splits"], "masked": row["masked"],
            "masked_out": row["masked_out"], "classes": len(row["classes"]), "first": row["first"], "last": row["last"],
        })
    cam_rows.sort(key=lambda r: (r["key"] is None, -r["images"], r["key"] or ""))

    n_test_masked = len(_test_ids(snap) & rects.keys())     # 영역이 있는 test 이미지 — 그 박스는 빼지 않았다(masked_out)
    sizes = Counter((im["width"], im["height"]) for im in images).most_common()
    domains = Counter(im["domain"] or "none" for im in images)

    return {
        "scope": scope, "imgsz": int(imgsz), "drop_masked": bool(drop_masked), "counts": counts or {},
        "images": len(images), "boxes": n_boxes,
        "per_image": {"mean": round(n_boxes / len(images), 2) if images else 0.0,
                      "median": median(values) if values else 0,
                      "max": per_image.get(busiest["id"], 0) if busiest else 0,
                      "max_id": busiest["id"] if busiest and per_image.get(busiest["id"], 0) else None},
        "empty": count_bins[0]["n"],
        "classes": class_rows,
        "size_bins": {"edges": list(SIZE_EDGES), "counts": size_all},
        "count_bins": count_bins,
        "splits": {"images": split_images, "boxes": split_boxes,
                   "target": {"train": round(1.0 - val_ratio - test_ratio, 4), "val": val_ratio, "test": test_ratio}},
        "cameras": cam_rows,
        "sizes": [{"w": w, "h": h, "n": n} for (w, h), n in sizes[:8]], "sizes_more": max(0, len(sizes) - 8),
        "domains": dict(domains),
        "masks": {"images": len(rects), "rects": len(snap["masks"]), "dropped": n_gone, "test_images": n_test_masked},
        "tags": snap["tags"], "untagged": snap["untagged"],
        "notes": _notes(class_rows, split_images, len(images), n_gone, drop_masked, val_ratio, test_ratio,
                        n_test_masked),
    }


def _notes(class_rows: list[dict], split_images: dict, n_images: int, n_gone: int, drop_masked: bool,
           val_ratio: float, test_ratio: float, n_test_masked: int = 0) -> list[dict]:
    """눈여겨볼 것 — 숫자로 확인되는 것만 적는다(많다·적다 같은 판단은 넣지 않는다)."""
    out: list[dict] = []
    if not n_images:
        return [{"level": "info", "text": "이 범위에 이미지가 없습니다."}]
    assigned = n_images - split_images["none"]
    if split_images["none"]:
        out.append({"level": "info", "text": (
            f"분할이 아직 없습니다({n_images:,}장 모두 미배정) — 다음 학습이나 labeler.py split 때 배정됩니다." if not assigned
            else f"분할 미배정 {split_images['none']:,}장 — 다음 학습이나 labeler.py split 때 배정됩니다.")})
    none = [r for r in class_rows if not r["boxes"]]
    if none:
        out.append({"level": "warn", "text": f"박스가 없는 클래스: {_names(none)}"})
    few = [r for r in class_rows if r["boxes"] and r["units"] < MIN_UNITS]
    if few:
        out.append({"level": "warn", "text": (
            f"카메라(묶음)가 {MIN_UNITS}곳 미만인 클래스: {_names(few, lambda r: r['units'])} — 분할이 카메라째라 "
            "train·val·test 에 모두 들어갈 수 없습니다.")})
    if assigned:
        for sp, ratio in (("val", val_ratio), ("test", test_ratio)):
            if ratio <= 0:
                continue
            gone = [r for r in class_rows if r["boxes"] and not r["split"][sp]]
            if gone:
                out.append({"level": "warn", "text": f"{sp} 에 박스가 없는 클래스: {_names(gone)}"})
    if n_gone:
        out.append({"level": "info", "text": (
            f"가릴 영역이 반 이상 덮은 박스 {n_gone:,}개는 세지 않았습니다 — 학습 라벨에서 빠지는 박스입니다(DB 에는 그대로)."
            if drop_masked else
            f"가릴 영역이 반 이상 덮어 학습 라벨에서 빠지는 박스 {n_gone:,}개가 수치에 들어 있습니다.")})
    if n_test_masked:
        out.append({"level": "info", "text": (
            f"가릴 영역이 있는 test 이미지 {n_test_masked:,}장 — test 는 칠하지 않으므로 그 아래의 박스도 그대로 셉니다"
            "(학습 세트의 test 라벨과 같게).")})
    return out


def heatmap(snap: dict, *, cls: int | None = None, camera: str | None = None,
            cell: tuple[int, int] | None = None, drop_masked: bool = True) -> dict:
    """박스 중심이 놓인 칸마다 박스 수 — `cells` 는 줄 순서(위 → 아래, 왼 → 오른)로 GRID[0]×GRID[1] 개.

    cls·camera(출처 없음은 `store.NO_CAMERA`)로 거른다. cell=(x, y) 를 주면 그 칸에 중심이 있는 박스의 이미지 id 도 준다.
    camera 를 골랐으면 `sample_id` 는 그 카메라의 가장 새 이미지 — 화면이 바탕에 흐리게 깐다.
    drop_masked 는 report 와 같다(가릴 영역이 빼는 박스는 세지 않는다).
    """
    gw, gh = GRID
    by_id = {im["id"]: im for im in snap["images"]}
    gone = masked_out(snap) if drop_masked else [False] * len(snap["boxes"])
    cells = [0] * (gw * gh)
    seen: set[int] = set()
    hit: set[int] = set()
    n = 0
    for (image_id, c, cx, cy, _w, _h), skip in zip(snap["boxes"], gone):
        im = by_id.get(image_id)
        if skip or im is None or (cls is not None and c != cls):
            continue
        if camera is not None and (im["camera"] or NO_CAMERA) != camera:
            continue
        gx = min(gw - 1, max(0, int(cx * gw)))
        gy = min(gh - 1, max(0, int(cy * gh)))
        cells[gy * gw + gx] += 1
        n += 1
        seen.add(image_id)
        if cell == (gx, gy):
            hit.add(image_id)
    sample = None
    if camera is not None:
        sample = max((im["id"] for im in snap["images"] if (im["camera"] or NO_CAMERA) == camera), default=None)
    out = {"grid": [gw, gh], "cells": cells, "max": max(cells), "boxes": n, "images": len(seen), "sample_id": sample}
    if cell is not None:
        out["cell"] = {"x": cell[0], "y": cell[1], "n": len(hit), **_capped(hit)}
    return out
