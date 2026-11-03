"""legacy 이미지의 **장면 종류·카메라 묶음·크롭 여부 초안**을 만든다 (DB 읽기 전용, 2026-11-03).

legacy(출처 없이 모인 지난 수집분, CLAUDE.md '현재 데이터 상태')는 촬영 카메라를 모른다. 사용자가
CCTV 이미지를 크롭인가 아닌가로 먼저 나누고 그 뒤 카메라·출처로 묶기로 해서(11-03), 사람이 확인할
초안을 낸다. **DB 에는 아무것도 쓰지 않는다** — 확정한 분류를 DB 에 넣는 것은 따로 한다.

    python tools/legacy_groups.py                   # seed 중 드론 아님 → data/visible/legacy_groups/
    python tools/legacy_groups.py --fine 0.65       # 카메라 후보 기준을 바꿔 다시 (임베딩은 캐시를 쓴다)
    python tools/legacy_groups.py --domain legacy   # 비CCTV(source='unlabeled')까지

**'seed 중 드론 아님' 은 전부 CCTV 가 아니다.** test_baseline 의 `cctv` 도메인이 이 조건인데, 시트를 보니
웹 사진(COCO 류 버스·자전거 사진)·폰 사진도 섞여 있다(11-03). 그래서 장면 종류(scene)를 먼저 낸다.

**크롭 추정** (`crop_guess`) — 크기만 본다:
  full    표준 영상·카메라 해상도(±2px — 1919×1079 같은 화면 캡처 포함)
  crop    그 밖의 크기. 다만 웹 사진은 원래 크기가 제각각(640×427 등)이라 CCTV 에서만 뜻이 있다
  unknown 정사각 — 로보플로가 640×640 등으로 늘린 것이라 크기로는 원래 모양을 모른다
**묶기** — 지금 규칙으로 다시 묶은 장면 묶음(grp — `grouping.group_images`, DB 에는 쓰지 않는다)을 단위로,
묶음마다 이미지 임베딩(사전학습 yolo26s 백본, `Model.embed`)의 평균을 내 평균 연결(average linkage)로
합친다. 날 임베딩은 서로 코사인 0.8 이상이라 가르지 못해 **전체 평균을 빼고** 다시 정규화한다. 합치는
도중 `--fine` 에서 끊은 것이 카메라 후보(cam), 더 합쳐 `--coarse` 에서 끊은 것이 장면 종류(scene)다 —
cam 은 scene 안에 들어간다. 임베딩은 장면 **내용**을 보지 카메라를 알아보지는 않는다 — 같은 카메라의 다른
부분을 자른 크롭은 따로 나오고, 비슷한 도로 장면은 다른 카메라라도 붙을 수 있다. 시트로 확인할 것.

나오는 것 (`--out`, 기본 `data/visible/legacy_groups/` — git 밖):
  images.csv   이미지마다 id·분할·묶음·scene·cam·크기·crop_guess·이름 계열·경로
  scenes.csv   장면 종류마다 장수·카메라 후보 수·분할·크롭 추정·이름 계열·시트 + 비워 둔 판정 칸(CCTV인가·크롭인가)
  cams.csv     카메라 후보마다 같은 것 + 비워 둔 판정 칸(카메라 이름)
  sheets/      scene_NNN.jpg(장면 종류별 — 카메라 후보마다 고르게) · cam_NNNN.jpg(묶음 둘 이상인 후보) ·
               crop_<판정>_NN.jpg(크롭 추정 확인용)
  emb_<가중치>_<imgsz>.npz  임베딩 캐시(id 가 같으면 다시 쓴다)
"""

from __future__ import annotations

import argparse
import csv
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # tools/ -> 저장소 루트

from autolabel.config import Config, pretrained_path, resolve
from autolabel.grouping import base_name, group_images

# 가로×세로 (세로 영상은 뒤집어서도 본다). 영상·CCTV·카메라에서 흔한 것만.
STD_SIZES = {
    (3840, 2160), (2560, 1440), (1920, 1080), (1600, 900), (1280, 720), (960, 540), (854, 480),
    (640, 360), (640, 352), (426, 240),
    (2688, 1520), (2592, 1944), (2048, 1536), (1600, 1200), (1280, 960), (1280, 1024), (1024, 768),
    (800, 600), (640, 480), (320, 240),
    (704, 576), (720, 576), (704, 480), (720, 480), (352, 288), (352, 240),
    (4000, 3000), (4032, 3024), (3264, 2448), (4000, 2250), (4032, 2268),
}
NEAR_PX = 2
DOMAINS = {
    "cctv": "source='seed' AND lower(path) NOT LIKE '%stabilized_%'",
    "legacy": "source IN ('seed', 'unlabeled') AND lower(path) NOT LIKE '%stabilized_%'",
}
THUMB = 220
SHEET_COLS = 6


def crop_guess(w: int, h: int) -> tuple[str, str]:
    """(full|crop|unknown, 이유). 크기만 본다 — 머리말 '크롭 추정'."""
    if w == h:
        return "unknown", f"정사각 {w}"
    for sw, sh in STD_SIZES:
        for a, b in ((sw, sh), (sh, sw)):
            if (w, h) == (a, b):
                return "full", f"표준 {a}x{b}"
            if abs(w - a) <= NEAR_PX and abs(h - b) <= NEAR_PX:
                return "full", f"표준±{NEAR_PX} {a}x{b}"
    return "crop", "비표준 크기"


def family(path: str) -> str:
    """이름 계열 — 꼬리표를 떼고 숫자를 # 로."""
    return re.sub(r"\d+", "#", base_name(path))[:40]


def load_rows(cfg: Config, where: str) -> list[dict]:
    con = sqlite3.connect(f"file:{cfg.db.as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(
        "SELECT id, path, split, grp, dhash, width, height FROM images"
        f" WHERE status='approved' AND {where} ORDER BY grp, id")]
    con.close()
    # DB 의 grp 는 마지막 분할 때 것이라 09-24 이름 계열 규칙 전의 `g000001`(1,433장)이 아직 한 덩어리다 —
    # 지금 규칙으로 다시 묶은 것을 단위로 쓴다(DB 에는 쓰지 않는다). 옛 묶음은 db_grp 로 남긴다.
    fresh = group_images(rows)
    for r in rows:
        r["db_grp"], r["grp"] = r["grp"], fresh[r["id"]]
    return rows


def embeddings(rows: list[dict], weights: Path, imgsz: int, cache: Path) -> np.ndarray:
    """(n, d) 임베딩 — 정규화 → 전체 평균을 빼고 → 다시 정규화. 캐시에 있는 id 는 다시 재지 않는다."""
    have: dict[int, np.ndarray] = {}
    if cache.exists():
        z = np.load(cache)
        have = dict(zip(z["ids"].tolist(), z["emb"]))
    todo = [r for r in rows if r["id"] not in have]
    if todo:
        from ultralytics import YOLO
        model = YOLO(str(weights))
        step = 32
        for i in range(0, len(todo), step):
            chunk = todo[i:i + step]
            out = model.embed([r["path"] for r in chunk], imgsz=imgsz, verbose=False)
            for r, e in zip(chunk, out):
                have[r["id"]] = e.detach().float().cpu().numpy().ravel()
            print(f"\r  임베딩 {min(i + step, len(todo))}/{len(todo)}", end="", flush=True)
        print()
        ids = np.array(sorted(have))
        np.savez(cache, ids=ids, emb=np.stack([have[i] for i in ids.tolist()]))
    emb = np.stack([have[r["id"]] for r in rows]).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True).clip(1e-8)
    emb -= emb.mean(0)
    return emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(1e-8)


def average_linkage(cen: np.ndarray, cuts: list[float]) -> list[np.ndarray]:
    """평균 연결로 합치며 유사도가 cuts(내림차순)의 각 값 아래로 떨어질 때의 군집 번호를 낸다.

    단위 n개(묶음)에 n×n 행렬을 갱신한다 — 3,000개에 몇 초. 묶음 크기가 아니라 묶음 수로 평균한다
    (큰 장면 묶음 하나가 군집 평균을 끌고 가지 않게).
    """
    s = (cen @ cen.T).astype(np.float64)
    n = len(s)
    np.fill_diagonal(s, -np.inf)
    size = np.ones(n)
    lab = np.arange(n)
    out = []
    for cut in cuts:
        while n > 1:
            k = int(np.argmax(s))
            i, j = divmod(k, n)
            if s[i, j] < cut:
                break
            row = (size[i] * s[i] + size[j] * s[j]) / (size[i] + size[j])
            s[i], s[:, i] = row, row
            s[i, i] = -np.inf
            s[j], s[:, j] = -np.inf, -np.inf
            size[i] += size[j]
            lab[lab == j] = i
        out.append(lab.copy())
    return out


def numbered(labels: dict[str, int], weight: Counter) -> dict[str, int]:
    """군집 번호를 큰 것(장수)부터 1, 2, … 로 바꾼다."""
    tot: Counter = Counter()
    for g, c in labels.items():
        tot[c] += weight[g]
    rank = {c: k for k, (c, _) in enumerate(tot.most_common(), 1)}
    return {g: rank[c] for g, c in labels.items()}


def font(size: int):
    for name in ("malgun.ttf", "C:/Windows/Fonts/malgun.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def sheet(items: list[dict], title: str, path: Path) -> None:
    """썸네일 대조 시트. 캡션은 id·분할·크기 / 크롭 추정·카메라 후보·이름 계열."""
    f_title, f_cap = font(20), font(13)
    n_rows = (len(items) + SHEET_COLS - 1) // SHEET_COLS
    cell_h = THUMB + 34
    canvas = Image.new("RGB", (SHEET_COLS * THUMB, 36 + n_rows * cell_h), "white")
    d = ImageDraw.Draw(canvas)
    d.text((8, 6), title, fill="black", font=f_title)
    for k, it in enumerate(items):
        x, y = (k % SHEET_COLS) * THUMB, 36 + (k // SHEET_COLS) * cell_h
        try:
            with Image.open(it["path"]) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((THUMB - 6, THUMB - 6))
                canvas.paste(im, (x + (THUMB - im.width) // 2, y + (THUMB - im.height) // 2))
        except OSError:
            d.text((x + 8, y + 8), "열 수 없음", fill="red", font=f_cap)
        d.text((x + 4, y + THUMB), f"{it['id']} {it['split'] or '-'} {it['width']}x{it['height']}",
               fill="black", font=f_cap)
        d.text((x + 4, y + THUMB + 16), f"{it['crop_guess']} · c{it['cam']} · {it['family'][:14]}",
               fill="#555", font=f_cap)
    canvas.save(path, quality=85)


def spread(items: list[dict], n: int, key: str = "grp") -> list[dict]:
    """key 마다 한 장씩 돌아가며 n 장 — 큰 것 하나가 시트를 다 차지하지 않게."""
    by: dict = defaultdict(list)
    for it in items:
        by[it[key]].append(it)
    out: list[dict] = []
    while len(out) < n and any(by.values()):
        for g in list(by):
            if by[g]:
                lst = by[g]
                out.append(lst.pop(len(lst) // 2))
                if len(out) >= n:
                    break
    return out


def summarize(its: list[dict]) -> dict:
    sp, cg = Counter(it["split"] for it in its), Counter(it["crop_guess"] for it in its)
    fams = Counter(it["family"] for it in its).most_common(3)
    return {
        "images": len(its), "groups": len({it["grp"] for it in its}),
        "train": sp.get("train", 0), "val": sp.get("val", 0), "test": sp.get("test", 0),
        "unassigned": sp.get(None, 0),
        "full": cg.get("full", 0), "crop": cg.get("crop", 0), "unknown": cg.get("unknown", 0),
        "families": " | ".join(f"{f}({n})" for f, n in fams),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:   # 엑셀이 한글을 읽게 BOM
        w = csv.DictWriter(fh, list(rows[0]), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="legacy 이미지의 장면 종류·카메라 묶음·크롭 여부 초안 (DB 읽기 전용).")
    p.add_argument("--config", default=None)
    p.add_argument("--domain", choices=sorted(DOMAINS), default="cctv")
    p.add_argument("--weights", default="yolo26s.pt", help="임베딩에 쓸 가중치 (weights/ 에서 찾는다)")
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--fine", type=float, default=0.6, help="카메라 후보를 끊는 평균 유사도 (높을수록 잘게)")
    p.add_argument("--coarse", type=float, default=0.4, help="장면 종류를 끊는 평균 유사도")
    p.add_argument("--per-sheet", type=int, default=30)
    p.add_argument("--out", default="data/visible/legacy_groups")
    p.add_argument("--no-sheets", action="store_true")
    args = p.parse_args()
    if args.coarse > args.fine:
        p.error("--coarse 는 --fine 보다 작아야 한다 (장면 종류가 카메라 후보를 담는다)")

    cfg = Config.load(args.config)
    out = resolve(args.out)
    (out / "sheets").mkdir(parents=True, exist_ok=True)
    rows = load_rows(cfg, DOMAINS[args.domain])
    missing = [r for r in rows if not Path(r["path"]).exists()]
    if missing:
        print(f"  파일 없음 {len(missing)}장 — 뺀다 (예: {missing[0]['path']})")
        rows = [r for r in rows if Path(r["path"]).exists()]
    print(f"{args.domain}: {len(rows)}장, 묶음 {len({r['grp'] for r in rows})}개 (DB 묶음 {len({r['db_grp'] for r in rows})}개)")

    for r in rows:
        r["crop_guess"], r["crop_reason"] = crop_guess(r["width"], r["height"])
        r["family"] = family(r["path"])
    weights = Path(args.weights) if Path(args.weights).exists() else Path(pretrained_path(args.weights))
    emb = embeddings(rows, weights, args.imgsz, out / f"emb_{weights.stem}_{args.imgsz}.npz")

    by_grp: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_grp[r["grp"]].append(i)
    grps = list(by_grp)
    cen = np.stack([emb[by_grp[g]].mean(0) for g in grps])
    cen /= np.linalg.norm(cen, axis=1, keepdims=True).clip(1e-8)
    fine, coarse = average_linkage(cen, [args.fine, args.coarse])
    weight = Counter({g: len(by_grp[g]) for g in grps})
    cam = numbered({g: int(fine[a]) for a, g in enumerate(grps)}, weight)
    scene = numbered({g: int(coarse[a]) for a, g in enumerate(grps)}, weight)
    for r in rows:
        r["cam"], r["scene"] = cam[r["grp"]], scene[r["grp"]]

    cols = ["id", "split", "db_grp", "grp", "scene", "cam", "crop_guess", "crop_reason",
            "width", "height", "family", "path"]
    write_csv(out / "images.csv", [{c: r[c] for c in cols}
                                   for r in sorted(rows, key=lambda r: (r["scene"], r["cam"], r["grp"], r["id"]))])

    by_scene: dict[int, list[dict]] = defaultdict(list)
    by_cam: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_scene[r["scene"]].append(r)
        by_cam[r["cam"]].append(r)
    scenes = [{"scene": s, "cams": len({it["cam"] for it in its}), **summarize(its),
               "sheet": f"scene_{s:03d}.jpg", "decision_cctv": "", "decision_crop": "", "note": ""}
              for s, its in sorted(by_scene.items())]
    multi = {c for c, its in by_cam.items() if len({it["grp"] for it in its}) > 1}
    cams = [{"cam": c, "scene": its[0]["scene"], **summarize(its),
             "sheet": f"cam_{c:04d}.jpg" if c in multi else "", "decision_camera": "", "note": ""}
            for c, its in sorted(by_cam.items())]
    write_csv(out / "scenes.csv", scenes)
    write_csv(out / "cams.csv", cams)

    if not args.no_sheets:
        sd = out / "sheets"
        for old in sd.glob("*.jpg"):
            old.unlink()
        for s, its in by_scene.items():
            sheet(spread(its, args.per_sheet, "cam"),
                  f"장면 종류 {s} — {len(its)}장 · 카메라 후보 {len({it['cam'] for it in its})}개"
                  f" (c=카메라 후보 번호, 후보마다 돌아가며)", sd / f"scene_{s:03d}.jpg")
        for c in multi:
            its = by_cam[c]
            sheet(spread(its, args.per_sheet),
                  f"카메라 후보 {c} (장면 종류 {its[0]['scene']}) — {len(its)}장 · 묶음 {len({it['grp'] for it in its})}개",
                  sd / f"cam_{c:04d}.jpg")
        for g in ("full", "crop", "unknown"):
            its = [r for r in rows if r["crop_guess"] == g]
            sample = spread(its, args.per_sheet * 2, "scene")
            for k in range(0, len(sample), args.per_sheet):
                sheet(sample[k:k + args.per_sheet],
                      f"크롭 추정 '{g}' 표본 {k + 1}–{min(k + args.per_sheet, len(sample))} (전체 {len(its)}장)",
                      sd / f"crop_{g}_{k // args.per_sheet + 1:02d}.jpg")

    cg = Counter(r["crop_guess"] for r in rows)
    big = sorted((len(its) for its in by_scene.values()), reverse=True)
    print(f"크롭 추정: full {cg['full']} · crop {cg['crop']} · unknown {cg['unknown']}")
    print(f"장면 종류 {len(by_scene)}개(큰 것 {big[:8]}) · 카메라 후보 {len(by_cam)}개(묶음 둘 이상 {len(multi)}개)")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
