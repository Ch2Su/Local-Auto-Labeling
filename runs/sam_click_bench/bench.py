"""SAM 클릭 품질 비교 — sam2.1_t(운영)·sam2.1_s·SAM 3.1·t + 클릭 주변 잘라 넣기 (2026-11-25 사용자 '받아서 재기').

`src/autolabel/segmenter.py` 와 같은 길로 부른다 — Ultralytics `SAM(가중치)` 의 predictor 에 `set_image` 한 번, 프롬프트마다
`predictor(bboxes=…, points=…, labels=[1])`, 돌아온 첫 마스크의 외접 박스. 정답은 지금 DB 의 **승인 박스**(직접 모은 그림만 —
AI Hub `ah-*`·출처 aihub_pm 은 뺀다. 가릴 영역이 반 넘게 덮은 박스도 뺀다 — 학습 라벨과 같은 `masks.drops`).

프롬프트 셋 (박스마다 같은 값을 모든 갈래에 준다 — 박스 번호로 seed):
  center  박스 정중앙 한 점 (09월 측정의 '중심 1클릭')
  inner   박스 가운데 절반(가로·세로 50%) 안의 무작위 한 점 — 사람이 물체 위 아무 데나 찍는 것에 가깝게
  rough   네 변을 각각 박스 폭·높이의 ±15% 안에서 무작위로 민 박스 (09월 측정의 '거친 박스')

갈래(--variant):
  t      weights/sam2.1_t.pt, imgsz 1024 (지금 검수 화면)
  s      weights/sam2.1_s.pt, imgsz 1024
  sam31  sam3.1_multiplex.pt (--sam3 경로), imgsz 1008, 패키지는 --deps 폴더에서 (YOLO_AUTOINSTALL=false)
  tcrop  t 와 같은 가중치, 프롬프트 주변 정사각형만 잘라 넣는다 — 점: 한 변 --crop(512), 박스: max(--crop, 박스 긴 변 × 2).
         마스크가 잘린 칸 가장자리(2px)에 닿으면 통째 그림으로 다시 묻는다(그 시간까지 센다).
  thybrid 먼저 t 로 통째 그림에 묻고, 나온 박스의 짧은 변이 --small(원본 px, 기본 64) 미만일 때만 tcrop 처럼 잘라 다시
         묻는다(정답 크기가 아니라 t 가 낸 박스로 고른다 — 실제 화면에서도 할 수 있는 판단). 잘라 낸 결과가 칸 가장자리에
         닿으면 통째 결과를 그대로 쓴다. 시간은 통째 묻기 + (잘랐으면) 잘라 인코딩·묻기.

한 갈래 = 한 프로세스(VRAM 최대치를 따로 재려고). 박스별 결과는 `out/<갈래>.jsonl.gz`(git 밖 — runs/ 규칙에 .gz 가 없다),
요약은 summarize.py. 그림 파일 이름·카메라 키는 남기지 않는다(이미지 id·클래스·크기·IoU 만).

  .venv\\Scripts\\python.exe -I runs\\sam_click_bench\\bench.py --variant t [--limit 5]
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sqlite3
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from autolabel.masks import drops, from_row  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
WEIGHTS = {"t": ROOT / "weights" / "sam2.1_t.pt", "s": ROOT / "weights" / "sam2.1_s.pt", "tcrop": ROOT / "weights" / "sam2.1_t.pt",
           "thybrid": ROOT / "weights" / "sam2.1_t.pt"}
IMGSZ = {"t": 1024, "s": 1024, "tcrop": 1024, "thybrid": 1024, "sam31": 1008}
PROMPTS = ("center", "inner", "rough")


def load_set(db: Path, limit: int | None):
    """승인·직접 모은 그림의 박스 — 가려져 빠질 박스는 뺀다. DB 는 읽기 전용 스냅숏으로."""
    src = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    con = sqlite3.connect(":memory:")
    src.backup(con)
    src.close()
    imgs = con.execute(
        "SELECT id, path, width, height FROM images WHERE status='approved' "
        "AND coalesce(camera,'') NOT LIKE 'ah-%' AND coalesce(source,'') != 'aihub_pm' ORDER BY id").fetchall()
    if limit:
        imgs = imgs[:limit]
    masks: dict[int, list] = {}
    for iid, x1, y1, x2, y2, poly in con.execute("SELECT image_id, x1, y1, x2, y2, poly FROM masks"):
        masks.setdefault(iid, []).append(from_row(x1, y1, x2, y2, poly))
    out, dropped = [], 0
    for iid, path, W, H in imgs:
        boxes = []
        for k, (cls, cx, cy, w, h, origin) in enumerate(con.execute(
                "SELECT cls, cx, cy, w, h, origin FROM boxes WHERE image_id=? ORDER BY id", (iid,))):
            if drops(SimpleNamespace(cx=cx, cy=cy, w=w, h=h), masks.get(iid, [])):
                dropped += 1
                continue
            gt = ((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H)
            boxes.append({"k": k, "cls": cls, "origin": origin, "gt": gt})
        if boxes:
            out.append({"id": iid, "path": path, "W": W, "H": H, "boxes": boxes})
    return out, dropped


def prompts_for(iid: int, k: int, gt, W, H):
    x1, y1, x2, y2 = gt
    w, h = x2 - x1, y2 - y1
    rng = random.Random(iid * 100003 + k)
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    inner = (x1 + w * (0.25 + 0.5 * rng.random()), y1 + h * (0.25 + 0.5 * rng.random()))
    j = [rng.uniform(-0.15, 0.15) for _ in range(4)]
    r = [max(0.0, x1 + j[0] * w), max(0.0, y1 + j[1] * h), min(W, x2 + j[2] * w), min(H, y2 + j[3] * h)]
    if r[2] - r[0] < 1:
        r[2] = min(W, r[0] + 1)
    if r[3] - r[1] < 1:
        r[3] = min(H, r[1] + 1)
    return {"center": center, "inner": inner, "rough": tuple(r)}


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


class Runner:
    def __init__(self, variant: str, weights: Path, imgsz: int, crop: int, half: bool):
        import torch
        from ultralytics import SAM

        self.torch, self.variant, self.imgsz, self.crop = torch, variant, imgsz, crop
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        self.model = SAM(str(weights))
        import numpy as np

        extra = {"quantize": 16} if half else {}          # 8.4.212: half 는 낡은 이름이고 quantize=16 이 fp16 이다
        self.model(source=np.zeros((64, 64, 3), np.uint8), bboxes=[[8, 8, 40, 40]], device=0, verbose=False,
                   save=False, imgsz=imgsz, **extra)
        self.pred = self.model.predictor
        torch.cuda.synchronize()
        self.load_s = time.perf_counter() - t0
        self.cur = None

    def _sync(self):
        self.torch.cuda.synchronize()

    def encode(self, img) -> float:
        """img: 경로(str) 또는 BGR 배열. 걸린 초."""
        self._sync()
        t0 = time.perf_counter()
        self.pred.reset_image()
        self.pred.set_image(img)
        self._sync()
        return time.perf_counter() - t0

    def ask(self, point=None, box=None):
        """(외접 박스 xyxy | None, 걸린 초) — 지금 set_image 한 그림의 좌표로."""
        kw = {}
        if box is not None:
            kw["bboxes"] = [list(box)]
        if point is not None:
            kw["points"] = [list(point)]
            kw["labels"] = [1]
        self._sync()
        t0 = time.perf_counter()
        res = self.pred(**kw)
        self._sync()
        dt = time.perf_counter() - t0
        if not res or res[0].masks is None or not len(res[0].masks.data):
            return None, dt
        m = res[0].masks.data[0]
        ys, xs = self.torch.nonzero(m, as_tuple=True)
        if not len(xs):
            return None, dt
        return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)), dt


def crop_window(cx, cy, side, W, H):
    side = int(min(side, W, H))
    x0 = int(round(min(max(cx - side / 2, 0), W - side)))
    y0 = int(round(min(max(cy - side / 2, 0), H - side)))
    return x0, y0, side


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["t", "s", "sam31", "tcrop", "thybrid"])
    ap.add_argument("--small", type=float, default=64)
    ap.add_argument("--db", default=str(ROOT / "data" / "visible" / "labeling.db"))
    ap.add_argument("--sam3", default=r"C:\datasets\sam3_test\weights\sam3.1_multiplex.pt")
    ap.add_argument("--deps", default=r"C:\datasets\sam3_test\pydeps")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    if args.variant == "sam31":
        os.environ["YOLO_AUTOINSTALL"] = "false"
        sys.path.insert(0, args.deps)
    data, dropped = load_set(Path(args.db), args.limit)
    nbox = sum(len(d["boxes"]) for d in data)
    print(f"그림 {len(data)}장 · 박스 {nbox}개 (가려져 뺀 박스 {dropped})", flush=True)

    weights = Path(args.sam3) if args.variant == "sam31" else WEIGHTS[args.variant]
    run = Runner(args.variant, weights, IMGSZ[args.variant], args.crop, args.half)
    print(f"불러오기 {run.load_s:.1f}초", flush=True)

    import cv2

    OUT.mkdir(exist_ok=True)
    name = args.variant + (f"_{args.tag}" if args.tag else "")
    path = OUT / f"{name}.jsonl.gz"
    enc_s, ask_s, fallbacks, nfail = [], [], 0, 0
    t_all = time.perf_counter()
    def crop_ask(whole, W, H, point, box):
        """프롬프트 주변 정사각형만 넣고 묻는다 → (원본 좌표 박스 | None, 걸린 초, 가장자리에 닿음, 칸이 그림 전체)."""
        if box is not None:
            side = max(args.crop, 2 * max(box[2] - box[0], box[3] - box[1]))
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        else:
            side, (cx, cy) = args.crop, point
        x0, y0, side = crop_window(cx, cy, side, W, H)
        dt = run.encode(whole[y0:y0 + side, x0:x0 + side])
        lp = None if point is None else (point[0] - x0, point[1] - y0)
        lb = None if box is None else (box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0)
        out, dt2 = run.ask(point=lp, box=lb)
        full = side >= W and side >= H
        if out is None:
            return None, dt + dt2, False, full
        touch = out[0] <= 2 or out[1] <= 2 or out[2] >= side - 2 or out[3] >= side - 2
        return (out[0] + x0, out[1] + y0, out[2] + x0, out[3] + y0), dt + dt2, touch, full

    with gzip.open(path, "wt", encoding="utf-8") as f:
        for n, d in enumerate(data, 1):
            W, H = d["W"], d["H"]
            whole = None
            if args.variant in ("tcrop", "thybrid"):
                whole = cv2.imread(d["path"])
                if whole is None or whole.shape[1] != W or whole.shape[0] != H:
                    raise SystemExit(f"그림 크기가 DB 와 다르다: id {d['id']}")
            if args.variant != "tcrop":
                enc_s.append(run.encode(d["path"]))
            items = []                                   # (rec, kind, point, box)
            for b in d["boxes"]:
                p = prompts_for(d["id"], b["k"], b["gt"], W, H)
                bw, bh = b["gt"][2] - b["gt"][0], b["gt"][3] - b["gt"][1]
                rec = {"id": d["id"], "k": b["k"], "cls": b["cls"], "origin": b["origin"], "W": W, "H": H,
                       "bw": round(bw, 1), "bh": round(bh, 1), "rough_in": round(iou(p["rough"], b["gt"]), 4), "_gt": b["gt"]}
                for kind in PROMPTS:
                    items.append((rec, kind, p[kind] if kind != "rough" else None, p[kind] if kind == "rough" else None))
            res = {}                                     # (id(rec), kind) → [out, dt, flag]
            if args.variant in ("t", "s", "sam31", "thybrid"):
                for rec, kind, point, box in items:     # 통째 그림 한 번 인코딩한 채로 전부
                    out, dt = run.ask(point=point, box=box)
                    res[(id(rec), kind)] = [out, dt, None]
            if args.variant == "thybrid":                # 작게 나온 것만 잘라 다시
                for rec, kind, point, box in items:
                    r = res[(id(rec), kind)]
                    out = r[0]
                    if out is not None and min(out[2] - out[0], out[3] - out[1]) >= args.small:
                        continue
                    c_out, c_dt, touch, _ = crop_ask(whole, W, H, point, box)
                    r[1] += c_dt
                    if c_out is not None and not touch:
                        r[0], r[2] = c_out, "crop"
                        fallbacks += 1               # thybrid 에서는 '잘라 쓴 수'
            if args.variant == "tcrop":
                for rec, kind, point, box in items:
                    out, dt, touch, full = crop_ask(whole, W, H, point, box)
                    flag = None
                    if out is not None and touch and not full:
                        flag = "fb"
                        fallbacks += 1
                        dt += run.encode(whole)
                        out, dt3 = run.ask(point=point, box=box)
                        dt += dt3
                    res[(id(rec), kind)] = [out, dt, flag]
            for rec, kind, point, box in items:
                out, dt, flag = res[(id(rec), kind)]
                ask_s.append(dt)
                if out is None:
                    nfail += 1
                rec[kind] = None if out is None else round(iou(out, rec["_gt"]), 4)
                rec[kind + "_ms"] = round(dt * 1000, 1)
                if out is not None:
                    rec[kind + "_short"] = round(min(out[2] - out[0], out[3] - out[1]), 1)
                if flag:
                    rec[kind + "_" + flag] = 1
            seen = set()
            for rec, *_ in items:
                if id(rec) not in seen:
                    seen.add(id(rec))
                    rec.pop("_gt")
                    f.write(json.dumps(rec) + "\n")
            if n % 20 == 0 or n == len(data):
                print(f"  {n}/{len(data)}장 · {time.perf_counter() - t_all:.0f}초", flush=True)

    torch = run.torch
    meta = {"variant": args.variant, "tag": args.tag, "weights": weights.name, "imgsz": IMGSZ[args.variant],
            "half": args.half, "crop": args.crop if args.variant in ("tcrop", "thybrid") else None,
            "small": args.small if args.variant == "thybrid" else None, "images": len(data), "boxes": nbox,
            "mask_dropped": dropped, "load_s": round(run.load_s, 2),
            "encode_ms_mean": round(1000 * sum(enc_s) / len(enc_s), 1) if enc_s else None,
            "ask_ms_mean": round(1000 * sum(ask_s) / len(ask_s), 1),
            "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
            "no_mask": nfail, "fallbacks": fallbacks, "wall_s": round(time.perf_counter() - t_all, 1),
            "gpu": torch.cuda.get_device_name(0)}
    (OUT / f"{name}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
