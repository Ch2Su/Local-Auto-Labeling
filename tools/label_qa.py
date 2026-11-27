"""승인된 라벨과 모델 예측이 어긋나는 곳을 뽑아 라벨 오류 후보로 정리한다.

사람이 승인 이미지 6천여 장을 다시 볼 수는 없으니, 모델이 라벨과 다르게 보는 곳만 먼저 본다.
DB 는 읽기 전용으로 열고(`mode=ro`) 아무것도 고치지 않는다. 결과는 `--out` 에만 쓴다.

    python tools/label_qa.py                  # 예측 + 분석 + 대조 시트
    python tools/label_qa.py --reuse          # 저장된 예측으로 분석만 다시 (임계값 조정용)
    python tools/label_qa.py --weights models/best_visible2.pt

**후보 종류** (예측은 conf 0.1 이상을 모두 저장하고, 후보 판정은 `--high` 이상만 쓴다)

- `missing` — 모델이 확신하는데(conf ≥ high) 같은 클래스 라벨이 근처에 없다(IoU < 0.3).
  빠진 박스일 가능성. 다른 클래스 라벨과 IoU ≥ 0.5 로 겹치면 여기가 아니라 아래로 간다.
- `class`   — 라벨이 있는데 같은 클래스 예측은 없고, 다른 클래스 예측이 확신하며 겹친다.
  클래스를 잘못 붙였을 가능성 (motorcycle/scooter/bicycle 같은 경계 사례가 많을 것).
- `box`     — 같은 클래스인데 IoU 가 0.3~0.5 뿐이다. 박스가 헐겁거나 잘렸을 가능성.
- `unseen`  — 라벨이 있는데 모델이 어떤 클래스로도 거기 아무것도 안 본다(conf ≥ 0.1,
  IoU ≥ 0.3 인 예측 없음). 잘못 그린 박스일 수도, 그냥 어려운 객체일 수도 있다.
  짧은 변 `--min-side` 픽셀 미만은 뺀다 — 작은 객체는 모델이 원래 놓친다.
- `duplicate` — **라벨끼리** 사실상 같은 자리다(IoU ≥ 0.85, 클래스 불문). 같은 물체를 두 번
  그렸거나 클래스를 바꾸려다 하나를 더 그린 흔적이다(2026-10-25 추가). 예측과는 상관없어서
  모델 없이도 잰다(`--dup-only`). 이 행의 `pred_cls`·`pred_box` 는 예측이 아니라 **짝이 된 다른
  라벨**이고 `iou` 가 두 라벨의 IoU 다. 오토바이와 탄 사람처럼 진짜로 겹친 것도 걸린다.

**가릴 영역(검수 화면 `M`)이 덮은 자리는 견주지 않는다 (2026-11-16).** 영역이 넓이의 반 이상을 덮은 라벨은 학습 라벨에서
빠지는 박스이고(`masks.drops` — 학습 세트와 같은 판정), 그 자리의 예측은 사람이 '애매해서 안 그린다' 고 덮어 둔 물체다. 전에는
영역을 몰라서 덮어 둔 자리의 남은 자동 라벨 박스가 `unseen` 으로, 모델이 거기서 찾은 물체가 `missing` 으로 쏟아졌다(2026-11-15 의
DB 는 승인 박스 1,277개 중 619개가 가려진 박스였다). **test 분할도 같다**(2026-11-24 — test 의 영역은 칠하지 않고 채점에서 빼는
자리다, masks.py. 그 전에는 test 는 영역을 적용하지 않아 그대로 견줬다).
전부 견주려면 `--keep-masked`.

**모델이 모르는 클래스의 라벨은 견주지 않는다 (2026-11-16).** 검수 화면의 예측 비교(P, app.js `computeFindings`)와 같은 규칙이다 —
그 라벨은 어느 클래스의 예측과도 짝이 되고(그 자리의 예측을 `missing` 으로 띄우지 않게), `class`·`unseen` 후보로 내지 않는다.
전에는 이 도구만 그 규칙이 없어서, 운영 모델(8클래스)이 모르는 responder·cone 을 그린 뒤로 responder 라벨마다
'responder → car/truck' 이, cone 라벨마다 `unseen` 이 나왔다(그날 실제 DB 로 재 보니 후보 81건 가운데 28건). 모델이 아는
클래스는 예측할 때 `meta.json` 의 `model_classes` 에 남기고 `--reuse` 는 그것을 읽는다(그 전에 만든 예측 폴더는 전부 아는 것으로 본다).

**train 분할의 후보는 적게 나온다.** 모델이 그 라벨을 외웠기 때문이다. 거기서도
어긋나면 오히려 강한 신호다. val·test·미배정은 학습에 안 쓴 이미지라 편향이 없다.
어느 모델로 돌렸는지(학습 분할)에 따라 이 해석이 달라지니 `--weights` 를 바꿀 때 주의.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from collections import Counter, defaultdict, namedtuple
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # tools/ -> 저장소 루트

from autolabel.config import Config, resolve
from autolabel.masks import drops, from_row
from autolabel.predictor import build_class_map

_Box = namedtuple("_Box", "cx cy w h")      # masks.drops 가 읽는 꼴(정규화)

STORE_CONF = 0.10       # 이 이상 예측은 전부 저장 (분석 임계값은 나중에 바꿀 수 있게)
MATCH_IOU = 0.5
NEAR_IOU = 0.3
DUP_IOU = 0.85          # 라벨끼리 이 이상 겹치면 duplicate 후보 (검수 UI 의 SELF_OVERLAP 과 같다)


def parse_args():
    p = argparse.ArgumentParser(description="라벨과 모델 예측이 어긋나는 곳을 뽑는다 (DB 읽기 전용).")
    p.add_argument("--config", default=None, help="설정 파일 (기본: 저장소의 config.yaml)")
    p.add_argument("--weights", default=None, help="가중치 (기본: models/best.pt)")
    p.add_argument("--imgsz", type=int, default=None, help="추론 해상도 (기본: predict.imgsz)")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--high", type=float, default=0.5, help="후보 판정에 쓸 예측 신뢰도 (기본 0.5)")
    p.add_argument("--min-side", type=float, default=24, help="unseen 후보의 최소 짧은 변 픽셀 (기본 24)")
    p.add_argument("--out", default=None, help="결과 폴더 (기본: runs/label_qa/<날짜>_<가중치>)")
    p.add_argument("--reuse", action="store_true", help="저장된 preds.jsonl 로 분석만 다시")
    p.add_argument("--sheet", type=int, default=48, help="종류별 대조 시트에 넣을 상위 후보 수 (0=안 만듦)")
    p.add_argument("--dup-only", action="store_true",
                   help="모델 없이 라벨끼리 겹친 곳(duplicate)만 찾는다 (GPU 를 쓰지 않음)")
    p.add_argument("--keep-masked", action="store_true",
                   help="가릴 영역이 덮은 라벨·예측도 견준다 (기본: 학습 라벨에서 빠지는 자리는 견주지 않는다)")
    return p.parse_args()


def load_db(db: Path):
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    classes = [r["name"] for r in con.execute("SELECT idx, name FROM classes ORDER BY idx")]
    images = [dict(r) for r in con.execute(
        "SELECT id, path, name, width, height, split FROM images WHERE status='approved' ORDER BY id")]
    gt = defaultdict(list)
    for r in con.execute(
            "SELECT b.image_id, b.cls, b.cx, b.cy, b.w, b.h FROM boxes b "
            "JOIN images i ON i.id = b.image_id WHERE i.status='approved'"):
        gt[r["image_id"]].append((r["cls"], r["cx"], r["cy"], r["w"], r["h"]))
    masks = defaultdict(list)
    try:
        # 다각형 꼭짓점 열(poly, 2026-11-19)이 생기기 전의 DB 는 읽기 전용이라 열을 더하지 않고 NULL(사각형)로 읽는다
        cols = {r["name"] for r in con.execute("PRAGMA table_info(masks)")}
        poly = "m.poly" if "poly" in cols else "NULL"
        for r in con.execute(
                f"SELECT m.image_id, m.x1, m.y1, m.x2, m.y2, {poly} FROM masks m "
                "JOIN images i ON i.id = m.image_id WHERE i.status='approved' ORDER BY m.image_id, m.id"):
            masks[r[0]].append(from_row(*tuple(r)[1:]))       # 사각형 (x1, y1, x2, y2)·다각형 ((x, y), …) — masks.py
    except sqlite3.OperationalError:    # masks 표가 생기기 전(2026-11-09)의 DB — 보관해 둔 옛 DB 도 읽을 수 있게
        pass
    con.close()
    return classes, images, gt, masks


def applied_rects(images: list[dict], masks) -> dict[int, list]:
    """영역이 있는 이미지의 영역 {이미지 id: [영역(사각형·다각형)]} — 분할과 상관없다(2026-11-24 부터 test 도: 라벨 파일에서
    빠지고 그 자리의 예측은 채점에서 빠지는 자리다 — masks.py 'test 의 영역')."""
    return {im["id"]: masks[im["id"]] for im in images if masks.get(im["id"])}


def unmasked_labels(gt_rows: list, rects) -> list:
    """영역이 반 이상 덮은 라벨을 뺀 목록 — 학습 라벨에 남는 박스만(`masks.drops`)."""
    return [r for r in gt_rows if not drops(_Box(*r[1:]), rects)]


def unmasked_preds(preds: list, rects) -> list:
    """영역이 반 이상 덮은 예측을 뺀 목록. preds 의 항목은 [클래스, conf, x1, y1, x2, y2](정규화 xyxy)."""
    return [p for p in preds
            if not drops(_Box((p[2] + p[4]) / 2, (p[3] + p[5]) / 2, p[4] - p[2], p[5] - p[3]), rects)]


def predict_all(cfg: Config, weights: str, images: list[dict], classes: list[str], args, out: Path):
    from ultralytics import YOLO

    model = YOLO(weights)
    cmap = build_class_map(model.names, classes)
    imgsz = args.imgsz or int(cfg.get("predict.imgsz") or cfg.imgsz)
    todo = [im for im in images if Path(im["path"]).exists()]
    missing = len(images) - len(todo)
    print(f"가중치 {weights} / imgsz {imgsz} / 대상 {len(todo)}장 (파일 없음 {missing})", flush=True)

    with open(out / "preds.jsonl", "w", encoding="utf-8") as f:
        for start in range(0, len(todo), args.batch):
            chunk = todo[start:start + args.batch]
            results = model.predict(
                [im["path"] for im in chunk], conf=STORE_CONF, iou=float(cfg.get("predict.iou", 0.6)),
                max_det=300, imgsz=imgsz, device=cfg.device, verbose=False)
            for im, res in zip(chunk, results):
                h, w = res.orig_shape
                preds = []
                if res.boxes is not None:
                    for (x1, y1, x2, y2), c, cf in zip(
                            res.boxes.xyxyn.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist()):
                        mc = cmap.get(int(c))
                        if mc is not None:
                            preds.append([mc, round(cf, 4), round(x1, 5), round(y1, 5), round(x2, 5), round(y2, 5)])
                f.write(json.dumps({"id": im["id"], "w": w, "h": h, "preds": preds}) + "\n")
            done = min(start + args.batch, len(todo))
            if done % (args.batch * 50) < args.batch or done == len(todo):
                print(f"  {done}/{len(todo)}", flush=True)
    (out / "meta.json").write_text(json.dumps(
        {"weights": weights, "imgsz": imgsz, "store_conf": STORE_CONF, "date": str(date.today()),
         "model_classes": sorted(set(cmap.values()))},      # 모델이 낼 수 있는 클래스(DB 번호) — 모르는 클래스의 라벨은 안 견준다
        ensure_ascii=False, indent=1), encoding="utf-8")


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def analyze_image(gt_rows, rec, high, min_side, known=None):
    """한 이미지의 후보 목록. 좌표는 원본 픽셀 xyxy.

    known 은 모델이 아는 클래스(DB 번호)의 집합 — None 이면 전부 안다. 모르는 클래스의 라벨은 어느 클래스 예측과도 짝이 되고
    `class`·`unseen` 후보로 내지 않는다(app.js `computeFindings` 와 같은 규칙 — 고치면 둘 다).
    """
    W, H = rec["w"], rec["h"]
    g = np.array([[(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]
                  for _, cx, cy, w, h in gt_rows]).reshape(-1, 4)
    gc = np.array([r[0] for r in gt_rows], dtype=int)
    g_known = np.array([known is None or int(c) in known for c in gc], dtype=bool)
    preds = sorted(rec["preds"], key=lambda p: -p[1])
    p = np.array([[x1 * W, y1 * H, x2 * W, y2 * H] for _, _, x1, y1, x2, y2 in preds]).reshape(-1, 4)
    pc = np.array([q[0] for q in preds], dtype=int)
    pf = np.array([q[1] for q in preds])
    ious = iou_matrix(p, g)

    # 1) 같은 클래스끼리 신뢰도 순으로 탐욕 매칭. 모델이 모르는 클래스의 라벨은 어느 클래스 예측과도 짝이 된다
    g_matched = np.zeros(len(g), bool)
    p_matched = np.zeros(len(p), bool)
    for i in range(len(p)):
        cand = np.where(((gc == pc[i]) | ~g_known) & ~g_matched & (ious[i] >= MATCH_IOU))[0] if len(g) else []
        if len(cand):
            j = cand[np.argmax(ious[i, cand])]
            g_matched[j] = p_matched[i] = True

    out = []
    used_g = set()
    # 2) 확신하는데 짝이 없는 예측
    for i in np.where(~p_matched & (pf >= high))[0]:
        if len(g):
            other = np.where(gc != pc[i])[0]
            same = np.where(gc == pc[i])[0]
            jo = other[np.argmax(ious[i, other])] if len(other) else None
            js = same[np.argmax(ious[i, same])] if len(same) else None
            io = ious[i, jo] if jo is not None else 0.0
            is_ = ious[i, js] if js is not None else 0.0
        else:
            jo = js = None
            io = is_ = 0.0
        if io >= MATCH_IOU:
            if not g_matched[jo] and jo not in used_g and g_known[jo]:
                used_g.add(jo)
                out.append(("class", int(gc[jo]), int(pc[i]), float(pf[i]), float(io), g[jo], p[i]))
            continue            # 이미 제 클래스로 잡힌 라벨 위의 다른 클래스 가설 — 후보 아님
        if is_ >= NEAR_IOU:
            if not g_matched[js] and js not in used_g:
                used_g.add(js)
                out.append(("box", int(gc[js]), int(pc[i]), float(pf[i]), float(is_), g[js], p[i]))
            continue            # 군중 속 중복 예측
        out.append(("missing", -1, int(pc[i]), float(pf[i]), float(max(io, is_)), None, p[i]))

    # 3) 모델이 아무것도 안 보는 라벨
    for j in np.where(~g_matched)[0]:
        if j in used_g or not g_known[j]:
            continue
        side = min(g[j, 2] - g[j, 0], g[j, 3] - g[j, 1])
        if side < min_side:
            continue
        best = ious[:, j].max() if len(p) else 0.0
        if best < NEAR_IOU:
            out.append(("unseen", int(gc[j]), -1, 0.0, float(best), g[j], None))
    return out


def self_overlaps(gt_rows) -> list[tuple[int, int, float]]:
    """라벨끼리 IoU ≥ DUP_IOU 인 쌍 (j, k, IoU). 정규화 좌표로 잰다 — IoU 는 가로·세로를 따로
    늘려도 변하지 않아서 이미지 크기가 필요 없다."""
    if len(gt_rows) < 2:
        return []
    g = np.array([[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2] for _, cx, cy, w, h in gt_rows])
    m = iou_matrix(g, g)
    j, k = np.where(np.triu(m, 1) >= DUP_IOU)
    return [(int(a), int(b), float(m[a, b])) for a, b in zip(j, k)]


def duplicate_rows(images: list[dict], gt, classes) -> list[dict]:
    from autolabel.yolo_io import display_size

    rows = []
    for im in images:
        pairs = self_overlaps(gt[im["id"]])
        if not pairs:
            continue
        try:
            W, H = display_size(Path(im["path"]))    # EXIF 회전 4장은 DB 크기와 가로·세로가 다르다
        except Exception:
            W, H = im["width"], im["height"]
        px = lambda r: [(r[1] - r[3] / 2) * W, (r[2] - r[4] / 2) * H, (r[1] + r[3] / 2) * W, (r[2] + r[4] / 2) * H]
        for j, k, v in pairs:
            a, b = px(gt[im["id"]][j]), px(gt[im["id"]][k])
            rows.append({
                "kind": "duplicate", "image_id": im["id"], "split": im["split"] or "미배정", "name": im["name"],
                "gt_cls": classes[gt[im["id"]][j][0]], "pred_cls": classes[gt[im["id"]][k][0]],
                "conf": 0.0, "iou": round(v, 3),
                "side_px": round(float(min(a[2] - a[0], a[3] - a[1])), 1),
                "gt_box": " ".join(f"{x:.1f}" for x in a), "pred_box": " ".join(f"{x:.1f}" for x in b),
                "path": im["path"],
            })
    return rows


def render_sheet(rows, images_by_id, gt, classes, path: Path, title: str):
    from PIL import Image, ImageDraw, ImageOps

    from autolabel.yolo_io import plain_image_open

    tile, cols = 256, 6
    n = len(rows)
    if not n:
        return
    sheet = Image.new("RGB", (cols * tile, ((n + cols - 1) // cols) * (tile + 28) + 24), "white")
    d = ImageDraw.Draw(sheet)
    d.text((6, 6), title, fill="black")
    for k, r in enumerate(rows):
        im = images_by_id[r["image_id"]]
        # 예측 박스(res.orig_shape)도 라벨도 EXIF 회전을 적용한 화면 기준이다. 그냥 열면
        # 회전 사진 49장에서 크롭이 엉뚱한 곳을 자른다 (CLAUDE.md 'EXIF 회전 이미지').
        # 이 프로세스에는 Ultralytics 가 올라와 있다 — 갈아 끼운 열기는 못 알아보는 파일에서 패키지를 깔려 든다(yolo_io)
        img = ImageOps.exif_transpose(plain_image_open()(im["path"])).convert("RGB")
        W, H = img.size
        box = [float(v) for v in (r["pred_box"] or r["gt_box"]).split()]
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        side = max(box[2] - box[0], box[3] - box[1]) * 3
        side = min(max(side, 160), max(W, H))
        x0 = min(max(cx - side / 2, 0), max(W - side, 0))
        y0 = min(max(cy - side / 2, 0), max(H - side, 0))
        crop = img.crop((x0, y0, x0 + side, y0 + side))
        s = tile / side
        crop = crop.resize((tile, tile))
        cd = ImageDraw.Draw(crop)
        for c, bcx, bcy, bw, bh in gt[r["image_id"]]:
            gx1, gy1 = ((bcx - bw / 2) * W - x0) * s, ((bcy - bh / 2) * H - y0) * s
            gx2, gy2 = ((bcx + bw / 2) * W - x0) * s, ((bcy + bh / 2) * H - y0) * s
            cd.rectangle([gx1, gy1, gx2, gy2], outline=(0, 200, 0), width=1)
        if r["gt_box"]:
            b = [float(v) for v in r["gt_box"].split()]
            cd.rectangle([(b[0] - x0) * s, (b[1] - y0) * s, (b[2] - x0) * s, (b[3] - y0) * s],
                         outline=(255, 220, 0), width=2)
        if r["pred_box"]:
            b = [float(v) for v in r["pred_box"].split()]
            cd.rectangle([(b[0] - x0) * s, (b[1] - y0) * s, (b[2] - x0) * s, (b[3] - y0) * s],
                         outline=(255, 0, 0), width=2)
        x, y = (k % cols) * tile, 24 + (k // cols) * (tile + 28)
        sheet.paste(crop, (x, y))
        split = r["split"] if r["split"] != "미배정" else "unassigned"   # 기본 폰트는 한글을 못 그린다
        cap = f"#{k + 1} id{r['image_id']} {split} {r['side_px']:.0f}px"
        if r["kind"] == "duplicate":
            cap2 = f"{r['gt_cls']}+{r['pred_cls']} iou {r['iou']}"
        else:
            cap2 = f"{r['gt_cls'] or '-'}>{r['pred_cls'] or '-'} {r['conf']}"
        d.text((x + 3, y + tile + 1), cap, fill="black")
        d.text((x + 3, y + tile + 13), cap2, fill="black")
    sheet.save(path)


def main():
    args = parse_args()
    cfg = Config.load(args.config)
    weights = str(resolve(args.weights)) if args.weights else cfg.active_weights()
    tag = "dup" if args.dup_only else Path(weights).stem
    out = resolve(args.out) if args.out else resolve(f"runs/label_qa/{date.today():%Y%m%d}_{tag}")
    out.mkdir(parents=True, exist_ok=True)

    classes, images, gt, masks = load_db(cfg.db)
    images_by_id = {im["id"]: im for im in images}
    if not args.reuse and not args.dup_only:
        predict_all(cfg, weights, images, classes, args, out)       # 예측은 그대로 다 남긴다 — 거르는 것은 분석할 때

    # 가릴 영역이 덮은 자리는 견주지 않는다(머리말). 라벨은 여기서, 예측은 아래에서 장마다 거른다
    rects = {} if args.keep_masked else applied_rects(images, masks)
    skipped_labels = skipped_preds = 0
    for image_id, rr in rects.items():
        kept = unmasked_labels(gt[image_id], rr)
        skipped_labels += len(gt[image_id]) - len(kept)
        gt[image_id] = kept

    recs = [] if args.dup_only else [json.loads(line) for line in open(out / "preds.jsonl", encoding="utf-8")]
    # 모델이 아는 클래스 — 예측할 때 meta.json 에 남긴 것(머리말). 없으면(옛 예측 폴더) 전부 아는 것으로 본다
    known = None
    if not args.dup_only and (out / "meta.json").exists():
        listed = json.loads((out / "meta.json").read_text(encoding="utf-8")).get("model_classes")
        known = None if listed is None else {int(c) for c in listed}
    unknown_labels = 0
    rows = duplicate_rows(images, gt, classes)      # 라벨끼리 겹침 — 예측과 상관없이 승인 이미지 전부
    for rec in recs:
        im = images_by_id.get(rec["id"])
        if im is None:
            continue
        if rec["id"] in rects:
            kept = unmasked_preds(rec["preds"], rects[rec["id"]])
            skipped_preds += len(rec["preds"]) - len(kept)
            rec = {**rec, "preds": kept}
        if known is not None:
            unknown_labels += sum(1 for r in gt[rec["id"]] if r[0] not in known)
        for kind, gcls, pcls, conf, iou, gbox, pbox in analyze_image(gt[rec["id"]], rec, args.high, args.min_side, known):
            ref = pbox if pbox is not None else gbox
            rows.append({
                "kind": kind, "image_id": im["id"], "split": im["split"] or "미배정", "name": im["name"],
                "gt_cls": classes[gcls] if gcls >= 0 else "", "pred_cls": classes[pcls] if pcls >= 0 else "",
                "conf": round(conf, 3), "iou": round(iou, 3),
                "side_px": round(float(min(ref[2] - ref[0], ref[3] - ref[1])), 1),
                "gt_box": " ".join(f"{v:.1f}" for v in gbox) if gbox is not None else "",
                "pred_box": " ".join(f"{v:.1f}" for v in pbox) if pbox is not None else "",
                "path": im["path"],
            })

    order = {"missing": 0, "class": 1, "box": 2, "unseen": 3, "duplicate": 4}
    if args.dup_only:
        order = {"duplicate": 4}
    sort_key = {"unseen": "side_px", "duplicate": "iou"}
    rows.sort(key=lambda r: (order[r["kind"]], -r[sort_key.get(r["kind"], "conf")]))
    with open(out / "candidates.csv", "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["kind"])
        wr.writeheader()
        wr.writerows(rows)

    # --- 요약 ---
    if args.dup_only:       # 예측을 안 했으니 분모는 승인 이미지 전부
        split_n = Counter((im["split"] or "미배정") for im in images)
    else:
        split_n = Counter((images_by_id[r["id"]]["split"] or "미배정") for r in recs if r["id"] in images_by_id)
    print(f"\n결과: {out}")
    if args.dup_only:
        print(f"승인 이미지 {len(images)}장, 라벨끼리 겹침(IoU ≥ {DUP_IOU}) {len(rows)}쌍")
    else:
        print(f"예측한 이미지 {len(recs)}장, 후보 {len(rows)}건 (예측 conf ≥ {args.high}, duplicate 는 라벨끼리)")
    if args.keep_masked:
        print("가릴 영역이 덮은 라벨·예측도 견줬습니다 (--keep-masked)")
    elif rects:
        print(f"가릴 영역이 있는 {len(rects)}장에서 덮인 라벨 {skipped_labels}개"
              + ("" if args.dup_only else f"·예측 {skipped_preds}개(conf ≥ {STORE_CONF})")
              + "는 견주지 않았습니다 — 학습 라벨에서 빠지는 자리입니다 (전부 보려면 --keep-masked)")
    if known is not None and unknown_labels:
        names = ", ".join(c for i, c in enumerate(classes) if i not in known)
        print(f"모델이 모르는 클래스({names})의 라벨 {unknown_labels}개는 견주지 않았습니다 — 그 자리의 예측도 후보로 내지 않습니다")
    splits = ["train", "val", "test", "미배정"]
    print(f"\n{'':10}" + "".join(f"{s:>12}" for s in splits))
    print(f"{'이미지':10}" + "".join(f"{split_n[s]:>12}" for s in splits))
    for kind in order:
        c = Counter(r["split"] for r in rows if r["kind"] == kind)
        print(f"{kind:10}" + "".join(f"{c[s]:>6} ({c[s] / max(split_n[s], 1):4.0%})" for s in splits))
    imgs_with = Counter(r["split"] for r in {(r["image_id"], r["split"]): r for r in rows}.values())
    print(f"{'후보 있는 장':10}" + "".join(f"{imgs_with[s]:>6} ({imgs_with[s] / max(split_n[s], 1):4.0%})" for s in splits))

    print("\nclass 후보: 라벨 -> 예측")
    for (a, b), n in Counter((r["gt_cls"], r["pred_cls"]) for r in rows if r["kind"] == "class").most_common(12):
        print(f"  {a:>10} -> {b:<10} {n}")
    print("missing 후보: 예측 클래스")
    for k, n in Counter(r["pred_cls"] for r in rows if r["kind"] == "missing").most_common():
        print(f"  {k:<10} {n}")
    print("unseen 후보: 라벨 클래스")
    for k, n in Counter(r["gt_cls"] for r in rows if r["kind"] == "unseen").most_common():
        print(f"  {k:<10} {n}")
    print("duplicate 후보: 라벨 · 라벨 (IoU)")
    for r in [r for r in rows if r["kind"] == "duplicate"][:20]:
        print(f"  id{r['image_id']:<6} {r['gt_cls']:>10} · {r['pred_cls']:<10} {r['iou']:.3f}  {r['split']}")

    if args.sheet:
        for kind in order:
            top = [r for r in rows if r["kind"] == kind][:args.sheet]
            legend = ("yellow/red=the two overlapping labels" if kind == "duplicate"
                      else "yellow=label in question  red=prediction")
            render_sheet(top, images_by_id, gt, classes, out / f"sheet_{kind}.png",
                         f"{kind} top {len(top)}  green=labels  {legend}")
        print(f"\n대조 시트: {out}\\sheet_*.png")


if __name__ == "__main__":      # Windows spawn 이 스크립트를 다시 import 한다 (CLAUDE.md 함정 7)
    main()
