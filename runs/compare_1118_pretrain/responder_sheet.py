"""검수 라벨 갈래(FZ11R)가 responder 로 잘못 낸 것을 대조 시트로 만든다 — 평가 라벨의 실수인지 모델의 실수인지 눈으로 가르려고.

시트 A: 정답이 차량(car·truck·bus·pickup)인데 responder 예측이 짝이 된 정답 박스(seed 셋을 합쳐 정답 박스 하나에 칸 하나).
시트 B: 짝이 없는 responder 예측(헛 박스) — seed 셋의 예측을 IoU 0.5 로 묶어 칸 하나. 가장 많이 겹치는 정답 박스로 갈래를 단다:
  '같은 차에 겹침' = 그 정답이 이미 다른(신뢰도 높은) 예측과 짝이 됨(IoU ≥ 0.5) · '일부만 겹침' = 0.1 ≤ IoU < 0.5 · '정답 없음' = IoU < 0.1.
짝짓기는 cmp_common.count 와 같다(conf 0.25, conf 순 탐욕 1:1, IoU 0.5). 칸 아래에 기준(FZ11)·검수 전(FZ11U) 갈래가 같은 자리를
responder 로 낸 seed 수도 적는다. 시트와 칸 번호 목록(tsv — 파일 이름이 있다)은 저장소 밖(또는 git 이 빼는 export/)에만 쓴다.

  python -I responder_sheet.py <출력 폴더>
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cmp_common as cc  # noqa: E402

ARMS = {"R": "FZ11R", "U": "FZ11U", "B": "FZ11"}       # B = 기준
SEEDS = (0, 1, 2)
VEHICLES = {"car", "truck", "bus", "pickup"}
TILE, COLS = 240, 7
SHORT_KIND = {"정답 없음": "없음", "일부만 겹침": "일부", "같은 차에 겹침": "중복"}


def weights(arm: str, fold: int, seed: int) -> Path:
    return cc.ROOT / "runs" / (f"{ARMS[arm]}_f{fold}_e25" + (f"_s{seed}" if seed else "")) / "weights" / "last.pt"


def iou(a, b) -> float:
    return float(cc.iou_one(np.asarray(a, dtype=float), np.asarray([b], dtype=float))[0])


def run_model(model, files):
    """파일마다 [(xyxy, conf, 예측 클래스, 짝 정답 번호 또는 None)] 와 정답 목록."""
    out = {}
    for i in range(0, len(files), 8):
        chunk = files[i:i + 8]
        res = model.predict([str(f) for f in chunk], imgsz=640, conf=cc.CONF, iou=0.6, max_det=300, device=0, verbose=False)
        for f, r in zip(chunk, res):
            h, w = r.orig_shape
            gts = cc.load_gt(f, w, h)
            gt_xy = np.array([g[1:] for g in gts], dtype=float).reshape(-1, 4)
            used = np.zeros(len(gts), dtype=bool)
            b = r.boxes
            conf = b.conf.cpu().numpy(); xy = b.xyxy.cpu().numpy(); cls = b.cls.cpu().numpy().astype(int)
            preds = []
            for j in np.argsort(-conf):
                k_hit = None
                if len(gts):
                    v = cc.iou_one(xy[j], gt_xy)
                    v[used] = -1
                    k = int(v.argmax())
                    if v[k] >= cc.MATCH_IOU:
                        used[k] = True
                        k_hit = k
                preds.append((xy[j].tolist(), float(conf[j]), model.names[int(cls[j])], k_hit))
            out[str(f)] = (preds, gts, (w, h))
    return out


def font(size):
    for p in ("C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/consola.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def tile(img_path, focus, boxes):
    """focus 박스 둘레를 잘라 TILE 크기로 — boxes = [(xyxy, 색, 굵기, 글)]."""
    with Image.open(img_path) as src:
        src = src.convert("RGB")
    x1, y1, x2, y2 = focus
    side = max(max(x2 - x1, y2 - y1) * 2.2, 160)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    box = (int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2))
    t = src.crop(box).resize((TILE, TILE), Image.BICUBIC)
    k = TILE / (box[2] - box[0])
    d = ImageDraw.Draw(t)
    f = font(11)
    for (bx1, by1, bx2, by2), color, width, label in boxes:
        r = ((bx1 - box[0]) * k, (by1 - box[1]) * k, (bx2 - box[0]) * k, (by2 - box[1]) * k)
        if r[2] < 0 or r[3] < 0 or r[0] > TILE or r[1] > TILE:
            continue
        d.rectangle(r, outline=color, width=width)
        if label:
            d.text((max(0, r[0]) + 2, max(0, r[1]) + 1), label, fill=color, font=f)
    return t


def sheet(tiles, title, legend, out_path):
    cap_h = 36
    rows = max(1, (len(tiles) + COLS - 1) // COLS)
    lines = legend.split("\n")
    head = 30 + 17 * len(lines)
    im = Image.new("RGB", (COLS * (TILE + 2) + 2, head + rows * (TILE + cap_h + 2) + 2), (24, 24, 24))
    d = ImageDraw.Draw(im)
    d.text((6, 4), title, fill=(255, 255, 255), font=font(16))
    for n, line in enumerate(lines):
        d.text((6, 25 + 17 * n), line, fill=(200, 200, 200), font=font(12))
    f = font(12)
    for i, (t, line1, line2) in enumerate(tiles):
        x, y = 2 + (i % COLS) * (TILE + 2), head + (i // COLS) * (TILE + cap_h + 2)
        im.paste(t, (x, y))
        d.text((x + 3, y + TILE + 2), line1, fill=(240, 240, 240), font=f)
        d.text((x + 3, y + TILE + 18), line2, fill=(255, 215, 120), font=f)
    im.save(out_path, "JPEG", quality=90)
    return im.size


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    from ultralytics import YOLO

    res = {}                                            # (arm, seed) -> {파일: (preds, gts, (w, h))}
    fold_of = {}
    for fold in (0, 1, 2):
        files = cc.fold_files(fold, "val")
        for f in files:
            fold_of[str(f)] = fold
        for arm in ARMS:
            for seed in SEEDS:
                res.setdefault((arm, seed), {}).update(run_model(YOLO(str(weights(arm, fold, seed))), files))
    files = sorted(fold_of)

    def resp_on_gt(arm, f, k):
        """그 정답 박스와 짝이 된 예측이 responder 인 seed 수와 그때 conf·예측 박스."""
        n, best = 0, None
        for seed in SEEDS:
            for xy, c, name, hit in res[(arm, seed)][f][0]:
                if hit == k and name == "responder":
                    n += 1
                    if best is None or c > best[0]:
                        best = (c, xy)
        return n, best

    def resp_near(arm, f, box):
        """그 자리(IoU ≥ 0.5)에 responder 를 낸 seed 수(짝이 됐든 안 됐든)."""
        return sum(1 for seed in SEEDS if any(name == "responder" and iou(xy, box) >= 0.5
                                              for xy, c, name, hit in res[(arm, seed)][f][0]))

    # 시트 A — 차량 정답을 responder 로
    A = []
    for f in files:
        gts = res[("R", 0)][f][1]
        for k, g in enumerate(gts):
            if g[0] not in VEHICLES:
                continue
            n, best = resp_on_gt("R", f, k)
            if n:
                A.append({"f": f, "k": k, "gt": g, "n": n, "conf": best[0], "pred": best[1],
                          "U": resp_on_gt("U", f, k)[0], "B": resp_on_gt("B", f, k)[0]})
    A.sort(key=lambda x: (-x["n"], -x["conf"]))

    # 시트 B — 짝 없는 responder 예측, seed 셋을 자리로 묶음
    B = []
    per_seed = Counter()
    for f in files:
        clusters = []
        for seed in SEEDS:
            preds, gts, _ = res[("R", seed)][f]
            matched = {hit for *_, hit in preds if hit is not None}
            for xy, c, name, hit in preds:
                if name != "responder" or hit is not None:
                    continue
                if gts:
                    ious = [iou(xy, g[1:]) for g in gts]
                    j = int(np.argmax(ious))
                    best_iou, best_cls, best_used = ious[j], gts[j][0], j in matched
                else:
                    best_iou, best_cls, best_used = 0.0, None, False
                kind = ("같은 차에 겹침" if best_iou >= 0.5 and best_used else
                        "일부만 겹침" if best_iou >= 0.1 else "정답 없음")
                per_seed[(seed, kind)] += 1
                if kind == "같은 차에 겹침":         # 그 차의 정답이 responder 인가 다른 차량인가, 먼저 짝이 된 예측은 무엇인가
                    per_seed[(seed, "겹침:" + ("responder" if best_cls == "responder" else "다른 클래스"))] += 1
                    first = next(nm for _, _, nm, h in preds if h == j)
                    per_seed[("짝", best_cls, first)] += 1
                for cl in clusters:
                    if iou(xy, cl["pred"]) >= 0.5:
                        cl["seeds"].add(seed)
                        if c > cl["conf"]:
                            cl.update(conf=c, pred=xy, kind=kind, gt_iou=best_iou, gt_cls=best_cls)
                        break
                else:
                    clusters.append({"f": f, "seeds": {seed}, "conf": c, "pred": xy, "kind": kind, "gt_iou": best_iou, "gt_cls": best_cls})
        for cl in clusters:
            cl["n"] = len(cl["seeds"])
            cl["U"] = resp_near("U", f, cl["pred"])
            cl["B"] = resp_near("B", f, cl["pred"])
            B.append(cl)
    order = {"정답 없음": 0, "일부만 겹침": 1, "같은 차에 겹침": 2}
    B.sort(key=lambda x: (order[x["kind"]], -x["n"], -x["conf"]))

    print(f"시트 A(차량 정답 → responder): 정답 박스 {len(A)}개 · R seed 수 분포 {dict(Counter(x['n'] for x in A))} · 정답 클래스 {dict(Counter(x['gt'][0] for x in A))}")
    print(f"  같은 정답을 기준·검수 전 갈래도 responder 로 낸 것(seed 하나 이상): 기준 {sum(1 for x in A if x['B'])} · 검수 전 {sum(1 for x in A if x['U'])}")
    print(f"시트 B(짝 없는 responder): 자리 {len(B)}개 · 갈래 {dict(Counter(x['kind'] for x in B))} · R seed 수 분포 {dict(Counter(x['n'] for x in B))}")
    for seed in SEEDS:
        print(f"  seed {seed}: " + ", ".join(f"{k} {per_seed[(seed, k)]}" for k in order) +
              f" (그중 정답이 responder {per_seed[(seed, '겹침:responder')]} · 다른 클래스 {per_seed[(seed, '겹침:다른 클래스')]})")
    print("  '같은 차에 겹침' 자리의 정답 클래스:", dict(Counter(x["gt_cls"] for x in B if x["kind"] == "같은 차에 겹침")))
    print("  중복 예측(seed 셋 합)의 (정답 클래스 ← 그 정답과 먼저 짝이 된 예측 클래스):",
          {f"{k[1]}←{k[2]}": v for k, v in sorted(per_seed.items(), key=lambda kv: -kv[1]) if k[0] == "짝"})
    print(f"  같은 자리에 기준·검수 전 갈래도 responder 를 낸 것(seed 하나 이상): 기준 {sum(1 for x in B if x['B'])} · 검수 전 {sum(1 for x in B if x['U'])}")

    args.out.mkdir(parents=True, exist_ok=True)
    rows = ["시트\t칸\t묶음\t파일\t정답 클래스\t정답 xyxy\t예측 xyxy\tR seed\tconf\t기준 seed\t검수 전 seed\t갈래"]

    def short640(xy, f):
        w, h = res[("R", 0)][f][2]
        return min(xy[2] - xy[0], xy[3] - xy[1]) * 640 / max(w, h)

    tiles = []
    for i, x in enumerate(A, 1):
        gts = res[("R", 0)][x["f"]][1]
        others = [(g[1:], (90, 200, 90), 1, g[0][:2]) for j, g in enumerate(gts) if j != x["k"]]
        t = tile(x["f"], x["gt"][1:], others + [(x["gt"][1:], (0, 230, 255), 2, x["gt"][0]), (x["pred"], (255, 60, 60), 2, "")])
        tiles.append((t, f"#{i} 정답 {x['gt'][0]} · {short640(x['gt'][1:], x['f']):.0f}px",
                      f"R {x['n']}/3 ({x['conf']:.2f}) · 기준 {x['B']}/3 · 검수전 {x['U']}/3"))
        rows.append(f"A\t{i}\t{fold_of[x['f']]}\t{x['f']}\t{x['gt'][0]}\t{[round(v) for v in x['gt'][1:]]}\t{[round(v) for v in x['pred']]}\t"
                    f"{x['n']}\t{x['conf']:.3f}\t{x['B']}\t{x['U']}\t")
    pa = args.out / "resp_vehicle_as_responder.jpg"
    sa = sheet(tiles, f"A. 정답은 차량인데 검수 라벨 갈래(R)가 responder 로 낸 것 — {len(A)}개",
               "하늘색 = 정답 박스(클래스) · 빨강 = R 의 responder 예측 · 초록 = 다른 정답 박스 · 아랫줄 = 그렇게 낸 seed 수(가장 높은 conf)", pa)

    tiles = []
    for i, x in enumerate(B, 1):
        gts = res[("R", 0)][x["f"]][1]
        others = [(g[1:], (90, 200, 90), 1, g[0][:2]) for g in gts]
        t = tile(x["f"], x["pred"], others + [(x["pred"], (255, 60, 60), 2, "")])
        near = f"{x['gt_cls']} {x['gt_iou']:.2f}" if x["gt_cls"] else "-"
        tiles.append((t, f"#{i} {SHORT_KIND[x['kind']]} · 정답 {near}",
                      f"R {x['n']}/3 ({x['conf']:.2f}) · 기준 {x['B']}/3 · 검수전 {x['U']}/3"))
        rows.append(f"B\t{i}\t{fold_of[x['f']]}\t{x['f']}\t{x['gt_cls']}\t\t{[round(v) for v in x['pred']]}\t"
                    f"{x['n']}\t{x['conf']:.3f}\t{x['B']}\t{x['U']}\t{x['kind']}")
    pb = args.out / "resp_unmatched_responder.jpg"
    sb = sheet(tiles, f"B. 짝이 없는 responder 예측(헛 박스) — 자리 {len(B)}개",
               "빨강 = R 의 responder 예측 · 초록 = 정답 박스(앞 두 글자) · '정답' 뒤 = 가장 많이 겹친 정답 박스의 클래스와 IoU\n"
               "중복 = 그 정답 박스는 다른(신뢰도 높은) 예측과 이미 짝이 됨 · 일부 = 정답과 IoU 0.1–0.5 · 없음 = 근처에 정답 박스 없음", pb)
    (args.out / "resp_sheet_index.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("시트:", pa.name, sa, "·", pb.name, sb, "· 목록:", "resp_sheet_index.tsv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
