"""검증 쪽 responder·cone 정답 박스 하나하나를 갈래마다 무엇으로 찾았는지 본다(conf 0.25, sweep_eval 과 같은 짝짓기).

출력은 로컬에만(파일 이름·카메라 키가 있다): gt_detail.jsonl, 그리고 정답 박스를 잘라 붙인 대조 시트(인자로 준 폴더).
칸 아래 글: 번호 · 640 입력에서의 짧은 변(px) · 갈래마다 예측 클래스 앞 두 글자(-- = 못 찾음).
  python -I gt_detail.py --sets FZ11:25,FZ11X2:25,FZ11X1:25 --sheet <폴더>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cmp_common as cc  # noqa: E402

WANT = ("responder", "cone")
SHORT = {"bicycle": "bi", "bus": "bu", "car": "ca", "motorcycle": "mo", "person": "pe", "scooter": "sc", "truck": "tr",
         "pickup": "pi", "responder": "RE", "cone": "CO"}


def weights(name, fold):
    fixed = {"operating": cc.REPO / "models" / "best_y26m_8cls.pt"}
    if name in fixed:
        return fixed[name]
    arm, _, seed = name.partition("@")
    arm, _, ep = arm.partition(":")
    return cc.ROOT / "runs" / (f"{arm}_f{fold}_e{ep or 100}" + (f"_s{seed}" if seed else "")) / "weights" / "last.pt"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--sheet", type=Path, default=None)
    args = ap.parse_args()
    from ultralytics import YOLO

    sets = args.sets.split(",")
    gts = {}                                     # (fold, 파일, 정답 번호) -> 기록
    for fold in (0, 1, 2):
        files = cc.fold_files(fold, "val")
        for name in sets:
            model = YOLO(str(weights(name, fold)))
            for i in range(0, len(files), 8):
                chunk = files[i:i + 8]
                res = model.predict([str(f) for f in chunk], imgsz=args.imgsz, conf=cc.CONF, iou=0.6, max_det=300, device=0, verbose=False)
                for f, r in zip(chunk, res):
                    h, w = r.orig_shape
                    g = cc.load_gt(f, w, h)
                    gt_xy = np.array([x[1:] for x in g], dtype=float).reshape(-1, 4)
                    used = np.zeros(len(g), dtype=bool)
                    hit = {}
                    b = r.boxes
                    conf = b.conf.cpu().numpy(); xy = b.xyxy.cpu().numpy(); cls = b.cls.cpu().numpy().astype(int)
                    for j in np.argsort(-conf):
                        if not len(g):
                            break
                        v = cc.iou_one(xy[j], gt_xy)
                        v[used] = -1
                        k = int(v.argmax())
                        if v[k] >= cc.MATCH_IOU:
                            used[k] = True
                            hit[k] = (model.names[int(cls[j])], float(conf[j]))
                    for k, x in enumerate(g):
                        if x[0] not in WANT:
                            continue
                        rec = gts.setdefault((fold, str(f), k), {"fold": fold, "file": str(f), "cls": x[0], "xyxy": [float(t) for t in x[1:]],
                                                               "short640": round(min(x[3] - x[1], x[4] - x[2]) * args.imgsz / max(w, h), 1),
                                                               "camera": Path(f).name.split("_")[0], "pred": {}})
                        rec["pred"][name] = hit.get(k)
    recs = sorted(gts.values(), key=lambda r: (r["cls"], r["camera"], r["file"], r["xyxy"][0]))
    with open(cc.ROOT / "gt_detail.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for c in WANT:
        rs = [r for r in recs if r["cls"] == c]
        print(f"== {c} {len(rs)}개 ==")
        for name in sets:
            got = [r["pred"].get(name) for r in rs]
            from collections import Counter
            cnt = Counter(p[0] if p else "(못 찾음)" for p in got)
            print(f"  {name}: {dict(cnt.most_common())}")
        for i, r in enumerate(rs, 1):
            print(f"   {i:2d} f{r['fold']} {r['camera']:9s} {r['short640']:6.1f}px  " +
                  "  ".join(f"{n}={(r['pred'].get(n) or ('--',))[0]}" for n in sets))
    if args.sheet:
        font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 12)
        for c, size, grow, mins in (("responder", 200, 1.5, 96), ("cone", 160, 5.0, 96)):
            rs = [r for r in recs if r["cls"] == c]
            cols = 9
            rows = (len(rs) + cols - 1) // cols
            im = Image.new("RGB", (cols * (size + 2) + 2, rows * (size + 30) + 2), (24, 24, 24))
            d = ImageDraw.Draw(im)
            for i, r in enumerate(rs):
                with Image.open(r["file"]) as src:
                    src = src.convert("RGB")
                x1, y1, x2, y2 = r["xyxy"]
                side = max(max(x2 - x1, y2 - y1) * grow, mins)
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                box = (int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2))
                t = src.crop(box).resize((size, size), Image.BICUBIC)
                k = size / (box[2] - box[0])
                dt = ImageDraw.Draw(t)
                dt.rectangle(((x1 - box[0]) * k, (y1 - box[1]) * k, (x2 - box[0]) * k, (y2 - box[1]) * k), outline=(255, 40, 40) if c == "responder" else (0, 235, 255))
                x, y = 2 + (i % cols) * (size + 2), 2 + (i // cols) * (size + 30)
                im.paste(t, (x, y))
                d.text((x + 2, y + size), f"{i + 1} f{r['fold']} {r['short640']:.0f}px {r['camera']}", fill=(230, 230, 230), font=font)
                d.text((x + 2, y + size + 13), " ".join(SHORT.get((r["pred"].get(n) or ("--",))[0], "--") for n in sets), fill=(255, 220, 120), font=font)
            args.sheet.mkdir(parents=True, exist_ok=True)
            im.save(args.sheet / f"current_{c}_gt.jpg", "JPEG", quality=90)
            print("시트:", f"current_{c}_gt.jpg", im.size, "· 둘째 줄 순서:", " ".join(sets))
    return 0


if __name__ == "__main__":
    sys.exit(main())
