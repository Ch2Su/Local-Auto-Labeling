"""sam3_propose.py 가 쓴 후보(jsonl)를 이름별·문턱별로 나눠 세고, 놓친 정답과 헛 후보를 대조 시트로 붙인다 — 읽기만.

  python -I sam3_analyze.py <후보.jsonl> [--sheet <폴더>] [--sheet-conf 0.5]
칸 아래 글: 프롬프트 이름 앞 두 글자·점수 · 닿은 곳(정답 클래스 또는 none) · 카메라 키. 시트는 파일 이름·화면이 있어 로컬에만 둔다.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cmp_common as cc  # noqa: E402
from sam3_propose import TO_CLASS  # noqa: E402

CONFS = (0.25, 0.35, 0.5, 0.6, 0.7, 0.8)


def match(rec, conf):
    """[(후보, 닿은 정답 번호 또는 None)] — conf 순 탐욕 1:1, IoU 0.5 (sam3_propose.score 와 같다)."""
    gts = rec["gt"]
    gt_xy = np.array([g[1:] for g in gts], dtype=float).reshape(-1, 4)
    used = np.zeros(len(gts), dtype=bool)
    out = []
    for d in sorted((d for d in rec["dets"] if d[1] >= conf), key=lambda t: -t[1]):
        k = None
        if len(gts):
            v = cc.iou_one(np.array(d[2:], dtype=float), gt_xy)
            v[used] = -1
            j = int(v.argmax())
            if v[j] >= cc.MATCH_IOU:
                used[j] = True
                k = j
        out.append((d, k))
    return out


def tile(img, box, size, grow, mins, color):
    x1, y1, x2, y2 = box
    side = max(max(x2 - x1, y2 - y1) * grow, mins)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    b = (int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2))
    t = img.crop(b).resize((size, size), Image.BICUBIC)
    k = size / (b[2] - b[0])
    ImageDraw.Draw(t).rectangle(((x1 - b[0]) * k, (y1 - b[1]) * k, (x2 - b[0]) * k, (y2 - b[1]) * k), outline=color)
    return t


def sheet(items, size, cols, dst):
    """items = [(파일, 박스, 글 한 줄, 색)]"""
    if not items:
        return
    font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 12)
    rows = (len(items) + cols - 1) // cols
    im = Image.new("RGB", (cols * (size + 2) + 2, rows * (size + 16) + 2), (24, 24, 24))
    d = ImageDraw.Draw(im)
    cache = {}
    for i, (f, box, cap, color) in enumerate(items):
        if f not in cache:
            cache.clear()
            cache[f] = Image.open(f).convert("RGB")
        x, y = 2 + (i % cols) * (size + 2), 2 + (i // cols) * (size + 16)
        im.paste(tile(cache[f], box, size, 1.6 if color[0] else 5.0, 96, color), (x, y))
        d.text((x + 2, y + size + 1), cap, fill=(230, 230, 230), font=font)
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, "JPEG", quality=88)
    print("시트:", dst.name, im.size, len(items), "칸")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--sheet", type=Path, default=None)
    ap.add_argument("--sheet-conf", type=float, default=0.5)
    args = ap.parse_args()
    recs = [json.loads(x) for x in args.jsonl.read_text(encoding="utf-8").split("\n") if x.strip()]
    total = Counter(g[0] for r in recs for g in r["gt"])
    print(f"이미지 {len(recs)}장 · 정답 responder {total['responder']} · cone {total['cone']}")
    print("| conf | responder 찾음 | 후보 | 맞음 | 다른 클래스 정답 | 정답 없는 자리 | cone 찾음 | 후보 | 맞음 | 정답 없는 자리 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for conf in CONFS:
        c = Counter()
        for r in recs:
            for d, k in match(r, conf):
                cls = TO_CLASS[d[0]]
                where = "none" if k is None else ("hit" if r["gt"][k][0] == cls else "other")
                c[(cls, where)] += 1
        rr, cn = [c[("responder", w)] for w in ("hit", "other", "none")], [c[("cone", w)] for w in ("hit", "other", "none")]
        print(f"| {conf} | {rr[0]}/{total['responder']} | {sum(rr)} | {rr[0]} | {rr[1]} | {rr[2]} | {cn[0]}/{total['cone']} | {sum(cn)} | {cn[0]} | {cn[1] + cn[2]} |")
    print("\n이름별 (conf 0.5 / 0.7) — 후보 · 맞음 · 다른 클래스 정답 · 정답 없는 자리")
    for conf in (0.5, 0.7):
        c = defaultdict(Counter)
        for r in recs:
            for d, k in match(r, conf):
                where = "none" if k is None else ("hit" if r["gt"][k][0] == TO_CLASS[d[0]] else "other:" + r["gt"][k][0])
                c[d[0]][where] += 1
        for name in TO_CLASS:
            v = c[name]
            other = {k.split(":")[1]: n for k, n in v.items() if k.startswith("other:")}
            print(f"  conf {conf} {name:13s} 후보 {sum(v.values()):3d} · 맞음 {v['hit']:2d} · 다른 클래스 {sum(other.values()):3d} {dict(sorted(other.items(), key=lambda t: -t[1]))} · 없는 자리 {v['none']}")
    # 놓친 정답 (sheet-conf 에서)
    missed, wrong, nowhere = [], [], []
    for r in recs:
        m = match(r, args.sheet_conf)
        got = {k for d, k in m if k is not None and r["gt"][k][0] == TO_CLASS[d[0]]}
        cam = Path(r["file"]).name.split("_")[0]
        for k, g in enumerate(r["gt"]):
            if g[0] in ("responder", "cone") and k not in got:
                short = min(g[3] - g[1], g[4] - g[2])
                missed.append((r["file"], g[1:], f"MISS {g[0][:4]} {short:.0f}px {cam}", (255, 40, 40) if g[0] == "responder" else (0, 235, 255)))
        for d, k in m:
            cls = TO_CLASS[d[0]]
            color = (255, 40, 40) if cls == "responder" else (0, 235, 255)
            cap = f"{d[0][:6]} {d[1]:.2f}"
            if k is None:
                nowhere.append((r["file"], d[2:], f"{cap} none {cam}", color))
            elif r["gt"][k][0] != cls:
                wrong.append((r["file"], d[2:], f"{cap} gt={r['gt'][k][0][:5]} {cam}", color))
    print(f"\nconf {args.sheet_conf}: 놓친 정답 {len(missed)} · 다른 클래스 정답에 닿은 후보 {len(wrong)} · 정답 없는 자리 후보 {len(nowhere)}")
    for cap in sorted(x[2] for x in missed):
        print("   ", cap)
    if args.sheet:
        tag = args.jsonl.stem
        sheet(sorted(missed, key=lambda t: t[2]), 200, 9, args.sheet / f"{tag}_missed.jpg")
        sheet(sorted(wrong, key=lambda t: t[2]), 176, 12, args.sheet / f"{tag}_on_other_class.jpg")
        sheet(sorted(nowhere, key=lambda t: t[2]), 150, 12, args.sheet / f"{tag}_no_gt.jpg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
