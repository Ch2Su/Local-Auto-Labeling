"""모델이 찾은 이륜 탈것 박스 중 어느 PM 박스에도 들지 않는 것(신뢰도 높은 순)의 크롭 시트 — 세워 둔 탈것이 라벨에 없는지 본다."""
import sys, json, os
from PIL import Image, ImageDraw
pred, img_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(pred, encoding="utf-8"))
def inter(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
def area(a):
    return max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
rows = []
n_all = 0
conf_bins = {"<0.5": 0, "0.5-0.7": 0, ">=0.7": 0}
for it in d["items"]:
    pms = [(x, y, x + w, y + h) for code, (x, y, w, h) in it["pm"]]
    for name, cf, bb in it["pred"]:
        if name in ("bicycle", "motorcycle", "scooter") and area(bb) > 0:
            n_all += 1
            if not any(inter(P, bb) / area(bb) >= 0.5 for P in pms):
                rows.append((cf, it["stem"], name, bb, pms))
                conf_bins["<0.5" if cf < 0.5 else ("0.5-0.7" if cf < 0.7 else ">=0.7")] += 1
rows.sort(reverse=True)
print("two-wheeler predictions", n_all, "outside every PM box (less than half inside)", len(rows), conf_bins)
CELL = 250
pick = rows[:10] + rows[len(rows) // 2 - 5: len(rows) // 2 + 5]
sheet = Image.new("RGB", (CELL * 5, CELL * 4), (30, 30, 30))
for k, (cf, stem, name, bb, pms) in enumerate(pick[:20]):
    im = Image.open(os.path.join(img_dir, stem + ".jpg")).convert("RGB")
    cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
    half = max(bb[2] - bb[0], bb[3] - bb[1]) * 1.1
    box = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
    s = CELL / (2 * half)
    crop = im.crop(box).resize((CELL, CELL), Image.LANCZOS)
    dr = ImageDraw.Draw(crop)
    t = lambda b: [(b[0] - box[0]) * s, (b[1] - box[1]) * s, (b[2] - box[0]) * s, (b[3] - box[1]) * s]
    dr.rectangle(t(bb), outline=(255, 80, 80), width=2)
    for P in pms:
        dr.rectangle(t(P), outline=(255, 255, 0), width=2)
    dr.text((4, 4), "%s %.2f %dx%d" % (name, cf, bb[2] - bb[0], bb[3] - bb[1]), fill=(255, 255, 255))
    sheet.paste(crop, ((k % 5) * CELL, (k // 5) * CELL))
sheet.save(out, quality=88)
