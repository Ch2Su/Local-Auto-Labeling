"""킥보드 코드(28-36)인데 모델이 bicycle 로 잡았거나 탈것을 못 찾은 곳의 크롭 시트."""
import sys, json, os
from PIL import Image, ImageDraw
pred, img_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(pred, encoding="utf-8"))
def inter(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
def area(a):
    return max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
rows = {"bicycle": [], "none": []}
for it in d["items"]:
    for code, (x, y, w, h) in it["pm"]:
        if int(code) < 28:
            continue
        P = (x, y, x + w, y + h)
        best = None
        for name, cf, bb in it["pred"]:
            if name in ("bicycle", "motorcycle", "scooter") and area(bb) > 0:
                i = inter(P, bb)
                if i / area(bb) >= 0.7 and i / area(P) >= 0.10 and (best is None or cf > best[1]):
                    best = (name, cf, bb)
        if best is None:
            rows["none"].append((it["stem"], code, P, None, it["pred"]))
        elif best[0] == "bicycle":
            rows["bicycle"].append((it["stem"], code, P, best, it["pred"]))
CELL = 250
for key, lst in rows.items():
    lst = lst[:15]
    sheet = Image.new("RGB", (CELL * 5, CELL * 3), (30, 30, 30))
    for k, (stem, code, P, best, preds) in enumerate(lst):
        im = Image.open(os.path.join(img_dir, stem + ".jpg")).convert("RGB")
        cx, cy = (P[0] + P[2]) / 2, (P[1] + P[3]) / 2
        half = max(P[2] - P[0], P[3] - P[1]) * 0.75
        box = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
        s = CELL / (2 * half)
        crop = im.crop(box).resize((CELL, CELL), Image.LANCZOS)
        dr = ImageDraw.Draw(crop)
        t = lambda b: [(b[0] - box[0]) * s, (b[1] - box[1]) * s, (b[2] - box[0]) * s, (b[3] - box[1]) * s]
        for name, cf, bb in preds:
            if inter(P, bb) > 0:
                col = (0, 255, 255) if name == "person" else ((255, 80, 80) if name in ("bicycle", "motorcycle", "scooter") else (0, 200, 0))
                dr.rectangle(t(bb), outline=col, width=2)
                dr.text((t(bb)[0] + 2, t(bb)[1] + 2), "%s %.2f" % (name[:5], cf), fill=col)
        dr.rectangle(t(P), outline=(255, 255, 0), width=1)
        dr.text((4, CELL - 14), "code %s  h=%d" % (code, P[3] - P[1]), fill=(255, 255, 0))
        sheet.paste(crop, ((k % 5) * CELL, (k // 5) * CELL))
    sheet.save(os.path.join(out, "staged_scooter_%s.jpg" % key), quality=88)
    print(key, len(rows[key]), "shown", len(lst))
