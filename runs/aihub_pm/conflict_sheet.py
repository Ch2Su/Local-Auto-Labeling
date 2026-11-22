"""코드 종류와 모델 클래스가 다른 곳 전부의 크롭 시트 + PM 박스 안에서 모델의 탈것·사람 박스가 놓이는 자리(중앙값)."""
import sys, json, os, statistics
from PIL import Image, ImageDraw
out = sys.argv[1]
srcs = [(sys.argv[i], sys.argv[i + 1]) for i in range(2, len(sys.argv), 2)]
def grp(c):
    c = int(c); return "motorcycle" if c <= 19 else ("bicycle" if c <= 27 else "scooter")
def inter(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
def area(a):
    return max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
conf = []
geo = {}
for pred, img_dir in srcs:
    d = json.load(open(pred, encoding="utf-8"))
    for it in d["items"]:
        for code, (x, y, w, h) in it["pm"]:
            P = (x, y, x + w, y + h); T = grp(code)
            best = None; rider = None
            for name, cf, bb in it["pred"]:
                if area(bb) <= 0: continue
                i = inter(P, bb)
                if name in ("bicycle", "motorcycle", "scooter") and i / area(bb) >= 0.7 and i / area(P) >= 0.10 and (best is None or cf > best[1]):
                    best = (name, cf, bb)
                if name == "person" and i / area(bb) >= 0.6 and i / area(P) >= 0.15 and (rider is None or cf > rider[1]):
                    rider = (name, cf, bb)
            if best and best[0] != T:
                conf.append((T, best[0], code, it["stem"], P, best, img_dir, it["pred"]))
            if best and best[0] == T and rider:
                n = lambda b: ((b[0] - P[0]) / w, (b[1] - P[1]) / h, (b[2] - P[0]) / w, (b[3] - P[1]) / h)
                geo.setdefault(T, {"v": [], "r": []})
                geo[T]["v"].append(n(best[2])); geo[T]["r"].append(n(rider[2]))
for T, g in geo.items():
    med = lambda rows: [round(statistics.median(r[k] for r in rows), 3) for k in range(4)]
    q = lambda rows, p: [round(sorted(r[k] for r in rows)[int(len(rows) * p)], 2) for k in range(4)]
    print("%-10s n=%d  vehicle in PM (x1,y1,x2,y2) median %s  p25 %s p75 %s" % (T, len(g["v"]), med(g["v"]), q(g["v"], .25), q(g["v"], .75)))
    print("%-10s        person  in PM              median %s  p25 %s p75 %s" % ("", med(g["r"]), q(g["r"], .25), q(g["r"], .75)))
print("conflicts", len(conf), {(a, b): sum(1 for c in conf if c[0] == a and c[1] == b) for a, b in {(c[0], c[1]) for c in conf}})
CELL = 220
conf.sort(key=lambda c: (c[0], c[1]))
cols = 8
rows_n = (len(conf) + cols - 1) // cols
sheet = Image.new("RGB", (CELL * cols, CELL * rows_n), (30, 30, 30))
for k, (T, M, code, stem, P, best, img_dir, preds) in enumerate(conf):
    im = Image.open(os.path.join(img_dir, stem + ".jpg")).convert("RGB")
    cx, cy = (P[0] + P[2]) / 2, (P[1] + P[3]) / 2
    half = max(P[2] - P[0], P[3] - P[1]) * 0.7
    box = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
    s = CELL / (2 * half)
    crop = im.crop(box).resize((CELL, CELL), Image.LANCZOS)
    dr = ImageDraw.Draw(crop)
    dr.rectangle([0, CELL - 16, CELL, CELL], fill=(0, 0, 0))
    dr.text((3, CELL - 14), "code %s=%s / model %s" % (code, T[:5], M[:5]), fill=(255, 255, 0))
    sheet.paste(crop, ((k % cols) * CELL, (k // cols) * CELL))
sheet.save(out, quality=88)
print("sheet", out, sheet.size)
