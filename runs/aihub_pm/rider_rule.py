"""우리 DB 에서 탈것(bicycle·motorcycle·scooter) 박스와 탄 사람(person) 박스의 관계를 잰다 + 예시 시트. 읽기 전용."""
import sqlite3, sys, os, statistics
from PIL import Image, ImageDraw, ImageOps

db, imm, remap_from, remap_to, out = sys.argv[1], sys.argv[2] == "1", sys.argv[3], sys.argv[4], sys.argv[5]
con = sqlite3.connect("file:" + db + ("?mode=ro&immutable=1" if imm else "?mode=ro"), uri=True)
names = dict(con.execute("select idx, name from classes"))
VEH = {k for k, v in names.items() if v in ("bicycle", "motorcycle", "scooter")}
PER = [k for k, v in names.items() if v == "person"][0]
rows = con.execute("select b.image_id, b.cls, b.cx, b.cy, b.w, b.h, i.path, i.width, i.height from boxes b join images i on i.id=b.image_id where i.status='approved'").fetchall()
by_img = {}
for iid, cls, cx, cy, w, h, path, W, H in rows:
    by_img.setdefault(iid, {"path": path, "W": W, "H": H, "b": []})["b"].append((cls, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))

def inter(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
def area(a):
    return (a[2] - a[0]) * (a[3] - a[1])

stats = {}
examples = {}
for iid, d in by_img.items():
    veh = [b for b in d["b"] if b[0] in VEH]
    per = [b for b in d["b"] if b[0] == PER]
    for v in veh:
        vb = v[1:]
        best, bi = None, 0
        for p in per:
            pb = p[1:]
            i = inter(vb, pb)
            if i > bi:
                best, bi = pb, i
        s = stats.setdefault(names[v[0]], {"n": 0, "rider": 0, "top": [], "p_in_v": [], "v_in_p": []})
        s["n"] += 1
        # 탄 사람으로 볼 조건: 사람 박스의 30% 이상이 탈것 박스와 겹치거나 탈것의 30% 이상이 사람과 겹침
        if best and (bi / area(best) > 0.3 or bi / area(vb) > 0.3):
            s["rider"] += 1
            ph = best[3] - best[1]
            s["top"].append((vb[1] - best[1]) / ph)       # 0 = 탈것 박스 위가 사람 머리와 같다, >0 = 탈것 박스가 머리보다 아래에서 시작
            s["p_in_v"].append(bi / area(best))
            s["v_in_p"].append(bi / area(vb))
        examples.setdefault(names[v[0]], []).append((area(vb) * d["W"] * d["H"], iid, vb, best if best and (bi / area(best) > 0.3 or bi / area(vb) > 0.3) else None))
for k, s in stats.items():
    q = lambda xs, p: round(sorted(xs)[int(len(xs) * p)], 2) if xs else None
    print("%-10s boxes %5d  with rider %5d (%.0f%%) | (vehicle_top - person_top)/person_h p10/p50/p90 = %s/%s/%s | person-in-vehicle p50 %s | vehicle-in-person p50 %s" % (
        k, s["n"], s["rider"], 100 * s["rider"] / max(1, s["n"]), q(s["top"], .1), q(s["top"], .5), q(s["top"], .9), q(s["p_in_v"], .5), q(s["v_in_p"], .5)))

# 예시 시트: 클래스마다 큰 것 10개 (탄 사람이 있는 것 우선)
CELL = 260
for k, ex in examples.items():
    ex = sorted([e for e in ex if e[3]], reverse=True)[:40]
    ex = ex[::max(1, len(ex) // 10)][:10]
    sheet = Image.new("RGB", (CELL * 5, CELL * 2), (30, 30, 30))
    n = 0
    for a, iid, vb, pb in ex:
        d = by_img[iid]
        path = d["path"]
        if remap_from and path.lower().startswith(remap_from.lower()):
            path = remap_to + path[len(remap_from):]
        if not os.path.exists(path):
            continue
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        W, H = im.size
        x1, y1, x2, y2 = vb[0] * W, vb[1] * H, vb[2] * W, vb[3] * H
        px = [pb[0] * W, pb[1] * H, pb[2] * W, pb[3] * H]
        ux1, uy1, ux2, uy2 = min(x1, px[0]), min(y1, px[1]), max(x2, px[2]), max(y2, px[3])
        cx, cy = (ux1 + ux2) / 2, (uy1 + uy2) / 2
        half = max(ux2 - ux1, uy2 - uy1) * 0.75
        box = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
        s = CELL / (2 * half)
        crop = im.crop(box).resize((CELL, CELL), Image.LANCZOS)
        dr = ImageDraw.Draw(crop)
        dr.rectangle([(x1 - box[0]) * s, (y1 - box[1]) * s, (x2 - box[0]) * s, (y2 - box[1]) * s], outline=(255, 255, 0), width=2)
        dr.rectangle([(px[0] - box[0]) * s, (px[1] - box[1]) * s, (px[2] - box[0]) * s, (px[3] - box[1]) * s], outline=(0, 255, 255), width=1)
        dr.text((4, 4), "#%d %s %dx%d" % (iid, k, x2 - x1, y2 - y1), fill=(255, 255, 255))
        sheet.paste(crop, ((n % 5) * CELL, (n // 5) * CELL))
        n += 1
    sheet.save(os.path.join(out, "ours_%s.jpg" % k), quality=88)
    print("sheet", k, n)
