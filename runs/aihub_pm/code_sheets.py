"""풀어 둔 CCTV/주간/맑음/실증 에서 라벨을 표본으로 읽어 PM_code 별 크롭 대조 시트를 만든다 (읽기 전용)."""
import os, sys, json, random, collections
from PIL import Image, ImageDraw

lab_dir, img_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
n_sample = int(sys.argv[4]) if len(sys.argv) > 4 else 6000
os.makedirs(out_dir, exist_ok=True)
random.seed(7)
names = [f for f in os.listdir(lab_dir) if f.endswith(".json")]
random.shuffle(names)
by_code = collections.defaultdict(list)
codes = collections.Counter()
for f in names[:n_sample]:
    try:
        with open(os.path.join(lab_dir, f), encoding="utf-8") as fh:
            d = json.load(fh)
    except Exception:
        continue
    for p in d["annotations"].get("PM") or []:
        x, y, w, h = p["points"]
        codes[p["PM_code"]] += 1
        by_code[p["PM_code"]].append((w * h, f[:-5], (x, y, w, h)))
print("sampled", n_sample, "codes", sorted(codes.items(), key=lambda kv: int(kv[0])))
CELL = 300
for code in sorted(by_code, key=int):
    items = by_code[code]
    # 다른 영상에서 고르게: 영상마다 가장 큰 박스 하나씩, 그중 큰 순서로
    best = {}
    for a, stem, box in items:
        v = stem.split("_")[0]
        if v not in best or a > best[v][0]:
            best[v] = (a, stem, box)
    pick = sorted(best.values(), reverse=True)[:10]
    sheet = Image.new("RGB", (CELL * 5, CELL * 2), (30, 30, 30))
    k = 0
    for a, stem, (x, y, w, h) in pick:
        ip = os.path.join(img_dir, stem + ".jpg")
        if not os.path.exists(ip):
            continue
        im = Image.open(ip).convert("RGB")
        pad = max(w, h) * 0.6
        cx, cy = x + w / 2, y + h / 2
        half = max(w, h) / 2 + pad
        box = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
        crop = im.crop(box).resize((CELL, CELL), Image.LANCZOS)
        s = CELL / (2 * half)
        dr = ImageDraw.Draw(crop)
        dr.rectangle([(x - box[0]) * s, (y - box[1]) * s, (x + w - box[0]) * s, (y + h - box[1]) * s], outline=(255, 255, 0), width=2)
        dr.text((4, 4), "%s %dx%d" % (stem.split("_")[0], w, h), fill=(0, 255, 255))
        sheet.paste(crop, ((k % 5) * CELL, (k // 5) * CELL))
        k += 1
    sheet.save(os.path.join(out_dir, "code_%s.jpg" % code), quality=88)
    print("code", code, "boxes", len(items), "videos", len(best), "tiles", k)
