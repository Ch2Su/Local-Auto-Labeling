"""축소본 상관이 문턱 근처인 영상 쌍을 나란히 그려 본다 — '같은 화면' 문턱을 눈으로 정하려고."""
import os, sys, collections, random
import numpy as np
from PIL import Image, ImageDraw
from concurrent.futures import ThreadPoolExecutor

img_dir, out = sys.argv[1], sys.argv[2]
by_vid = collections.defaultdict(list)
for f in os.listdir(img_dir):
    if f.endswith(".jpg"):
        by_vid[f.split("_")[0]].append(f)
vids = sorted(by_vid)
def thumb(v):
    fs = sorted(by_vid[v]); f = fs[len(fs) // 2]
    im = Image.open(os.path.join(img_dir, f)); im.draft("L", (240, 135))
    a = np.asarray(im.convert("L").resize((32, 18), Image.BILINEAR), dtype=np.float32).ravel()
    return v, f, (a - a.mean()) / (a.std() + 1e-6)
with ThreadPoolExecutor(12) as ex:
    res = list(ex.map(thumb, vids))
X = np.stack([r[2] for r in res]); C = (X @ X.T) / X.shape[1]
random.seed(4)
bands = [(0.90, 0.95), (0.80, 0.85), (0.70, 0.75), (0.60, 0.65)]
W, H = 300, 169
sheet = Image.new("RGB", (W * 2 * 4 + 30, (H + 4) * len(bands) * 2), (20, 20, 20))
for bi, (lo, hi) in enumerate(bands):
    ii, jj = np.where((np.triu(C, 1) >= lo) & (np.triu(C, 1) < hi))
    idx = list(range(len(ii))); random.shuffle(idx)
    print("band %.2f-%.2f pairs %d" % (lo, hi, len(ii)))
    for k, p in enumerate(idx[:8]):
        a, b = res[ii[p]], res[jj[p]]
        row, col = bi * 2 + k // 4, k % 4
        for s, r in enumerate((a, b)):
            im = Image.open(os.path.join(img_dir, r[1])); im.draft("RGB", (480, 270))
            im = im.convert("RGB").resize((W, H), Image.BILINEAR)
            d = ImageDraw.Draw(im); d.rectangle([0, 0, 150, 13], fill=(0, 0, 0))
            d.text((3, 1), "%s  %.2f" % (r[0], C[ii[p], jj[p]]), fill=(255, 255, 0))
            sheet.paste(im, (col * (2 * W + 10) + s * W, row * (H + 4)))
sheet.save(out, quality=85)
print(out, sheet.size)
