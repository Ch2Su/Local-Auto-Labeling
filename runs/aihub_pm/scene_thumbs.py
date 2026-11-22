"""영상마다 프레임 한 장의 작은 흑백 축소본을 떠서 같은 화면(카메라 자리)끼리 묶어 본다 — 카메라 수 어림. 읽기 전용."""
import os, sys, collections, json, time
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

img_dir, out = sys.argv[1], sys.argv[2]
by_vid = collections.defaultdict(list)
for f in os.listdir(img_dir):
    if f.endswith(".jpg"):
        by_vid[f.split("_")[0]].append(f)
vids = sorted(by_vid)
def thumb(v):
    fs = sorted(by_vid[v])
    f = fs[len(fs) // 2]
    try:
        im = Image.open(os.path.join(img_dir, f))
        im.draft("L", (240, 135))
        im = im.convert("L").resize((32, 18), Image.BILINEAR)
        a = np.asarray(im, dtype=np.float32).ravel()
        a = (a - a.mean()) / (a.std() + 1e-6)
        return v, f, a
    except Exception as e:
        return v, f, None
t = time.time()
with ThreadPoolExecutor(12) as ex:
    res = list(ex.map(thumb, vids))
res = [r for r in res if r[2] is not None]
X = np.stack([r[2] for r in res])
C = (X @ X.T) / X.shape[1]
print("videos", len(res), "thumbs in", round(time.time() - t), "s")
ids = [r[0] for r in res]
for thr in (0.95, 0.90, 0.85, 0.80, 0.70):
    parent = list(range(len(ids)))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    ii, jj = np.where(np.triu(C, 1) >= thr)
    for a, b in zip(ii.tolist(), jj.tolist()):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    groups = collections.defaultdict(list)
    for k in range(len(ids)):
        groups[find(k)].append(ids[k])
    sizes = sorted((len(g) for g in groups.values()), reverse=True)
    print("corr >= %.2f: clusters %d, singletons %d, top sizes %s" % (thr, len(groups), sum(1 for s in sizes if s == 1), sizes[:12]))
    if thr == 0.90:
        keep = {ids[k]: find(k) for k in range(len(ids))}
        json.dump({"thr": thr, "cluster": keep, "frame": {r[0]: r[1] for r in res}}, open(out, "w"))
        # 묶음이 영상 번호로 이어져 있나: 묶음마다 접두 글자와 번호 범위
        big = sorted(groups.values(), key=len, reverse=True)[:8]
        for g in big:
            pref = collections.Counter(x[0] for x in g)
            nums = sorted(int(x[1:]) for x in g)
            print("   size %4d prefix %s  id range %d..%d" % (len(g), dict(pref), nums[0], nums[-1]))
