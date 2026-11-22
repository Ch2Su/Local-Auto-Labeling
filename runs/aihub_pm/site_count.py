"""실증·연출 영상을 표본으로 뽑아 '같은 카메라 자리'(도구의 same_camera)로 묶으면 자리가 몇 개인지 — 표본이 늘 때 자리 수가 어떻게 느는지."""
import sys, os, random, time, zipfile, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))   # 저장소의 tools/
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import import_aihub_pm as ah

cv2.setNumThreads(1)
mode, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
root = ah.AIHUB_ROOT / "1.Training" / "원천데이터"
random.seed(31)
frames = ah.Frames(root)
if mode == "real":
    rel = "CCTV/주간/맑음/실증"
    by_vid = collections.defaultdict(list)
    for f in os.listdir(root / rel):
        by_vid[f.split("_")[0]].append(f[:-4])
else:
    rel = "CCTV/주간/맑음/연출"
    by_vid = collections.defaultdict(list)
    for name in frames._zip_names():
        if name.startswith(rel + "/"):
            stem = name.rsplit("/", 1)[1][:-4]
            by_vid[stem.split("_")[0]].append(stem)
vids = sorted(by_vid); random.shuffle(vids); vids = vids[:n]
print(mode, "videos in folder", len(by_vid), "sampled", len(vids), flush=True)
t = time.time()
def load(v):
    stem = sorted(by_vid[v])[len(by_vid[v]) // 2]
    return v, stem, ah.scene_of(frames.read(rel, stem))
scenes = [load(v) for v in vids]            # zip 읽기는 한 줄로(같은 핸들)
print("scenes %.0fs" % (time.time() - t), flush=True)
parent = list(range(len(scenes)))
def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]; a = parent[a]
    return a
ex = ThreadPoolExecutor(14)
curve = {}
links = 0
t = time.time()
for i in range(len(scenes)):
    js = list(range(i))
    hits = list(ex.map(lambda j: ah.same_camera(scenes[i][2], scenes[j][2]), js))
    for j, h in zip(js, hits):
        if h:
            links += 1
            ra, rb = find(i), find(j)
            if ra != rb: parent[ra] = rb
    if (i + 1) in (25, 50, 100, 150, 200, 300, 400, 600):
        curve[i + 1] = len({find(k) for k in range(i + 1)})
        print("  first %d videos -> %d camera spots (%.0fs)" % (i + 1, curve[i + 1], time.time() - t), flush=True)
groups = collections.defaultdict(list)
for k in range(len(scenes)):
    groups[find(k)].append(scenes[k][0])
sizes = sorted((len(g) for g in groups.values()), reverse=True)
print("spots", len(groups), "singletons", sum(1 for s in sizes if s == 1), "top sizes", sizes[:15], "links", links)
import json
json.dump({"mode": mode, "curve": curve, "groups": sorted(groups.values(), key=len, reverse=True)}, open(out, "w"))
