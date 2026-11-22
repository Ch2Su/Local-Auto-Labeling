"""학습 라벨 색인(58만)으로 폴더별 종류 분포·박스 크기·프레임당 PM 수·이상값을 센다."""
import sys, json, collections, math
import numpy as np

def group(code):
    c = int(code)
    return "motorcycle" if c <= 19 else ("bicycle" if c <= 27 else "scooter")

frames = collections.Counter()
boxes = collections.defaultdict(collections.Counter)
vids = collections.defaultdict(lambda: collections.defaultdict(set))
frames_with = collections.defaultdict(collections.Counter)
codes = collections.Counter()
err = 0; nan = 0; neg = 0; oob = 0; empty = 0; noenv = 0
hs = collections.defaultdict(list)
per_frame = collections.Counter()
sizes = collections.Counter()
for line in open(sys.argv[1], encoding="utf-8"):
    rel, stem, W, H, pm, nenv = json.loads(line)
    frames[rel] += 1
    if isinstance(pm, str):
        err += 1
        continue
    sizes[(W, H)] += 1
    if not pm:
        empty += 1
    if not nenv:
        noenv += 1
    per_frame[min(len(pm), 8)] += 1
    seen = set()
    vid = stem.split("_")[0]
    for p in pm:
        code = p[0]
        pts = p[1:]
        if len(pts) != 4 or any(v is None or (isinstance(v, float) and math.isnan(v)) for v in pts):
            nan += 1
            continue
        x, y, w, h = pts
        if w <= 0 or h <= 0:
            neg += 1
            continue
        if x < -2 or y < -2 or x + w > (W or 1920) + 2 or y + h > (H or 1080) + 2:
            oob += 1
        g = group(code)
        codes[code] += 1
        boxes[rel][g] += 1
        vids[rel][g].add(vid)
        hs[g].append(h)
        seen.add(g)
    for g in seen:
        frames_with[rel][g] += 1

tot_frames = sum(frames.values())
print("frames", tot_frames, "json errors", err, "frames with no PM", empty, "frames with no environment polygon", noenv)
print("declared sizes", dict(sizes))
print("bad boxes: nan/missing", nan, " non-positive w/h", neg, " outside frame", oob)
print()
print("%-22s %8s | %-27s | %-27s | %-27s" % ("folder", "frames", "motorcycle box/frames/videos", "bicycle box/frames/videos", "scooter box/frames/videos"))
T = collections.Counter(); TF = collections.Counter()
for rel in sorted(frames):
    cells = []
    for g in ("motorcycle", "bicycle", "scooter"):
        cells.append("%7d /%7d /%5d" % (boxes[rel][g], frames_with[rel][g], len(vids[rel][g])))
        T[g] += boxes[rel][g]; TF[g] += frames_with[rel][g]
    print("%-22s %8d | %-27s | %-27s | %-27s" % (rel, frames[rel], *cells))
n = sum(T.values())
print("TOTAL boxes", n, {g: "%d (%.0f%%)" % (T[g], 100 * T[g] / n) for g in T}, "frames containing", dict(TF))
print("PM per frame (8 = 8+):", sorted(per_frame.items()))
for g in ("motorcycle", "bicycle", "scooter"):
    a = np.array(hs[g])
    print("%-10s box height px p10/p25/p50/p75/p90: %s   share h<48: %.0f%%  h>=96: %.0f%%" % (g, np.percentile(a, [10, 25, 50, 75, 90]).round().astype(int).tolist(), 100 * (a < 48).mean(), 100 * (a >= 96).mean()))
print("codes:", sorted(codes.items(), key=lambda kv: int(kv[0])))
