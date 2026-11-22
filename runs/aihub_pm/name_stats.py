"""파일 이름만으로 영상·클립·프레임 수를 센다 (라벨 JSON 을 열지 않는다)."""
import os, sys, collections, statistics

root = sys.argv[1]
rows = []
for dp, dn, fn in os.walk(root):
    rel = os.path.relpath(dp, root).replace("\\", "/")
    for f in fn:
        if f.endswith(".json"):
            rows.append((rel, f[:-5]))
print("label files", len(rows))
by_folder = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
pref = collections.Counter()
bad = 0
for rel, stem in rows:
    p = stem.split("_")
    if len(p) != 7:
        bad += 1
        continue
    vid, clip, frame = p[0], p[1], p[2]
    by_folder[rel][vid][clip].append(int(frame))
    pref[(rel, vid[0])] += 1
print("names not matching 7 parts:", bad)
print("%-28s %8s %7s %7s  %s" % ("folder", "frames", "videos", "clips", "frames/video p10/p50/p90/max ; prefix"))
allvid = collections.defaultdict(set)
for rel in sorted(by_folder):
    vids = by_folder[rel]
    nfr = sum(len(fr) for v in vids.values() for fr in v.values())
    ncl = sum(len(v) for v in vids.values())
    per = sorted(sum(len(fr) for fr in v.values()) for v in vids.values())
    q = lambda p: per[min(len(per) - 1, int(len(per) * p))]
    pf = {k[1]: n for k, n in pref.items() if k[0] == rel}
    print("%-28s %8d %7d %7d  %d/%d/%d/%d ; %s" % (rel, nfr, len(vids), ncl, q(.1), q(.5), q(.9), per[-1], pf))
    for v in vids:
        allvid[v].add(rel)
print("distinct video ids", len(allvid), " ids in >1 folder:", sum(1 for v in allvid.values() if len(v) > 1))
# 프레임 간격
gaps = collections.Counter()
for rel, vids in by_folder.items():
    for v, clips in vids.items():
        for c, fr in clips.items():
            fr.sort()
            for a, b in zip(fr, fr[1:]):
                gaps[b - a] += 1
print("frame gap top", gaps.most_common(12))
# 가장 큰 영상 몇 개
big = sorted(((sum(len(fr) for fr in clips.values()), rel, v, len(clips)) for rel, vids in by_folder.items() for v, clips in vids.items()), reverse=True)[:8]
for n, rel, v, nc in big:
    print("  big", n, rel, v, "clips", nc)
