"""VL1.zip (검증 라벨 56,423개) 전수 통계 — 읽기 전용."""
import sys, zipfile, json, collections, math

z = zipfile.ZipFile(sys.argv[1])
top_keys = collections.Counter()
ann_keys = collections.Counter()
info_vals = collections.defaultdict(collections.Counter)
desc_size = collections.Counter()
pm_codes = collections.Counter()
pm_shape = collections.Counter()
pm_keys = collections.Counter()
area_codes = collections.Counter()
area_shape = collections.Counter()
env_keys = collections.Counter()
pm_per_img = collections.Counter()
other = collections.defaultdict(collections.Counter)
videos = collections.defaultdict(set)
n = 0
bad = 0
wh = []
oob = 0
for i in z.infolist():
    if i.is_dir():
        continue
    try:
        d = json.loads(z.read(i))
    except Exception as e:
        bad += 1
        continue
    n += 1
    top_keys.update(d.keys())
    for k, v in d.get("info", {}).items():
        if k not in ("video_id", "clip_id"):
            info_vals[k][v] += 1
    videos[d["info"].get("device")].add(d["info"].get("video_id"))
    ds = d.get("description", {})
    W, H = ds.get("imageWidth"), ds.get("imageHeight")
    desc_size[(W, H)] += 1
    a = d.get("annotations", {})
    ann_keys.update(a.keys())
    pms = a.get("PM", []) or []
    pm_per_img[min(len(pms), 10)] += 1
    for p in pms:
        pm_keys.update(p.keys())
        pm_codes[p.get("PM_code")] += 1
        pm_shape[p.get("shape_type")] += 1
        pts = p.get("points")
        if p.get("shape_type") == "bbox" and isinstance(pts, list) and len(pts) == 4:
            x, y, w, h = pts
            wh.append((w, h))
            if W and H and (x < -1 or y < -1 or x + w > W + 1 or y + h > H + 1):
                oob += 1
    for e in a.get("environment", []) or []:
        env_keys.update(e.keys())
        area_codes[e.get("area_code")] += 1
        area_shape[e.get("shape_type")] += 1
    for k, v in a.items():
        if k not in ("PM", "environment"):
            other[k][type(v).__name__] += 1

def show(name, c, key=None):
    print(name)
    items = sorted(c.items(), key=key or (lambda kv: (-kv[1], str(kv[0]))))
    for k, v in items:
        print("   %8d  %r" % (v, k))

print("files", n, "bad", bad)
show("top keys", top_keys)
show("annotation keys", ann_keys)
for k, c in info_vals.items():
    show("info." + k, c)
print("videos per device", {k: len(v) for k, v in videos.items()})
show("imageWidth,imageHeight", desc_size)
show("PM keys", pm_keys)
show("PM shape", pm_shape)
show("PM_code", pm_codes, key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 999)
show("PM per image (10 = 10+)", pm_per_img, key=lambda kv: kv[0])
show("env keys", env_keys)
show("env shape", area_shape)
show("area_code", area_codes, key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 999)
print("other annotation keys", {k: dict(v) for k, v in other.items()})
print("PM boxes", len(wh), "out-of-bounds vs declared size", oob)
# 검증 라벨에 NaN 좌표가 하나 있다 — NaN 이 섞이면 sorted 가 순서를 못 맞춰 분위수가 뒤죽박죽이 된다 (2026-11-22 에 겪음).
ok = [(w, h) for w, h in wh if all(isinstance(v, (int, float)) and math.isfinite(v) for v in (w, h))]
print("non-finite w/h", len(wh) - len(ok))
wh = ok
if wh:
    ws = sorted(w for w, h in wh); hs = sorted(h for w, h in wh)
    q = lambda s, p: round(s[int(len(s) * p)], 1)
    print("bbox w  p5/p25/p50/p75/p95", q(ws, .05), q(ws, .25), q(ws, .5), q(ws, .75), q(ws, .95))
    print("bbox h  p5/p25/p50/p75/p95", q(hs, .05), q(hs, .25), q(hs, .5), q(hs, .75), q(hs, .95))
