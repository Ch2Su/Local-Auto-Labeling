"""model_vs_pm.py 결과를 센다: PM 박스 자리에서 탈것·탄 사람을 찾았나, 코드↔클래스, 프레임당 박스 수."""
import sys, json, collections, math

d = json.load(open(sys.argv[1], encoding="utf-8"))
items = d["items"]
VEH = ("bicycle", "motorcycle", "scooter")
def group(code):
    c = int(code)
    return "motorcycle" if c <= 19 else ("bicycle" if c <= 27 else "scooter")
def inter(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
def area(a):
    return max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])

per_code = collections.defaultdict(collections.Counter)
tot = collections.Counter()
by_size = collections.defaultdict(collections.Counter)
by_type = collections.defaultdict(collections.Counter)
n_pred = collections.Counter()
per_frame = []
nan = 0
pm_heights = []
unmatched_veh = 0
total_veh_pred = 0
for it in items:
    preds = it["pred"]
    per_frame.append(len(preds))
    for name, cf, bb in preds:
        n_pred[name] += 1
    used = set()
    for code, pts in it["pm"]:
        if any(isinstance(v, float) and math.isnan(v) for v in pts):
            nan += 1
            continue
        x, y, w, h = pts
        P = (x, y, x + w, y + h)
        pm_heights.append(h)
        g = group(code)
        best = None
        for k, (name, cf, bb) in enumerate(preds):
            if name in VEH and area(bb) > 0:
                i = inter(P, bb)
                if i / area(bb) >= 0.7 and i / max(1e-9, area(P)) >= 0.10:
                    if best is None or cf > best[1]:
                        best = (name, cf, k)
        rider = any(name == "person" and area(bb) > 0 and inter(P, bb) / area(bb) >= 0.6 and inter(P, bb) / max(1e-9, area(P)) >= 0.15 for name, cf, bb in preds)
        sz = "h<48" if h < 48 else ("48-96" if h < 96 else ("96-192" if h < 192 else "192+"))
        key = "veh+rider" if best and rider else ("veh only" if best else ("rider only" if rider else "nothing"))
        tot[key] += 1
        by_size[sz][key] += 1
        by_type[g][key] += 1
        if best:
            used.add(best[2])
            per_code[code][best[0]] += 1
            by_type[g]["cls=" + best[0]] += 1
        else:
            per_code[code]["(none)"] += 1
    for k, (name, cf, bb) in enumerate(preds):
        if name in VEH:
            total_veh_pred += 1
            if k not in used:
                unmatched_veh += 1

n = sum(tot.values())
print("weights", d["weights"], "imgsz", d["imgsz"], "frames", len(items), "PM boxes", n, "NaN boxes skipped", nan)
print("PM boxes: " + ", ".join("%s %d (%.0f%%)" % (k, v, 100 * v / n) for k, v in tot.most_common()))
print("by PM box height (px):")
for sz in ("h<48", "48-96", "96-192", "192+"):
    c = by_size[sz]; m = sum(c.values())
    if m:
        print("   %-7s n=%4d  veh found %.0f%%  rider found %.0f%%  nothing %.0f%%" % (sz, m, 100 * (c["veh+rider"] + c["veh only"]) / m, 100 * (c["veh+rider"] + c["rider only"]) / m, 100 * c["nothing"] / m))
print("by dataset type (code group):")
for g in ("motorcycle", "bicycle", "scooter"):
    c = by_type[g]; m = c["veh+rider"] + c["veh only"] + c["rider only"] + c["nothing"]
    f = c["veh+rider"] + c["veh only"]
    print("   %-10s n=%4d  veh found %d (%.0f%%)  -> model class: %s" % (g, m, f, 100 * f / max(1, m), {k[4:]: v for k, v in c.items() if k.startswith("cls=")}))
print("per code -> model vehicle class at that box:")
for code in sorted(per_code, key=int):
    print("   %s (%s): %s" % (code, group(code), dict(per_code[code].most_common())))
print("model vehicle boxes (bicycle/motorcycle/scooter) not inside any PM box: %d of %d" % (unmatched_veh, total_veh_pred))
print("predictions per frame: mean %.1f  p50 %d  p90 %d  max %d" % (sum(per_frame) / len(per_frame), sorted(per_frame)[len(per_frame) // 2], sorted(per_frame)[int(len(per_frame) * .9)], max(per_frame)))
print("predicted boxes by class:", dict(n_pred.most_common()))
hs = sorted(pm_heights)
print("PM box height px p10/p50/p90:", [round(hs[int(len(hs) * p)]) for p in (.1, .5, .9)])
