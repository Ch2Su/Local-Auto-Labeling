"""(1) 같은 물체에 PM 박스가 둘 붙은 곳(IoU>=0.8) (2) AI Hub 의 Training/Validation 이 영상을 공유하는지 (3) 검증 라벨의 이상값."""
import sys, json, zipfile, collections, math

index, vl = sys.argv[1], sys.argv[2]
def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    i = max(0.0, min(ax2, bx2) - max(a[0], b[0])) * max(0.0, min(ay2, by2) - max(a[1], b[1]))
    u = a[2] * a[3] + b[2] * b[3] - i
    return i / u if u > 0 else 0.0
def grp(c):
    c = int(c)
    return "M" if c <= 19 else ("B" if c <= 27 else "S")
frames = dup_frames = boxes = dup_boxes = 0
dup_kind = collections.Counter()
train_vids = set()
for line in open(index, encoding="utf-8"):
    rel, stem, W, H, pm, nenv = json.loads(line)
    train_vids.add(stem.split("_")[0])
    frames += 1
    boxes += len(pm)
    hit = set()
    for i in range(len(pm)):
        for j in range(i + 1, len(pm)):
            if iou(pm[i][1:], pm[j][1:]) >= 0.8:
                hit.add(j)
                dup_kind["same type" if grp(pm[i][0]) == grp(pm[j][0]) else "different type"] += 1
    if hit:
        dup_frames += 1
        dup_boxes += len(hit)
print("train frames %d, PM boxes %d; boxes that repeat another box (IoU>=0.8): %d (%.1f%%) in %d frames (%.1f%%); pairs: %s" % (
    frames, boxes, dup_boxes, 100 * dup_boxes / boxes, dup_frames, 100 * dup_frames / frames, dict(dup_kind)))
z = zipfile.ZipFile(vl)
val_vids = collections.Counter()
nan = bad = n = 0
for i in z.infolist():
    if i.is_dir():
        continue
    val_vids[i.filename.rsplit("/", 1)[-1].split("_")[0]] += 1
    d = json.loads(z.read(i))
    for p in d["annotations"].get("PM") or []:
        n += 1
        pts = p.get("points") or []
        if len(pts) != 4 or any(not isinstance(v, (int, float)) or (isinstance(v, float) and math.isnan(v)) for v in pts):
            nan += 1
        elif pts[2] <= 0 or pts[3] <= 0:
            bad += 1
shared = [v for v in val_vids if v in train_vids]
print("validation: videos %d, of which also in training %d (%.1f%%) — frames in those %d of %d" % (
    len(val_vids), len(shared), 100 * len(shared) / len(val_vids), sum(val_vids[v] for v in shared), sum(val_vids.values())))
print("validation PM boxes %d, nan/malformed %d, non-positive size %d" % (n, nan, bad))
