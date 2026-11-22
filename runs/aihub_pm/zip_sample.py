"""zip 에서 고르게 흩어진 항목을 끝까지 읽어 CRC 를 확인하고(zipfile 이 읽을 때 검사한다), 폴더별 예시 몇 장을 PM 박스와 함께 저장."""
import sys, zipfile, random, os, json, io, collections
from PIL import Image, ImageDraw

zpath, lab_root, out, n_crc = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
os.makedirs(out, exist_ok=True)
z = zipfile.ZipFile(zpath)
infos = [i for i in z.infolist() if not i.is_dir()]
infos.sort(key=lambda i: i.header_offset)
random.seed(3)
step = max(1, len(infos) // n_crc)
bad = 0
sizes = collections.Counter()
for k in range(0, len(infos), step):
    i = infos[k]
    try:
        data = z.read(i)            # CRC 불일치면 BadZipFile
        im = Image.open(io.BytesIO(data)); im.load()
        sizes[im.size] += 1
    except Exception as e:
        bad += 1
        print("  BAD", i.filename, type(e).__name__, e)
print(os.path.basename(zpath), "entries", len(infos), "checked", len(range(0, len(infos), step)), "bad", bad, "sizes", dict(sizes),
      "offset span GB %.1f..%.1f" % (infos[0].header_offset / 1e9, infos[-1].header_offset / 1e9))
# 폴더별 예시
by_dir = collections.defaultdict(list)
for i in infos:
    by_dir[os.path.dirname(i.filename)].append(i)
for d, lst in sorted(by_dir.items()):
    by_vid = collections.defaultdict(list)
    for i in lst:
        by_vid[os.path.basename(i.filename).split("_")[0]].append(i)
    vids = sorted(by_vid)
    random.shuffle(vids)
    for v in vids[:2]:
        i = random.choice(by_vid[v])
        stem = os.path.basename(i.filename)[:-4]
        lp = os.path.join(lab_root, d, stem + ".json")
        im = Image.open(io.BytesIO(z.read(i))).convert("RGB")
        dr = ImageDraw.Draw(im)
        if os.path.exists(lp):
            lab = json.load(open(lp, encoding="utf-8"))
            for p in lab["annotations"].get("PM") or []:
                x, y, w, h = p["points"]
                dr.rectangle([x, y, x + w, y + h], outline=(255, 255, 0), width=4)
                dr.text((x + 3, y + h + 2), "PM %s" % p["PM_code"], fill=(255, 255, 0))
        else:
            dr.text((10, 10), "NO LABEL FILE", fill=(255, 0, 0))
        im = im.resize((960, 540), Image.LANCZOS)
        name = d.replace("/", "_") + "__" + stem + ".jpg"
        im.save(os.path.join(out, name), quality=82)
        print("  example", d, stem)
