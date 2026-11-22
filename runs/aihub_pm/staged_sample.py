"""연출 CCTV(주간·맑음) 중 킥보드가 든 프레임을 영상마다 한 장씩 zip 에서 꺼내 운영 가중치로 재 본다 (스크래치에만 쓴다)."""
import os, sys, json, random, zipfile, collections

def main():
    index, zips, out_dir, weights, out_json, n = sys.argv[1], sys.argv[2].split("|"), sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6])
    folder = "CCTV/주간/맑음/연출"
    random.seed(21)
    by_vid = collections.defaultdict(list)
    for line in open(index, encoding="utf-8"):
        rel, stem, W, H, pm, nenv = json.loads(line)
        if rel == folder and any(int(p[0]) >= 28 for p in pm):
            by_vid[stem.split("_")[0]].append((stem, pm))
    vids = sorted(by_vid)
    random.shuffle(vids)
    picks = {}
    for v in vids[:n]:
        stem, pm = random.choice(by_vid[v])
        picks[stem] = pm
    os.makedirs(out_dir, exist_ok=True)
    found = {}
    for zp in zips:
        z = zipfile.ZipFile(zp)
        for i in z.infolist():
            if i.is_dir() or not i.filename.startswith(folder + "/"):
                continue
            stem = os.path.basename(i.filename)[:-4]
            if stem in picks and stem not in found:
                p = os.path.join(out_dir, stem + ".jpg")
                if not os.path.exists(p):
                    with open(p, "wb") as fo:
                        fo.write(z.read(i))
                found[stem] = p
    print("videos with scooter in folder", len(vids), "picked", len(picks), "extracted", len(found), flush=True)
    from ultralytics import YOLO
    model = YOLO(weights)
    names = model.names
    stems = sorted(found)
    out = []
    for s in range(0, len(stems), 16):
        chunk = stems[s:s + 16]
        res = model.predict([found[c] for c in chunk], imgsz=640, conf=0.25, iou=0.6, max_det=100, verbose=False)
        for stem, r in zip(chunk, res):
            preds = [(names[int(c)], float(cf), [float(v) for v in xyxy]) for c, cf, xyxy in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())]
            out.append({"stem": stem, "shape": list(r.orig_shape), "pm": [(p[0], p[1:]) for p in picks[stem]], "pred": preds})
    json.dump({"imgsz": 640, "weights": weights, "names": names, "items": out}, open(out_json, "w", encoding="utf-8"))
    print("predicted", len(out))

if __name__ == "__main__":
    main()
