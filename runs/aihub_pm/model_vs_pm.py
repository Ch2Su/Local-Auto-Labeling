"""운영 가중치를 AI Hub PM 표본 프레임에 돌려 PM 박스와 견준다 (읽기 전용 — 그림·DB 를 바꾸지 않는다)."""
import os, sys, json, random, collections

def main():
    lab_dir, img_dir, weights, out_json, n_videos, imgsz = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
    random.seed(11)
    by_vid = collections.defaultdict(list)
    for f in os.listdir(img_dir):
        if f.endswith(".jpg"):
            by_vid[f.split("_")[0]].append(f[:-4])
    vids = sorted(by_vid)
    random.shuffle(vids)
    picks = [random.choice(by_vid[v]) for v in vids[:n_videos]]
    from ultralytics import YOLO
    model = YOLO(weights)
    names = model.names
    out = []
    B = 16
    for s in range(0, len(picks), B):
        chunk = picks[s:s + B]
        res = model.predict([os.path.join(img_dir, c + ".jpg") for c in chunk], imgsz=imgsz, conf=0.25, iou=0.6, max_det=100, verbose=False)
        for stem, r in zip(chunk, res):
            with open(os.path.join(lab_dir, stem + ".json"), encoding="utf-8") as fh:
                d = json.load(fh)
            pm = [(p["PM_code"], p["points"]) for p in d["annotations"].get("PM") or []]
            preds = [(names[int(c)], float(cf), [float(v) for v in xyxy]) for c, cf, xyxy in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())]
            out.append({"stem": stem, "shape": list(r.orig_shape), "pm": pm, "pred": preds})
        print("done", min(s + B, len(picks)), "/", len(picks), flush=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump({"imgsz": imgsz, "weights": weights, "names": names, "items": out}, fh)

if __name__ == "__main__":
    main()
