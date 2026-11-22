"""학습 라벨 58만 개를 한 번 읽어 PM 박스만 한 파일로 모은다 (JSONL: 폴더, 이름, [[code,x,y,w,h],...]). 읽기 전용."""
import os, sys, json, time
from concurrent.futures import ThreadPoolExecutor

root, out = sys.argv[1], sys.argv[2]
files = []
for dp, dn, fn in os.walk(root):
    rel = os.path.relpath(dp, root).replace("\\", "/")
    for f in fn:
        if f.endswith(".json"):
            files.append((rel, f))
files.sort()
print("files", len(files), flush=True)

def load(item):
    rel, f = item
    try:
        with open(os.path.join(root, rel, f), "rb") as fh:
            d = json.loads(fh.read())
        pm = [[p.get("PM_code")] + list(p.get("points") or []) for p in d["annotations"].get("PM") or []]
        ds = d.get("description", {})
        return rel, f[:-5], ds.get("imageWidth"), ds.get("imageHeight"), pm, len(d["annotations"].get("environment") or [])
    except Exception as e:
        return rel, f[:-5], None, None, "ERR " + type(e).__name__, 0

t = time.time()
with ThreadPoolExecutor(16) as ex, open(out, "w", encoding="utf-8") as fo:
    for k, r in enumerate(ex.map(load, files, chunksize=256)):
        fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        if k % 50000 == 0:
            print(k, round(time.time() - t), "s", flush=True)
print("done", round(time.time() - t), "s", flush=True)
