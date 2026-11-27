"""yolo26n 의 해상도별 최대 batch (2026-11-27, docs/experiments.md '16GB 에서 1280' 의 n 표) — wsl/probe_vram.py 의 한 설정 재기(run_child)를 그대로 쓰고, batch 만 찾아 나간다.

1280 에서 batch 둘로 장당 VRAM 과 바닥(컨텍스트·모델)을 재고, 해상도마다 면적에 비례해 어림한 batch 부터 시작해
들어가면 하나씩 올리고 넘치면 하나씩 내린다. 판정은 probe 그대로(GPU 전체 94% 초과 = 넘침, OOM).
"""
import json, math, shutil, sys, types
from pathlib import Path
sys.path.insert(0, "/mnt/c/local_auto_labeling/wsl")
import probe_vram as pv

OUT = Path("/mnt/c/local_auto_labeling/runs/vram_probe/results_16gb.jsonl")
args = types.SimpleNamespace(steps=30, data=str(Path.home() / "datasets/probe_copy/data.yaml"), workers=6)
pv.WORKDIR.mkdir(parents=True, exist_ok=True)
(pv.WORKDIR / "weights").mkdir(exist_ok=True)
if not (pv.WORKDIR / "weights/yolo26n.pt").exists():
    shutil.copy2(pv.PROJECT / "weights/yolo26n.pt", pv.WORKDIR / "weights/yolo26n.pt")
seen = {}
for line in OUT.read_text(encoding="utf-8").splitlines():      # 앞서 잰 n 기록은 다시 재지 않는다
    r = json.loads(line)
    if r["model"] == "n" and r.get("total") == 15.9:
        seen[(r["imgsz"], r["batch"])] = r


def probe(sz, b):
    if (sz, b) in seen:
        return seen[(sz, b)]
    r = pv.run_child("n", sz, b, args)
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(pv.fmt(r) + f"  [{r['wall_s']}s]", flush=True)
    seen[(sz, b)] = r
    return r


ok = lambda r: r["status"] == "ok"
a, b = probe(1280, 8), probe(1280, 16)        # 첫 시도의 b24 는 넘쳤다(15.9/15.9) — 둘 다 들어가는 둘로 기울기를 잰다
assert ok(a) and ok(b), "1280 b8·b16 가 안 들어간다 — 어림을 다시"
k1280 = (b["device_used"] - a["device_used"]) / 8
base = a["device_used"] - 8 * k1280
limit = pv.LIMIT_FRAC * a["total"]
print(f"1280: 장당 {k1280:.3f} GiB, 바닥 {base:.2f} GiB, 한도 {limit:.2f} GiB", flush=True)

best = {}
for sz in (640, 960, 1280, 1600, 1920, 2240, 2560):
    k = k1280 * (sz / 1280) ** 2
    g = max(1, math.floor((limit - base) / k))
    if ok(probe(sz, g)):
        while ok(probe(sz, g + 1)):
            g += 1
        best[sz] = g
    else:
        while g > 1 and not ok(probe(sz, g - 1)):
            g -= 1
        best[sz] = g - 1 if g > 1 else (1 if ok(probe(sz, 1)) else 0)
    print(f"  -> n {sz}: 최대 batch {best[sz]}", flush=True)
    if best[sz] == 0:
        break
print("요약 " + json.dumps(best), flush=True)
shutil.rmtree(pv.WORKDIR / "runs", ignore_errors=True)
