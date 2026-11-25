"""분할이 세는 박스 — DB 의 박스 대신 라벨에 남는 박스로 (2026-11-25, docs/experiments.md 의 같은 이름 절).

실제 DB 의 읽기 전용 사본으로: (1) 승인 박스의 DB 수·라벨에 남는 수, (2) class_rarity·approved_class_counts 걸리는 시간,
(3) 첫 분할을 사본에서 돌렸을 때 분할마다 라벨에 남는 박스 수. 실제 DB 는 mode=ro 로 읽어 사본만 쓴다.

    python splitcmp.py <tag>      # tag 는 결과 파일 이름 — 저장소의 지금 코드로 돈다

그날은 `git stash push src/autolabel/store.py` 로 고치기 전 코드에서 `old`, 고친 코드에서 `new` 를 돌렸다(같은 사본 snap.db).
"""
import json
import re
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent                # 결과(<tag>.json)만 여기에 남긴다
# DB 사본·설정 사본은 저장소 밖(그날의 Claude 임시 폴더)에 둔다 — 다시 돌리려면 이 경로만 바꾼다
WORK = Path(r"C:\Users\chisu\AppData\Local\Temp\claude\3b97b6605d596f29\2e54f18b-e378-4fcb-8689-43102bf14dc5\scratchpad\splitcmp")
sys.path.insert(0, str(REPO / "src"))

from autolabel import masks  # noqa: E402
from autolabel.config import Config  # noqa: E402
from autolabel.split import ensure_splits  # noqa: E402
from autolabel.store import Store  # noqa: E402

tag = sys.argv[1]
WORK.mkdir(parents=True, exist_ok=True)
src = REPO / "data" / "visible" / "labeling.db"
snap = WORK / "snap.db"
if not snap.exists():                       # 실제 DB 를 읽기 전용으로 한 번만 뜬다 — old·new 가 같은 시점을 본다
    a = sqlite3.connect(f"file:{src.as_posix()}?mode=ro", uri=True)
    b = sqlite3.connect(snap)
    a.backup(b)
    b.close()
    a.close()
work = WORK / f"{tag}.db"
work.write_bytes(snap.read_bytes())
con = sqlite3.connect(work)
con.execute("UPDATE images SET split=NULL")          # 사본에서만 — 첫 분할을 다시 돌려 본다(실제 DB 도 미배정이다)
con.commit()
con.close()

t = (REPO / "config.yaml").read_text(encoding="utf-8")
t, a = re.subn(r"(?m)^  db: .*$", "  db: " + work.as_posix(), t, count=1)
t, b = re.subn(r"(?m)^  dataset: data/visible/dataset$", "  dataset: " + (WORK / f"set_{tag}").as_posix(), t, count=1)
t, c = re.subn(r"(?m)^  keep: \d+", "  keep: 0", t, count=1)
assert (a, b, c) == (1, 1, 1), (a, b, c)
cfgp = WORK / f"config_{tag}.yaml"
cfgp.write_text(t, encoding="utf-8")
cfg = Config.load(cfgp)
assert Path(cfg.db) == work, cfg.db
store = Store(cfg.db)
classes = store.get_classes()

# 라벨에 남는 박스는 이 스크립트가 따로 센다 — store 의 새 계산과 독립된 기준(학습 세트와 같은 masks.split_boxes)
rects = store.masks_by_image()
eff: dict[int, Counter] = {}
raw: dict[int, Counter] = {}
meta = {}
for im, boxes in store.approved_with_boxes():
    keep, _ = masks.split_boxes(boxes, rects.get(im["id"], []))
    eff[im["id"]] = Counter(b.cls for b in keep)
    raw[im["id"]] = Counter(b.cls for b in boxes)
    meta[im["id"]] = im
tot_raw, tot_eff = Counter(), Counter()
for i in raw:
    tot_raw.update(raw[i])
    tot_eff.update(eff[i])
out = {"tag": tag, "approved": len(raw), "raw": sum(tot_raw.values()), "eff": sum(tot_eff.values()),
       "raw_by_class": {classes[k]: v for k, v in sorted(tot_raw.items())},
       "eff_by_class": {classes[k]: v for k, v in sorted(tot_eff.items())}}

t0 = time.perf_counter()
items = store.approved_class_counts()
out["counts_ms"] = round((time.perf_counter() - t0) * 1000, 1)
t0 = time.perf_counter()
rar = store.class_rarity()
out["rarity_ms"] = round((time.perf_counter() - t0) * 1000, 1)
out["rarity_groups"] = {classes[k]: v["groups"] for k, v in rar.items() if k < len(classes)}
got = Counter()
for it in items:
    got.update(it["counts"])
out["store_counts_total"] = sum(got.values())

assign, report = ensure_splits(cfg, store)
per = {s: Counter() for s in ("train", "val", "test")}
n_img = Counter()
own = Counter()
for i, s in assign.items():
    per[s].update(eff[i])
    n_img[s] += 1
    if not str(meta[i].get("camera") or "").startswith("ah-"):
        own[s] += 1
out["images"] = dict(n_img)
out["own_images"] = dict(own)
out["eff_by_split"] = {s: {classes[k]: v for k, v in sorted(per[s].items())} for s in per}
out["missing"] = {s: [classes[k] for k in sorted(tot_eff) if tot_eff[k] and not per[s][k]] for s in ("val", "test")}
out["report_shares_worst"] = round(report.worst(), 4)
(HERE / f"{tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False))
