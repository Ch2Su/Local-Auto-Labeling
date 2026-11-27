"""AI Hub 그림을 '같은 자리' 묶음으로 모으고(영상 id 가 같거나 same_camera), 그 묶음째 분할하면 어떻게 갈리는지 — DB 사본으로만.

실제 DB 는 mode=ro 로 사본을 뜨고, 등록 파일도 사본(ah-* 의 split: train 을 비운 것)을 쓴다. 그림은 코드가 읽기만 한다.
"""
import collections, json, re, sqlite3, sys, time
from pathlib import Path
REPO = Path(r"C:\local_auto_labeling"); W = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "tools"))
import import_aihub_pm as ah
from autolabel.config import Config
from autolabel.store import Store
from autolabel.split import ensure_splits
from autolabel import masks as M

def copy_db(dst):
    a = sqlite3.connect(f"file:{(REPO / 'data/visible/labeling.db').as_posix()}?mode=ro", uri=True); b = sqlite3.connect(dst)
    a.backup(b); b.close(); a.close()

def make_cfg(tag, db, cams):
    t = (REPO / "config.yaml").read_text(encoding="utf-8")
    t, n1 = re.subn(r"(?m)^  db: .*$", "  db: " + db.as_posix(), t, count=1)
    t, n2 = re.subn(r"(?m)^  cameras: .*$", "  cameras: " + cams.as_posix(), t, count=1)
    t, n3 = re.subn(r"(?m)^  dataset: data/visible/dataset$", "  dataset: " + (W / f"ds_{tag}").as_posix(), t, count=1)
    t, n4 = re.subn(r"(?m)^  keep: \d+", "  keep: 0", t, count=1)
    assert (n1, n2, n3, n4) == (1, 1, 1, 1), (n1, n2, n3, n4)
    p = W / f"config_{tag}.yaml"; p.write_text(t, encoding="utf-8")
    cfg = Config.load(p); assert Path(cfg.db) == db and Path(cfg.cameras) == cams
    return cfg

base_db = W / "ah_base.db"; copy_db(base_db)
store = Store(base_db)
t0 = time.time()
rows = ah.known(store)
print(f"AI Hub 그림 {len(rows)}장 (지문 {sum(k.scene is not None for k in rows)}장) {time.time()-t0:.0f}s", flush=True)

parent = list(range(len(rows)))
def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]; i = parent[i]
    return i
def union(i, j):
    parent[find(i)] = find(j)
why = collections.Counter()
by_video = collections.defaultdict(list)
for i, k in enumerate(rows):
    if k.video: by_video[k.video].append(i)
for ids in by_video.values():
    for j in ids[1:]:
        if find(j) != find(ids[0]): why["같은 영상"] += 1
        union(j, ids[0])
t0 = time.time()
for i in range(len(rows)):
    for j in range(i + 1, len(rows)):
        a, b = rows[i].scene, rows[j].scene
        if a is None or b is None or find(i) == find(j): continue
        if ah.same_camera(a, b):
            union(i, j); why["같은 자리"] += 1
print(f"짝 견주기 {time.time()-t0:.0f}s, 이은 것 {dict(why)}", flush=True)
groups = collections.defaultdict(list)
for i in range(len(rows)): groups[find(i)].append(i)
sizes = sorted((len(g) for g in groups.values()), reverse=True)
appr = [g for g in groups.values() if any(rows[i].status == "approved" for i in g)]
print(f"묶음 {len(groups)}개 (승인 그림이 든 묶음 {len(appr)}개) · 큰 순 {sizes[:12]} · 한 장짜리 {sizes.count(1)}", flush=True)

# 라벨에 남는 박스 — 승인분, 묶음별 scooter
cls = store.get_classes(); rects = store.masks_by_image()
eff = {}
for im, boxes in store.approved_with_boxes():
    keep, _ = M.split_boxes(boxes, rects.get(im["id"], []))
    eff[im["id"]] = collections.Counter(b.cls for b in keep)
sc = cls.index("scooter")
sc_groups = sorted((sum(eff.get(rows[i].id, {}).get(sc, 0) for i in g) for g in groups.values()), reverse=True)
print(f"scooter 라벨 박스가 있는 AI Hub 묶음 {sum(1 for s in sc_groups if s)}개 · 묶음별 {[s for s in sc_groups if s][:20]}", flush=True)

def simulate(tag, regroup):
    db = W / f"ah_{tag}.db"; db.write_bytes(base_db.read_bytes())
    cams_src = (REPO / "data/visible/cameras.yaml").read_text(encoding="utf-8")
    out, key = [], None
    for line in cams_src.splitlines(keepends=True):
        m = re.match(r"^  ([A-Za-z0-9][A-Za-z0-9-]*):\s*$", line)
        if m: key = m.group(1)
        if regroup and key and key.startswith("ah-") and line.strip() == "split: train":
            line = line.replace("split: train", "split:")
        out.append(line)
    cams = W / f"cameras_{tag}.yaml"; cams.write_text("".join(out), encoding="utf-8")
    con = sqlite3.connect(db)
    con.execute("UPDATE images SET split=NULL, grp=NULL")
    if regroup:   # 묶음의 그림을 한 카메라 키로 — 분할이 카메라째라 묶음이 한 분할로 간다(사본에서만, 파일은 그대로)
        for g in groups.values():
            k0 = min(rows[i].camera for i in g)
            con.executemany("UPDATE images SET camera=? WHERE id=?", [(k0, rows[i].id) for i in g])
    con.commit(); con.close()
    cfg = make_cfg(tag, db, cams); st = Store(db)
    assign, rep = ensure_splits(cfg, st)
    src = {r["id"]: (r["camera"] or "") for r in sqlite3.connect(db).execute("SELECT id, camera FROM images").fetchall() for r in [dict(zip(("id","camera"), r))]}
    per = {s: collections.Counter() for s in ("train", "val", "test")}
    n = {s: collections.Counter() for s in per}
    for i, s in assign.items():
        per[s].update(eff.get(i, {}))
        n[s]["AI Hub" if src[i].startswith("ah-") else "직접"] += 1
    res = {"images": {s: dict(n[s]) for s in per},
           "boxes": {s: {cls[c]: per[s][c] for c in range(len(cls))} for s in per}, "worst": round(rep.worst(), 4)}
    print(tag, json.dumps(res, ensure_ascii=False), flush=True)
    return res

out = {"groups": len(groups), "groups_with_approved": len(appr), "sizes": sizes, "links": dict(why),
       "scooter_groups": [s for s in sc_groups if s],
       "pinned": simulate("pinned", False), "grouped": simulate("grouped", True)}
(W / "ah_groups.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
