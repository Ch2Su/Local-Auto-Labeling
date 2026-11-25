"""지금 승인분(260장)으로 출발 가중치·얼리기를 3-fold 로 견준다 (2026-11-25, 사용자 'k-fold 로도 비교해 주세요').

fixed 분할 하나(test 37장)로 잰 new260_old10·얼리기 결과가 흔들려서, 직접 모은 194장을 카메라·같은 장면 묶음째 3묶음으로 나눠
묶음마다 '나머지 둘 + AI Hub 66장' 으로 학습하고 빼 둔 묶음에서 잰다. AI Hub(`ah-*`)는 늘 학습 쪽이다(11-23 사용자 결정).

- 세트 둘을 프로젝트의 `trainer.build_dataset` 로 만든다(DB 사본 둘): **P** = 승인 전부 train(가릴 영역을 칠함 — 학습 쪽),
  **U** = 직접 모은 것 test·AI Hub train(칠하지 않고 `ignore/` 그림을 남김 — 평가 쪽, 11-24 의 test 방식 그대로).
- 묶음은 DB 사본의 `images.grp`(분할이 계산한 카메라째·이름 계열·dHash 묶음)를 그대로 쓰고, 클래스별 박스 수와 이미지 수가
  fold 마다 비슷해지게 탐욕적으로 배정한다(runs/compare_1118_pretrain/make_folds.py 와 같은 식).
- 학습 인자는 프로젝트 학습 버튼과 같다(`trainer.train_args` — 640·b10·100 epoch·close_mosaic 10·cache disk). 바꾸는 것은
  data·name·project·freeze 뿐. **평가는 last.pt** — 학습 중 val 이 빼 둔 묶음이라 best.pt 는 평가 세트로 고른 셈이 된다.
- 평가는 `wsl/eval_test.py`(conf 0.25 의 찾음·놓침·헛 박스, 가릴 영역 자리 빼기)로, 학습 안 한 출발 가중치·운영 가중치도 같이 잰다.

  python kfold.py prepare            # DB 사본·세트 둘·fold 목록
  python kfold.py train ARM.K ...    # ARM = F | FZ11 | FZ23, K = 0..2 (끝난 것은 건너뜀)
  python kfold.py eval               # fold 마다 last.pt 다섯 가중치 → eval.jsonl
  python kfold.py summary            # 표
작업 폴더(DB 사본·세트·학습 결과·가중치)는 저장소 밖 WORK — 그림·가중치는 git 에 넣지 않는다. 결과 표·fold 목록(이미지 id)만 여기.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(r"C:\local_auto_labeling")
HERE = Path(__file__).resolve().parent
WORK = Path(r"C:\Users\chisu\AppData\Local\Temp\claude\3b97b6605d596f29\2e54f18b-e378-4fcb-8689-43102bf14dc5\scratchpad\kfold")
SRC_DB = WORK.parent / "splitsim" / "labeling.db"          # new260_old10 이 쓴 DB 사본(분할·묶음 계산됨)
SRC_CFG = WORK.parent / "splitsim" / "config.yaml"
START = r"C:\datasets\lal_review_old\models\best.pt"       # 옛 데이터 10클래스(11-24), 출발 가중치
OPERATING = str(REPO / "models" / "best.pt")
K = 3
FREEZE = {"F": None, "FZ11": 11, "FZ23": 23}
NAMES = ["bicycle", "bus", "car", "motorcycle", "person", "scooter", "truck", "pickup", "responder", "cone"]
PY = str(REPO / ".venv" / "Scripts" / "python.exe")

sys.path.insert(0, str(REPO / "src"))


def stem_of(path: str) -> str:
    src = Path(path)
    return f"{src.stem}__{hashlib.sha1(str(src).encode('utf-8')).hexdigest()[:8]}"


def make_cfg(name: str) -> Path:
    t = SRC_CFG.read_text(encoding="utf-8")
    # data: 절의 것만 (wsl: 절에도 dataset: 이 있다)
    t, a = re.subn(r"(?m)^  db: .*splitsim/labeling\.db$", "  db: " + (WORK / f"{name}.db").as_posix(), t)
    t, b = re.subn(r"(?m)^  dataset: .*splitsim/dataset$", "  dataset: " + (WORK / f"set_{name}").as_posix(), t)
    assert (a, b) == (1, 1), (a, b)
    p = WORK / f"config_{name}.yaml"
    p.write_text(t, encoding="utf-8")
    return p


def prepare() -> int:
    from autolabel import trainer
    from autolabel.config import Config
    from autolabel.store import Store

    WORK.mkdir(parents=True, exist_ok=True)
    for name, own_split in (("P", "train"), ("U", "test")):
        db = WORK / f"{name}.db"
        if db.exists():
            db.unlink()
        s = sqlite3.connect(f"file:{SRC_DB.as_posix()}?mode=ro", uri=True)
        d = sqlite3.connect(db)
        s.backup(d)
        s.close()
        d.execute("UPDATE images SET split = CASE WHEN camera LIKE 'ah-%' THEN 'train' ELSE ? END WHERE status='approved'",
                  (own_split,))
        d.commit()
        d.close()
        cfg = Config.load(make_cfg(name))
        rep = trainer.build_dataset(cfg, Store(cfg.db))
        print(f"[{name}] train {rep.train} · val {rep.val} · test {rep.test} · 칠함 {rep.masked} · ignore {rep.masked_test} · "
              f"뺀 박스 {rep.mask_dropped}", flush=True)

    con = sqlite3.connect(f"file:{SRC_DB.as_posix()}?mode=ro", uri=True)
    rows = con.execute("SELECT id, path, camera, grp FROM images WHERE status='approved' ORDER BY id").fetchall()
    own = [(i, p, c, g) for i, p, c, g in rows if not (c or "").startswith("ah-")]
    ah = [(i, p, c, g) for i, p, c, g in rows if (c or "").startswith("ah-")]
    U, P = WORK / "set_U", WORK / "set_P"
    items = []
    for iid, path, cam, grp in own:
        st = stem_of(path)
        test_img = [q for q in (U / "images" / "test").glob(f"{st}.*")]
        assert len(test_img) == 1, (iid, test_img)
        cls = Counter()
        lab = U / "labels" / "test" / f"{st}.txt"
        for line in lab.read_text().split("\n") if lab.exists() else []:
            if line.strip():
                cls[int(line.split()[0])] += 1
        items.append({"id": iid, "grp": grp or f"img{iid}", "camera": cam, "eval": test_img[0], "cls": cls,
                      "train": P / "images" / "train" / f"{st}.jpg" if (P / "images" / "train" / f"{st}.jpg").exists()
                      else next((P / "images" / "train").glob(f"{st}.*"))})
    ah_train = []
    for iid, path, cam, grp in ah:
        st = stem_of(path)
        hit = list((P / "images" / "train").glob(f"{st}.*"))
        assert len(hit) == 1, (iid, hit)
        ah_train.append(hit[0])

    groups: dict[str, list[dict]] = {}
    for it in items:
        groups.setdefault(it["grp"], []).append(it)

    def vec(members):
        c = Counter()
        for it in members:
            c.update(it["cls"])
        return np.array([c[i] for i in range(len(NAMES))] + [len(members)], dtype=float)

    gv = {g: vec(m) for g, m in groups.items()}
    total = sum(gv.values())
    total[total == 0] = 1
    order = sorted(groups, key=lambda g: (-float((gv[g] / total).max()), g))
    load = np.zeros((K, len(NAMES) + 1))
    fold_of = {}
    for g in order:
        best, best_cost = 0, None
        for k in range(K):
            trial = load.copy()
            trial[k] += gv[g]
            cost = float(((trial / total).std(axis=0)).sum())
            if best_cost is None or cost < best_cost - 1e-12:
                best, best_cost = k, cost
        fold_of[g] = best
        load[best] += gv[g]

    import yaml
    folds = {}
    for k in range(K):
        ev = [it for it in items if fold_of[it["grp"]] == k]
        tr = [it for it in items if fold_of[it["grp"]] != k]
        assert not {it["id"] for it in ev} & {it["id"] for it in tr}
        assert not {it["camera"] for it in ev} & {it["camera"] for it in tr}, "카메라가 두 쪽에 걸침"
        (WORK / f"f{k}_train.txt").write_text("\n".join(str(it["train"]) for it in tr) + "\n"
                                              + "\n".join(str(p) for p in ah_train) + "\n", encoding="utf-8")
        (WORK / f"f{k}_val.txt").write_text("\n".join(str(it["eval"]) for it in ev) + "\n", encoding="utf-8")
        spec = {"path": str(WORK), "train": str(WORK / f"f{k}_train.txt"), "val": str(WORK / f"f{k}_val.txt"),
                "names": {i: n for i, n in enumerate(NAMES)}}
        (WORK / f"f{k}.yaml").write_text(yaml.safe_dump(spec, allow_unicode=True, sort_keys=False), encoding="utf-8")
        folds[k] = sorted(it["id"] for it in ev)
        print(f"fold {k}: 평가 {len(ev)}장(카메라 {len({it['camera'] for it in ev})}) / 학습 {len(tr)} + AI Hub {len(ah_train)}장 · "
              f"평가 박스 " + ", ".join(f"{NAMES[i]} {int(load[k][i])}" for i in range(len(NAMES))), flush=True)
    (HERE / "folds.json").write_text(json.dumps({"note": "fold 마다 평가 쪽 이미지 id (직접 모은 승인분). 나머지 + AI Hub 가 학습 쪽",
                                                 "eval_ids": folds}, ensure_ascii=False, indent=0), encoding="utf-8", newline="\n")
    return 0


def train_one(arm: str, k: int) -> int:
    from ultralytics import YOLO
    from autolabel import trainer
    from autolabel.config import Config

    name = f"{arm}_f{k}"
    out = WORK / "runs" / name
    if (out / "weights" / "last.pt").exists() and (out / "results.csv").exists() \
            and len((out / "results.csv").read_text().strip().split("\n")) > 100:
        print(f"[건너뜀] {name}", flush=True)
        return 0
    cfg = Config.load(SRC_CFG)
    args = trainer.train_args(cfg, WORK / f"f{k}.yaml")
    args.update(project=str(WORK / "runs"), name=name, exist_ok=True)
    args.pop("freeze", None)
    if FREEZE[arm]:
        args["freeze"] = FREEZE[arm]
    print(f"=== START {name} {json.dumps({x: args.get(x) for x in ('epochs', 'batch', 'imgsz', 'freeze', 'seed', 'cache')})}", flush=True)
    YOLO(START).train(**args)
    shutil.copy2(out / "results.csv", HERE / f"{name}_results.csv")
    print(f"=== DONE {name}", flush=True)
    return 0


def evaluate() -> int:
    out = HERE / "eval.jsonl"
    for k in range(K):
        ws = [str(WORK / "runs" / f"{a}_f{k}" / "weights" / "last.pt") for a in FREEZE] + [START, OPERATING]
        rc = subprocess.run([PY, "wsl/eval_test.py", "--weights", *ws, "--data", f"f{k}={WORK / f'f{k}.yaml'}", "--imgsz", "640",
                             "--split", "val", "--out", str(out), "--project", str(WORK / "evalproj"), "--device", "0",
                             "--note", f"kfold_new260 f{k}"], cwd=REPO).returncode
        if rc:
            return rc
    return 0


def label(w: str) -> str:
    p = Path(w)
    if p.name == "last.pt":
        return p.parent.parent.name.split("_f")[0]
    return "start" if "lal_review_old" in w else "operating"


def summary() -> int:
    old8 = ["car", "truck", "bus", "pickup", "person", "bicycle", "motorcycle", "scooter"]
    rows = [json.loads(x) for x in (HERE / "eval.jsonl").read_text(encoding="utf-8").split("\n") if x.strip()]
    by = {}
    for r in rows:
        by.setdefault(label(r["weights"]), {})[r["set"]] = r
    print(f"{'weights':<10} {'8cls mAP50-95 per fold':<26} {'mean':>6} | {'hit/missed/false (8cls, pooled)':<30} | responder hit/n · cone hit/n")
    for lab in ("F", "FZ11", "FZ23", "start", "operating"):
        if lab not in by:
            continue
        ms, H, M, Fa, rh, rn, ch, cn = [], 0, 0, 0, 0, 0, 0, 0
        for k in range(K):
            r = by[lab].get(f"f{k}")
            if r is None:
                continue
            c, pc = r["classes"], r["at_conf"]["per_class"]
            present = [x for x in old8 if x in c and c[x]["n"]]
            ms.append(sum(c[x]["map"] for x in present) / len(present))
            H += sum(pc[x]["hit"] for x in old8 if x in pc)
            M += sum(pc[x]["missed"] for x in old8 if x in pc)
            Fa += sum(pc[x]["false"] for x in old8 if x in pc)
            if "responder" in pc:
                rh += pc["responder"]["hit"]; rn += c["responder"]["n"]
                ch += pc["cone"]["hit"]; cn += c["cone"]["n"]
        print(f"{lab:<10} {' / '.join(f'{m:.3f}' for m in ms):<26} {np.mean(ms):>6.3f} | {f'{H} / {M} / {Fa}':<30} | "
              + (f"{rh}/{rn} · {ch}/{cn}" if rn else "—"))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prepare":
        sys.exit(prepare())
    if cmd == "train":
        for job in sys.argv[2:]:
            a, k = job.split(".")
            # 실행마다 따로 띄운다(메모리·Windows spawn 워커를 실행마다 정리)
            rc = subprocess.run([PY, __file__, "_train", a, k], cwd=REPO).returncode
            if rc:
                sys.exit(rc)
        sys.exit(0)
    if cmd == "_train":
        sys.exit(train_one(sys.argv[2], int(sys.argv[3])))
    if cmd == "eval":
        sys.exit(evaluate())
    if cmd == "summary":
        sys.exit(summary())
    raise SystemExit(f"모르는 명령: {cmd}")
