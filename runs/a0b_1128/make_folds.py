"""A0b — 크롭 세트의 카메라째 5-fold 와 같은 나눔으로 검출 학습 세트 다섯 벌을 만든다 (2026-11-28, 사용자 'A0b 돌리기').

docs/vehicle-refiner.md 4.2 의 A0b: 분류기(train_refiner)와 **같은 데이터**로 배운 검출기를 fold 마다 두고, 그 fold 의 크롭에서
검출기의 클래스를 분류기의 OOF 와 견준다(구조의 몫만 보려고 — A0 운영 m 은 옛 데이터 3,725장으로 배웠다).

fold f 의 세트: train = 다른 fold 의 그룹 + 차량 크롭이 없는 그룹(평가하지 않는 이미지), val = fold f 의 그룹(학습 중 보기만 —
평가는 정해 둔 epoch 끝의 last.pt 로, best.pt 를 쓰지 않는다), test = 크롭 세트에서 test 로 뺀 그룹(학습에 안 쓴다). 그룹·fold 는
크롭 세트(`manifest.csv`)의 것을 그대로 쓰고, 그룹은 `build_vehicle_crops.groups_of` 로 다시 셈해 목록의 그룹과 같은지 본다.

실제 DB 는 읽기 전용으로 사본을 뜨고(`build_vehicle_crops.copy_db`), fold 마다 그 사본에 분할만 적은 뒤 프로젝트의 학습 세트
만들기(`trainer.build_dataset` — 가릴 영역 칠하기·라벨 빼기·test 의 ignore 그림까지 평소와 같다)로 쓴다. 등록 파일은 빈 것으로
준다(이 실험의 나눔은 카메라 고정을 따르지 않는다 — 고정은 진짜 분할의 것). 모두 저장소 밖 `C:\\datasets\\a0b_1128\\f<N>\\`.

  .venv\\Scripts\\python.exe runs\\a0b_1128\\make_folds.py
"""
import collections
import csv
import json
import re
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))
from autolabel.config import Config  # noqa: E402
from autolabel.store import Store  # noqa: E402
from autolabel.trainer import build_dataset  # noqa: E402
from build_vehicle_crops import copy_db, groups_of, load_pairs, ref_splits  # noqa: E402

CROPS = Path(r"C:\datasets\vehicle_crops_20261128")
OUT = Path(r"C:\datasets\a0b_1128")


def config_for(fold_dir: Path, db: Path) -> Config:
    t = (REPO / "config.yaml").read_text(encoding="utf-8")
    cams = fold_dir / "cameras.yaml"
    cams.write_text("cameras: {}\n", encoding="utf-8")
    subs = [(r"(?m)^  db: .*$", "  db: " + db.as_posix()),
            (r"(?m)^  dataset: data/visible/dataset$", "  dataset: " + (fold_dir / "dataset").as_posix()),
            (r"(?m)^  keep: \d+", "  keep: 0")]
    for pat, rep in subs:
        t, n = re.subn(pat, rep, t, count=1)
        assert n == 1, pat
    t, n = re.subn(r"(?m)^  cameras: .*$", "  cameras: " + cams.as_posix(), t, count=1)
    if n == 0:                                  # data.cameras 를 안 적은 설정 — data: 바로 아래에 넣는다
        t, n = re.subn(r"(?m)^data:\n", "data:\n  cameras: " + cams.as_posix() + "\n", t, count=1)
    assert n == 1
    p = fold_dir / "config.yaml"
    p.write_text(t, encoding="utf-8")
    c = Config.load(p)
    assert Path(c.db) == db and Path(c.dataset) == fold_dir / "dataset" and Path(c.cameras) == cams
    return c


def main() -> int:
    info = json.loads((CROPS / "crops_info.json").read_text(encoding="utf-8"))
    with open(CROPS / "manifest.csv", encoding="utf-8", newline="") as f:
        mani = list(csv.DictReader(f))
    fold_of_group = {}
    for r in mani:
        g, fd = int(r["group"]), r["fold"]
        assert fold_of_group.setdefault(g, fd) == fd, f"그룹 {g} 의 fold 가 둘"
    OUT.mkdir(parents=True, exist_ok=True)
    base = OUT / "base.db"
    copy_db(Path(info["source_db"]), base)
    con = sqlite3.connect(base)
    con.row_factory = sqlite3.Row
    items = [dict(r) for r in con.execute("SELECT id, path, camera, grp FROM images WHERE status = 'approved' ORDER BY id")]
    con.close()
    group = groups_of(items, load_pairs(Path(info["same_place"])))
    for r in mani:                                              # 다시 센 그룹이 크롭 세트의 것과 같은가
        assert group[int(r["image_id"])] == int(r["group"]), f"이미지 {r['image_id']} 의 그룹이 다르다"
    ref = ref_splits(Path(info["ref_dataset"]))
    test_groups = {group[i["id"]] for i in items if ref.get(Path(i["path"]).stem) == info["ref_exclude"]}
    assert test_groups == {g for g, fd in fold_of_group.items() if fd == "test"} | (test_groups - set(fold_of_group))
    folds = sorted({fd for fd in fold_of_group.values() if fd != "test"}, key=int)
    in_mani = {int(r["image_id"]) for r in mani}
    summary = {"crops_made_at": info["made_at"], "approved": len(items),
               "approved_not_in_manifest": sum(1 for i in items if i["id"] not in in_mani), "folds": {}}
    for f in folds:
        fdir = OUT / f"f{f}"
        fdir.mkdir(exist_ok=True)
        db = fdir / "labeling.db"
        db.write_bytes(base.read_bytes())
        split = {}
        for i in items:
            g = group[i["id"]]
            if g in test_groups:
                split[i["id"]] = "test"
            elif fold_of_group.get(g) == f:
                split[i["id"]] = "val"
            else:
                split[i["id"]] = "train"
        store = Store(db)
        store.set_splits(list(split.items()))
        rep = build_dataset(config_for(fdir, db), store)
        n = collections.Counter(split.values())
        summary["folds"][f] = {"images": dict(n), "boxes": {s: dict(c) for s, c in rep.per_split.items()}
                               if hasattr(rep, "per_split") else None}
        print(f"fold {f}: 이미지 {dict(n)}")
        print("  " + rep.summary().split("\n  카메라별")[0].replace("\n", "\n  "))
    (OUT / "folds.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    base.unlink()
    print(f"썼습니다: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
