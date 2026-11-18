"""실험용 세트(175장)를 카메라·같은 장면 묶음째 K 묶음으로 나눈다 — 프로젝트 DB 는 읽지 않고 세트의 DB 사본만 읽는다.

묶음은 프로젝트의 `grouping.group_images`(카메라째 + 이름 계열 + dHash)를 그대로 쓴다. 묶음을 통째로 한 fold 에 넣되
클래스별 박스 수와 이미지 수가 fold 마다 비슷해지도록 탐욕적으로 배정한다.
"""
import hashlib
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

import cmp_common as cc

sys.path.insert(0, str(cc.REPO / "src"))
from autolabel import grouping  # noqa: E402

K = 3


def main() -> int:
    con = sqlite3.connect("file:" + str(cc.ROOT / "set" / "labeling.db").replace("\\", "/") + "?mode=ro", uri=True)
    rows = con.execute("SELECT id, path, width, height, camera FROM images ORDER BY id").fetchall()
    items = []
    for iid, path, w, h, cam in rows:
        src = Path(path)
        stem = f"{src.stem}__{hashlib.sha1(str(src).encode('utf-8')).hexdigest()[:8]}"
        cands = [cc.DS / "images" / "test" / f"{stem}{ext}" for ext in dict.fromkeys([".jpg", src.suffix])]
        files = [p for p in cands if p.exists()]
        if len(files) != 1:
            raise SystemExit(f"세트에서 이미지를 못 찾음: id {iid} ({len(files)}개)")
        cls = Counter()
        lab = cc.DS / "labels" / "test" / f"{stem}.txt"
        if lab.exists():
            for line in lab.read_text().split("\n"):
                if line.strip():
                    cls[int(line.split()[0])] += 1
        items.append({"id": iid, "path": path, "width": w, "height": h, "camera": cam, "dhash": grouping.dhash(src),
                      "file": files[0], "cls": cls})
    grp = grouping.group_images(items)
    groups: dict[str, list[dict]] = {}
    for it in items:
        groups.setdefault(grp[it["id"]], []).append(it)
    cams = {it["camera"] for it in items}
    print(f"이미지 {len(items)} · 카메라 {len(cams)} · 묶음 {len(groups)} (카메라보다 적으면 dHash 로 이어진 카메라가 있다)")

    feats = list(range(len(cc.NAMES))) + ["img"]

    def vec(members):
        c = Counter()
        for it in members:
            c.update(it["cls"])
        return np.array([c[i] for i in range(len(cc.NAMES))] + [len(members)], dtype=float)

    gv = {g: vec(m) for g, m in groups.items()}
    total = sum(gv.values())
    total[total == 0] = 1
    order = sorted(groups, key=lambda g: (-float((gv[g] / total).max()), g))      # 한 클래스를 많이 쥔 묶음부터
    load = np.zeros((K, len(feats)))
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

    out = cc.ROOT / "folds"
    out.mkdir(exist_ok=True)
    for k in range(K):
        val = sorted(str(it["file"]) for g, m in groups.items() if fold_of[g] == k for it in m)
        train = sorted(str(it["file"]) for g, m in groups.items() if fold_of[g] != k for it in m)
        assert not set(val) & set(train)
        (out / f"f{k}_val.txt").write_text("\n".join(val) + "\n", encoding="utf-8")
        (out / f"f{k}_train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
        spec = {"path": str(cc.DS), "train": str(out / f"f{k}_train.txt"), "val": str(out / f"f{k}_val.txt"),
                "names": {i: n for i, n in enumerate(cc.NAMES)}}
        (out / f"f{k}.yaml").write_text(yaml.safe_dump(spec, allow_unicode=True, sort_keys=False), encoding="utf-8")
        n_groups = sum(1 for g in groups if fold_of[g] == k)
        print(f"fold {k}: 평가 {len(val)}장(묶음 {n_groups}) / 학습 {len(train)}장 · 평가 박스 {int(load[k][:-1].sum())} — "
              + ", ".join(f"{cc.NAMES[i]} {int(load[k][i])}" for i in range(len(cc.NAMES))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
