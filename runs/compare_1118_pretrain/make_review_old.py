"""보관 폴더의 옛 데이터를 **어느 시점의 DB 사본 그대로** 담은 검수용 작업 사본을 만든다 — 실제 프로젝트 DB 와 보관 폴더는
건드리지 않는다. 기본 시점은 OWLv2 후보(responder·cone)를 넣기 직전(`labeling.db.bak_owl_20261109_113205`)이다: 클래스는 10개인데
두 클래스의 박스는 하나도 없고, 승인 이미지의 박스는 운영 가중치(8클래스, 11-08 학습)가 배운 것과 같다(11-23 에 학습 직전 사본
`bak_pretrain_20261108_050927` 과 견줌 — 다른 399장은 전부 검수 대기 VIRAT 프레임).

만드는 것(<out>): data/labeling.db(그 시점 사본 — 라벨·검수 상태·분할·태그·변경 이력이 그때 것) · data/<보관 폴더와 같은 자리>/…
(그림 사본 — 그 DB 가 가리키는 것만) · data/cameras.yaml · data/classes.txt · models/best.pt(운영 가중치 사본) · config.yaml(경로
전부 이 폴더 안의 절대경로) · README.txt.

그 시점 뒤에 바뀐 것 가운데 **그림 자리만** 마지막 DB(`labeling.db`)를 따른다 — 11-09 20:40 에 고속도로 카메라 키를 hw-NN 으로
바꾸며 폴더·파일 이름이 바뀌어서, `path`·`name`·`camera`·`grp` 는 id 로 마지막 DB 것을 쓴다(그림 내용·크기는 같다: 그 시점에 있던
이미지 가운데 뒤에 잘린 것이 없음을 폭·높이로 확인한다 — 다르면 멈춘다). 그 시점에는 있었는데 뒤에 지워져 파일이 없는 이미지는
사본에서 뺀다. 뒤에 들어온 이미지(11:32 뒤의 고속도로 프레임 등)는 그 시점에 없었으니 들어가지 않는다.

찾기 쉽게 태그 둘을 더한다(이 사본에만) — 뒤에 OWLv2 가 responder·cone 후보를 낸 이미지. 라벨은 건드리지 않는다.

보관 DB 들은 immutable 로 열어 읽기만 한다. <out> 이 비어 있지 않으면 멈춘다(덮어쓰지 않는다).

  python -I make_review_old.py <…/20261110> <out> [--snapshot <data/visible 아래 DB 사본 이름>] [--port 7872]
띄우기:  .venv\\Scripts\\python.exe labeler.py --config <out>\\config.yaml review --port 7872
"""
import argparse
import json
import re
import shutil
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from autolabel.store import Store  # noqa: E402

MARK = "\\data\\visible\\"
SNAPSHOT = "labeling.db.bak_owl_20261109_113205"
TAGS = {"responder": "OWLv2 후보 responder", "cone": "OWLv2 후보 cone"}


def ro(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro&immutable=1", uri=True)
    con.row_factory = sqlite3.Row
    return con


def rel(path: str) -> str:
    """보관 DB 의 절대경로(노트북 자리)에서 data\\visible\\ 뒤쪽만."""
    at = path.lower().find(MARK)
    if at < 0:
        raise SystemExit(f"data\\visible 아래가 아닌 경로: {path}")
    return path[at + len(MARK):]


def owl_candidates(fin: sqlite3.Connection, names: list[str]) -> tuple[dict[int, set[str]], Counter]:
    """마지막 DB 의 변경 이력에서, OWLv2 묶음이 이미지마다 무엇을 넣었는지. {이미지: {클래스 이름}}, (클래스, 한 일) 셈.

    import_proposals 는 차량 박스와 겹친 responder 후보는 **그 박스의 클래스만** 바꾸고(좌표 그대로) 나머지는 새 박스로 넣는다 —
    그래서 묶음 직전 목록에 같은 좌표의 박스가 있으면 '클래스만 바꿈', 없으면 '새 박스' 다.
    """
    want = {names.index(k): k for k in TAGS}
    per: dict[int, set[str]] = {}
    did: Counter = Counter()
    for iid, before, after in fin.execute(
            "SELECT image_id, before, after FROM box_history WHERE source = 'bulk' AND batch LIKE 'OWLv2 후보%' ORDER BY id"):
        prev: dict[tuple, list[int]] = {}
        for row in json.loads(before):
            prev.setdefault(tuple(round(float(v), 6) for v in row[1:5]), []).append(int(row[0]))
        for row in json.loads(after):
            cls = int(row[0])
            if cls not in want or row[6] != "owl":
                continue
            was = prev.get(tuple(round(float(v), 6) for v in row[1:5]), [])
            if cls in was:                              # 앞 묶음이 이미 넣은 후보
                continue
            per.setdefault(int(iid), set()).add(want[cls])
            did[(want[cls], f"클래스만 바꿈 ← {names[was[0]]}" if was else "새 박스")] += 1
    return per, did


def unpin_test(text: str, cams: set[str]) -> tuple[str, list[str]]:
    """등록 파일에서 이 사본에 이미지가 있는 카메라의 `split: test` 를 train 으로(이 사본 안에서만 — 가릴 영역을 쓸 수 있게)."""
    lines = text.split("\n")
    cur, changed = None, []
    for i, ln in enumerate(lines):
        m = re.match(r"^  ([A-Za-z0-9_-]+):\s*$", ln)
        if m:
            cur = m.group(1)
        elif cur in cams and re.match(r"^    split:\s*test\s*$", ln):
            lines[i] = "    split: train"
            changed.append(cur)
    return "\n".join(lines), sorted(set(changed))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("archive", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--snapshot", default=SNAPSHOT, help="data/visible 아래의 DB 사본 이름(backups/… 도 된다)")
    ap.add_argument("--port", type=int, default=7872)
    args = ap.parse_args()
    vis = args.archive / "data" / "visible"
    snap_path = vis / args.snapshot
    if not snap_path.is_file():
        raise SystemExit(f"사본이 없습니다: {snap_path}")
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} 에 이미 파일이 있습니다 — 덮어쓰지 않습니다")
    if REPO in out.parents or out == REPO:
        raise SystemExit("저장소 안에는 만들지 않습니다(그림·DB 사본이 git·스캔에 걸린다)")
    data = out / "data"

    fin = ro(vis / "labeling.db")
    final = {r["id"]: dict(r) for r in fin.execute("SELECT id, path, name, width, height, camera, grp FROM images")}
    snap = ro(snap_path)
    rows = [dict(r) for r in snap.execute("SELECT id, path, name, width, height, camera, grp, status FROM images")]
    names = [r[0] for r in snap.execute("SELECT name FROM classes ORDER BY idx")]

    # 그림을 어디서 가져올지부터 전부 정한다 — 하나라도 어긋나면 아무것도 만들기 전에 멈춘다
    plan, gone = [], []
    for r in rows:
        f = final.get(r["id"])
        if f is not None and (f["width"], f["height"]) != (r["width"], r["height"]):
            raise SystemExit(f"이미지 {r['id']} 의 크기가 그 시점과 마지막 DB 에서 다릅니다(뒤에 잘린 그림) — 이 시점으로는 만들 수 없습니다")
        where = rel(f["path"] if f is not None else r["path"])
        if (vis / where).is_file():
            plan.append((r, f, where))
        else:
            gone.append(r)
    if len(gone) > 50:
        raise SystemExit(f"파일이 없는 이미지가 {len(gone)}장입니다 — 보관 폴더가 맞는지 볼 것")

    data.mkdir(parents=True, exist_ok=True)
    dst = sqlite3.connect(data / "labeling.db")
    try:
        snap.backup(dst)
    finally:
        dst.close()
        snap.close()

    store = Store(data / "labeling.db")                     # 지금 코드의 표·열(가릴 영역·이력 목록 등)을 더한다
    store.delete_images([r["id"] for r in gone])
    moved, size = [], 0
    for r, f, where in plan:
        s, d = vis / where, data / where
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)
        size += d.stat().st_size
        keep = f if f is not None else r
        moved.append((str(d), keep["name"], keep["camera"], keep["grp"], r["id"]))
    with store.connect() as con:
        # 파일과 함께 옮기는 것이라 camera 도 같이 바꾼다(파일은 마지막 DB 의 카메라 폴더에 있다)
        con.executemany("UPDATE images SET path = ?, name = ?, camera = ?, grp = ? WHERE id = ?", moved)
    renamed = sum(1 for r, f, _ in plan if f is not None and f["camera"] != r["camera"])

    have = {r["id"] for r, _, _ in plan}
    per, did = owl_candidates(fin, [r[0] for r in fin.execute("SELECT name FROM classes ORDER BY idx")])
    fin.close()
    tagged = {}
    for key, tag in TAGS.items():
        ids = sorted(i for i, kinds in per.items() if key in kinds and i in have)
        if key in names and ids:
            tagged[tag] = store.set_image_tag(ids, store.create_tag(tag), True)
    later = sorted(i for i in per if i not in have)

    con = sqlite3.connect(data / "labeling.db")
    con.execute("VACUUM")
    n_img = con.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    status = dict(con.execute("SELECT status, COUNT(*) FROM images GROUP BY 1").fetchall())
    split = dict(con.execute("SELECT COALESCE(split, '미배정'), COUNT(*) FROM images WHERE status = 'approved' GROUP BY 1").fetchall())
    n_box = con.execute("SELECT COUNT(*) FROM boxes").fetchone()[0]
    new_cls = {k: con.execute("SELECT COUNT(*) FROM boxes WHERE cls = ?", (names.index(k),)).fetchone()[0] for k in TAGS if k in names}
    cams = {r[0] for r in con.execute("SELECT DISTINCT camera FROM images WHERE camera IS NOT NULL")}
    last = con.execute("SELECT MAX(at) FROM box_history").fetchone()[0]
    con.close()
    (data / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")

    text, unpinned = unpin_test((vis / "cameras.yaml").read_text(encoding="utf-8"), cams)
    (data / "cameras.yaml").write_text(text, encoding="utf-8")

    (out / "models").mkdir(exist_ok=True)
    shutil.copy2(REPO / "models" / "best_y26m_8cls.pt", out / "models" / "best.pt")

    cfg = yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8"))
    cfg["data"].update({"seed_images": str(data / "seed"), "seed_labels": str(data / "seed" / "labels"), "seed_coco": str(data / "seed"),
                        "unlabeled": str(data / "unlabeled"), "cameras": str(data / "cameras.yaml"), "labeled": str(data / "labeled"),
                        "dataset": str(data / "dataset"), "db": str(data / "labeling.db")})
    cfg["classes"] = names
    cfg["model"]["weights"] = str(out / "models" / "best.pt")
    cfg["train"]["overrides"].update({"project": str(out / "runs"), "name": "review_old"})
    cfg["wsl"].update({"enabled": False, "dataset": "~/datasets/review_old", "name": "review_old"})
    cfg["server"]["port"] = args.port
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    head = (f"# 검수용 작업 사본의 설정 — make_review_old.py 가 저장소의 config.yaml 에서 만들었다({now}).\n"
            "# 경로는 전부 이 폴더 안의 절대경로다(상대경로는 저장소 기준으로 풀린다). 학습 결과·가중치도 이 폴더 안으로 간다.\n")
    (out / "config.yaml").write_text(head + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    did_lines = "".join(f"            {k[0]:<10} {k[1]:<24} {v:>5}\n" for k, v in sorted(did.items()))
    (out / "README.txt").write_text(
        "옛 데이터 검수용 작업 사본 — 시점: " + args.snapshot + "\n"
        "=================================================================\n\n"
        f"만든 때   {now}\n"
        f"원본      {args.archive} (읽기만 했다 — 그 폴더와 프로젝트의 실제 DB 는 그대로다)\n"
        f"시점      {args.snapshot} — 이 DB 의 마지막 변경 이력 {last} (UTC)\n"
        f"담긴 것   이미지 {n_img}장({status}) · 승인분의 분할 {split} · 박스 {n_box}개 · 클래스 {len(names)}개\n"
        f"          responder·cone 박스 {new_cls}\n"
        f"          그림 {size / 2**30:.2f}GB 는 보관 폴더와 같은 자리(legacy\\·labeled\\·unlabeled\\)에 복사했다.\n"
        "          data\\labeling.db 가 이 사본의 DB 다 — 여기서 고친 라벨은 이 파일에만 남는다.\n\n"
        "그 시점과 다른 것\n"
        f"  - 그림 자리: 뒤에 카메라 키가 바뀐 {renamed}장은 path·name·camera·grp 를 마지막 DB(11-10)의 것으로 적었다(그림은 같다).\n"
        f"  - 그 시점에는 있었지만 뒤에 지워져 파일이 없는 {len(gone)}장은 뺐다: id {[r['id'] for r in gone]}\n"
        f"  - 태그 둘을 더했다(찾기용 — 라벨은 그대로): {tagged}\n"
        "    뒤에(11-09 11:32–11:44) OWLv2 가 그 클래스 후보를 낸 이미지다. 그때 넣은 것(마지막 DB 의 이력으로 셈):\n"
        + did_lines +
        f"    후보가 났지만 이 시점에 없던 이미지(뒤에 들어온 프레임) {len(later)}장은 이 사본에 없다.\n"
        + (f"  - 옛 등록에서 test 고정이던 카메라 {unpinned} 는 이 사본에서만 train 으로 바꿨다(가릴 영역을 쓸 수 있게).\n" if unpinned
           else "  - 카메라 등록(cameras.yaml)은 보관 폴더의 것 그대로다(이 사본의 이미지에 test 고정 카메라가 없어 바꾼 것이 없다).\n") +
        "    분할이 test 인 승인 이미지에는 서버가 가릴 영역을 받지 않는다(그때의 분할 그대로 두었다).\n\n"
        "띄우기    저장소 폴더에서\n"
        f"          .venv\\Scripts\\python.exe labeler.py --config {out}\\config.yaml review --port {args.port}\n"
        f"          → http://127.0.0.1:{args.port}/\n\n"
        "주의      옛 화면에는 사용 권리를 확인하지 않은 것이 있다 — 저장소(git)에 넣지 말 것.\n"
        "          학습·스캔·정리 버튼은 누를 필요가 없다(눌러도 이 폴더 안에서만 바뀐다).\n", encoding="utf-8")

    print("만든 곳", out)
    print("시점", args.snapshot, "· 마지막 이력(UTC)", last)
    print("이미지", n_img, status, "· 승인분 분할", split, "· 박스", n_box, "· 클래스", len(names))
    print("responder·cone 박스", new_cls, "· 그림", f"{size / 2**30:.2f}GB", "· 키가 바뀐 카메라의 이미지", renamed)
    print("뺀 이미지(파일 없음)", [(r["id"], r["status"]) for r in gone])
    print("태그", tagged, "· 후보가 났지만 이 시점에 없던 이미지", len(later))
    for k, v in sorted(did.items()):
        print("  OWLv2 가 넣은 것:", k[0], "·", k[1], v)
    print("test 고정을 푼 카메라", unpinned)
    return 0


if __name__ == "__main__":
    sys.exit(main())
