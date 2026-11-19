"""보관 폴더의 고속도로 프레임(hw-*)만 담은 **검수용 작업 사본**을 만든다 — 실제 프로젝트 DB 와 보관 폴더는 건드리지 않는다.

만드는 것(<out>): data/labeling.db(보관 DB 의 사본에서 hw 이미지만 남기고 경로를 사본으로 바꾼 것) · data/unlabeled/<카메라>/…(그림
사본) · data/cameras.yaml · data/classes.txt · models/best.pt(운영 가중치 사본 — 이 설정으로 학습을 눌러도 저장소의 models/ 가
바뀌지 않게) · config.yaml(경로 전부 절대경로) · README.txt.
보관 DB 는 immutable 로 열어 backup() 으로만 읽는다. <out> 이 비어 있지 않으면 멈춘다(덮어쓰지 않는다).

  python -I make_review_copy.py <…/20261110> <out> [--port 7871]
띄우기:  .venv\\Scripts\\python.exe labeler.py --config <out>\\config.yaml review --port 7871
"""
import argparse
import re
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import yaml

REPO = Path(r"C:\local_auto_labeling")
sys.path.insert(0, str(REPO / "src"))
from autolabel.store import Store  # noqa: E402

MARK = "\\data\\visible\\"
UNFIX = ("hw-06", "hw-07", "hw-08", "hw-09")      # 옛 등록의 test 고정 — 사본에서는 풀어 가릴 영역을 쓸 수 있게 한다


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("archive", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--port", type=int, default=7871)
    args = ap.parse_args()
    vis = args.archive / "data" / "visible"
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} 에 이미 파일이 있습니다 — 덮어쓰지 않습니다")
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)

    src = sqlite3.connect(f"file:{(vis / 'labeling.db').as_posix()}?mode=ro&immutable=1", uri=True)
    dst = sqlite3.connect(data / "labeling.db")
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()

    store = Store(data / "labeling.db")
    with store.connect() as con:
        rows = con.execute("SELECT id, path, camera FROM images").fetchall()
    keep = [r for r in rows if (r["camera"] or "").startswith("hw-")]
    store.delete_images([r["id"] for r in rows if not (r["camera"] or "").startswith("hw-")])
    moved = []
    for r in keep:
        p = r["path"]
        rel = p[p.lower().find(MARK) + len(MARK):]
        s, d = vis / rel, data / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)
        moved.append((str(d), r["id"]))
    with store.connect() as con:
        con.executemany("UPDATE images SET path = ? WHERE id = ?", moved)
        con.execute("DELETE FROM mask_templates")           # 다른 이미지에서 만든 템플릿은 가져오지 않는다
    con = sqlite3.connect(data / "labeling.db")
    con.execute("VACUUM")
    names = [r[0] for r in con.execute("SELECT name FROM classes ORDER BY idx")]
    n_img = con.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    status = dict(con.execute("SELECT status, COUNT(*) FROM images GROUP BY 1").fetchall())
    n_box = con.execute("SELECT COUNT(*) FROM boxes").fetchone()[0]
    con.close()
    (data / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")

    # 카메라 등록 — 옛 파일 그대로, test 고정 넷만 train 으로(이 사본 안에서만)
    lines = (vis / "cameras.yaml").read_text(encoding="utf-8").split("\n")
    cur, changed = None, []
    for i, ln in enumerate(lines):
        m = re.match(r"^  ([A-Za-z0-9_-]+):\s*$", ln)
        if m:
            cur = m.group(1)
        elif cur in UNFIX and re.match(r"^    split:\s*test\s*$", ln):
            lines[i] = "    split: train"
            changed.append(cur)
    (data / "cameras.yaml").write_text("\n".join(lines), encoding="utf-8")

    (out / "models").mkdir(exist_ok=True)
    shutil.copy2(REPO / "models" / "best_y26m_8cls.pt", out / "models" / "best.pt")

    cfg = yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8"))
    cfg["data"].update({"seed_images": str(data / "seed"), "seed_labels": str(data / "seed" / "labels"), "seed_coco": str(data / "seed"),
                        "unlabeled": str(data / "unlabeled"), "cameras": str(data / "cameras.yaml"), "labeled": str(data / "labeled"),
                        "dataset": str(data / "dataset"), "db": str(data / "labeling.db")})
    cfg["classes"] = names
    cfg["model"]["weights"] = str(out / "models" / "best.pt")
    cfg["train"]["overrides"].update({"project": str(out / "runs"), "name": "review_hw"})
    cfg["wsl"].update({"enabled": False, "dataset": "~/datasets/review_hw", "name": "review_hw"})
    cfg["server"]["port"] = args.port
    head = ("# 검수용 작업 사본의 설정 — make_review_copy.py 가 저장소의 config.yaml 에서 만들었다(" + datetime.now().strftime("%Y-%m-%d %H:%M") + ").\n"
            "# 경로는 전부 이 폴더 안의 절대경로다(상대경로는 저장소 기준으로 풀린다). 학습 결과·가중치도 이 폴더 안으로 간다.\n")
    (out / "config.yaml").write_text(head + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    (out / "README.txt").write_text(
        "옛 데이터의 고속도로 프레임(hw-01 … hw-09) 검수용 작업 사본\n"
        "=====================================================\n\n"
        f"만든 때   {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"원본      {args.archive} (읽기만 했다 — 그 폴더와 프로젝트의 실제 DB 는 그대로다)\n"
        f"담긴 것   이미지 {n_img}장({status}) · 박스 {n_box}개 · 클래스 {len(names)}개\n"
        "          data\\labeling.db 가 이 사본의 DB 다 — 여기서 고친 라벨은 이 파일에만 남는다.\n"
        f"          옛 등록에서 test 고정이던 {', '.join(sorted(set(changed)))} 는 이 사본에서만 train 으로 바꿨다(가릴 영역을 쓸 수 있게).\n"
        "          이 프레임들은 사용 권리를 확인하지 않은 화면이다 — 저장소(git)에 넣지 말 것.\n\n"
        "띄우기    저장소 폴더에서\n"
        f"          .venv\\Scripts\\python.exe labeler.py --config {out}\\config.yaml review --port {args.port}\n"
        f"          → http://127.0.0.1:{args.port}/\n\n"
        "쓰임      responder(출동 차량) 라벨을 고쳐 승인한 프레임만 학습 쪽에 더해, 검수 전 라벨로 잰 결과와 견준다\n"
        "          (저장소 docs/experiments.md '그 798장을 학습 쪽에만 더해 쟀다').\n"
        "          학습·스캔·정리 버튼은 누를 필요가 없다(눌러도 이 폴더 안에서만 바뀐다).\n", encoding="utf-8")
    print("만든 곳", out)
    print("이미지", n_img, status, "· 박스", n_box, "· 클래스", names)
    print("test 고정을 푼 카메라", sorted(set(changed)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
