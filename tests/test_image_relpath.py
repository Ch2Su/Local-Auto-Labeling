"""이미지 경로의 DB 폴더 기준 상대경로 (store `images.rel`·`_relocate`, 2026-11-28 포터블 앱 6단계 — 스키마 2).

`images.path` 는 절대경로 그대로이고 모두가 그것을 읽는다. `rel` 은 path 를 쓰는 길이 함께 적고 열 때마다 다시 맞추며, DB 폴더째
옮긴 뒤 열면 **path 의 파일이 없고 `DB 폴더/rel` 에 파일이 있는 행만** path 를 그 자리로 바꾼다(그 전에 사본). 파일이 있는 행은
건드리지 않는다 — '파일 없음' 이 늘면 prune 이 그 행을 라벨째 지운다. 그림 내용은 보지 않으니 파일은 아무 바이트나 쓴다.
"""

from __future__ import annotations

import ast
import os
import re
import shutil
import sqlite3
from pathlib import Path

import pytest

import autolabel.store as store_mod
from autolabel.store import Store
from helpers import ROOT, box

NAMES = ("a.jpg", "b.jpg", "c.jpg")


def project(root: Path) -> tuple[Path, list[int]]:
    """root/data/labeling.db 와 그 아래 그림 셋(카메라 cam-01) — 박스·영역·이력·상태가 조금씩 든 DB. (DB 경로, id 들)."""
    data = root / "data"
    db = data / "labeling.db"
    s = Store(db)
    s.set_classes(["car", "person"])
    items = []
    for k, name in enumerate(NAMES):
        f = data / "unlabeled" / "cam-01" / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(bytes([k]) * 16)
        items.append({"path": str(f.resolve()), "source": "collected", "width": 64, "height": 36,
                      "status": "pending", "boxes": [box(cls=0)], "camera": "cam-01"})
    s.ingest_batch(items)
    with s.connect() as con:
        ids = [int(r[0]) for r in con.execute("SELECT id FROM images ORDER BY id")]
    s.set_boxes(ids[0], [box(cls=1), box(cls=0)])                       # 이력 한 줄
    s.set_masks(ids[1], [(0.1, 0.1, 0.4, 0.4)])
    s.set_status(ids[2], "approved")
    return db, ids


def snapshot(db: Path) -> dict:
    """경로 말고 지켜야 할 것 — 행마다 (이름·상태·updated_at·카메라), 박스·영역·이력 수."""
    con = sqlite3.connect(db)
    try:
        return {
            "rows": [tuple(r) for r in con.execute(
                "SELECT id, name, status, updated_at, camera FROM images ORDER BY id")],
            **{t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
               for t in ("boxes", "masks", "box_history")},
        }
    finally:
        con.close()


def paths(db: Path) -> list[tuple[str, str | None]]:
    con = sqlite3.connect(db)
    try:
        return [tuple(r) for r in con.execute("SELECT path, rel FROM images ORDER BY id")]
    finally:
        con.close()


def backups(db: Path) -> list[Path]:
    return sorted((db.parent / "backups").glob("*_before_relocate_*.db"))


# ================================================================= rel 을 적는 것


def test_rows_inside_the_db_folder_get_a_relative_path(tmp_path):
    db, _ = project(tmp_path)
    assert [rel for _, rel in paths(db)] == [f"unlabeled/cam-01/{n}" for n in NAMES]
    s = Store(db)
    base = db.parent.resolve()
    assert s.rel_path(base / "x" / "y.jpg") == "x/y.jpg"
    assert s.rel_path(str(base / "x" / "y.jpg").upper() if os.name == "nt" else base / "x" / "y.jpg") is not None
    assert s.rel_path(base) is None                                     # DB 폴더 자체
    assert s.rel_path(tmp_path / "elsewhere.jpg") is None               # DB 폴더 밖
    assert s.rel_path(str(base) + "x" + os.sep + "a.jpg") is None       # 이름이 DB 폴더로 시작할 뿐인 옆 폴더
    other = "/home/someone/data/a.jpg" if os.name == "nt" else "C:\\data\\a.jpg"
    assert s.rel_path(other) is None                                    # 다른 OS 꼴


def test_every_writer_keeps_rel_with_path(tmp_path):
    db, ids = project(tmp_path)
    s = Store(db)
    base = db.parent.resolve()

    def rel_ok() -> None:
        for path, rel in paths(db):
            assert rel == s.rel_path(path), (path, rel)

    a = s.get_image(ids[0])["path"]
    moved = base / "labeled" / "images" / "cam-01" / "a.jpg"
    assert s.move_row(ids[0], a, moved, ("pending",), lambda: None) is not None
    assert paths(db)[0] == (str(moved.resolve()), "labeled/images/cam-01/a.jpg")
    s.update_path(ids[0], base / "unlabeled" / "cam-01" / "a2.jpg")
    rel_ok()
    b = s.get_image(ids[1])["path"]
    b2 = str(base / "unlabeled" / "cam-09" / "cam-09_b.jpg")
    undo = s.rename_camera_rows([(ids[1], b, b2)], "cam-01", "cam-09")
    assert paths(db)[1][1] == "unlabeled/cam-09/cam-09_b.jpg"
    s.undo_camera_rows([(ids[1], b, b2)], undo, "cam-01")
    assert paths(db)[1][1] == "unlabeled/cam-01/b.jpg"
    c = s.get_image(ids[2])["path"]
    c2 = str(base / "unlabeled" / "cam-02" / "cam-02_c.jpg")
    s.move_camera_rows([(ids[2], c, c2, "cam-01", "cam-02", None)])
    assert paths(db)[2][1] == "unlabeled/cam-02/cam-02_c.jpg"
    img = s.get_image(ids[2])
    outside = str((tmp_path / "crop" / "c.jpg").resolve())             # DB 폴더 밖으로 — rel 은 비운다
    assert s.replace_file(ids[2], outside, 32, 18, [box()], expect=(img["status"], img["updated_at"]),
                          uncertainty=0.0, model_version="t", keep_status=True)
    assert paths(db)[2] == (outside, None)
    rel_ok()


def _sql_strings(path: Path) -> list[tuple[int, str]]:
    out = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.append((node.lineno, node.value))
    return out


def test_every_sql_that_writes_a_path_writes_rel_too():
    """패키지에서 images.path 를 쓰는 SQL(넣기·SET path)은 rel 도 적는다 — 빠뜨리면 그 행은 열 때까지 옮길 단서가 없다."""
    found = []
    for py in sorted((ROOT / "src" / "autolabel").glob("*.py")):
        for line, sql in _sql_strings(py):
            if re.search(r"\bINSERT\b[^;]*\bINTO\s+images\s*\(", sql, re.I):
                found.append(py.name)
                assert re.search(r"\brel\b", sql), f"{py.name}:{line} 넣기에 rel 이 없다: {sql[:80]}"
            m = re.search(r"\bUPDATE\s+images\s+SET\b(.*?)(\bWHERE\b|$)", sql, re.I | re.S)
            if m and re.search(r"(^|[\s,])path\s*=", m.group(1)):
                found.append(py.name)
                assert re.search(r"\brel\s*=", m.group(1)), f"{py.name}:{line} SET path 에 rel 이 없다: {sql[:80]}"
    assert found.count("store.py") >= 7          # 이 검사가 실제로 무언가를 보고 있는지 (넣기 1 + SET path 6)


def test_rel_is_put_right_when_old_code_moved_a_file(tmp_path):
    """rel 을 모르는 옛 코드(이 열 전의 정리 등)가 파일을 옮기고 path 만 바꾼 행 — 다음에 열 때 rel 이 그 path 에 맞춰진다."""
    db, ids = project(tmp_path)
    new = str((db.parent / "labeled" / "images" / "cam-01" / "a.jpg").resolve())
    Path(new).parent.mkdir(parents=True)
    shutil.move(paths(db)[0][0], new)
    with sqlite3.connect(db) as con:
        con.execute("UPDATE images SET path=? WHERE id=?", (new, ids[0]))
    assert paths(db)[0][1] == "unlabeled/cam-01/a.jpg"                  # 아직 옛 자리
    Store(db)
    assert paths(db)[0] == (new, "labeled/images/cam-01/a.jpg")


# ================================================================= 옮긴 DB 에서 다시 찾기


def test_a_moved_folder_is_found_again(tmp_path):
    db, ids = project(tmp_path / "A")
    before = snapshot(db)
    shutil.move(str(tmp_path / "A"), str(tmp_path / "B"))              # 폴더째 옮겼다 — 옛 자리에는 아무것도 없다
    db2 = tmp_path / "B" / "data" / "labeling.db"
    old = [p for p, _ in paths(db2)]
    s = Store(db2)
    base = db2.parent.resolve()
    assert paths(db2) == [(str((base / "unlabeled" / "cam-01" / n).resolve()), f"unlabeled/cam-01/{n}") for n in NAMES]
    assert snapshot(db2) == before                                      # 이름·상태·updated_at·카메라·박스·영역·이력 그대로
    assert s.relocated["moved"] == 3 and s.relocated["missing"] == 0 and s.relocated["taken"] == 0
    [copy] = backups(db2)
    assert s.relocated["backup"] == copy and [p for p, _ in paths(copy)] == old    # 사본은 고치기 전의 경로
    assert set(s.known_statuses()) == {str((base / "unlabeled" / "cam-01" / n).resolve()) for n in NAMES}  # 스캔이 다시 넣지 않는다
    assert any("이미지 경로 3개" in line for line in s.open_notes())
    again = Store(db2)                                                  # 다시 열면 할 일이 없다
    assert again.relocated is None and again.open_notes() == [] and len(backups(db2)) == 1


def test_a_db_moved_up_into_its_parent_folder(tmp_path):
    """DB 폴더가 옛 DB 폴더의 위 폴더가 됐다(data/ 안의 것을 한 칸 위로) — 옛 경로가 새 DB 폴더 아래라 rel 을 먼저 맞추면
    'data/unlabeled/…' 로 바뀌어 못 찾는다. 다시 찾기가 rel 맞추기보다 먼저다."""
    db, _ = project(tmp_path / "X")
    for child in list(db.parent.iterdir()):
        shutil.move(str(child), str(tmp_path / "X" / child.name))
    db.parent.rmdir()
    db2 = tmp_path / "X" / "labeling.db"
    s = Store(db2)
    base = db2.parent.resolve()
    assert s.relocated["moved"] == 3
    assert paths(db2) == [(str((base / "unlabeled" / "cam-01" / n).resolve()), f"unlabeled/cam-01/{n}") for n in NAMES]


def test_rows_whose_file_is_still_there_are_left_alone(tmp_path):
    """복사만 했고 옛 자리에 그림이 그대로면 그 행은 그대로 — DB 사본으로 띄우는 시험 서버도 이 꼴이다(원래 그림을 가리킨다)."""
    db, _ = project(tmp_path / "A")
    shutil.copytree(tmp_path / "A", tmp_path / "B")
    db2 = tmp_path / "B" / "data" / "labeling.db"
    want = paths(db2)
    s = Store(db2)
    assert paths(db2) == want and s.relocated is None and not backups(db2)
    lone = tmp_path / "C" / "labeling.db"                               # DB 파일만 다른 폴더로
    lone.parent.mkdir()
    shutil.copy2(db, lone)
    s = Store(lone)
    assert [p for p, _ in paths(lone)] == [p for p, _ in want] and s.relocated is None and not backups(lone)


def test_images_left_behind_are_reported_and_kept_for_later(tmp_path):
    """DB 만 옮기고 그림 일부를 빠뜨렸다 — 찾은 것만 고치고, 못 찾은 행은 경로·rel 을 그대로 둔다(그림을 마저 옮기고 다시
    열면 찾는다). 옛 자리에도 없으니 '파일 없음' 수는 고치기 전과 같거나 적다."""
    db, _ = project(tmp_path / "A")
    shutil.move(str(tmp_path / "A"), str(tmp_path / "B"))
    db2 = tmp_path / "B" / "data" / "labeling.db"
    late = tmp_path / "later_c.jpg"
    shutil.move(str(db2.parent / "unlabeled" / "cam-01" / "c.jpg"), str(late))
    old = paths(db2)
    s = Store(db2)
    assert s.relocated["moved"] == 2 and s.relocated["missing"] == 1
    assert paths(db2)[2] == old[2]
    assert any("[경고] 이미지 1개" in line for line in s.open_notes())
    shutil.move(str(late), str(db2.parent / "unlabeled" / "cam-01" / "c.jpg"))
    s = Store(db2)
    assert s.relocated["moved"] == 1 and s.relocated["missing"] == 0 and len(backups(db2)) == 2
    assert all(os.path.exists(p) for p, _ in paths(db2))


def test_nothing_found_means_nothing_written(tmp_path):
    db, _ = project(tmp_path / "A")
    lone = tmp_path / "C" / "labeling.db"
    lone.parent.mkdir()
    shutil.copy2(db, lone)
    shutil.rmtree(tmp_path / "A")                                       # 그림은 어디에도 없다
    raw = lone.read_bytes()
    s = Store(lone)
    assert s.relocated == {"moved": 0, "missing": 3, "taken": 0, "backup": None}
    assert lone.read_bytes() == raw and not backups(lone)              # 바이트 그대로 (사본도 없다)


def test_a_spot_already_taken_is_skipped(tmp_path):
    """옛 코드로 새 자리를 스캔해 같은 사진이 새 행으로 들어간 DB — 그 자리를 쓰는 행이 있으면 옛 행은 고치지 않는다(UNIQUE)."""
    db, ids = project(tmp_path / "A")
    shutil.move(str(tmp_path / "A"), str(tmp_path / "B"))
    db2 = tmp_path / "B" / "data" / "labeling.db"
    dup = str((db2.parent / "unlabeled" / "cam-01" / "a.jpg").resolve())
    with sqlite3.connect(db2) as con:                                   # 옛 코드의 넣기 — rel 없이
        con.execute("INSERT INTO images(path, name, width, height, source, status, updated_at)"
                    " VALUES (?, 'a.jpg', 64, 36, 'collected', 'unlabeled', 'x')", (dup,))
    old_a = paths(db2)[0][0]
    s = Store(db2)
    assert s.relocated["moved"] == 2 and s.relocated["taken"] == 1
    assert paths(db2)[0][0] == old_a and paths(db2)[3] == (dup, "unlabeled/cam-01/a.jpg")
    assert any("[알림]" in line for line in s.open_notes())


def test_a_db_from_another_os_is_found_again(tmp_path):
    """다른 OS 에서 만든 DB — path 는 그 OS 꼴의 절대경로(여기서는 없는 파일), rel 은 '/' 로 이은 것."""
    data = tmp_path / "data"
    db = data / "labeling.db"
    Store(db)
    f = data / "unlabeled" / "cam-01" / "a.jpg"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"x")
    there = "/home/someone/lal/data/unlabeled/cam-01/a.jpg" if os.name == "nt" else "Q:\\lal\\data\\unlabeled\\cam-01\\a.jpg"
    with sqlite3.connect(db) as con:
        con.execute("INSERT INTO images(path, name, width, height, source, status, updated_at, rel)"
                    " VALUES (?, 'a.jpg', 64, 36, 'collected', 'approved', 'x', 'unlabeled/cam-01/a.jpg')", (there,))
    s = Store(db)
    assert s.relocated["moved"] == 1 and paths(db) == [(str(f.resolve()), "unlabeled/cam-01/a.jpg")]


def test_without_the_copy_no_path_is_changed(tmp_path, monkeypatch):
    db, _ = project(tmp_path / "A")
    shutil.move(str(tmp_path / "A"), str(tmp_path / "B"))
    db2 = tmp_path / "B" / "data" / "labeling.db"
    raw = db2.read_bytes()

    def full(src, dst):
        raise OSError("디스크가 찼습니다")

    monkeypatch.setattr(store_mod, "copy_db", full)
    with pytest.raises(RuntimeError, match="DB 를 바꾸지 않았습니다"):
        Store(db2)
    assert db2.read_bytes() == raw
