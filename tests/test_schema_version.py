"""DB 스키마 버전 (`PRAGMA user_version` — store.SCHEMA_VERSION, 2026-11-28 포터블 앱 5단계).

`Store` 가 열 때 열 맞추기를 마친 뒤 지금 번호를 적고, 이 프로그램보다 새 번호의 DB 는 아무것도 쓰기 전에 거절한다(옛 코드가 모르는
꼴의 값을 덮지 않게). 사본(`copy_db`)은 번호까지 그대로 옮긴다. 번호마다(store.py 의 SCHEMA_VERSION 머리말):
  0 → 1 (5단계) 번호만 — 값은 그대로.
  1 → 2 (6단계) 열 `images.rel`(DB 폴더 기준 상대경로)을 더하고 채운다 — 더하기 전에 사본, `images.path` 는 그대로.
    상대경로를 쓰는 규칙(옮긴 DB 의 경로 다시 찾기)은 `tests/test_image_relpath.py`.
"""

from __future__ import annotations

import sqlite3

import pytest

from autolabel.store import SCHEMA_VERSION, Store, copy_db, needs_rel_migration, schema_version
from helpers import box
from test_workbatch import old_db


def counts(db) -> dict:
    con = sqlite3.connect(db)
    try:
        return {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("images", "boxes", "box_history")}
    finally:
        con.close()


def test_a_new_db_gets_the_current_version(tmp_path):
    db = tmp_path / "new.db"
    assert schema_version(db) == 0                      # 없는 파일
    Store(db)
    assert SCHEMA_VERSION == 2 and schema_version(db) == 2
    Store(db)                                           # 다시 열어도 그대로
    assert schema_version(db) == 2


def test_an_old_db_only_gets_the_number(tmp_path, store, add_image):
    """그 전 코드가 만든 DB(번호 0)는 열 때 번호만 생긴다 — 이미지·박스·이력 수는 그대로."""
    add_image("a.jpg", boxes=[box(), box(cls=0)])
    with sqlite3.connect(store.db_path) as con:
        con.execute("PRAGMA user_version = 0")
    before = counts(store.db_path)
    assert schema_version(store.db_path) == 0
    Store(store.db_path)
    assert schema_version(store.db_path) == SCHEMA_VERSION and counts(store.db_path) == before


def test_a_db_older_than_work_batches_also_ends_at_the_current_version(tmp_path):
    db = tmp_path / "old.db"
    old_db(db)                                          # 작업 배치 열이 없는 옛 DB — 열을 더하고 사본도 뜬다(11-23)
    s = Store(db)
    assert s.migrated_from is not None and schema_version(db) == SCHEMA_VERSION
    assert schema_version(s.migrated_from) == 0         # 사본은 더하기 전의 모습 그대로
    assert len(list((tmp_path / "backups").glob("*.db"))) == 1      # 작업 배치·상대경로를 한 사본으로


def test_a_version_1_db_gets_the_relative_path_column(tmp_path):
    """5단계 코드가 만든 DB(번호 1, rel 열 없음) — 열을 더하기 전에 사본(`_before_relpath_`), DB 폴더 아래 이미지는 rel 이 채워지고
    path·박스·이력은 그대로."""
    db = tmp_path / "v1.db"
    s = Store(db)
    s.set_classes(["car"])
    inside = tmp_path / "unlabeled" / "cam-01" / "a.jpg"
    s.ingest_batch([{"path": str(inside), "source": "collected", "width": 64, "height": 36, "status": "pending",
                     "boxes": [box(cls=0)]},
                    {"path": str(tmp_path.parent / "elsewhere.jpg"), "source": "legacy", "width": 64, "height": 36,
                     "status": "pending"}])
    with sqlite3.connect(db) as con:
        con.execute("ALTER TABLE images DROP COLUMN rel")
        con.execute("PRAGMA user_version = 1")
    before = counts(db)
    with sqlite3.connect(db) as con:
        paths = [r[0] for r in con.execute("SELECT path FROM images ORDER BY id")]
    assert needs_rel_migration(db) is True
    s = Store(db)
    assert schema_version(db) == 2 and counts(db) == before and not needs_rel_migration(db)
    assert s.migrated_from.name.startswith("v1_before_relpath_") and schema_version(s.migrated_from) == 1
    with sqlite3.connect(db) as con:
        assert [tuple(r) for r in con.execute("SELECT path, rel FROM images ORDER BY id")] == [
            (paths[0], "unlabeled/cam-01/a.jpg"), (paths[1], None)]
    assert any("상대경로 열" in line for line in s.open_notes())
    assert Store(db).migrated_from is None                  # 다시 열면 뜨지 않는다


def test_a_db_from_a_newer_program_is_refused_without_writing(tmp_path):
    db = tmp_path / "newer.db"
    Store(db)
    with sqlite3.connect(db) as con:
        con.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    raw = db.read_bytes()
    with pytest.raises(RuntimeError, match=rf"스키마 {SCHEMA_VERSION + 1}"):
        Store(db)
    assert db.read_bytes() == raw and not (tmp_path / "backups").exists()


def test_a_copy_keeps_the_version(tmp_path, store):
    dst = copy_db(store.db_path, tmp_path / "copy.db")
    assert schema_version(dst) == SCHEMA_VERSION
