"""DB 스키마 버전 (`PRAGMA user_version` — store.SCHEMA_VERSION, 2026-11-28 포터블 앱 5단계).

번호만 적는 단계다 — 값은 바꾸지 않는다. `Store` 가 열 때 열 맞추기를 마친 뒤 1 을 적고, 이 프로그램보다 새 번호의 DB 는 아무것도
쓰기 전에 거절한다(옛 코드가 모르는 꼴의 값을 덮지 않게). 사본(`copy_db`)은 번호까지 그대로 옮긴다.
"""

from __future__ import annotations

import sqlite3

import pytest

from autolabel.store import SCHEMA_VERSION, Store, copy_db, schema_version
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
    assert SCHEMA_VERSION == 1 and schema_version(db) == 1
    Store(db)                                           # 다시 열어도 그대로
    assert schema_version(db) == 1


def test_an_old_db_only_gets_the_number(tmp_path, store, add_image):
    """그 전 코드가 만든 DB(번호 0)는 열 때 번호만 생긴다 — 이미지·박스·이력 수는 그대로."""
    add_image("a.jpg", boxes=[box(), box(cls=0)])
    with sqlite3.connect(store.db_path) as con:
        con.execute("PRAGMA user_version = 0")
    before = counts(store.db_path)
    assert schema_version(store.db_path) == 0
    Store(store.db_path)
    assert schema_version(store.db_path) == 1 and counts(store.db_path) == before


def test_a_db_older_than_work_batches_also_ends_at_the_current_version(tmp_path):
    db = tmp_path / "old.db"
    old_db(db)                                          # 작업 배치 열이 없는 옛 DB — 열을 더하고 사본도 뜬다(11-23)
    s = Store(db)
    assert s.migrated_from is not None and schema_version(db) == 1
    assert schema_version(s.migrated_from) == 0         # 사본은 더하기 전의 모습 그대로


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
