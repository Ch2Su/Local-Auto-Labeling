"""휴지통 비우기·정리·없는 파일 걷기 (tidy.py) — 도는 사이 사람이 상태를 바꾼 이미지 (2026-11-27, Gemini 리뷰 묶음 8).

셋 다 시작할 때 목록을 한 번 읽고 한참 돈다(작업 큐). 그사이 검수 화면은 살아 있어서 휴지통의 이미지를 되살리거나 승인을
풀 수 있다. 고치기 전에는 처음 읽은 목록 그대로 파일을 지우고·옮기고 끝에 id 만으로 행을 지워, 되살린 이미지의 원본과
라벨이 사라졌다.

'도는 사이' 는 진행 알림(`progress` — 장마다 그 장을 다루기 직전에 불린다)에서 만든다: 첫 장의 알림에서 뒤의 장을 바꾼다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autolabel.tidy import prune_missing, purge_trashed, tidy_approved
from helpers import box


@pytest.fixture
def put(cfg, store):
    """자동 라벨링 폴더(unlabeled/cam-1/)에 파일과 행을 함께 둔다 — 박스 하나."""

    def add(name: str, status: str) -> int:
        path = cfg.unlabeled / "cam-1" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"jpeg " + name.encode())
        store.ingest_batch([{"path": str(path), "source": "collected", "width": 1280, "height": 720, "status": status,
                             "boxes": [box()], "camera": "cam-1"}])
        with store.connect() as con:
            return int(con.execute("SELECT id FROM images WHERE path=?", (str(path),)).fetchone()[0])

    return add


def at_first(fn):
    """첫 장의 진행 알림에서 한 번만 fn 을 부르는 progress."""
    def progress(n, total, name):
        if n == 1:
            fn()
    return progress


def test_purge_leaves_an_image_restored_while_it_runs(cfg, store, put):
    a, b, c = (put(f"{k}.jpg", "trashed") for k in "abc")
    paths = {i: Path(store.get_image(i)["path"]) for i in (a, b, c)}

    report = purge_trashed(cfg, store, progress=at_first(lambda: store.set_status(b, "pending")))

    assert store.get_image(b)["status"] == "pending" and paths[b].exists() and len(store.get_boxes(b)) == 1
    assert store.get_image(a) is None and store.get_image(c) is None
    assert not paths[a].exists() and not paths[c].exists()
    assert (report.rows_deleted, report.files_deleted, report.kept, report.boxes) == (2, 2, 1, 2)
    assert "휴지통에서 꺼낸 이미지 1장" in report.summary()


def test_purge_keeps_the_row_when_the_file_cannot_be_deleted(cfg, store, put, monkeypatch):
    a, b = put("a.jpg", "trashed"), put("b.jpg", "trashed")
    stuck = Path(store.get_image(a)["path"])
    real = Path.unlink

    def unlink(self, *args, **kwargs):
        if self == stuck:
            raise PermissionError("다른 프로그램이 쓰는 중")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    report = purge_trashed(cfg, store)
    assert store.get_image(a)["status"] == "trashed" and stuck.exists()        # 다시 비울 수 있게 남는다
    assert store.get_image(b) is None and report.rows_deleted == 1 and len(report.errors) == 1


def test_purge_removes_rows_whose_file_is_already_gone(cfg, store, put):
    a = put("a.jpg", "trashed")
    Path(store.get_image(a)["path"]).unlink()
    report = purge_trashed(cfg, store)
    assert store.get_image(a) is None and (report.rows_deleted, report.already_gone, report.files_deleted) == (1, 1, 0)


def test_tidy_does_not_move_an_image_unapproved_while_it_runs(cfg, store, put):
    a, b, c = (put(f"{k}.jpg", "approved") for k in "abc")
    before = Path(store.get_image(b)["path"])

    report = tidy_approved(cfg, store, progress=at_first(lambda: store.set_status(b, "pending")))

    moved = cfg.labeled / "images" / "cam-1"
    assert sorted(p.name for p in moved.iterdir()) == ["a.jpg", "c.jpg"]
    assert sorted(p.name for p in (cfg.labeled / "labels" / "cam-1").iterdir()) == ["a.txt", "c.txt"]   # 라벨 파일도 안 쓴다
    assert before.exists() and store.get_image(b)["path"] == str(before) and store.get_image(b)["status"] == "pending"
    assert Path(store.get_image(a)["path"]) == (moved / "a.jpg").resolve()
    assert (report.moved, report.changed, report.errors) == (2, ["b.jpg"], [])
    assert "옮기지 않은 이미지 1장" in report.summary()


def test_tidy_puts_the_file_back_when_the_database_cannot_be_updated(cfg, store, put, monkeypatch):
    """파일을 옮긴 뒤 DB 를 못 고치면 제자리로 돌린다(함정 13 — 파일만 옮겨지면 뒤이은 prune 이 그 행을 라벨째 지운다)."""
    a = put("a.jpg", "approved")
    before = Path(store.get_image(a)["path"])

    def broken():
        raise RuntimeError("disk full")

    monkeypatch.setattr("autolabel.store._now", broken)       # move_row 가 파일을 옮긴 뒤 경로를 고칠 때 부른다
    report = tidy_approved(cfg, store)
    assert before.exists() and store.get_image(a)["path"] == str(before)
    assert report.moved == 0 and "되돌렸습니다" in report.errors[0]
    assert not list((cfg.labeled / "images").rglob("*.jpg"))


def test_prune_deletes_only_rows_still_at_the_path_it_saw(cfg, store, put):
    a, b = put("a.jpg", "approved"), put("b.jpg", "pending")
    for i in (a, b):
        Path(store.get_image(i)["path"]).unlink()
    keep = put("keep1.jpg", "approved"), put("keep2.jpg", "approved"), put("keep3.jpg", "approved")
    seen = [(i, store.get_image(i)["path"]) for i in (a, b)]
    store.update_path(b, cfg.labeled / "images" / "b.jpg")       # 그새 다른 곳으로 옮겨졌다
    assert store.delete_missing(seen) == 1
    assert store.get_image(a) is None and store.get_image(b) is not None

    Path(store.get_image(keep[0])["path"]).unlink()
    report = prune_missing(cfg, store, max_ratio=1.0)            # b 도 새 경로에 파일이 없다
    assert (report.missing, report.deleted) == (2, 2) and store.get_image(keep[1]) is not None
