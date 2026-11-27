"""작업 배치 (2026-11-23, 사용자 요청 — store 의 work_batches·workbatch.py·server 의 /api/work-batches).

한 번에 넣은 묶음(영상 캡처의 넣기·이미지 넣기·폴더 스캔)이 '작업 N' 이 되고, 그 안에서 검수한 뒤 데이터셋(approved)에 넣는다.
여기서 지키는 것:

- **실제 DB 를 바꾸는 길**: 표 `work_batches` 와 열 `images.work_batch` 는 Store 를 열 때 더한다. 이미지가 든 옛 DB 는 더하기
  **전에** 통째로 떠 두고(사용자 결정 — '직전 사본'), 사본을 못 뜨면 DB 를 바꾸지 않는다. 기존 행은 그대로다.
- **'데이터셋' 은 approved 하나뿐이다**: 새 상태 reviewed('검수 끝')는 학습·분할·내보내기가 읽는 어느 길에도 들어가지 않고,
  한꺼번에 올리는 것(`add_reviewed_to_dataset`)은 reviewed 만 올린다 — 검수 대기·미라벨은 사람이 보지 않은 것이다.
- **배치가 생기는 때**: 새 미라벨 이미지가 있을 때만, 넣기 한 번에 하나. 이번에 옮긴 파일만 그 배치에 들고 같은 스캔에 걸린
  다른 새 파일은 '폴더 스캔' 배치로 따로 묶인다. 시드(곧바로 승인)는 배치에 넣지 않는다.

그림 파일은 작은 JPEG 을 실제로 쓴다(스캔이 열어 본다). 색이 다르면 바이트가 달라 중복 검사에 걸리지 않는다. 장소 이름은
지어낸 것이다.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import pytest
import yaml
from PIL import Image

from autolabel import store as store_mod
from autolabel.ingest import ingest
from autolabel.store import NO_WORK, Store, needs_work_batch_migration
from autolabel.workbatch import Intake, ingest_as_batch
from helpers import CLASSES, box

ORIGIN = "http://127.0.0.1:7860"
CAMERAS = """\
cameras:
  cam-01:
    kind: cctv
    place: 시험 장소 A
  cam-02:
    kind: cctv
    place: 시험 장소 B
"""
# 2026-11-23 전의 images 표 — 실제 DB 가 이 꼴이었다(work_batch 열이 없다)
OLD_IMAGES = """
CREATE TABLE images (
    id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT NOT NULL UNIQUE, name TEXT NOT NULL, width INTEGER NOT NULL,
    height INTEGER NOT NULL, source TEXT NOT NULL, status TEXT NOT NULL, uncertainty REAL NOT NULL DEFAULT 0,
    model_version TEXT, updated_at TEXT NOT NULL, split TEXT, grp TEXT, dhash TEXT, camera TEXT, captured_at TEXT,
    domain TEXT
);
"""


def put(cfg, camera: str | None, name: str, shade: int, sub: str = "") -> Path:
    """미라벨 폴더에 진짜 그림 하나 — shade 가 다르면 내용(바이트)이 다르다."""
    d = cfg.unlabeled / camera if camera else cfg.unlabeled
    if sub:
        d = d / sub
    d.mkdir(parents=True, exist_ok=True)
    path = d / name
    Image.new("RGB", (64, 36), (shade, 255 - shade, shade // 2)).save(path, quality=90)
    return path


def rows(store) -> list[dict]:
    with store.connect() as con:
        return [dict(r) for r in con.execute("SELECT id, name, status, camera, work_batch, updated_at FROM images ORDER BY id")]


@pytest.fixture
def cams(cfg):
    cfg.cameras.write_text(CAMERAS, encoding="utf-8")


# ================================================================= DB — 표와 열 더하기

def test_a_new_db_gets_the_table_and_the_column_and_no_copy(cfg, store):
    with store.connect() as con:
        cols = {r["name"] for r in con.execute("PRAGMA table_info(images)")}
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "work_batch" in cols and "work_batches" in tables
    assert store.migrated_from is None and not (cfg.db.parent / "backups").exists()
    assert Store(cfg.db).migrated_from is None              # 다시 열어도 뜨지 않는다


def old_db(path: Path, n: int = 2) -> None:
    con = sqlite3.connect(path)
    con.executescript(OLD_IMAGES)
    for i in range(n):
        con.execute("INSERT INTO images(path, name, width, height, source, status, updated_at, camera)"
                    " VALUES (?,?,?,?,?,?,?,?)", (f"/x/{i}.jpg", f"{i}.jpg", 64, 36, "collected", "approved",
                                                  "2026-11-22T10:00:00+00:00", "cam-01"))
    con.commit()
    con.close()


def test_an_old_db_is_copied_whole_before_the_column_is_added(cfg):
    """이미지가 든 옛 DB — 열을 더하기 **전에** 사본을 뜬다. 사본은 옛 모양 그대로이고, 원본의 행은 하나도 바뀌지 않는다."""
    old_db(cfg.db)
    assert needs_work_batch_migration(cfg.db) is True
    s = Store(cfg.db)
    copies = sorted((cfg.db.parent / "backups").glob("*.db"))
    assert s.migrated_from == copies[0] and len(copies) == 1
    assert copies[0].name.startswith("labeling_before_workbatch_")
    # 하루 사본의 이름 꼴(<이름>_YYYYMMDD_HHMMSS.db)이 아니다 — keep 정리가 지우지도, '오늘 것은 떴다' 로 치지도 않는다
    assert not list((cfg.db.parent / "backups").glob("labeling_????????_??????.db"))
    con = sqlite3.connect(copies[0])
    assert "work_batch" not in {r[1] for r in con.execute("PRAGMA table_info(images)")}
    assert not con.execute("SELECT 1 FROM sqlite_master WHERE name='work_batches'").fetchone()
    assert con.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 2
    con.close()
    got = rows(s)
    assert [(r["name"], r["status"], r["camera"], r["work_batch"], r["updated_at"]) for r in got] == [
        ("0.jpg", "approved", "cam-01", None, "2026-11-22T10:00:00+00:00"),
        ("1.jpg", "approved", "cam-01", None, "2026-11-22T10:00:00+00:00")]
    assert needs_work_batch_migration(cfg.db) is False
    assert Store(cfg.db).migrated_from is None and len(list((cfg.db.parent / "backups").glob("*.db"))) == 1


def test_an_old_db_without_images_is_not_copied(cfg):
    old_db(cfg.db, n=0)
    assert Store(cfg.db).migrated_from is None and not (cfg.db.parent / "backups").exists()


def test_without_the_copy_the_db_is_left_alone(cfg, monkeypatch):
    """사본을 못 뜨면(디스크가 찼다) 멈춘다 — 열도 표도 더하지 않는다."""
    old_db(cfg.db)

    def full(src, dst):
        raise OSError("디스크가 찼습니다")

    monkeypatch.setattr(store_mod, "copy_db", full)
    with pytest.raises(RuntimeError, match="DB 를 바꾸지 않았습니다"):
        Store(cfg.db)
    con = sqlite3.connect(cfg.db)
    assert "work_batch" not in {r[1] for r in con.execute("PRAGMA table_info(images)")}
    assert not con.execute("SELECT 1 FROM sqlite_master WHERE name='work_batches'").fetchone()
    con.close()


# ================================================================= DB — 배치

def test_batches_count_their_images_by_status_and_camera(store, add_image):
    a = store.create_work_batch("capture", note="clip.mp4")
    b = store.create_work_batch("scan", name="  둘째   묶음 ", hold=True)
    ids = [add_image(f"{i}.jpg", status=st, camera=cam) for i, (st, cam) in enumerate([
        ("unlabeled", "cam-01"), ("pending", "cam-01"), ("reviewed", "cam-02"), ("approved", "cam-01"),
        ("trashed", "cam-02")])]
    other = add_image("x.jpg", status="pending", camera="cam-02")
    loose = add_image("y.jpg", status="approved")
    assert store.set_work_batch(ids, a) == 5 and store.set_work_batch([other], b) == 1
    got = store.list_work_batches()
    assert [r["id"] for r in got] == [b, a]                                  # 새것부터
    first = got[1]
    assert (first["kind"], first["name"], first["note"], first["hold"]) == ("capture", None, "clip.mp4", False)
    assert first["counts"] == {"unlabeled": 1, "pending": 1, "reviewed": 1, "approved": 1, "skipped": 0, "trashed": 1}
    assert first["total"] == 5
    assert first["cameras"] == [{"key": "cam-01", "n": 3}, {"key": "cam-02", "n": 1}]   # 휴지통은 세지 않는다
    assert (got[0]["name"], got[0]["hold"], got[0]["total"]) == ("둘째 묶음", True, 1)
    assert store.get_work_batch(a) == first and store.get_work_batch(999) is None
    assert store.unbatched_counts() == {"unlabeled": 0, "pending": 0, "reviewed": 0, "approved": 1, "skipped": 0,
                                        "trashed": 0, "total": 1}
    assert store.get_image(loose)["work_batch"] is None


def test_the_list_and_the_count_filter_by_batch(store, add_image):
    a, b = store.create_work_batch("scan"), store.create_work_batch("upload")
    in_a = [add_image(f"a{i}.jpg", status=st) for i, st in enumerate(["pending", "pending", "approved"])]
    in_b = [add_image("b0.jpg", status="pending")]
    loose = [add_image("n0.jpg", status="pending")]
    store.set_work_batch(in_a, a)
    store.set_work_batch(in_b, b)
    ids = lambda **kw: sorted(r["id"] for r in store.list_images(limit=100, **kw))
    assert ids(status="all", work=a) == in_a
    assert ids(status="pending", work=a) == in_a[:2]
    assert ids(status="pending", work=str(b)) == in_b                        # 주소에서 온 글자여도 된다
    assert ids(status="all", work=NO_WORK) == loose
    assert ids(status="pending") == sorted(in_a[:2] + in_b + loose)          # 안 걸면 전부
    for kw in ({"status": "pending", "work": a}, {"status": "all", "work": NO_WORK}, {"status": "approved", "work": b}):
        assert store.count_images(**kw) == len(ids(**kw))                    # 목록 묶음이 이 수로 나뉜다
    for order in ("rare", "tag", "name", "recent"):                          # 정렬마다 같은 조건이다
        assert sorted(r["id"] for r in store.list_images(status="all", work=a, order=order, limit=100)) == in_a


def test_kinds_and_names_are_checked(store):
    with pytest.raises(ValueError):
        store.create_work_batch("mystery")
    with pytest.raises(ValueError):
        store.create_work_batch("scan", name="가" * 61)
    i = store.create_work_batch("scan", name="  ")                           # 빈 이름 = 이름 없음
    assert store.get_work_batch(i)["name"] is None
    assert store.update_work_batch(i, name="첫 묶음", set_name=True) is True
    assert store.update_work_batch(i, hold=True) is True                     # 이름은 그대로 두고 방식만
    assert (store.get_work_batch(i)["name"], store.get_work_batch(i)["hold"]) == ("첫 묶음", True)
    assert store.update_work_batch(i, name="", set_name=True) and store.get_work_batch(i)["name"] is None
    with pytest.raises(ValueError):
        store.update_work_batch(i, name="가" * 61, set_name=True)
    assert store.update_work_batch(999, hold=True) is False and store.update_work_batch(999) is False


def test_putting_images_in_a_batch_is_not_an_edit(store, add_image):
    """배치는 라벨이 아니다 — updated_at 을 건드리면 자동 라벨링의 (상태, updated_at) 확인이 '사람이 바꿨다' 로 읽는다."""
    i = add_image("a.jpg", status="unlabeled")
    before = store.get_image(i)
    a = store.create_work_batch("manual")
    assert store.set_work_batch([i, i, 99999], a) == 1                       # 없는 id 는 세지 않는다
    after = store.get_image(i)
    assert after["work_batch"] == a and after["updated_at"] == before["updated_at"]
    assert store.apply_prediction(i, [box()], expect=(before["status"], before["updated_at"]), uncertainty=0.1,
                                  model_version="t") is True
    with pytest.raises(ValueError):
        store.set_work_batch([i], 12345)                                     # 없는 배치
    assert store.set_work_batch([i], None) == 1 and store.get_image(i)["work_batch"] is None


def test_deleting_a_batch_keeps_its_images(store, add_image):
    a = store.create_work_batch("scan")
    i = add_image("a.jpg", status="pending", boxes=[box()])
    store.set_work_batch([i], a)
    assert store.delete_work_batch(a, only_if_empty=True) is False           # 이미지가 있으면 '빈 배치 걷기' 는 지우지 않는다
    assert store.delete_work_batch(a) is True
    got = store.get_image(i)
    assert got["work_batch"] is None and got["status"] == "pending" and len(store.get_boxes(i)) == 1
    assert store.delete_work_batch(a) is False


def test_a_batch_number_is_never_reused(store):
    a = store.create_work_batch("scan")
    store.delete_work_batch(a)
    assert store.create_work_batch("scan") == a + 1          # 열려 있던 화면이 엉뚱한 배치를 가리키지 않게 (AUTOINCREMENT)


# ================================================================= 검수 끝 → 데이터셋

def test_only_reviewed_images_of_that_batch_go_to_the_dataset(store, add_image):
    a, b = store.create_work_batch("scan", hold=True), store.create_work_batch("scan", hold=True)
    done = [add_image(f"d{i}.jpg", status="reviewed", boxes=[box()]) for i in range(2)]
    unseen = [add_image("p.jpg", status="pending", boxes=[box()]), add_image("u.jpg", status="unlabeled"),
              add_image("s.jpg", status="skipped"), add_image("t.jpg", status="trashed")]
    elsewhere = add_image("e.jpg", status="reviewed")
    loose = add_image("l.jpg", status="reviewed")
    store.set_work_batch(done + unseen, a)
    store.set_work_batch([elsewhere], b)
    assert store.add_reviewed_to_dataset(a) == 2
    assert [store.get_image(i)["status"] for i in done] == ["approved", "approved"]
    assert [store.get_image(i)["status"] for i in unseen] == ["pending", "unlabeled", "skipped", "trashed"]
    assert store.get_image(elsewhere)["status"] == "reviewed" and store.get_image(loose)["status"] == "reviewed"
    assert store.add_reviewed_to_dataset(a) == 0                             # 다시 눌러도 더 올라가지 않는다


def test_reviewed_images_are_not_in_the_dataset(cfg, store, add_image):
    """'검수 끝' 은 아직 데이터셋이 아니다 — 학습 세트·분할·내보내기·데이터 점검(승인만)이 읽는 것은 approved 뿐이다."""
    from autolabel.split import ensure_splits

    ok = add_image("ok.jpg", status="approved", boxes=[box()], camera="cam-01")
    held = add_image("held.jpg", status="reviewed", boxes=[box()], camera="cam-02")
    assert [im["id"] for im, _ in store.approved_with_boxes()] == [ok]
    assert [it["id"] for it in store.approved_class_counts()] == [ok]
    assert [im["id"] for im in store.health_snapshot(approved_only=True)["images"]] == [ok]
    ensure_splits(cfg, store)
    assert store.get_image(ok)["split"] is not None and store.get_image(held)["split"] is None
    counts = store.counts()
    assert (counts["reviewed"], counts["approved"], counts["total"]) == (1, 1, 2)
    # 자동 라벨링 '전부 다시'(label --all)가 고르는 것은 미라벨·검수 대기뿐이다 — 검수 끝낸 박스를 예측으로 덮지 않는다
    again = [r["id"] for st in ("unlabeled", "pending") for r in store.list_images(status=st, limit=100)]
    assert held not in again
    store.set_status(held, "reviewed")                                       # 받는 상태다
    with pytest.raises(ValueError):
        store.set_status(held, "done")


# ================================================================= 넣기 — 스캔이 배치를 만든다

def test_a_scan_makes_one_batch_of_the_new_images(cfg, store, cams):
    put(cfg, "cam-01", "a.jpg", 10)
    put(cfg, "cam-02", "b.jpg", 20)
    put(cfg, None, "loose.jpg", 30)                          # 카메라 폴더 없이 둔 것도 새 미라벨이다
    report = ingest_as_batch(cfg, store, "scan")
    batches = store.list_work_batches()
    assert len(batches) == 1 and report.work == {"main": {"id": batches[0]["id"], "n": 3}, "other": None}
    assert (batches[0]["kind"], batches[0]["counts"]["unlabeled"], batches[0]["hold"]) == ("scan", 3, False)
    assert {r["work_batch"] for r in rows(store)} == {batches[0]["id"]}
    assert f"작업 배치 {batches[0]['id']}: 새 이미지 3장" in report.summary()
    # 새 것이 없는 스캔은 배치를 만들지 않는다 — 스캔할 때마다 빈 '작업 N' 이 쌓이지 않게
    again = ingest_as_batch(cfg, store, "scan")
    assert again.work == {"main": None, "other": None} and len(store.list_work_batches()) == 1
    put(cfg, "cam-01", "c.jpg", 40)
    third = ingest_as_batch(cfg, store, "scan")
    assert third.work["main"]["n"] == 1 and third.work["main"]["id"] != batches[0]["id"]
    assert len(store.list_work_batches()) == 2


def test_a_plain_ingest_makes_no_batch(cfg, store, cams):
    put(cfg, "cam-01", "a.jpg", 10)
    report = ingest(cfg, store)
    assert report.work is None and store.list_work_batches() == []
    assert rows(store)[0]["work_batch"] is None


def test_seed_images_stay_out_of_the_batch(cfg, store, cams):
    """시드는 검수 없이 승인으로 들어간다 — 배치(넣고 → 검수하고 → 데이터셋에)에 넣을 것이 아니다."""
    cfg.seed_images.mkdir(parents=True)
    cfg.seed_labels.mkdir(parents=True)
    Image.new("RGB", (64, 36), (200, 10, 10)).save(cfg.seed_images / "s.jpg")
    (cfg.seed_labels / "s.txt").write_text("2 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    put(cfg, "cam-01", "a.jpg", 10)
    report = ingest_as_batch(cfg, store, "scan")
    got = {r["name"]: (r["status"], r["work_batch"]) for r in rows(store)}
    assert got["s.jpg"] == ("approved", None)
    assert got["a.jpg"] == ("unlabeled", report.work["main"]["id"]) and report.work["main"]["n"] == 1


def test_only_the_files_of_this_intake_join_its_batch(cfg, store, cams):
    """영상 캡처·이미지 넣기는 '이번에 옮긴 파일' 만 제 배치에 넣는다. 같은 스캔에 걸린 다른 새 파일(사람이 폴더에 따로 둔 것)은
    '폴더 스캔' 배치로 — 어느 새 이미지도 배치 없이 남지 않는다."""
    mine = [put(cfg, "cam-01", "f1.jpg", 10, "20261123"), put(cfg, "cam-01", "f2.jpg", 20, "20261123")]
    put(cfg, "cam-02", "stray.jpg", 30)
    report = ingest_as_batch(cfg, store, "capture", mine=mine, note="clip.mp4")
    main, other = report.work["main"], report.work["other"]
    assert main["n"] == 2 and other["n"] == 1 and main["id"] != other["id"]
    kinds = {b["id"]: (b["kind"], b["note"]) for b in store.list_work_batches()}
    assert kinds[main["id"]] == ("capture", "clip.mp4") and kinds[other["id"]] == ("scan", None)
    by_name = {r["name"]: r["work_batch"] for r in rows(store)}
    assert by_name == {"f1.jpg": main["id"], "f2.jpg": main["id"], "stray.jpg": other["id"]}
    assert f"작업 배치 {other['id']} (폴더 스캔)" in report.summary()


def test_an_intake_that_registers_nothing_leaves_no_batch(cfg, store, cams):
    """배치는 새 이미지가 처음 나올 때 만든다. 만들었는데 한 장도 못 넣었으면(스캔이 도중에 멈췄다) 걷어 낸다."""
    intake = Intake(store, "upload", mine=[cfg.unlabeled / "cam-01" / "a.jpg"], name="묶음")
    assert store.list_work_batches() == []                                   # 묻기 전에는 만들지 않는다
    made = intake(str(cfg.unlabeled / "cam-01" / "a.jpg"))
    assert [b["id"] for b in store.list_work_batches()] == [made]
    assert intake(str(cfg.unlabeled / "cam-01" / "A.JPG")) == made           # Windows 는 대소문자를 가리지 않는다 — 같은 파일
    assert intake.finish() == {"main": None, "other": None} and store.list_work_batches() == []


def test_a_scan_that_stops_halfway_leaves_no_empty_batch(cfg, store, cams, monkeypatch):
    put(cfg, "cam-01", "a.jpg", 10)

    def broken(items):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(store, "ingest_batch", broken)
    with pytest.raises(sqlite3.OperationalError):
        ingest_as_batch(cfg, store, "scan")
    assert store.list_work_batches() == [] and rows(store) == []


def test_the_cli_scan_makes_a_batch_too(cfg, cams, capsys):
    from autolabel import cli

    cfg.path.write_text(yaml.safe_dump(cfg.raw, allow_unicode=True), encoding="utf-8")
    put(cfg, "cam-01", "a.jpg", 10)
    assert cli.main(["--config", str(cfg.path), "scan"]) == 0
    s = Store(cfg.db)
    batches = s.list_work_batches()
    assert len(batches) == 1 and batches[0]["kind"] == "scan" and batches[0]["total"] == 1
    assert f"작업 배치 {batches[0]['id']}" in capsys.readouterr().out
    assert cli.main(["--config", str(cfg.path), "status"]) == 0
    out = capsys.readouterr().out
    assert "검수 끝" in out and "작업 배치 : 1개 (할 일이 남은 배치 1개)" in out


# ================================================================= 서버

@pytest.fixture
def client(cfg, store):
    from fastapi.testclient import TestClient

    from autolabel.server import create_app

    app = create_app(cfg, label="시험", bind_host="127.0.0.1")
    with TestClient(app, base_url=ORIGIN) as c:
        yield c


def wait_job(client, job: dict) -> dict:
    for _ in range(500):
        got = client.get(f"/api/jobs/{job['id']}").json()
        if got["status"] != "running":
            return got
        threading.Event().wait(0.02)
    raise AssertionError("작업이 끝나지 않았습니다")


def test_the_batch_list_reads_the_camera_file_once(client, store, add_image, cams, monkeypatch):
    """배치가 여럿이어도 등록 파일은 한 번만 읽는다 (2026-11-27, Gemini 리뷰 묶음 8) — 전에는 배치 줄마다 cameras.yaml 을
    다시 읽어, 필터 칸을 채우는 요청 하나가 배치 수만큼 읽었다."""
    import autolabel.server as server

    for k in range(5):
        store.set_work_batch([add_image(f"b{k}.jpg", status="pending", camera="cam-01")], store.create_work_batch("scan"))
    reads: list = []
    real = server.load_cameras
    monkeypatch.setattr(server, "load_cameras", lambda path: (reads.append(path), real(path))[1])
    got = client.get("/api/work-batches").json()["batches"]
    assert len(got) == 5 and all(b["cameras"][0]["place"] == "시험 장소 A" for b in got)
    assert len(reads) == 1


def test_the_api_lists_batches_and_filters_the_image_list(client, store, add_image, cams):
    a = store.create_work_batch("capture", note="clip.mp4")
    in_a = [add_image("a0.jpg", status="pending", camera="cam-01"), add_image("a1.jpg", status="approved", camera="cam-01")]
    loose = add_image("n.jpg", status="pending")
    store.set_work_batch(in_a, a)
    got = client.get("/api/work-batches").json()
    assert got["none_key"] == NO_WORK and got["unbatched"]["total"] == 1 and got["name_max"] == 60
    (b,) = got["batches"]
    assert (b["id"], b["kind"], b["kind_name"], b["note"], b["hold"], b["total"]) == (a, "capture", "영상 캡처", "clip.mp4", False, 2)
    assert b["cameras"] == [{"key": "cam-01", "n": 2, "place": "시험 장소 A"}]     # 장소는 화면에만 쓴다
    assert client.get(f"/api/work-batches/{a}").json()["id"] == a
    assert client.get("/api/work-batches/999").status_code == 404

    listed = lambda q: client.get(f"/api/images?limit=100&{q}").json()
    got = listed(f"status=pending&wb={a}")
    assert [im["id"] for im in got["images"]] == [in_a[0]] and got["total"] == 1 and got["images"][0]["work"] == a
    assert listed(f"status=all&wb={a}")["total"] == 2
    none = listed(f"status=all&wb={NO_WORK}")
    assert [im["id"] for im in none["images"]] == [loose] and none["images"][0]["work"] is None
    assert listed("status=pending")["total"] == 2                            # 안 걸면 전부
    assert listed("status=all&wb=999")["total"] == 0
    assert client.get("/api/images?wb=abc").status_code == 400

    meta = client.get(f"/api/images/{in_a[0]}").json()
    assert meta["work"] == {"id": a, "kind": "capture", "name": None, "hold": False}
    assert client.get(f"/api/images/{loose}").json()["work"] is None


def test_the_api_renames_a_batch_and_switches_how_it_adds(client, store, add_image):
    a = store.create_work_batch("scan")
    i = add_image("a.jpg", status="reviewed")
    store.set_work_batch([i], a)
    r = client.put(f"/api/work-batches/{a}", json={"name": "  첫   묶음 "})
    assert r.status_code == 200 and (r.json()["name"], r.json()["hold"]) == ("첫 묶음", False)
    r = client.put(f"/api/work-batches/{a}", json={"hold": True})
    assert (r.json()["name"], r.json()["hold"]) == ("첫 묶음", True)          # 안 보낸 칸은 그대로
    assert client.get(f"/api/images/{i}").json()["work"]["hold"] is True
    assert client.put(f"/api/work-batches/{a}", json={"name": ""}).json()["name"] is None
    assert client.put(f"/api/work-batches/{a}", json={"name": "가" * 61}).status_code == 400
    assert client.put("/api/work-batches/999", json={"hold": True}).status_code == 404
    # 방식을 바꿔도 이미지의 상태는 그대로다 — 이미 검수 끝인 장은 '데이터셋에 추가' 를 기다린다
    client.put(f"/api/work-batches/{a}", json={"hold": False})
    assert store.get_image(i)["status"] == "reviewed"


def test_the_api_adds_reviewed_images_to_the_dataset(client, store, add_image):
    a = store.create_work_batch("upload", hold=True)
    done = add_image("d.jpg", status="pending", boxes=[box()])
    unseen = add_image("p.jpg", status="pending", boxes=[box()])
    store.set_work_batch([done, unseen], a)
    # 검수 화면의 Space 가 보내는 것 — '모아서 한꺼번에' 배치에서는 검수 끝으로
    r = client.post(f"/api/images/{done}/status", json={"status": "reviewed"})
    assert r.status_code == 200
    state = client.get("/api/state").json()["counts"]
    assert (state["reviewed"], state["pending"], state["approved"]) == (1, 1, 0)
    assert client.get("/api/images?status=reviewed").json()["total"] == 1
    r = client.post(f"/api/work-batches/{a}/add").json()
    assert r["added"] == 1 and r["batch"]["counts"]["approved"] == 1 and r["batch"]["counts"]["pending"] == 1
    assert store.get_image(done)["status"] == "approved" and store.get_image(unseen)["status"] == "pending"
    assert client.post(f"/api/work-batches/{a}/add").json()["added"] == 0
    assert client.post("/api/work-batches/999/add").status_code == 404
    assert client.post(f"/api/images/{done}/status", json={"status": "done"}).status_code == 400


def test_the_scan_job_makes_a_batch(client, cfg, store, cams):
    put(cfg, "cam-01", "a.jpg", 10)
    put(cfg, "cam-01", "b.jpg", 20)
    job = wait_job(client, client.post("/api/jobs/scan").json())
    assert job["status"] == "done", job
    (b,) = store.list_work_batches()
    assert job["result"]["work"] == {"main": {"id": b["id"], "n": 2}, "other": None}
    assert (b["kind"], b["counts"]["unlabeled"]) == ("scan", 2)
    job = wait_job(client, client.post("/api/jobs/scan").json())             # 새 것이 없으면 배치도 없다
    assert job["result"]["work"] == {"main": None, "other": None} and len(store.list_work_batches()) == 1


class FakeLabeler:
    """자동 라벨링 대역 — 모델을 올리지 않고, 무엇을 돌리라고 받았는지만 적는다."""

    asked: list = []

    def __init__(self, cfg, store):
        pass

    def run(self, ids, progress=None, tiles=False):
        from autolabel.predictor import PredictReport

        FakeLabeler.asked.append((None if ids is None else sorted(ids), tiles))
        return PredictReport(processed=len(ids or []))


def test_the_predict_job_takes_only_the_images_of_that_batch(client, store, add_image, monkeypatch):
    monkeypatch.setattr("autolabel.server.AutoLabeler", FakeLabeler)
    FakeLabeler.asked = []
    a, b = store.create_work_batch("scan"), store.create_work_batch("scan")
    fresh = [add_image(f"u{i}.jpg", status="unlabeled") for i in range(2)]
    waiting = add_image("p.jpg", status="pending", boxes=[box()])
    held = add_image("r.jpg", status="reviewed", boxes=[box()])
    okd = add_image("ok.jpg", status="approved", boxes=[box()])
    store.set_work_batch([*fresh, waiting, held, okd], a)
    elsewhere = add_image("e.jpg", status="unlabeled")
    store.set_work_batch([elsewhere], b)
    add_image("loose.jpg", status="unlabeled")

    job = wait_job(client, client.post("/api/jobs/predict", json={"work_batch": a}).json())
    assert job["status"] == "done" and job["result"]["work_batch"] == a and job["result"]["processed"] == 2
    job = wait_job(client, client.post("/api/jobs/predict", json={"work_batch": a, "include_pending": True, "tiles": True}).json())
    assert job["status"] == "done"
    wait_job(client, client.post("/api/jobs/predict", json={}).json())       # 배치를 안 주면 전처럼 미라벨 전부(None)
    assert FakeLabeler.asked == [(fresh, False), (sorted([*fresh, waiting]), True), (None, False)]
    # 검수 끝·승인은 어느 쪽으로도 대상이 아니다 — 사람이 본 박스를 예측으로 덮지 않는다
    assert held not in FakeLabeler.asked[1][0] and okd not in FakeLabeler.asked[1][0]
    # 없는 배치는 작업을 만들기 전에 알린다
    assert client.post("/api/jobs/predict", json={"work_batch": 999}).status_code == 404
    assert len(FakeLabeler.asked) == 3
    assert client.post("/api/jobs/predict", json={"work_batch": 0}).status_code == 422


def test_an_empty_batch_runs_nothing_rather_than_everything(client, store, add_image, monkeypatch):
    """미라벨이 없는 배치의 자동 라벨링 — 대상이 빈 목록이다. None(= 미라벨 전부)으로 새면 다른 배치까지 돌린다."""
    monkeypatch.setattr("autolabel.server.AutoLabeler", FakeLabeler)
    FakeLabeler.asked = []
    a = store.create_work_batch("scan")
    store.set_work_batch([add_image("ok.jpg", status="approved")], a)
    add_image("loose.jpg", status="unlabeled")
    wait_job(client, client.post("/api/jobs/predict", json={"work_batch": a}).json())
    assert FakeLabeler.asked == [([], False)]


def test_a_capture_commit_puts_its_frames_in_one_batch(cfg, store, cams):
    """영상 캡처의 넣기(서버 전체) — 고른 프레임이 '작업 N' 하나가 되고, 폴더에 따로 둔 새 파일은 다른 배치가 된다."""
    from fastapi.testclient import TestClient

    from autolabel.server import create_app

    sid = "20261123_101500_ab12"
    d = cfg.db.parent / "capture" / sid
    (d / "frames").mkdir(parents=True)
    for i in range(3):
        Image.new("RGB", (64, 36), (60 * i, 90, 120)).save(d / "frames" / f"{i:05d}.jpg")
    data = {"created": "2026-11-23T10:15:00", "video": {"name": "clip.mp4", "file": "video.mp4", "sha1": "ab" * 20},
            "state": "done", "error": None, "note": None, "settings": {"every": 1.0},
            "frames": [{"i": i, "t": float(i), "c": 10.0, "s": None, "k": True} for i in range(3)],
            "run": 1, "pos": 3.0, "camera": "cam-01", "start": None, "start_guess": None, "start_mtime": None}
    (d / "session.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    put(cfg, "cam-02", "stray.jpg", 200)

    with TestClient(create_app(cfg, label="시험", bind_host="127.0.0.1"), base_url=ORIGIN) as client:
        r = client.post(f"/api/capture/{sid}/commit", json={"camera": "cam-01", "start": "2026-11-23 12:00:00", "keep": [0, 2]})
        assert r.status_code == 200, r.text
        out = r.json()
        main, other = out["work"]["main"], out["work"]["other"]
        assert (out["moved"], out["registered"], out["other_new"], main["n"], other["n"]) == (2, 2, 1, 2, 1)
        got = {b["id"]: b for b in client.get("/api/work-batches").json()["batches"]}
        assert (got[main["id"]]["kind"], got[main["id"]]["note"]) == ("capture", "clip.mp4")
        assert got[main["id"]]["cameras"] == [{"key": "cam-01", "n": 2, "place": "시험 장소 A"}]
        assert got[other["id"]]["kind"] == "scan"
    by_name = {r["name"]: r["work_batch"] for r in rows(store)}
    assert by_name == {"cam-01_20261123_120000_000.jpg": main["id"], "cam-01_20261123_120002_000.jpg": main["id"],
                       "stray.jpg": other["id"]}


def cut_short_session(cfg, sid: str, n: int, moved: int, t0: str = "2026-11-23 12:00:00") -> list[str]:
    """넣는 도중 서버가 꺼진 세션 — 프레임 n 장 가운데 앞의 moved 장은 이미 unlabeled/cam-01/ 로 옮겨졌고(등록은 안 됨)
    session.json 의 상태는 committing 이다. 넣기가 붙였을 이름 n 개를 돌려준다."""
    from datetime import timedelta

    from autolabel.capture import frame_name, parse_start

    d = cfg.db.parent / "capture" / sid
    (d / "frames").mkdir(parents=True)
    start = parse_start(t0)
    names = [frame_name("cam-01", start + timedelta(seconds=i)) for i in range(n)]
    for i, name in enumerate(names):
        img = Image.new("RGB", (64, 36), (50 * i + 10, 90, 120))
        if i < moved:
            dst = cfg.unlabeled / "cam-01" / f"{start:%Y%m%d}" / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            img.save(dst)
        else:
            img.save(d / "frames" / f"{i:05d}.jpg")
    data = {"created": "2026-11-23T10:15:00", "video": {"name": "clip.mp4", "file": "video.mp4", "sha1": "ab" * 20},
            "state": "committing", "error": None, "note": None, "settings": {"every": 1.0},
            "frames": [{"i": i, "t": float(i), "c": 10.0, "s": None, "k": True} for i in range(n)],
            "run": 1, "pos": float(n), "camera": "cam-01", "start": None, "start_guess": None, "start_mtime": None}
    (d / "session.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return names


@pytest.mark.parametrize("moved", [4, 2])
def test_a_commit_cut_short_is_finished_by_committing_again(cfg, store, cams, moved):
    """넣는 도중 서버가 꺼졌다 — 다시 넣으면 옮겨만 둔 프레임도 이 영상의 배치로 등록된다 (2026-11-25, Gemini 리뷰 묶음 6).

    고치기 전: 전부 옮긴 뒤 꺼졌으면(moved 4) 다시 넣을 때 옮길 것이 없어 스캔을 건너뛰고 세션만 지웠다 — 프레임이 등록도
    배치도 없이 폴더에 남았다. 일부만 옮기고 꺼졌으면(moved 2) 배치에는 4장이 들었는데 결과는 '등록 2장 + 폴더에 있던 다른
    새 이미지 2장' 이었다."""
    from fastapi.testclient import TestClient

    from autolabel.server import create_app

    sid = "20261123_101500_ab12"
    names = cut_short_session(cfg, sid, 4, moved)
    put(cfg, "cam-02", "stray.jpg", 200)                       # 사람이 폴더에 따로 둔 새 파일 — 이것만 '다른 새 이미지' 다
    with TestClient(create_app(cfg, label="시험", bind_host="127.0.0.1"), base_url=ORIGIN) as client:
        r = client.post(f"/api/capture/{sid}/commit",
                        json={"camera": "cam-01", "start": "2026-11-23 12:00:00", "keep": [0, 1, 2, 3]})
        assert r.status_code == 200, r.text
        out = r.json()
    assert "work" in out, out                                  # 스캔을 돌렸다(전부 옮긴 뒤 꺼진 넣기도)
    main, other = out["work"]["main"], out["work"]["other"]
    assert (out["registered"], out["other_new"], main["n"], other["n"]) == (4, 1, 4, 1)
    assert (out["moved"], out["exists"], out["stranded"]) == (4 - moved, moved, moved)
    by_name = {r["name"]: r["work_batch"] for r in rows(store)}
    assert by_name == {**{n: main["id"] for n in names}, "stray.jpg": other["id"]}
    assert not (cfg.db.parent / "capture" / sid).exists()


def test_frames_that_are_already_registered_do_not_start_a_scan(cfg, store, cams):
    """같은 영상을 두 번 넣었다 — 같은 이름이 전부 이미 등록돼 있으면 옮길 것도 등록할 것도 없다. 스캔하지 않는다(전과 같다:
    폴더에 따로 둔 새 파일을 이 넣기가 끌고 들어가지 않는다)."""
    from fastapi.testclient import TestClient

    from autolabel.server import create_app

    names = cut_short_session(cfg, "20261123_101500_ab12", 3, 3)
    ingest_as_batch(cfg, store, "scan")                        # 앞서 폴더 스캔으로 등록해 둔 프레임
    assert sorted(r["name"] for r in rows(store)) == names
    cut = cfg.db.parent / "capture" / "20261123_101500_ab12"
    for i in range(3):                                         # 같은 영상을 다시 뽑았다 — 프레임 파일이 세션에 다시 있다
        Image.new("RGB", (64, 36), (50 * i + 10, 90, 120)).save(cut / "frames" / f"{i:05d}.jpg")
    put(cfg, "cam-02", "stray.jpg", 200)
    with TestClient(create_app(cfg, label="시험", bind_host="127.0.0.1"), base_url=ORIGIN) as client:
        out = client.post("/api/capture/20261123_101500_ab12/commit",
                          json={"camera": "cam-01", "start": "2026-11-23 12:00:00", "keep": [0, 1, 2]}).json()
    assert (out["moved"], out["exists"], out["stranded"]) == (0, 3, 0)
    assert "summary" not in out and "work" not in out and "scan_error" not in out
    assert sorted(r["name"] for r in rows(store)) == names     # stray.jpg 는 그대로 미등록


def test_classes_are_untouched_by_all_this(store):
    assert store.get_classes() == CLASSES
