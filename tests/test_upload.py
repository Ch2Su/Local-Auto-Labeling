"""웹 이미지 올리기 (upload.py·server 의 /api/upload, 2026-11-23 — 작업 배치의 '＋ 이미지 넣기').

브라우저가 고른 이미지를 임시 폴더에 받아 두었다가 '넣기' 때만 `unlabeled/<카메라>/up<시각>/` 로 옮겨 스캔한다. 지키는 것:

- **넣기 전에는 unlabeled/ 에 아무것도 없다** — 올리다 만 묶음이 폴더 스캔에 걸려 엉뚱한 배치·출처로 들어가지 않는다.
- **출처(카메라) 없이는 넣지 않는다** — 등록 안 된 카메라는 파일을 받기 전에 거절하고, 카메라를 비우면 이 묶음용 카메라를
  등록한다(`vid-NN`). 한 장도 못 넣으면 그 등록을 되돌린다.
- **올린 이름을 그대로 경로로 쓰지 않는다** — 경로 조각·Windows 가 못 쓰는 글자를 떼고 세션 폴더 안에만 쓴다.
- 이미지가 아닌 것·열리지 않는 것·크기가 다른 것은 받지 않는다. 내용이 이미 있는 이미지와 같은 파일은 옮기지 않는다.
- 넣은 파일이 작업 배치 하나가 된다(이름을 줬으면 그 이름으로).

그림은 작은 JPEG 을 메모리에서 만들어 보낸다. 장소 이름은 지어낸 것이다.
"""

from __future__ import annotations

import asyncio
import io
import sqlite3
from pathlib import Path

import pytest
from PIL import Image

from autolabel import upload
from autolabel.provenance import load_cameras
from autolabel.upload import UploadError, UploadManager, clean_name
from helpers import StreamBody, post_endpoint

ORIGIN = "http://127.0.0.1:7860"
CAMERAS = """\
cameras:
  cam-01:
    kind: cctv
    place: 시험 장소 A
"""


def jpeg(shade: int, size=(64, 36)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (shade, 255 - shade, shade // 2)).save(buf, "JPEG", quality=90)
    return buf.getvalue()


@pytest.fixture
def cams(cfg):
    cfg.cameras.write_text(CAMERAS, encoding="utf-8")


@pytest.fixture
def client(cfg, store, cams):
    from fastapi.testclient import TestClient

    from autolabel.server import create_app

    app = create_app(cfg, label="시험", bind_host="127.0.0.1")
    with TestClient(app, base_url=ORIGIN) as c:
        yield c


def send(client, sid: str, name: str, data: bytes, size: int | None = -1):
    q = {"name": name}
    if size != -1:
        if size is not None:
            q["size"] = size
    else:
        q["size"] = len(data)
    return client.post(f"/api/upload/{sid}/file", params=q, content=data)


def begin(client, camera="cam-01", name=None) -> str:
    r = client.post("/api/upload", json={"camera": camera, "name": name})
    assert r.status_code == 200, r.text
    return r.json()["sid"]


def images(store) -> list[dict]:
    with store.connect() as con:
        return [dict(r) for r in con.execute(
            "SELECT id, path, name, status, source, camera, work_batch, width, height FROM images ORDER BY id")]


# ================================================================= 이름 다듬기

@pytest.mark.parametrize("sent, kept", [
    ("a.jpg", "a.jpg"),
    ("IMG_0001.JPG", "IMG_0001.JPG"),                        # 확장자 대소문자는 그대로 둔다(스캔이 가리지 않는다)
    ("폴더/하위/프레임 1.png", "프레임 1.png"),               # 폴더째 올리면 브라우저가 경로를 준다
    ("..\\..\\Windows\\evil.jpg", "evil.jpg"),
    ("../../etc/passwd.jpg", "passwd.jpg"),
    ("C:\\Users\\x\\b.webp", "b.webp"),
    ('a<b>:"c|d?e*.jpg', "a_b___c_d_e_.jpg"),                # Windows 가 파일 이름에 못 쓰는 글자
    ("tab\tname\x00.jpg", "tab_name_.jpg"),
    ("  spaced .jpg", "spaced.jpg"),
    ("trailing.. .jpeg. ", "trailing.jpeg"),
    ("CON.jpg", "_CON.jpg"),                                 # Windows 의 장치 이름
    ("hw-01_20261123_120000_000.jpg", "hw-01_20261123_120000_000.jpg"),     # 이름의 시각은 건드리지 않는다
    ("한글.jpg", "한글.jpg"),         # macOS 가 자모로 풀어 준 이름 — NFC 로 모은다
])
def test_names_are_cleaned(sent, kept):
    assert clean_name(sent) == kept


@pytest.mark.parametrize("sent", ["notes.txt", "noext", "", ".jpg.exe", "a.heic", "a.jpg/", "movie.mp4", "x.svg", "a."])
def test_other_formats_are_refused(sent):
    with pytest.raises(UploadError) as err:
        clean_name(sent)
    assert err.value.status == 415


def test_a_name_made_only_of_an_extension_still_gets_a_stem():
    assert clean_name(".jpg") == "image.jpg" and clean_name("...jpg") == "image.jpg"
    assert len(clean_name("가" * 400 + ".jpg")) == upload.NAME_MAX + 4


# ================================================================= 받기

def test_files_wait_outside_unlabeled_until_commit(client, cfg, store):
    sid = begin(client)
    for i, name in enumerate(["a.jpg", "b.jpg"]):
        r = send(client, sid, name, jpeg(10 + i))
        assert r.status_code == 200 and r.json()["name"] == name and (r.json()["width"], r.json()["height"]) == (64, 36)
    staged = sorted(p.name for p in (cfg.db.parent / "upload" / sid / "files").iterdir())
    assert staged == ["a.jpg", "b.jpg"]
    assert not cfg.unlabeled.exists() or not any(cfg.unlabeled.rglob("*.jpg"))      # 아직 아무것도 안 갔다
    assert images(store) == [] and store.list_work_batches() == []
    got = client.get("/api/upload").json()
    assert [(s["sid"], s["n"], s["camera"], s["state"]) for s in got["sessions"]] == [(sid, 2, "cam-01", "open")]
    assert ".jpg" in got["limits"]["exts"] and got["limits"]["max_files"] == upload.MAX_FILES


def test_bad_files_are_refused_one_by_one(client, cfg):
    sid = begin(client)
    assert send(client, sid, "notes.txt", b"hello").status_code == 415
    r = send(client, sid, "broken.jpg", b"this is not a jpeg at all")
    assert r.status_code == 415 and "읽을 수 없습니다" in r.json()["detail"]
    assert send(client, sid, "short.jpg", jpeg(10), size=len(jpeg(10)) + 5).status_code == 400    # 받은 크기가 다르다
    assert send(client, sid, "empty.jpg", b"", size=0).status_code == 400
    assert send(client, sid, "huge.jpg", jpeg(10), size=upload.MAX_FILE_BYTES + 1).status_code == 413
    assert send(client, sid, "ok.jpg", jpeg(20)).status_code == 200
    files = cfg.db.parent / "upload" / sid / "files"
    assert sorted(p.name for p in files.iterdir()) == ["ok.jpg"]                    # 거절한 것은 남기지 않는다
    assert client.post(f"/api/upload/{sid}/file", params={"name": "x.jpg"}, content=jpeg(30)).status_code == 200   # 크기를 안 줘도 받는다


def test_an_uploaded_name_never_leaves_the_session_folder(client, cfg):
    sid = begin(client)
    for i, name in enumerate(["../../../evil.jpg", "..\\..\\evil2.jpg", "C:\\Windows\\evil3.jpg", "/etc/evil4.jpg"]):
        assert send(client, sid, name, jpeg(10 + i)).status_code == 200
    root = cfg.db.parent / "upload" / sid / "files"
    assert sorted(p.name for p in root.iterdir()) == ["evil.jpg", "evil2.jpg", "evil3.jpg", "evil4.jpg"]
    everything = [p for p in cfg.db.parent.rglob("evil*")]
    assert all(root in p.parents for p in everything)
    # 세션 번호도 정해진 꼴만 받는다
    for bad in ("20261123", "abc", "20261123_101500_ZZZZ", "20261123_101500_ab12x"):
        assert client.post(f"/api/upload/{bad}/file", params={"name": "a.jpg"}, content=jpeg(1)).status_code == 404
        assert client.post(f"/api/upload/{bad}/commit").status_code == 404
        assert client.delete(f"/api/upload/{bad}").status_code == 404
    # 빗금을 숨겨 보낸 번호(%2F)는 경로가 달라져 이 주소에 닿지도 않는다 — 어느 쪽이든 받지 않는다
    assert client.post("/api/upload/..%2F..%2Fx/file", params={"name": "a.jpg"}, content=jpeg(1)).status_code in (404, 405)
    assert sorted(p.name for p in cfg.db.parent.rglob("a.jpg")) == []


def test_the_same_name_twice_in_one_bundle_gets_a_number(client, cfg):
    """폴더째 올리면 다른 하위 폴더의 같은 이름이 온다 — 덮지 않고 번호를 붙인다."""
    sid = begin(client)
    names = [send(client, sid, n, jpeg(10 + i)).json()["name"] for i, n in enumerate(["x/frame.jpg", "y/frame.jpg", "z/FRAME.JPG"])]
    assert names == ["frame.jpg", "frame_2.jpg", "FRAME_3.JPG"]                     # Windows 는 대소문자를 가리지 않는다


# ================================================================= 넣기

def test_commit_moves_the_files_and_makes_one_batch(client, cfg, store):
    sid = begin(client, name="  현장 사진  ")
    for i, name in enumerate(["a.jpg", "b.jpg", "hw_20261123_120000_000.jpg"]):
        assert send(client, sid, name, jpeg(10 + i)).status_code == 200
    r = client.post(f"/api/upload/{sid}/commit")
    assert r.status_code == 200, r.text
    out = r.json()
    main = out["work"]["main"]
    assert (out["camera"], out["auto"], out["moved"], out["duplicates"], out["registered"], main["n"]) == ("cam-01", None, 3, 0, 3, 3)
    dest = cfg.unlabeled / "cam-01" / f"up{sid[:15]}"
    assert sorted(p.name for p in dest.iterdir()) == ["a.jpg", "b.jpg", "hw_20261123_120000_000.jpg"]
    assert not (cfg.db.parent / "upload" / sid).exists()                            # 넣으면 임시 폴더를 지운다
    got = images(store)
    assert {(im["status"], im["source"], im["camera"], im["work_batch"], im["width"]) for im in got} == {
        ("unlabeled", "collected", "cam-01", main["id"], 64)}
    assert all(Path(im["path"]).parent == dest.resolve() for im in got)
    (b,) = store.list_work_batches()
    assert (b["id"], b["kind"], b["name"], b["hold"], b["counts"]["unlabeled"]) == (main["id"], "upload", "현장 사진", False, 3)
    with store.connect() as con:                                                    # 이름의 시각은 촬영 시각으로 읽힌다
        when = con.execute("SELECT captured_at FROM images WHERE name LIKE 'hw_%'").fetchone()[0]
    assert when.startswith("2026-11-23T12:00:00")
    assert client.post(f"/api/upload/{sid}/commit").status_code == 404              # 넣은 묶음은 없어졌다
    assert client.get("/api/upload").json()["sessions"] == []


def test_one_image_is_a_bundle_too(client, store):
    sid = begin(client)
    send(client, sid, "single.jpg", jpeg(10))
    out = client.post(f"/api/upload/{sid}/commit").json()
    assert out["work"]["main"]["n"] == 1 and len(images(store)) == 1


def test_a_camera_is_required_and_must_be_registered(client, cfg, store):
    r = client.post("/api/upload", json={"camera": "cam-99"})
    assert r.status_code == 400 and "cam-99" in r.json()["detail"]
    assert not (cfg.db.parent / "upload").exists() or not any((cfg.db.parent / "upload").iterdir())
    assert client.post("/api/upload", json={"camera": "cam-01", "name": "가" * 61}).status_code == 400
    # 받는 사이 그 카메라가 등록에서 빠졌다 — 넣지 않는다(파일은 임시 폴더에 그대로)
    sid = begin(client)
    send(client, sid, "a.jpg", jpeg(10))
    cfg.cameras.write_text("cameras:\n  cam-02:\n    kind: cctv\n    place: 시험 장소 B\n", encoding="utf-8")
    r = client.post(f"/api/upload/{sid}/commit")
    assert r.status_code == 400 and "그사이 지워졌습니다" in r.json()["detail"]
    assert images(store) == [] and not cfg.unlabeled.exists()
    assert (cfg.db.parent / "upload" / sid / "files" / "a.jpg").exists()
    assert client.delete(f"/api/upload/{sid}").json() == {"ok": True, "left": None}
    assert not (cfg.db.parent / "upload" / sid).exists()


def test_a_bundle_follows_its_camera_when_the_key_is_renamed(client, cfg, store):
    """카메라 키를 바꾸면 넣기 전인 묶음도 새 키로 간다 (2026-11-27, Gemini 리뷰 묶음 8) — 전에는 영상 캡처 세션만 따라가고
    올리기 묶음은 옛 키로 남아, 넣을 때 '그사이 지워졌습니다'(400)로 막혔다(묶음의 카메라를 바꾸는 길은 없다)."""
    sid = begin(client)
    send(client, sid, "a.jpg", jpeg(10))
    r = client.put("/api/cameras/cam-01", json={"new_key": "cam-07"})
    assert r.status_code == 200 and r.json()["key"] == "cam-07", r.text
    assert [s["camera"] for s in client.get("/api/upload").json()["sessions"]] == ["cam-07"]
    done = client.post(f"/api/upload/{sid}/commit")
    assert done.status_code == 200, done.text
    assert [(im["name"], im["camera"]) for im in images(store)] == [("a.jpg", "cam-07")]


def test_a_camera_picked_by_a_waiting_bundle_cannot_be_removed(client, cfg, store):
    """등록을 지우면 그 묶음은 넣을 수 없다 — 묶음을 넣거나 버리기 전에는 지우기를 막는다(409, 아무것도 안 지움)."""
    sid = begin(client)
    send(client, sid, "a.jpg", jpeg(10))
    before = cfg.cameras.read_bytes()
    r = client.post("/api/cameras/remove", json={"keys": ["cam-01"]})
    assert r.status_code == 409 and "cam-01(1묶음)" in r.json()["detail"]
    assert cfg.cameras.read_bytes() == before
    assert client.delete(f"/api/upload/{sid}").status_code == 200          # 버리면 지울 수 있다
    assert client.post("/api/cameras/remove", json={"keys": ["cam-01"]}).json()["removed"] == ["cam-01"]


def test_without_a_camera_one_is_registered_for_the_bundle(client, cfg, store):
    """카메라를 모르면 비워 둔다 — 이 묶음용 카메라를 다음 번호로 등록해 넣는다(분할이 카메라째라 출처 없이는 넣지 않는다)."""
    sid = begin(client, camera=None)
    assert "vid-01" not in load_cameras(cfg.cameras)                                # 등록은 넣을 때 한다
    send(client, sid, "first.jpg", jpeg(10))
    send(client, sid, "second.jpg", jpeg(20))
    out = client.post(f"/api/upload/{sid}/commit").json()
    assert (out["camera"], out["auto"], out["moved"]) == ("vid-01", "new", 2)
    cam = load_cameras(cfg.cameras)["vid-01"]
    assert cam.kind == "cctv" and cam.place == "모름 — 이미지 묶음 first.jpg 외 1장"
    assert {im["camera"] for im in images(store)} == {"vid-01"}
    assert "cam-01" in load_cameras(cfg.cameras)                                    # 있던 등록은 그대로
    # 다음 묶음은 다음 번호
    sid = begin(client, camera="")
    send(client, sid, "third.jpg", jpeg(30))
    assert client.post(f"/api/upload/{sid}/commit").json()["camera"] == "vid-02"


def test_a_discarded_bundle_leaves_no_camera_and_no_files(client, cfg, store):
    sid = begin(client, camera=None)
    send(client, sid, "a.jpg", jpeg(10))
    assert client.delete(f"/api/upload/{sid}").status_code == 200
    assert list(load_cameras(cfg.cameras)) == ["cam-01"] and images(store) == []
    assert not cfg.unlabeled.exists() and client.delete(f"/api/upload/{sid}").status_code == 404


def test_files_already_in_the_data_are_not_moved(client, cfg, store):
    """내용(바이트)이 이미 있는 이미지와 같은 파일 — 옮기지 않는다. 옮긴 뒤에 스캔이 막으면 등록 안 된 사본이 unlabeled/ 에
    남아 스캔할 때마다 '중복' 으로 다시 뜬다."""
    sid = begin(client)
    send(client, sid, "a.jpg", jpeg(10))
    first = client.post(f"/api/upload/{sid}/commit").json()
    old = images(store)[0]

    sid = begin(client)
    send(client, sid, "again.jpg", jpeg(10))                 # 같은 그림, 다른 이름
    send(client, sid, "new.jpg", jpeg(20))
    send(client, sid, "new_copy.jpg", jpeg(20))              # 같은 묶음 안의 같은 그림
    out = client.post(f"/api/upload/{sid}/commit").json()
    assert (out["moved"], out["duplicates"], out["registered"]) == (1, 2, 1)
    dups = {d["name"]: d["same_as"] for d in out["duplicate_names"]}
    assert dups == {"again.jpg": old["id"], "new_copy.jpg": None}
    on_disk = sorted(p.name for p in cfg.unlabeled.rglob("*.jpg"))
    assert on_disk == ["a.jpg", "new.jpg"]                                          # 겹친 둘은 unlabeled 로 가지 않았다
    assert out["work"]["main"]["id"] != first["work"]["main"]["id"]

    # 전부 이미 있는 그림이면 옮길 것도, 만들 배치도, 등록할 카메라도 없다
    sid = begin(client, camera=None)
    send(client, sid, "same.jpg", jpeg(10))
    out = client.post(f"/api/upload/{sid}/commit").json()
    assert (out["moved"], out["duplicates"], out["camera"], out["auto"]) == (0, 1, None, None)
    assert "work" not in out and len(store.list_work_batches()) == 2 and list(load_cameras(cfg.cameras)) == ["cam-01"]
    assert not (cfg.db.parent / "upload" / sid).exists()


def test_an_empty_bundle_is_not_committed(client):
    sid = begin(client)
    assert client.post(f"/api/upload/{sid}/commit").status_code == 400
    assert client.delete(f"/api/upload/{sid}").status_code == 200


def test_nothing_is_added_while_files_are_being_tidied(client, cfg, store, monkeypatch):
    """정리·휴지통 비우기가 도는 동안은 넣지 않는다(파일을 옮기는 중이다 — 영상 캡처의 넣기와 같은 약속)."""
    import threading

    started, release = threading.Event(), threading.Event()

    def slow_tidy(*args, **kwargs):
        started.set()
        release.wait(10)
        raise RuntimeError("시험용 정리 — 아무것도 하지 않는다")

    sid = begin(client)
    send(client, sid, "a.jpg", jpeg(10))
    monkeypatch.setattr("autolabel.server.tidy_approved", slow_tidy)
    job = client.post("/api/jobs/tidy").json()
    try:
        assert started.wait(5)
        r = client.post(f"/api/upload/{sid}/commit")
        assert r.status_code == 409 and "정리" in r.json()["detail"]
        assert client.post("/api/upload", json={"camera": "cam-01"}).status_code == 409   # 새 묶음도 시작하지 않는다
        assert images(store) == [] and not cfg.unlabeled.exists()
    finally:
        release.set()
    for _ in range(300):
        if client.get(f"/api/jobs/{job['id']}").json()["status"] != "running":
            break
        threading.Event().wait(0.02)
    assert client.post(f"/api/upload/{sid}/commit").json()["moved"] == 1              # 끝난 뒤에는 들어간다


# ================================================================= 관리자 (서버 없이)

def manager(cfg, asked: list | None = None, **kw) -> UploadManager:
    def ingest(mine=None, note=None, name=None):
        if asked is not None:
            asked.append((sorted(p.name for p in mine), note, name))
        raise RuntimeError("시험에서는 스캔하지 않는다")      # commit 은 scan_error 로 적고 계속한다(파일은 이미 옮겼다)

    return UploadManager(cfg, ingest=ingest, **kw)


def stage(man: UploadManager, sid: str, name: str, data: bytes) -> str:
    s, path, final = man.reserve(sid, name, len(data))
    path.write_bytes(data)
    man.accept(s, path, len(data), len(data))
    return final


def test_starting_the_server_clears_bundles_that_were_never_added(cfg, cams):
    """서버를 다시 켜면 넣지 않은 묶음은 지운다 — 사본일 뿐이다. 그 이름 꼴의 폴더만 지운다."""
    root = cfg.db.parent / "upload"
    (root / "20261123_101500_ab12" / "files").mkdir(parents=True)
    (root / "20261123_101500_ab12" / "files" / "a.jpg").write_bytes(jpeg(10))
    (root / "내 폴더").mkdir()
    (root / "내 폴더" / "keep.jpg").write_bytes(jpeg(20))
    man = manager(cfg)
    assert not (root / "20261123_101500_ab12").exists() and (root / "내 폴더" / "keep.jpg").exists()
    assert man.list() == []


def test_a_move_that_stops_halfway_can_be_finished_by_committing_again(cfg, cams, monkeypatch):
    """옮기다 멈췄다(디스크·권한) — 묶음은 열린 채로 남고, 다시 넣으면 남은 것만 옮기며 앞에서 옮긴 것도 같은 배치로 넘긴다."""
    asked: list = []
    man = manager(cfg, asked)
    sid = man.begin("cam-01", "묶음")["sid"]
    for i, name in enumerate(["a.jpg", "b.jpg", "c.jpg"]):
        stage(man, sid, name, jpeg(10 + i))
    real, moved = upload._rename, []

    def failing(src, dst):
        if len(moved) == 1:
            raise PermissionError("다른 프로그램이 쓰는 중")
        real(src, dst)
        moved.append(dst)

    monkeypatch.setattr(upload, "_rename", failing)
    with pytest.raises(UploadError) as err:
        man.commit(sid)
    assert err.value.status == 500 and "1장을 옮긴 뒤 멈췄습니다" in str(err.value)
    assert man.get(sid).state == "open" and asked == []                             # 스캔은 부르지 않았다
    dest = cfg.unlabeled / "cam-01" / f"up{sid[:15]}"
    assert sorted(p.name for p in dest.iterdir()) == ["a.jpg"]

    monkeypatch.setattr(upload, "_rename", real)
    out = man.commit(sid)
    assert (out["moved"], out["missing"], out["exists"]) == (2, 0, 0) and "scan_error" in out
    assert sorted(p.name for p in dest.iterdir()) == ["a.jpg", "b.jpg", "c.jpg"]
    assert asked == [(["a.jpg", "b.jpg", "c.jpg"], "이미지 3장", "묶음")]            # 앞에서 옮긴 a 도 이번 묶음의 파일이다
    assert not (cfg.db.parent / "upload" / sid).exists()


def test_an_auto_camera_is_taken_back_when_nothing_could_be_moved(cfg, cams, monkeypatch):
    man = manager(cfg)
    sid = man.begin(None)["sid"]
    stage(man, sid, "a.jpg", jpeg(10))

    def failing(src, dst):
        raise PermissionError("다른 프로그램이 쓰는 중")

    monkeypatch.setattr(upload, "_rename", failing)
    with pytest.raises(UploadError):
        man.commit(sid)
    assert list(load_cameras(cfg.cameras)) == ["cam-01"]                            # 빈 등록을 남기지 않는다
    assert man.get(sid).state == "open"


def test_a_bundle_has_a_size_limit(cfg, cams, monkeypatch):
    monkeypatch.setattr(upload, "MAX_FILES", 2)
    man = manager(cfg)
    sid = man.begin("cam-01")["sid"]
    stage(man, sid, "a.jpg", jpeg(10))
    stage(man, sid, "b.jpg", jpeg(20))
    with pytest.raises(UploadError) as err:
        man.reserve(sid, "c.jpg", 10)
    assert err.value.status == 413


def test_a_closed_bundle_takes_no_more_files(cfg, cams):
    man = manager(cfg)
    sid = man.begin("cam-01")["sid"]
    s, path, _ = man.reserve(sid, "late.jpg", 100)
    man.discard(sid)                                         # 쓰는 사이 버렸다
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(jpeg(10))
    with pytest.raises(UploadError) as err:
        man.accept(s, path, len(jpeg(10)), None)
    assert err.value.status == 409 and not path.exists()     # 늦게 온 파일은 남기지 않는다
    with pytest.raises(UploadError):
        man.reserve(sid, "x.jpg", 10)


def test_the_db_is_untouched_by_staging(cfg, store, cams):
    man = manager(cfg)
    sid = man.begin("cam-01")["sid"]
    stage(man, sid, "a.jpg", jpeg(10))
    con = sqlite3.connect(cfg.db)
    assert con.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM work_batches").fetchone()[0] == 0
    con.close()


# ================================================================= 받는 중인 파일과 넣기 (2026-11-25, Gemini 리뷰 묶음 6)

def test_a_bundle_is_not_committed_while_a_file_is_still_arriving(cfg, cams):
    """파일 하나를 아직 쓰는 중에 '넣기' 가 왔다 — 넣지 않는다(409). 고치기 전에는 다 받은 것만 옮기고 세션 폴더를 지워, 쓰던
    파일은 묶음에서 빠지고 그 요청은 409 로 끝났다."""
    asked: list = []
    man = manager(cfg, asked)
    sid = man.begin("cam-01")["sid"]
    stage(man, sid, "a.jpg", jpeg(10))
    s, path, _ = man.reserve(sid, "b.jpg", len(jpeg(20)))      # 자리를 잡고 쓰는 중
    assert man.get(sid).summary()["receiving"] == 1
    with pytest.raises(UploadError) as err:
        man.commit(sid)
    assert err.value.status == 409 and "받는 중인 파일이 1개" in str(err.value)
    assert man.get(sid).state == "open" and asked == []
    assert not (cfg.unlabeled / "cam-01").exists()             # 하나도 옮기지 않았다

    path.write_bytes(jpeg(20))
    man.accept(s, path, len(jpeg(20)), len(jpeg(20)))          # 다 받았다
    assert man.get(sid).summary()["receiving"] == 0
    out = man.commit(sid)
    assert out["moved"] == 2 and asked[0][0] == ["a.jpg", "b.jpg"]


def test_a_refused_upload_does_not_keep_the_bundle_from_being_committed(cfg, cams):
    """받다 만 파일(끊김·형식 오류)은 자리를 돌려준다 — '받는 중' 으로 남아 넣기를 막지 않는다."""
    man = manager(cfg)
    sid = man.begin("cam-01")["sid"]
    stage(man, sid, "a.jpg", jpeg(10))
    s, path, _ = man.reserve(sid, "cut.jpg", 100)
    man.reject(s, path)                                        # 올리기가 끊겼다
    s, path, _ = man.reserve(sid, "text.jpg", 5)
    path.write_bytes(b"hello")
    with pytest.raises(UploadError):
        man.accept(s, path, 5, 5)                              # 이미지가 아니다
    assert man.get(sid).summary()["receiving"] == 0
    assert man.commit(sid)["moved"] == 1


def test_a_file_that_finishes_just_before_the_commit_starts_is_in_it(cfg, cams):
    """넣기가 확인(정리 중인가·카메라)을 하는 사이에 다 받은 파일 — 받았다는 응답이 나갔으니 묶음에 든다. 고치기 전에는 넣을
    목록을 확인 **앞**에서 떠 두어 이 파일이 목록에 없었고, 세션 폴더와 함께 지워졌다."""
    asked: list = []
    armed: list = []                                           # 묶음을 시작할 때(begin)도 blocker 를 묻는다 — 넣기 때만 끼운다

    def blocker():
        if armed:                                              # 넣기의 첫 확인과 상태 바꾸기 사이
            stage(man, armed.pop(), "b.jpg", jpeg(20))
        return None

    man = manager(cfg, asked, blocker=blocker)
    sid = man.begin("cam-01")["sid"]
    stage(man, sid, "a.jpg", jpeg(10))
    armed.append(sid)
    out = man.commit(sid)
    assert out["moved"] == 2 and asked[0][0] == ["a.jpg", "b.jpg"]
    dest = cfg.unlabeled / "cam-01" / f"up{sid[:15]}"
    assert sorted(p.name for p in dest.iterdir()) == ["a.jpg", "b.jpg"]


def test_a_file_that_starts_arriving_just_before_the_commit_starts_stops_it(cfg, cams):
    armed: list = []

    def blocker():
        if armed:
            man.reserve(armed.pop(), "b.jpg", 100)             # 확인하는 사이 새 파일을 받기 시작했다
        return None

    man = manager(cfg, blocker=blocker)
    sid = man.begin("cam-01")["sid"]
    stage(man, sid, "a.jpg", jpeg(10))
    armed.append(sid)
    with pytest.raises(UploadError) as err:
        man.commit(sid)
    assert err.value.status == 409 and "받는 중" in str(err.value)
    assert man.get(sid).state == "open" and not (cfg.unlabeled / "cam-01").exists()


def test_a_file_turned_away_by_a_busy_bundle_gives_its_name_back(cfg, cams):
    """넣는 중인 묶음에 늦게 닿은 파일 — 지우고, 잡아 둔 이름도 돌려준다. 고치기 전에는 파일만 지워서, 넣기가 실패해 묶음이
    다시 열리면 그 이름이 예약된 채였다(같은 이름을 다시 올리면 `late_2.jpg` 가 됐다)."""
    man = manager(cfg)
    sid = man.begin("cam-01")["sid"]
    s, path, final = man.reserve(sid, "late.jpg", len(jpeg(10)))
    path.write_bytes(jpeg(10))
    s.state = "committing"                                     # 그사이 넣기가 시작됐다
    with pytest.raises(UploadError) as err:
        man.accept(s, path, len(jpeg(10)), len(jpeg(10)))
    assert err.value.status == 409 and not path.exists()
    assert not s.reserved and not s.inflight
    s.state = "open"                                           # 넣기가 실패해 묶음이 다시 열렸다
    assert man.reserve(sid, "late.jpg", 10)[2] == final == "late.jpg"


@pytest.mark.parametrize("fail", [RuntimeError("뜻밖의 오류"), asyncio.CancelledError()])
def test_an_upload_that_dies_midway_gives_its_place_back(client, fail):
    """받는 도중 요청이 취소되거나(서버를 끄는 중) 뜻밖의 오류가 났다 — 자리를 돌려준다. 안 돌려주면 그 이름이 '받는 중' 으로
    남아 이 묶음을 넣을 수 없다."""
    sid = begin(client)
    assert send(client, sid, "a.jpg", jpeg(10)).status_code == 200
    fn = post_endpoint(client.app, "/api/upload/{sid}/file")
    with pytest.raises(type(fail)):
        asyncio.run(fn(StreamBody([jpeg(20)[:100]], fail), sid=sid, name="b.jpg", size=len(jpeg(20))))
    bundle = client.get("/api/upload").json()["sessions"][0]
    assert (bundle["n"], bundle["receiving"]) == (1, 0)
    assert client.post(f"/api/upload/{sid}/commit").json()["moved"] == 1
