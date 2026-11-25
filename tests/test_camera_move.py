"""고른 이미지만 다른 카메라로 옮기기 (camera_move.py, 2026-11-15) — 파일 이름·폴더·DB 를 함께 바꾸고, 막히면 아무것도 안 바꾼다.

**DB 행이 없는 파일을 가리키는 때가 없어야 한다**(정리 뒤의 prune 이 그 행을 라벨째 지운다) — 그래서 실패하는 길마다 '파일과
행이 처음 그대로' 인지 본다. 여기의 장소 이름은 지어낸 것이다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autolabel.camera_move import LOG_NAME, MoveBlocked, apply_move, plan_move
from autolabel.provenance import load_cameras
from autolabel.split import ensure_splits
from helpers import box, far_hashes, plain_name

CAMERAS = """\
cameras:
  cam-01:
    kind: cctv
    place: 시험 장소 A
  cam-02:
    kind: cctv
    place: 시험 장소 B
  cam-03:
    kind: cctv
    place: 시험 장소 C
    split: test
  cam-04:
    kind: cctv
    place: 시험 장소 D
    tz: "+00:00"
"""
ORIGIN = "http://127.0.0.1:7860"


def row(store, image_id: int) -> dict:
    return store.get_image(image_id)


@pytest.fixture
def cams(cfg):
    cfg.cameras.write_text(CAMERAS, encoding="utf-8")
    return load_cameras(cfg.cameras)


@pytest.fixture
def frame(cfg, store):
    """카메라 폴더에 진짜 파일을 두고 행을 넣는다 — 영상 캡처가 넣는 꼴(unlabeled/<카메라>/<날짜>/<카메라>_<시각>_<ms>.jpg)."""

    def make(camera: str, hms: str = "164534", *, status: str = "pending", split: str | None = None,
             grp: str | None = None, tidied: bool = False) -> int:
        name = f"{camera}_20261115_{hms}_000.jpg"
        path = (cfg.labeled / "images" / camera / name) if tidied else (cfg.unlabeled / camera / "20261115" / name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"그림 {camera} {hms}".encode())
        if tidied:
            lbl = cfg.labeled / "labels" / camera / f"{path.stem}.txt"
            lbl.parent.mkdir(parents=True, exist_ok=True)
            lbl.write_text("2 0.5 0.5 0.25 0.25\n", encoding="utf-8")
        p = str(path.resolve())
        store.ingest_batch([{"path": p, "source": "collected", "width": 1920, "height": 1080, "status": status,
                             "boxes": [box()], "camera": camera,
                             "captured_at": f"2026-11-15T{hms[:2]}:{hms[2:4]}:{hms[4:]}+09:00"}])
        with store.connect() as con:
            i = int(con.execute("SELECT id FROM images WHERE path=?", (p,)).fetchone()[0])
        if split:
            store.set_splits([(i, split)])
        if grp:
            store.set_groups([(i, grp)])
        return i

    return make


def move(cfg, store, cams, ids, new):
    return apply_move(cfg, store, plan_move(cfg, store, ids, new, cams), cams, log=lambda _: None)


# ----------------------------------------------------------------- 옮기기

def test_one_image_moves_with_its_file_name_and_row(cfg, store, cams, frame):
    stay = frame("cam-01", "164428", grp="g000001")
    go = frame("cam-01", "164534", grp="g000001")
    before = row(store, go)
    data = Path(before["path"]).read_bytes()

    report = move(cfg, store, cams, [go], "cam-02")

    new_path = cfg.unlabeled.resolve() / "cam-02" / "20261115" / "cam-02_20261115_164534_000.jpg"
    after = row(store, go)
    assert (after["path"], after["name"], after["camera"]) == (str(new_path), new_path.name, "cam-02")
    assert new_path.read_bytes() == data and not Path(before["path"]).exists()
    assert after["grp"] is None                                   # 묶음은 카메라째다 — 다음 분할이 다시 센다
    assert after["captured_at"] == before["captured_at"]          # 시간대가 같으면 그대로
    assert after["updated_at"] == before["updated_at"]            # 라벨이 바뀐 것이 아니다
    assert len(store.get_boxes(go)) == 1
    assert (row(store, stay)["camera"], row(store, stay)["grp"]) == ("cam-01", "g000001")
    assert Path(row(store, stay)["path"]).is_file()               # 남은 이미지의 폴더는 그대로
    assert report["moved"] == 1 and report["renamed"] == 1 and report["left_n"] == 0
    assert report["rows"] == [{"id": go, "name": new_path.name, "camera": "cam-02", "captured_at": before["captured_at"]}]
    log = [json.loads(s) for s in (cfg.db.parent / "backups" / LOG_NAME).read_text(encoding="utf-8").splitlines()]
    assert [(e["id"], e["from"], e["to"], e["path_before"], e["grp_before"]) for e in log] \
        == [(go, "cam-01", "cam-02", before["path"], "g000001")]


def test_the_emptied_camera_folder_goes_but_nothing_above_it(cfg, store, cams, frame):
    go = frame("cam-01")
    move(cfg, store, cams, [go], "cam-02")
    assert not (cfg.unlabeled / "cam-01").exists() and cfg.unlabeled.is_dir()


def test_a_tidied_image_takes_its_label_file_along(cfg, store, cams, frame):
    go = frame("cam-01", tidied=True, status="approved")
    move(cfg, store, cams, [go], "cam-02")
    labeled = cfg.labeled.resolve()
    assert row(store, go)["path"] == str(labeled / "images" / "cam-02" / "cam-02_20261115_164534_000.jpg")
    assert (labeled / "labels" / "cam-02" / "cam-02_20261115_164534_000.txt").read_text(encoding="utf-8").startswith("2 ")
    assert not (labeled / "labels" / "cam-01").exists() and not (labeled / "images" / "cam-01").exists()


def test_the_capture_time_is_read_again_in_the_new_cameras_timezone(cfg, store, cams, frame):
    go = frame("cam-01")
    move(cfg, store, cams, [go], "cam-04")
    assert row(store, go)["captured_at"] == "2026-11-15T16:45:34+00:00"


def test_images_already_there_are_skipped_and_outside_files_stay(cfg, store, cams, frame, add_image):
    there = frame("cam-02", "101010")
    outside = add_image("loose.jpg", status="pending", camera="cam-01")       # 카메라 폴더 밖 — 행만 있다
    path = row(store, outside)["path"]
    report = move(cfg, store, cams, [there, outside], "cam-02")
    assert (report["moved"], report["same"], report["kept"]) == (1, 1, 1)
    assert (row(store, outside)["camera"], row(store, outside)["path"]) == ("cam-02", path)


def test_a_moved_image_follows_its_new_camera_at_the_next_split(cfg, store, cams, frame, add_image):
    """묶음을 비우지 않으면 옛 카메라의 묶음에 남아 그쪽 분할로 간다."""
    hashes = iter(far_hashes(40, seed=3))
    held = [add_image(f"t{plain_name(k)}", camera="cam-03", boxes=[box()], dhash=next(hashes)) for k in range(4)]
    loose = [add_image(plain_name(k), boxes=[box(cls=k % 4)], dhash=next(hashes)) for k in range(30)]
    stay = frame("cam-01", "164428", status="approved")
    go = frame("cam-01", "164534", status="approved")
    store.set_dhashes([(stay, next(hashes)), (go, next(hashes))])
    # 모두 한 번 묶였던 것처럼 — 묶음이 빈 이미지가 하나도 없으면 분할은 묶음을 다시 세지 않는다
    store.set_groups([(stay, "g-old"), (go, "g-old")] + [(i, "g-held") for i in held] + [(i, f"g{i:06d}") for i in loose])

    move(cfg, store, cams, [go], "cam-03")
    assign, _ = ensure_splits(cfg, store)
    assert assign[go] == "test" and {assign[i] for i in held} == {"test"}
    assert row(store, go)["grp"] == row(store, held[0])["grp"] != row(store, stay)["grp"]


# ----------------------------------------------------------------- 막는 것 — 아무것도 안 바꾼다

def unchanged(store, image_id: int, before: dict) -> bool:
    return row(store, image_id) == before and Path(before["path"]).is_file()


def test_unknown_camera_unknown_image_and_images_without_a_camera(cfg, store, cams, frame, add_image):
    go = frame("cam-01")
    before = row(store, go)
    bare = add_image("legacy.jpg", status="pending")
    with pytest.raises(MoveBlocked, match="등록돼 있지 않은"):
        plan_move(cfg, store, [go], "cam-77", cams)
    with pytest.raises(MoveBlocked, match="없는 이미지"):
        plan_move(cfg, store, [go, 99999], "cam-02", cams)
    with pytest.raises(MoveBlocked, match="출처"):
        plan_move(cfg, store, [go, bare], "cam-02", cams)
    assert unchanged(store, go, before)


def test_an_assigned_image_cannot_make_a_camera_span_two_splits(cfg, store, cams, frame):
    trained = frame("cam-01", "164428", status="approved", split="train")
    frame("cam-02", "101010", status="approved", split="val")
    before = row(store, trained)
    with pytest.raises(MoveBlocked, match="두 분할"):
        plan_move(cfg, store, [trained], "cam-02", cams)          # 그 카메라의 이미지는 val 이다
    with pytest.raises(MoveBlocked, match="고정"):
        plan_move(cfg, store, [trained], "cam-03", cams)          # test 로 고정한 카메라
    assert unchanged(store, trained, before)
    assert plan_move(cfg, store, [trained], "cam-04", cams).items                 # 빈 카메라로는 간다
    frame("cam-04", "111111", status="approved", split="train")
    assert plan_move(cfg, store, [trained], "cam-04", cams).items                 # 같은 분할의 카메라로도


def test_unassigned_images_may_go_to_a_pinned_camera_masks_and_all(cfg, store, cams, frame):
    """2026-11-24 — 가릴 영역이 있어도 test 고정 카메라로 옮긴다(test 의 영역은 채점에서 빼는 자리 — 전에는 막았다)."""
    plain, masked = frame("cam-01", "164428"), frame("cam-01", "164534")
    store.set_boxes(masked, [box()], uncertainty=0.0, source="ui", masks=[(0.1, 0.1, 0.2, 0.2)])
    assert len(plan_move(cfg, store, [plain, masked], "cam-03", cams).items) == 2


def test_an_existing_file_or_row_at_the_new_path_blocks(cfg, store, cams, frame):
    go = frame("cam-01")
    before = row(store, go)
    taken = cfg.unlabeled / "cam-02" / "20261115" / "cam-02_20261115_164534_000.jpg"
    taken.parent.mkdir(parents=True)
    taken.write_bytes(b"other")
    with pytest.raises(MoveBlocked, match="이미 파일"):
        plan_move(cfg, store, [go], "cam-02", cams)
    assert unchanged(store, go, before) and taken.read_bytes() == b"other"


def test_same_moment_frames_from_two_cameras_collide(cfg, store, cams, frame):
    a, b = frame("cam-01"), frame("cam-04")
    with pytest.raises(MoveBlocked, match="겹칩니다"):
        plan_move(cfg, store, [a, b], "cam-02", cams)


def test_a_row_that_changed_after_planning_undoes_the_new_names(cfg, store, cams, frame):
    go, other = frame("cam-01", "164428"), frame("cam-01", "164534")
    plan = plan_move(cfg, store, [go, other], "cam-02", cams)
    with store.connect() as con:                                  # 계획과 적용 사이에 다른 손이 카메라를 바꿨다
        con.execute("UPDATE images SET camera='cam-04' WHERE id=?", (other,))
    before = [row(store, go), row(store, other)]
    with pytest.raises(RuntimeError, match="그사이"):
        apply_move(cfg, store, plan, cams, log=lambda _: None)
    assert unchanged(store, go, before[0]) and unchanged(store, other, before[1])
    assert not (cfg.unlabeled / "cam-02").exists()                # 만든 새 이름·폴더는 지웠다
    assert not (cfg.db.parent / "backups" / LOG_NAME).exists()


def test_a_mask_template_made_from_the_image_follows_it(cfg, store, cams, frame):
    go = frame("cam-01")
    tid, _ = store.save_mask_template("화면 A", [(0.1, 0.1, 0.2, 0.2)], width=1920, height=1080, image_id=go, camera="cam-01")
    assert move(cfg, store, cams, [go], "cam-02")["templates"] == 1
    assert store.get_mask_template(tid)["camera"] == "cam-02"


# ----------------------------------------------------------------- 서버 (/api/bulk/camera)

@pytest.fixture
def client(cfg, store):
    from fastapi.testclient import TestClient

    from autolabel.server import create_app

    app = create_app(cfg, label="시험", bind_host="127.0.0.1")
    with TestClient(app, base_url=ORIGIN) as c:
        yield c


def test_the_endpoint_counts_first_and_moves_when_told(client, cfg, store, cams, frame):
    stay, go = frame("cam-01", "164428"), frame("cam-01", "164534")
    before = row(store, go)

    got = client.post("/api/bulk/camera", json={"ids": [go], "camera": "cam-02", "dry_run": True}).json()
    assert (got["moved"], got["from"], got["rows"]) == (1, {"cam-01": 1}, [])
    assert got["names"] == [["cam-01_20261115_164534_000.jpg", "cam-02_20261115_164534_000.jpg"]]
    assert unchanged(store, go, before)                           # 세기만 했다

    r = client.post("/api/bulk/camera", json={"ids": [go], "camera": "cam-02"})
    assert r.status_code == 200 and r.json()["moved"] == 1 and r.json()["place"] == "시험 장소 B"
    assert row(store, go)["camera"] == "cam-02" and Path(row(store, go)["path"]).is_file()
    assert client.get(f"/api/images/{go}").json()["name"] == "cam-02_20261115_164534_000.jpg"
    assert client.get(f"/api/images/{go}/file").status_code == 200
    assert row(store, stay)["camera"] == "cam-01"

    again = client.post("/api/bulk/camera", json={"ids": [go], "camera": "cam-02"}).json()
    assert (again["moved"], again["same"]) == (0, 1)              # 이미 그 카메라 — 작업도 만들지 않는다


def test_the_endpoint_refuses_without_touching_anything(client, cfg, store, cams, frame):
    go = frame("cam-01", status="approved", split="train")
    before = row(store, go)
    assert client.post("/api/bulk/camera", json={"ids": [go], "camera": "cam-77"}).status_code == 409
    assert client.post("/api/bulk/camera", json={"ids": [go], "camera": "cam-03"}).status_code == 409
    assert client.post("/api/bulk/camera", json={"ids": [], "camera": "cam-02"}).status_code == 422
    assert unchanged(store, go, before)


def test_the_endpoint_waits_its_turn_behind_a_running_job(client, cfg, store, cams, frame, monkeypatch):
    """정리·학습·자동 라벨링은 파일 경로를 읽는다 — 도는 동안에는 옮기지 않는다(409, 아무것도 안 바꿈)."""
    import threading

    go = frame("cam-01")
    before = row(store, go)
    started, release = threading.Event(), threading.Event()

    def slow_tidy(*args, **kwargs):
        started.set()
        release.wait(10)
        raise RuntimeError("시험용 정리 — 아무것도 하지 않는다")

    monkeypatch.setattr("autolabel.server.tidy_approved", slow_tidy)
    job = client.post("/api/jobs/tidy").json()
    try:
        assert started.wait(5)
        r = client.post("/api/bulk/camera", json={"ids": [go], "camera": "cam-02"})
        assert r.status_code == 409 and "정리" in r.json()["detail"]
        assert unchanged(store, go, before)
        assert client.post("/api/bulk/camera", json={"ids": [go], "camera": "cam-02", "dry_run": True}).json()["moved"] == 1
    finally:
        release.set()
    for _ in range(200):                                          # 작업이 끝난 뒤에는 옮겨진다
        if client.get(f"/api/jobs/{job['id']}").json()["status"] != "running":
            break
        threading.Event().wait(0.02)
    assert client.post("/api/bulk/camera", json={"ids": [go], "camera": "cam-02"}).json()["moved"] == 1
    assert row(store, go)["camera"] == "cam-02"


def test_registered_cameras_are_listed_even_without_images(client, cams, frame):
    frame("cam-01")
    got = client.get("/api/cameras").json()
    assert [c["key"] for c in got["cameras"]] == ["cam-01"]                       # 필터는 이미지가 있는 카메라만
    assert [(c["key"], c["n"]) for c in got["registered"]] == [("cam-01", 1), ("cam-02", 0), ("cam-03", 0), ("cam-04", 0)]
