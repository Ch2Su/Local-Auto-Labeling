"""데이터 점검 (health.py, 2026-11-15) — 화면의 숫자가 DB 와 맞는지. 읽기만 하므로 '무엇을 세는가' 를 지킨다.

범위(승인만 / 휴지통만 뺀 전부), 클래스마다 카메라·묶음 수와 분할별 몫, 학습 입력에서의 박스 크기 칸, 이미지당 객체 수 칸,
가릴 영역이 빼는 박스, 히트맵 칸. 여기의 장소 이름은 지어낸 것이다.
"""

from __future__ import annotations

import pytest

from autolabel import health
from autolabel.provenance import load_cameras
from helpers import CLASSES, box

CAMERAS = """\
cameras:
  cam-01:
    kind: cctv
    place: 시험 장소 A
  cam-02:
    kind: cctv
    place: 시험 장소 B
    split: test
  cam-03:
    kind: cctv
    place: 시험 장소 C
"""
ORIGIN = "http://127.0.0.1:7860"
PERSON, CAR, TRUCK, CONE = CLASSES.index("person"), CLASSES.index("car"), CLASSES.index("truck"), CLASSES.index("cone")


@pytest.fixture
def cams(cfg):
    cfg.cameras.write_text(CAMERAS, encoding="utf-8")
    return load_cameras(cfg.cameras)


@pytest.fixture
def data(add_image):
    """승인 4장(카메라 둘 + 출처 없음) · 검수 대기 1장 · 휴지통 1장. 그림은 1280×720 — 640 입력이면 px 가 반이 된다."""
    big = box(CAR, 0.5, 0.5, 0.25, 0.25)            # 짧은 변 180px → 640 입력에서 90px (64 이상 칸)
    small = box(PERSON, 0.1, 0.9, 0.01, 0.02)       # 12.8px → 6.4px (8 미만 칸)
    mid = box(TRUCK, 0.9, 0.1, 0.05, 0.1)           # 64px → 32px (32~64 칸의 아래 끝)
    ids = {
        "a": add_image("a.jpg", camera="cam-01", boxes=[big, small]),
        "b": add_image("b.jpg", camera="cam-01", boxes=[big, mid, small, small, small, small]),
        "c": add_image("c.jpg", camera="cam-02", boxes=[big]),
        "legacy": add_image("old.jpg", boxes=[]),
        "pending": add_image("p.jpg", camera="cam-01", status="pending", boxes=[box(CONE, 0.5, 0.5, 0.02, 0.02)]),
        "trash": add_image("t.jpg", camera="cam-03", status="trashed", boxes=[big, big]),
    }
    return ids


def rep(store, cams=None, scope="approved", imgsz=640, val=0.2, test=0.15, drop=True):
    snap = store.health_snapshot(approved_only=scope == "approved")
    return health.report(snap, store.get_classes(), cams or {}, scope=scope, imgsz=imgsz,
                         val_ratio=val, test_ratio=test, counts=store.counts(), drop_masked=drop)


def by_name(rows):
    return {r["name"]: r for r in rows}


def test_bins_put_the_edges_in_the_upper_cell():
    assert [health.size_bin(v) for v in (0, 7.99, 8, 15.9, 16, 32, 63.9, 64, 500)] == [0, 0, 1, 1, 2, 3, 3, 4, 4]
    assert [health.count_bin(n) for n in (0, 1, 5, 6, 10, 11, 20, 21, 30, 31, 400)] == [0, 1, 5, 6, 6, 7, 7, 8, 8, 9, 9]
    # 긴 변을 입력 크기에 맞춘다 — 세로로 긴 그림도 같은 식
    assert health.short_side(0.25, 0.25, 1280, 720, 640) == pytest.approx(90.0)
    assert health.short_side(0.25, 0.25, 720, 1280, 640) == pytest.approx(90.0)
    assert health.short_side(0.5, 0.5, 0, 0, 640) == 0.0


def test_approved_scope_counts_only_what_training_sees(store, data, cams):
    r = rep(store, cams)
    assert (r["images"], r["boxes"], r["empty"]) == (4, 9, 1)
    assert r["per_image"] == {"mean": 2.25, "median": 1.5, "max": 6, "max_id": data["b"]}
    assert r["counts"]["pending"] == 1 and r["counts"]["trashed"] == 1      # 머리의 상태별 수는 범위와 상관없이 전부

    c = by_name(r["classes"])
    assert [row["name"] for row in r["classes"]] == CLASSES                 # DB 의 클래스 순서 그대로, 박스가 없어도 줄은 있다
    assert (c["car"]["boxes"], c["car"]["images"], c["car"]["cameras"], c["car"]["units"]) == (3, 3, 2, 2)
    assert (c["person"]["boxes"], c["person"]["images"], c["person"]["cameras"]) == (5, 2, 1)
    assert c["person"]["top"] == {"camera": "cam-01", "share": 1.0}
    assert c["car"]["top"] == {"camera": "cam-01", "share": round(2 / 3, 4)}
    assert c["cone"]["boxes"] == 0 and c["cone"]["top"] is None and c["cone"]["median_px"] is None
    assert c["car"]["median_px"] == 90.0 and c["person"]["median_px"] == 6.4

    assert r["size_bins"] == {"edges": [8, 16, 32, 64], "counts": [5, 0, 0, 1, 3]}
    assert c["truck"]["sizes"] == [0, 0, 0, 1, 0]

    bins = {(b["lo"], b["hi"]): b for b in r["count_bins"]}
    assert bins[(0, 0)]["ids"] == [data["legacy"]]
    assert (bins[(1, 1)]["ids"], bins[(2, 2)]["ids"]) == ([data["c"]], [data["a"]])
    assert bins[(6, 10)]["ids"] == [data["b"]] and bins[(6, 10)]["more"] == 0
    assert sum(b["n"] for b in r["count_bins"]) == r["images"]

    assert r["sizes"] == [{"w": 1280, "h": 720, "n": 4}] and r["sizes_more"] == 0
    assert r["domains"] == {"none": 4}


def test_all_scope_adds_pending_but_never_the_trash(store, data):
    r = rep(store, scope="all")
    assert (r["images"], r["boxes"]) == (5, 10)
    assert by_name(r["classes"])["cone"]["boxes"] == 1
    assert "cam-03" not in {row["key"] for row in r["cameras"]}


def test_camera_rows_carry_the_registry_and_the_no_camera_row_goes_last(store, data, cams):
    store.set_splits([(data["a"], "train"), (data["c"], "test")])
    rows = rep(store, cams)["cameras"]
    assert [row["key"] for row in rows] == ["cam-01", "cam-02", None]
    one, two, none = rows
    assert (one["images"], one["boxes"], one["classes"], one["place"], one["pinned"]) == (2, 8, 3, "시험 장소 A", None)
    assert one["splits"] == {"train": 1, "val": 0, "test": 0, "none": 1}
    assert (two["pinned"], two["splits"]["test"], two["registered"]) == ("test", 1, True)
    assert (none["images"], none["boxes"], none["registered"], none["place"]) == (1, 0, False, None)


def test_camera_rows_survive_an_unreadable_registry(store, data):
    """등록 파일을 못 읽어도(cameras {}) 표는 DB 로 나온다 — 장소·종류만 빈다."""
    rows = rep(store)["cameras"]
    assert [(row["key"], row["registered"], row["kind"]) for row in rows[:2]] == [("cam-01", False, None), ("cam-02", False, None)]


def test_notes_say_only_what_the_numbers_show(store, data, cams):
    notes = [n["text"] for n in rep(store, cams)["notes"]]
    assert any("분할이 아직 없습니다(4장 모두 미배정)" in t for t in notes)
    few = next(t for t in notes if "3곳 미만" in t)
    assert "car(2)" in few and "person(1)" in few and "truck(1)" in few and "cone" not in few
    none = next(t for t in notes if t.startswith("박스가 없는 클래스"))
    assert "cone" in none and "car" not in none
    assert not any(t.startswith(("val 에", "test 에")) for t in notes)      # 분할이 없으면 분할별 경고는 내지 않는다

    store.set_splits([(data["a"], "train"), (data["b"], "train"), (data["c"], "test")])
    r = rep(store, cams)
    notes = [n["text"] for n in r["notes"]]
    assert any(t.startswith("분할 미배정 1장") for t in notes)
    val = next(t for t in notes if t.startswith("val 에 박스가 없는 클래스"))
    assert "car" in val and "person" in val and "cone" not in val            # 박스가 아예 없는 클래스는 위에서 이미 알렸다
    test = next(t for t in notes if t.startswith("test 에 박스가 없는 클래스"))
    assert "person" in test and "truck" in test and "car" not in test
    assert r["splits"]["images"] == {"train": 2, "val": 0, "test": 1, "none": 1}
    assert r["splits"]["boxes"] == {"train": 8, "val": 0, "test": 1, "none": 0}
    assert r["splits"]["target"] == {"train": 0.65, "val": 0.2, "test": 0.15}
    assert by_name(r["classes"])["car"]["split"] == {"train": 2, "val": 0, "test": 1, "none": 0}

    # test 를 쓰지 않는 설정이면 test 경고도 없다
    assert not any(n["text"].startswith("test 에") for n in rep(store, cams, test=0.0)["notes"])


def test_legacy_images_count_their_scene_group_as_one_unit(store, add_image):
    """출처 없는 이미지는 장면 묶음(grp)이 분할 단위다 — 같은 묶음 두 장은 한 곳, 묶음이 없으면 장마다 한 곳."""
    a = add_image("a.jpg", boxes=[box(CAR)])
    b = add_image("b.jpg", boxes=[box(CAR)])
    add_image("c.jpg", boxes=[box(CAR)])
    car = by_name(rep(store)["classes"])["car"]
    assert (car["units"], car["cameras"], car["no_camera_boxes"]) == (3, 0, 3)
    store.set_groups([(a, "g1"), (b, "g1")])
    assert by_name(rep(store)["classes"])["car"]["units"] == 2


def test_boxes_the_masks_drop_from_training_are_left_out_of_every_number(store, data):
    """영역이 박스 넓이의 반 이상을 덮으면 학습 라벨에서 빠진다(masks.DROP) — 그 박스는 기본으로 어느 수치에도 안 들어간다.

    2026-11-15 의 실제 DB 는 승인 박스 1,277개 가운데 619개가 그런 박스였다(영역으로 덮은 자리에 자동 라벨 박스가 남아 있다) —
    넣고 세면 클래스 균형과 박스 크기가 학습이 보는 것과 달라진다.
    """
    assert rep(store)["masks"] == {"images": 0, "rects": 0, "dropped": 0, "test_images": 0}
    # a 의 큰 차(0.375~0.625)의 왼쪽 60% 를 덮고, 작은 사람은 건드리지 않는다
    store.set_masks(data["a"], [(0.375, 0.375, 0.525, 0.625)])
    # c 의 큰 차는 40% 만 덮는다 — 남는다
    store.set_masks(data["c"], [(0.375, 0.375, 0.475, 0.625), (0.0, 0.0, 0.01, 0.01)])

    r = rep(store)
    assert r["drop_masked"] is True and r["masks"] == {"images": 2, "rects": 3, "dropped": 1, "test_images": 0}
    c = by_name(r["classes"])
    assert (r["boxes"], c["car"]["boxes"], c["car"]["images"], c["car"]["masked_out"], c["person"]["masked_out"]) == (8, 2, 2, 1, 0)
    assert r["size_bins"]["counts"] == [5, 0, 0, 1, 2] and r["splits"]["boxes"]["none"] == 8
    assert r["per_image"]["mean"] == 2.0                                     # a 는 이제 한 개(사람)만 센다
    assert any("박스 1개는 세지 않았습니다" in n["text"] for n in r["notes"])
    cams = {row["key"]: row for row in r["cameras"]}
    assert {k: (v["masked"], v["masked_out"], v["boxes"]) for k, v in cams.items()} == {
        "cam-01": (1, 1, 7), "cam-02": (1, 0, 1), None: (0, 0, 0)}

    keep = rep(store, drop=False)
    k = by_name(keep["classes"])
    assert keep["drop_masked"] is False and (keep["boxes"], k["car"]["boxes"], k["car"]["masked_out"]) == (9, 3, 1)
    assert any("박스 1개가 수치에 들어 있습니다" in n["text"] for n in keep["notes"])

    snap = store.health_snapshot()
    centre = (health.GRID[1] // 2) * health.GRID[0] + health.GRID[0] // 2
    assert health.heatmap(snap)["cells"][centre] == 2
    assert health.heatmap(snap, drop_masked=False)["cells"][centre] == 3


def test_an_image_whose_boxes_are_all_masked_counts_as_empty(store, add_image):
    """박스가 전부 가려진 장은 학습에서 배경 이미지다 — '박스 없는 이미지' 로 센다(넣고 세면 한 개짜리)."""
    i = add_image("m.jpg", camera="cam-01", boxes=[box(CAR)])
    store.set_masks(i, [(0.0, 0.0, 1.0, 1.0)])
    assert (rep(store)["empty"], rep(store)["boxes"]) == (1, 0)
    assert (rep(store, drop=False)["empty"], rep(store, drop=False)["boxes"]) == (0, 1)


def test_boxes_under_masks_on_test_images_are_not_dropped(store, add_image):
    """test 분할은 학습 세트·내보내기가 칠하지도, 박스를 빼지도 않는다(trainer.build_dataset·export_yolo 의 split == "test").

    영역은 검수하며 먼저 덮고 분할은 나중에 정해지므로 영역이 있는 이미지가 test 로 갈 수 있다. 2026-11-15 리뷰에서 고침:
    전에는 분할을 안 보고 빼서 test 의 박스가 labels/test 보다 적게 세어지고 'test 에 박스가 없는 클래스' 가 잘못 떴다.
    """
    tr = add_image("tr.jpg", camera="cam-01", boxes=[box(CAR)])
    te = add_image("te.jpg", camera="cam-02", boxes=[box(CAR)])
    add_image("va.jpg", camera="cam-03", boxes=[box(CAR)], split="val")
    for i in (tr, te):                                # 둘 다 박스를 통째로 덮는다 — 그 뒤에 분할이 정해진다
        store.set_masks(i, [(0.0, 0.0, 1.0, 1.0)])
    store.set_splits([(tr, "train"), (te, "test")])

    r = rep(store)
    car = by_name(r["classes"])["car"]
    assert (r["boxes"], car["boxes"], car["masked_out"]) == (2, 2, 1)         # train 의 한 개만 빠진다
    assert car["split"] == {"train": 0, "val": 1, "test": 1, "none": 0}
    assert r["splits"]["boxes"] == {"train": 0, "val": 1, "test": 1, "none": 0}
    assert r["masks"] == {"images": 2, "rects": 2, "dropped": 1, "test_images": 1}
    cams = {row["key"]: (row["masked"], row["masked_out"], row["boxes"]) for row in r["cameras"]}
    assert cams == {"cam-01": (1, 1, 0), "cam-02": (1, 0, 1), "cam-03": (0, 0, 1)}
    notes = [n["text"] for n in r["notes"]]
    assert not any(t.startswith("test 에 박스가 없는 클래스") for t in notes)
    assert any("가릴 영역이 있는 test 이미지 1장" in t for t in notes)

    snap = store.health_snapshot()
    assert health.masked_out(snap) == [True, False, False]                     # 박스 순서 = 이미지 id 순서(tr, te, va)
    centre = (health.GRID[1] // 2) * health.GRID[0] + health.GRID[0] // 2
    assert health.heatmap(snap)["cells"][centre] == 2
    assert health.heatmap(snap, drop_masked=False)["cells"][centre] == 3


def test_tag_counts_follow_the_scope(store, data):
    day = store.create_tag("day")
    fog = store.create_tag("fog")
    store.set_image_tag([data["a"], data["pending"], data["trash"]], day, True)
    store.set_image_tag([data["pending"]], fog, True)
    r = rep(store)
    assert r["tags"] == [{"id": day, "name": "day", "n": 1}, {"id": fog, "name": "fog", "n": 0}]
    assert r["untagged"] == 3
    r = rep(store, scope="all")
    assert [t["n"] for t in r["tags"]] == [2, 1] and r["untagged"] == 3


def test_a_box_of_a_class_the_list_does_not_know_still_gets_a_row(store, add_image):
    """클래스 표에 이름이 없는 번호(함정 12 같은 사고)도 세다가 죽지 않는다 — 'cls_N' 줄로 보인다."""
    add_image("x.jpg", camera="cam-01", boxes=[box(len(CLASSES) + 1)])
    rows = rep(store)["classes"]
    assert len(rows) == len(CLASSES) + 2 and rows[-1]["name"] == f"cls_{len(CLASSES) + 1}" and rows[-1]["boxes"] == 1


def test_an_empty_database_gives_zeros_not_errors(store):
    r = rep(store)
    assert (r["images"], r["boxes"], r["empty"], r["per_image"]["max_id"]) == (0, 0, 0, None)
    assert [n["text"] for n in r["notes"]] == ["이 범위에 이미지가 없습니다."]
    h = health.heatmap(store.health_snapshot())
    assert (h["max"], h["boxes"], len(h["cells"])) == (0, 0, health.GRID[0] * health.GRID[1])


def test_heatmap_bins_box_centres_and_filters(store, data):
    gw, gh = health.GRID
    snap = store.health_snapshot()
    h = health.heatmap(snap)
    assert (h["grid"], h["boxes"], h["images"], sum(h["cells"]), h["sample_id"]) == ([gw, gh], 9, 3, 9, None)
    centre = (gh // 2) * gw + gw // 2                 # (0.5, 0.5) — 큰 차 셋
    person = int(0.9 * gh) * gw + int(0.1 * gw)       # (0.1, 0.9) — 사람 다섯
    truck = int(0.1 * gh) * gw + int(0.9 * gw)
    assert (h["cells"][centre], h["cells"][person], h["cells"][truck], h["max"]) == (3, 5, 1, 5)

    cars = health.heatmap(snap, cls=CAR)
    assert (cars["boxes"], cars["cells"][centre], cars["cells"][person]) == (3, 3, 0)

    two = health.heatmap(snap, camera="cam-02", cell=(gw // 2, gh // 2))
    assert (two["boxes"], two["sample_id"]) == (1, data["c"])
    assert two["cell"] == {"x": gw // 2, "y": gh // 2, "n": 1, "ids": [data["c"]], "more": 0}

    none = health.heatmap(snap, camera="__none__")    # 출처 없는 이미지 — 박스는 없어도 바탕에 깔 그림은 준다
    assert (none["boxes"], none["sample_id"]) == (0, data["legacy"])

    hit = health.heatmap(snap, cell=(int(0.1 * gw), int(0.9 * gh)))["cell"]
    assert hit["ids"] == sorted([data["a"], data["b"]]) and hit["n"] == 2


def test_a_centre_on_the_far_edge_stays_inside_the_grid():
    """중심이 1.0(오른쪽·아래 끝)이어도 마지막 칸에 들어간다 — 칸 번호가 격자를 넘지 않는다."""
    snap = {"images": [{"id": 1, "camera": None}], "boxes": [(1, CAR, 1.0, 1.0, 0.01, 0.01), (1, CAR, 0.0, 0.0, 0.01, 0.01)]}
    h = health.heatmap(snap)
    assert (h["cells"][-1], h["cells"][0], h["boxes"]) == (1, 1, 2)


def test_id_lists_are_capped(store, add_image, monkeypatch):
    monkeypatch.setattr(health, "ID_CAP", 2)
    ids = [add_image(f"e{i}.jpg", boxes=[]) for i in range(5)]
    zero = rep(store)["count_bins"][0]
    assert (zero["n"], zero["ids"], zero["more"]) == (5, ids[:2], 3)


@pytest.fixture
def client(cfg, store):
    from fastapi.testclient import TestClient

    from autolabel.server import create_app

    app = create_app(cfg, label="시험", bind_host="127.0.0.1")
    with TestClient(app, base_url=ORIGIN) as c:
        yield c


def test_the_endpoint_serves_the_report_and_reads_only(client, cfg, store, data, cams):
    before = store.health_snapshot(approved_only=False)
    stamp = {i: store.get_image(i)["updated_at"] for i in data.values()}

    r = client.get("/api/health")
    assert r.status_code == 200
    d = r.json()
    assert (d["scope"], d["imgsz"], d["default_imgsz"], d["images"], d["boxes"]) == ("approved", 640, 640, 4, 9)
    assert d["cameras"][0]["place"] == "시험 장소 A" and d["camera_error"] is None
    assert (d["none_camera"], d["none_split"]) == ("__none__", "__none__")

    d = client.get("/api/health", params={"scope": "all", "imgsz": 1280}).json()
    assert (d["images"], d["imgsz"]) == (5, 1280)
    # 입력을 두 배로: 사람 6.4 → 12.8px(8~16), 트럭 32 → 64px(64 이상), 검수 대기의 라바콘은 짧은 변 14.4px(8~16)
    assert d["size_bins"]["counts"] == [0, 6, 0, 0, 4]
    assert client.get("/api/health", params={"scope": "trash"}).status_code == 400
    assert client.get("/api/health", params={"imgsz": 8}).status_code == 400
    assert client.get("/api/health", params={"masked": "maybe"}).status_code == 400
    assert client.get("/api/health/heatmap", params={"masked": "maybe"}).status_code == 400
    assert client.get("/api/health", params={"masked": "keep"}).json()["drop_masked"] is False

    h = client.get("/api/health/heatmap", params={"cls": CAR, "camera": "cam-01", "cell": "24,13"}).json()
    assert (h["boxes"], h["sample_id"], h["cell"]["ids"]) == (2, data["b"], sorted([data["a"], data["b"]]))
    for bad in ("48,0", "0,27", "a,b", "3"):
        assert client.get("/api/health/heatmap", params={"cell": bad}).status_code == 400, bad
    assert client.get("/api/health/heatmap", params={"cls": -1}).status_code == 400
    assert client.get("/api/health/heatmap", params={"scope": "x"}).status_code == 400

    assert store.health_snapshot(approved_only=False) == before
    assert {i: store.get_image(i)["updated_at"] for i in data.values()} == stamp


def test_the_list_status_of_a_scope_lists_exactly_that_scopes_images(client, store, data):
    """줄을 눌러 이미지 보기로 넘길 때 화면이 거는 상태 필터(`list_status`)는 그 범위와 **같은 이미지**를 준다.

    2026-11-15 리뷰에서 고침: 범위 '휴지통만 빼고 전부' 를 목록의 '전체'(`status=all` — 휴지통도 넣는다)로 넘겨, 표의 장수와
    이미지 보기의 장수가 달랐고 거기서 여러 장 바꾸기를 하면 휴지통 이미지에도 먹었다. 이제 `status=live`(store.LIVE).
    """
    def listed(**params):
        d = client.get("/api/images", params={"limit": 1000, **params}).json()
        return sorted(im["id"] for im in d["images"]), d["total"]

    for scope in health.SCOPES:
        status = client.get("/api/health", params={"scope": scope}).json()["list_status"]
        want = [im["id"] for im in store.health_snapshot(approved_only=scope == "approved")["images"]]
        assert listed(status=status) == (want, len(want)), scope                     # 필터 없는 길 — 상태별 수로 센다
        cam = [i for i in want if store.get_image(i)["camera"] == "cam-01"]
        assert listed(status=status, camera="cam-01") == (cam, len(cam)), scope      # 필터를 건 길 — count_images
    assert status == "live" and data["trash"] not in want
    assert listed(status="live", camera="cam-03") == ([], 0)                         # cam-03 은 휴지통의 한 장뿐이다
    # 목록의 '전체' 는 전과 같이 휴지통도 넣는다
    assert listed(status="all", camera="cam-03") == ([data["trash"]], 1)
    assert listed(status="all")[1] == len(data)
