"""검수 서버 (server.py) — 바깥에서 들어오는 요청을 거르는 곳. 임시 DB 로 띄우고 모델은 올리지 않는다.

TestClient 는 `base_url="http://127.0.0.1:7860"` 이어야 한다(기본 testserver 는 Host 검사에서 400).
"""

from __future__ import annotations

import pytest

from helpers import CLASSES, box, keys

ONE = {"cls": 2, "cx": 0.25, "cy": 0.5, "w": 0.25, "h": 0.25}
ORIGIN = "http://127.0.0.1:7860"


@pytest.fixture
def client(cfg, store):
    from fastapi.testclient import TestClient

    from autolabel.server import create_app

    app = create_app(cfg, label="시험", bind_host="127.0.0.1")
    with TestClient(app, base_url=ORIGIN) as c:
        yield c


def test_state(client, add_image):
    add_image("a.jpg", status="pending")
    got = client.get("/api/state").json()
    assert got["classes"] == CLASSES
    assert (got["counts"]["pending"], got["counts"]["total"]) == (1, 1)
    assert got["trained"] is False


# ----------------------------------------------------------------- 박스 저장

def test_save_boxes(client, store, add_image):
    i = add_image("a.jpg", status="pending", boxes=[box()])
    r = client.put(f"/api/images/{i}/boxes", json={"boxes": [ONE, {**ONE, "cx": 0.99, "conf": 0.5, "origin": "auto"}]})
    assert r.status_code == 200
    assert r.json() == {"ok": True, "n_boxes": 2, "n_before": 1, "n_history": 1}
    saved = store.get_boxes(i)
    assert keys(saved[:1]) == keys([box(cx=0.25, conf=None, origin="human")])
    assert saved[1].cx + saved[1].w / 2 == pytest.approx(1.0)           # 화면 밖으로 나간 몫은 잘라 넣는다
    assert store.box_history(i)[0]["source"] == "ui"


@pytest.mark.parametrize("body", [
    '{"boxes": [{"cls": 0, "cx": NaN, "cy": 0.5, "w": 0.1, "h": 0.1}]}',
    '{"boxes": [{"cls": 0, "cx": 0.5, "cy": 0.5, "w": Infinity, "h": 0.1}]}',
    '{"boxes": [{"cls": 0, "cx": 0.5, "cy": 0.5, "w": 0, "h": 0.1}]}',
    '{"boxes": [{"cls": -1, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}]}',
    '{"boxes": [], "masks": [{"x1": 0, "y1": 0, "x2": NaN, "y2": 1}]}',
])
def test_bad_numbers_are_rejected_not_saved(client, store, add_image, body):
    """2026-11-05 — NaN 리터럴을 받은 박스가 '이미지 전체' 박스로 둔갑해 저장됐다."""
    i = add_image("a.jpg", status="pending", boxes=[box()])
    r = client.put(f"/api/images/{i}/boxes", content=body, headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    assert keys(store.get_boxes(i)) == keys([box()]) and store.history_count(i) == 0


def test_unknown_class_and_unknown_image(client, store, add_image):
    i = add_image("a.jpg", status="pending", boxes=[box()])
    assert client.put(f"/api/images/{i}/boxes", json={"boxes": [{**ONE, "cls": len(CLASSES)}]}).status_code == 400
    assert client.put("/api/images/99999/boxes", json={"boxes": [ONE]}).status_code == 404
    assert keys(store.get_boxes(i)) == keys([box()])


# ----------------------------------------------------------------- 가릴 영역과 test

def test_masks_are_saved_only_when_sent(client, store, add_image):
    i = add_image("a.jpg", status="pending", boxes=[box()], split="train")
    area = {"x1": 0.5, "y1": 0.5, "x2": 0.75, "y2": 0.75}
    assert client.put(f"/api/images/{i}/boxes", json={"boxes": [ONE], "masks": [area]}).status_code == 200
    assert store.get_masks(i) == [(0.5, 0.5, 0.75, 0.75)]
    assert client.put(f"/api/images/{i}/boxes", json={"boxes": [ONE]}).status_code == 200       # 영역을 싣지 않은 저장
    assert store.get_masks(i) == [(0.5, 0.5, 0.75, 0.75)]
    assert client.put(f"/api/images/{i}/boxes", json={"boxes": [ONE], "masks": []}).status_code == 200
    assert store.get_masks(i) == []


def test_a_template_skips_a_photo_that_is_shown_turned(client, store, add_image):
    """여러 장에 템플릿을 넣을 때 화면 비율은 **파일의 EXIF 회전을 적용해** 본다.

    DB 의 크기는 회전 전 원시 크기다 — 16:9 로 찍은 폰 세로 사진은 원시 비율이 16:9 템플릿과 같지만 화면에서는 9:16 으로 돈다.
    2026-11-16 전에는 DB 비율이 맞으면 파일을 열지 않아 그런 사진에 영역이 엉뚱한 자리에 들어갔다. 파일을 못 여는 장은 DB 의
    크기를 믿고, 그것도 어긋나면 unreadable 로 센다.
    """
    from pathlib import Path

    from PIL import Image

    def make(name: str, *, turned: bool = False, file: bool = True, size=(192, 108)) -> int:
        image_id = add_image(name, camera="cam-01", boxes=[box()], size=size)      # DB 에는 원시 크기가 들어간다
        if file:
            path = Path(store.get_image(image_id)["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            exif = Image.Exif()
            if turned:
                exif[0x0112] = 6                    # Orientation 6 — 시계 방향으로 90도 돌려 본다
            Image.new("RGB", size, (90, 120, 150)).save(path, exif=exif)
        return image_id

    flat, turned = make("flat.jpg"), make("turned.jpg", turned=True)
    gone, odd = make("gone.jpg", file=False), make("odd.jpg", file=False, size=(100, 100))
    tpl, _ = store.save_mask_template("화면 글자", [(0.0, 0.0, 0.25, 0.1)], width=1920, height=1080)

    r = client.post("/api/bulk/mask-template", json={"ids": [flat, turned, gone, odd], "template_id": tpl})
    assert r.status_code == 200
    out = r.json()
    assert (out["changed"], out["aspect"], out["unreadable"]) == (2, 1, 1)
    assert store.get_masks(flat) == store.get_masks(gone) == [(0.0, 0.0, 0.25, 0.1)]
    assert store.get_masks(turned) == [] and store.get_masks(odd) == []


def test_test_images_take_no_masks(client, store, add_image):
    """test 는 칠하지 않는다 — 평가는 실제 화면 그대로여야 한다. 지우기만은 받는다."""
    i = add_image("a.jpg", status="approved", boxes=[box()], split="test")
    area = {"x1": 0.5, "y1": 0.5, "x2": 0.75, "y2": 0.75}
    r = client.put(f"/api/images/{i}/boxes", json={"boxes": [ONE], "masks": [area]})
    assert r.status_code == 409
    assert store.get_masks(i) == [] and keys(store.get_boxes(i)) == keys([box()])       # 박스도 저장하지 않는다
    assert client.put(f"/api/images/{i}/boxes", json={"boxes": [ONE], "masks": []}).status_code == 200


def test_images_of_a_camera_pinned_to_test_take_no_masks(client, cfg, store, add_image):
    cfg.cameras.write_text("cameras:\n  cam-01:\n    kind: cctv\n    place: 시험 장소\n    split: test\n", encoding="utf-8")
    i = add_image("a.jpg", status="pending", camera="cam-01")           # 아직 분할 전이어도 막는다
    area = {"x1": 0.5, "y1": 0.5, "x2": 0.75, "y2": 0.75}
    assert client.put(f"/api/images/{i}/boxes", json={"boxes": [], "masks": [area]}).status_code == 409


# ----------------------------------------------------------------- 다른 사이트에서 부르기 막기 (_RequestGuard)

def status_call(client, **headers):
    """상태 바꾸기 요청 — 없는 이미지라 거름망을 지나면 404 다."""
    return client.post("/api/images/99999/status", json={"status": "approved"}, headers=headers).status_code


def test_plain_and_same_origin_requests_pass(client):
    assert status_call(client) == 404                                   # 헤더 없는 요청(urllib·curl — scheduled_stop.py)
    assert status_call(client, **{"Sec-Fetch-Site": "same-origin"}) == 404
    assert status_call(client, **{"Sec-Fetch-Site": "none"}) == 404
    assert status_call(client, Origin=ORIGIN) == 404
    assert status_call(client, Origin="http://localhost:7860", Host="localhost:7860") == 404


@pytest.mark.parametrize("headers", [
    {"Sec-Fetch-Site": "cross-site"},
    {"Sec-Fetch-Site": "same-site"},
    {"Origin": "http://evil.example"},
    {"Origin": "http://127.0.0.1:9999"},                # 같은 PC 의 다른 포트도 다른 출처
    {"Origin": "null"},
])
def test_cross_site_writes_are_refused(client, headers):
    """2026-11-05 — 본문 없는 POST(학습·정리·휴지통 비우기)는 아무 사이트의 페이지나 보낼 수 있었다."""
    assert status_call(client, **headers) == 403


def test_cross_site_reads_are_not_blocked(client):
    assert client.get("/api/state", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 200


@pytest.mark.parametrize("host", ["evil.example:7860", "evil.example", "192.168.0.10:7860"])
def test_foreign_host_header_is_refused(client, host):
    """DNS rebinding — 남의 도메인을 127.0.0.1 로 돌려 들어오면 Host 가 그 도메인이다."""
    assert client.get("/api/state", headers={"Host": host}).status_code == 400


def test_static_files_are_revalidated(client):
    """함정 11 의 한 겹 — 화면 파일은 no-cache 라 서버를 다시 켠 뒤 옛 app.js 가 돌지 않는다."""
    r = client.get("/app.js")
    assert r.status_code == 200 and r.headers["cache-control"] == "no-cache"


def test_error_responses_are_never_cached(client, add_image):
    """크롬은 캐시 헤더가 없는 410 을 기한 없이 캐시한다 — 파일이 잠깐 없을 때 받은 '원본 파일이 사라졌습니다' 가 파일이
    돌아온 뒤에도 그 썸네일 주소에 남았다(2026-11-15 시험 서버: 같은 주소가 캐시로는 410, 캐시를 건너뛰면 200)."""
    gone = add_image("gone.jpg")                # 행만 있고 파일은 없다
    for url in (f"/api/images/{gone}/thumb?s=256", f"/api/images/{gone}/file"):
        r = client.get(url)
        assert (r.status_code, r.headers["cache-control"]) == (410, "no-store"), url
    r = client.get("/api/images/999999")
    assert (r.status_code, r.headers["cache-control"]) == (404, "no-store")
    assert client.get("/no-such-file.js").headers["cache-control"] == "no-store"
    assert "no-store" not in client.get("/api/state").headers.get("cache-control", "")      # 성공한 응답은 그대로


def test_error_responses_are_never_cached_under_a_project_tab(cfg, store, add_image):
    """프로젝트 탭(`/p/<이름>/`)으로 띄운 서버에서도 같다 — 평소 `labeler.py review` 가 이 꼴이다."""
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from autolabel.server import create_multi_app

    gone = add_image("gone.jpg")
    app = create_multi_app([SimpleNamespace(name="t", title="시험", cfg=cfg)], bind_host="127.0.0.1")
    with TestClient(app, base_url=ORIGIN) as c:
        r = c.get(f"/p/t/api/images/{gone}/thumb?s=256")
        assert (r.status_code, r.headers["cache-control"]) == (410, "no-store")
