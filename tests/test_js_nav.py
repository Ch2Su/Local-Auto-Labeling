"""누르고 있는 ↑·↓ 로 이미지를 계속 넘기기 (2026-11-12 사용자 요청) — 화면 코드(app.js)를 node 로 돌려 본다.

고치기 전에는 ↑·↓ 를 누르고 있어도 누르는 동안 한 장만 넘어갔다. 키 반복(약 33ms 마다)이 올 때마다 다음 장을 요청했는데,
이미지 정보를 받는 데 그보다 오래 걸려(실측 40~50ms) `S.cur` 가 바뀌기 전에 같은 장을 또 요청했고, 앞 요청은 '마지막 것만
반영'(`S.loadSeq`)에 걸려 버려졌다. 지금은 반복 입력이면 앞 장의 사진이 뜬 뒤에만 넘긴다(`step(delta, held)`·`imageLoading`).

`tests/nav_harness.js` 가 app.js 의 `step`·`imageLoading`·`openImage`·`loadImage`·`loadImageNow` 를 **그대로** 돌리고, 키 입력·
서버 응답·사진 도착만 가짜 시계로 일으킨다(브라우저는 띄우지 않는다 — 그림이 실제로 그려지는지는 보지 못한다). 함수는
test_js_parity.py 와 같은 규칙으로 글자로 뗀다 — 이름이나 모양을 바꾸면 여기도 맞출 것.

node 가 없으면 건너뛴다.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from helpers import WEB
from test_js_parity import find_node, js_function

HARNESS = Path(__file__).with_name("nav_harness.js")
PARTS = ("openImage", "loadImage", "loadImageNow", "imageLoading", "step")

# 목록 100줄의 start 번 줄에서 key 를 holdMs 동안 누른다. 첫 입력 뒤 delay ms 부터 repeat ms 마다 반복 입력이 온다(Windows 기본
# 쯤). meta·pic 은 이미지 정보·사진을 받는 데 걸리는 ms — 48·24 는 2026-11-12 에 시험 서버(2560×1440 133장)에서 잰 값이다.
BASE = {"rows": 100, "start": 1, "key": "down", "holdMs": 3000, "delay": 500, "repeat": 33, "meta": 48, "pic": 24,
        "passRepeat": True, "failPic": [], "failApi": [], "saveFails": False}
SCENARIOS = {
    "hold": {},
    "old_handler": {"passRepeat": False},
    "fast_server": {"meta": 5, "pic": 5},
    "slow_picture": {"pic": 400},
    "up": {"start": 60, "key": "up"},
    "bottom": {"start": 97, "holdMs": 2000},
    "top": {"start": 3, "key": "up", "holdMs": 2000},
    "broken_picture": {"failPic": [5]},
    "server_error": {"failApi": [4], "holdMs": 1500},
    "save_fails": {"saveFails": True, "holdMs": 1000},
}


def repeats(name: str) -> int:
    sc = {**BASE, **SCENARIOS[name]}
    return len(range(sc["delay"], sc["delay"] + sc["holdMs"], sc["repeat"]))


@pytest.fixture(scope="module")
def nav(tmp_path_factory) -> dict:
    node = find_node()
    if node is None:
        pytest.skip("node 가 없어 화면(JS) 쪽을 돌릴 수 없습니다")
    app = (WEB / "app.js").read_text(encoding="utf-8")
    d = tmp_path_factory.mktemp("nav")
    (d / "code.js").write_text("\n\n".join(js_function(app, name) for name in PARTS), encoding="utf-8")
    (d / "in.json").write_text(json.dumps([{**BASE, **sc} for sc in SCENARIOS.values()]), encoding="utf-8")
    done = subprocess.run([node, str(HARNESS), str(d / "code.js"), str(d / "in.json"), str(d / "out.json")],
                          capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert done.returncode == 0, f"node 실패:\n{done.stderr}"
    return dict(zip(SCENARIOS, json.loads((d / "out.json").read_text(encoding="utf-8"))))


def test_holding_an_arrow_keeps_moving(nav):
    """3초 누르고 있으면 서른 장쯤 넘어간다 — 한 장마다 정보 48ms + 사진 24ms 를 기다렸다가 그다음 반복 입력에서 넘긴다."""
    got = nav["hold"]
    moved = got["end"] - 1
    assert 25 <= moved <= 42                    # 위는 3000 / (48 + 24) — 기다리는 시간보다 빨리 넘어갈 수는 없다
    assert got["leftBeforePicture"] == 0        # 지나가는 장마다 사진이 뜬 것을 보고 넘어간다
    assert got["requests"] == moved             # 같은 장을 다시 요청하지 않는다
    assert got["end"] - got["atRelease"] <= 1   # 손을 떼면 선다 — 받던 한 장까지만
    assert got["pictureUp"] and not got["stillLoading"] and got["loading"] == 0


def test_the_bug_before_the_fix(nav):
    """대조 — 반복 입력을 held 없이 넘기면(2026-11-12 전의 keydown 처리기) 같은 조건에서 누르는 동안 한 장만 넘어간다.
    이 모의가 그 버그를 실제로 낸다는 확인이다(그래서 위 테스트가 뜻이 있다)."""
    got = nav["old_handler"]
    assert got["atRelease"] - 1 == 1                        # 처음 누른 한 장뿐
    assert got["end"] - 1 == 2                              # 손을 뗀 뒤에야 마지막 요청이 반영된다
    assert got["requests"] == 1 + repeats("old_handler")    # 반복마다 같은 장을 다시 요청하고 앞 요청은 버렸다


def test_a_fast_server_moves_on_every_repeat(nav):
    """정보·사진이 반복 간격보다 빨리 오면 반복 입력마다 한 장씩 — 기다리느라 더 느려지지 않는다."""
    got = nav["fast_server"]
    assert got["end"] - 1 == 1 + repeats("fast_server")
    assert got["leftBeforePicture"] == 0


def test_a_slow_picture_is_waited_for(nav):
    """사진이 늦게 오면(400ms) 그만큼 천천히 넘어간다 — 뜨지도 않은 장을 지나치지 않는다."""
    got = nav["slow_picture"]
    assert 3 <= got["end"] - 1 <= 9
    assert got["leftBeforePicture"] == 0
    assert got["pictureUp"] and got["loading"] == 0


def test_up_goes_the_other_way(nav):
    got = nav["up"]
    assert 25 <= 60 - got["end"] <= 42
    assert got["leftBeforePicture"] == 0


def test_stops_at_both_ends_without_reopening(nav):
    """목록 끝에 닿으면 선다 — 누르고 있는 동안 같은 장을 다시 열지 않는다(열면 되돌리기 기록·확대가 날아간다)."""
    assert (nav["bottom"]["end"], nav["bottom"]["requests"]) == (100, 3)
    assert (nav["top"]["end"], nav["top"]["requests"]) == (1, 2)
    assert nav["bottom"]["pictureUp"] and nav["top"]["pictureUp"]


def test_a_picture_that_fails_does_not_block(nav):
    """사진을 받지 못한 장에서 멈춰 버리지 않는다 — 알리고 다음 장으로 계속 간다."""
    got = nav["broken_picture"]
    assert got["end"] - 1 >= 25
    assert got["leftBeforePicture"] == 0
    assert any("불러오지 못했습니다" in t for t in got["toasts"])


def test_a_failed_request_does_not_leave_it_stuck(nav):
    """다음 장의 정보를 받지 못하면 그 앞 장에 머문다 — '받는 중' 으로 남아 그 뒤의 입력을 계속 버리지 않는다."""
    got = nav["server_error"]
    assert got["end"] == 3 and got["pictureUp"]
    assert got["loading"] == 0 and not got["stillLoading"]
    assert got["requests"] > 3                              # 뒤의 반복 입력이 다시 시도했다
    assert any("열지 못했습니다" in t for t in got["toasts"])


def test_a_failed_save_stays_on_the_image(nav):
    """저장하지 못한 수정이 있으면 넘어가지 않고, 다음 장을 요청하지도 않는다."""
    got = nav["save_fails"]
    assert (got["end"], got["requests"]) == (1, 0)
    assert got["loading"] == 0 and not got["stillLoading"]
