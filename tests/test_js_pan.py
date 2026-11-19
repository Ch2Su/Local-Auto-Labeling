"""다각형을 그리는 중의 W·A·S·D 화면 이동 (2026-11-20 사용자 요청) — 화면 코드(app.js)를 node 로 돌려 본다.

크게 확대해 넓은 자리를 다각형으로 덮을 때 꼭짓점을 찍는 마우스는 그대로 두고 화면만 옮기려고, 그리는 동안 W·A·S·D 를
누르고 있으면 그쪽을 보러 간다(이미지는 반대로 움직인다 — W 면 위쪽이 보인다). 고치기 전에는 그리는 동안의 W 가 도메인을
비CCTV 로 바꾸고, S 가 건너뛰기를 묻거나 다음 장으로 넘어가고, D 가 화면 끌기 모드로 바꿔 그리던 다각형을 버렸다.

`tests/pan_harness.js` 가 app.js 의 keydown 처리기(`onKeydown`)와 화면 이동 함수들을 **그대로** 돌리고, 시계·프레임(60fps)·키
입력만 가짜로 일으킨다(브라우저는 띄우지 않는다 — 그림이 실제로 그려지는지와 keyup·blur 처리기의 연결은 보지 못한다). 함수는
test_js_parity.py 와 같은 규칙으로 글자로 뗀다 — 이름이나 모양을 바꾸면 PARTS·CONSTS 도 맞출 것.

node 가 없으면 건너뛴다.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import pytest

from helpers import WEB
from test_js_parity import find_node, js_const, js_function

HARNESS = Path(__file__).with_name("pan_harness.js")
CONSTS = ("PAN_DIR", "PAN_KEY", "PAN_SPEED", "PAN_FAST", "PAN")
PARTS = ("panCode", "panKeyDown", "panKeyUp", "panStop", "panFrame", "panBy", "panClamp", "polyKey", "clampImg", "toImage",
         "onKeydown")
PAN_LINE = "  if (panKeyDown(e)) { e.preventDefault(); return; }"    # 고치기 전 꼴 = onKeydown 에서 이 줄을 뺀 것

FRAME = 1000 / 60
REPEAT_DELAY, REPEAT = 500, 33          # 누르고 있으면 500ms 뒤부터 33ms 마다 반복 입력(Windows 기본쯤 — test_js_nav 와 같다)
# 2560×1440 을 2배로 1200×800 캔버스에 — 화면에는 5120×2880. 이미지 가장자리가 화면 가운데까지 오는 오프셋은
# x [600-5120, 600] = [-4520, 600], y [400-2880, 400] = [-2480, 400]
BASE = {"W": 2560, "H": 1440, "scale": 2.0, "cw": 1200, "ch": 800, "ox": -1500.0, "oy": -1000.0, "drawing": True,
        "end": 2000}
ORIGINAL = {"setDomain", "skipOrAdvance", "setMode", "acceptFinding", "setStatus", "step"}    # 키의 원래 뜻
CODE = {"w": "KeyW", "a": "KeyA", "s": "KeyS", "d": "KeyD"}


def hold(t, key, ms, *, code=None, shift=False, up=True):
    """t 에 눌러 ms 동안 누르고 있다가 뗀다(up=False 면 떼지 않는다) — 그 사이의 반복 입력까지."""
    code = CODE[key] if code is None else code
    evs = [{"t": t, "type": "down", "code": code, "key": key, "shift": shift}]
    r = t + REPEAT_DELAY
    while r < t + ms:
        evs.append({"t": r, "type": "down", "code": code, "key": key, "shift": shift, "repeat": True})
        r += REPEAT
    if up:
        evs.append({"t": t + ms, "type": "up", "code": code, "key": key, "shift": shift})
    return evs


def tap(t, key):
    return hold(t, key, 60)


def mark(t, name):
    return {"t": t, "type": "mark", "name": name}


def shift_key(t, down):
    return {"t": t, "type": "down" if down else "up", "code": "ShiftLeft", "key": "Shift", "shift": down}


def unshift_after(evs, t):
    """t 뒤의 입력은 Shift 를 뗀 채로 온다."""
    return [{**ev, "shift": False} if ev["t"] >= t else ev for ev in evs]


SCENARIOS = {
    "hold_d": {"events": hold(0, "d", 1000) + [mark(1000, "release")]},
    "dir_w": {"events": hold(0, "w", 300)},
    "dir_a": {"events": hold(0, "a", 300)},
    "dir_s": {"events": hold(0, "s", 300)},
    "dir_d": {"events": hold(0, "d", 300)},
    "korean": {"events": hold(0, "ㅇ", 1000, code="KeyD")},          # 한글 입력 상태 — e.key 는 자모, 자리(e.code)는 같다
    "no_code": {"events": hold(0, "d", 1000, code="")},              # e.code 가 빈 키보드 — 글자로
    "no_code_jamo": {"events": hold(0, "ㅈ", 300, code="")},
    "shift": {"events": [shift_key(0, True)] + hold(0, "d", 1000, shift=True) + [shift_key(1000, False)]},
    "shift_half": {"events": [shift_key(0, True)] + unshift_after(hold(0, "d", 1000, shift=True), 500)
                   + [shift_key(500, False)]},
    "diagonal": {"events": hold(0, "w", 1000) + hold(0, "d", 1000)},
    "opposite": {"events": hold(0, "a", 1000) + hold(0, "d", 1000)},
    "edge_right": {"ox": -4420.0, "end": 3200, "events": hold(0, "d", 2000) + [mark(2000, "stopped")] + hold(2500, "a", 500)},
    "beyond": {"ox": -4800.0, "end": 1700, "events": hold(0, "d", 500) + [mark(500, "after_d")] + hold(1000, "a", 500)},
    "edge_up": {"oy": 350.0, "events": hold(0, "w", 1000)},
    "edge_down": {"oy": -2450.0, "events": hold(0, "s", 1000)},
    "not_drawing": {"drawing": False, "predOn": True, "events": tap(0, "w") + tap(200, "a") + tap(400, "s") + tap(600, "d")},
    "old_handler": {"code": "old", "predOn": True, "events": tap(0, "w") + tap(200, "a") + tap(400, "s") + tap(600, "d")},
    "close_enter": {"end": 1500, "events": hold(0, "d", 1000) + [{"t": 400, "type": "down", "code": "Enter", "key": "Enter"}]
                    + tap(1200, "d")},
    "close_mouse": {"end": 1500, "events": hold(0, "d", 1000) + [{"t": 400, "type": "closeByMouse"}] + tap(1200, "d")},
    "ctrl": {"events": [{"t": 0, "type": "down", "code": "ControlLeft", "key": "Control", "ctrl": True},
                        {"t": 10, "type": "down", "code": "KeyD", "key": "d", "ctrl": True},
                        {"t": 300, "type": "up", "code": "KeyD", "key": "d", "ctrl": True},
                        {"t": 310, "type": "up", "code": "ControlLeft", "key": "Control"}]},
    "blur": {"end": 1000, "events": hold(0, "d", 300, up=False) + [{"t": 300, "type": "blur"}]},
    "follow_cursor": {"mouse": [900, 300], "events": hold(0, "d", 500)},
    "drag_pan": {"dragPan": True, "events": hold(0, "d", 500)},
    "stall": {"noFrames": [200, 700], "events": hold(0, "d", 1000)},
}


@pytest.fixture(scope="module")
def pan(tmp_path_factory) -> dict:
    node = find_node()
    if node is None:
        pytest.skip("node 가 없어 화면(JS) 쪽을 돌릴 수 없습니다")
    app = (WEB / "app.js").read_text(encoding="utf-8")
    code = "\n\n".join([*(js_const(app, name) for name in CONSTS), *(js_function(app, name) for name in PARTS)])
    assert PAN_LINE + "\n" in code, "onKeydown 에서 W·A·S·D 를 받는 줄을 찾지 못했습니다 — 바꿨으면 PAN_LINE 을 맞출 것"
    d = tmp_path_factory.mktemp("pan")
    (d / "code.js").write_text(code, encoding="utf-8")
    (d / "old.js").write_text(code.replace(PAN_LINE + "\n", ""), encoding="utf-8")
    (d / "in.json").write_text(json.dumps([{**BASE, **sc} for sc in SCENARIOS.values()]), encoding="utf-8")
    done = subprocess.run([node, str(HARNESS), str(d / "code.js"), str(d / "old.js"), str(d / "in.json"), str(d / "out.json")],
                          capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert done.returncode == 0, f"node 실패:\n{done.stderr}"
    return dict(zip(SCENARIOS, json.loads((d / "out.json").read_text(encoding="utf-8"))))


def speed(got) -> float:
    """1초에 옮기는 화면 px — 캔버스 짧은 변 × PAN_SPEED."""
    return got["speed"] * min(BASE["cw"], BASE["ch"])


def near(moved, px_per_s, seconds):
    """누른 시간만큼 옮겼나 — 프레임이 16.7ms 마다라 마지막 한 프레임만큼 덜 갈 수 있다."""
    return px_per_s * (seconds - FRAME / 1000) - 1e-6 <= moved <= px_per_s * seconds + 1e-6


def originals(got):
    return [c for c in got["calls"] if c[0] in ORIGINAL]


def test_holding_d_moves_right_and_keeps_the_polygon(pan):
    """D 를 1초 누르고 있으면 오른쪽을 보러 간다(이미지가 왼쪽으로) — 그리던 다각형·다른 축은 그대로, 원래 뜻(D = 화면 끌기
    모드 — 그리던 다각형을 버렸다)으로 새지 않는다. 손을 떼면 서고 프레임도 더 돌지 않는다."""
    got = pan["hold_d"]
    assert near(BASE["ox"] - got["ox"], speed(got), 1.0)
    assert got["oy"] == BASE["oy"]
    assert got["poly"]["n"] == 2
    assert originals(got) == [] and got["calls"] == []
    assert got["prevented"] == got["keydowns"] == 1 + len(range(REPEAT_DELAY, 1000, REPEAT))   # 반복 입력까지 다 받았다
    assert got["marks"]["release"] == {"ox": got["ox"], "oy": got["oy"]}
    assert (got["held"], got["rafPending"], got["raf"]) == (0, 0, 0)
    assert got["draws"] >= 50                                       # 프레임마다 다시 그렸다


@pytest.mark.parametrize("name,axis,sign", [("dir_w", "oy", 1), ("dir_a", "ox", 1), ("dir_s", "oy", -1), ("dir_d", "ox", -1)])
def test_each_key_looks_its_way(pan, name, axis, sign):
    """W 위·A 왼쪽·S 아래·D 오른쪽을 보러 간다 — 이미지(오프셋)는 반대로: W 면 아래로, D 면 왼쪽으로."""
    got = pan[name]
    other = "ox" if axis == "oy" else "oy"
    assert near(sign * (got[axis] - BASE[axis]), speed(got), 0.3)
    assert got[other] == BASE[other]
    assert originals(got) == []


def test_korean_input_and_missing_code(pan):
    """한글 입력 상태(e.key 가 'ㅇ')여도 자리로 보고 같게 움직인다. e.code 가 없으면 글자(자모까지)로 본다."""
    assert pan["korean"]["ox"] == pan["no_code"]["ox"] == pan["hold_d"]["ox"]
    assert pan["korean"]["calls"] == pan["no_code"]["calls"] == []
    got = pan["no_code_jamo"]
    assert near(got["oy"] - BASE["oy"], speed(got), 0.3) and got["calls"] == []


def test_shift_goes_faster(pan):
    """Shift 를 함께 누르면 PAN_FAST 배 — 누르는 도중에 떼면 그때부터 원래 빠르기로."""
    got = pan["shift"]
    assert got["fast"] > 1
    assert near(BASE["ox"] - got["ox"], speed(got) * got["fast"], 1.0)
    half = pan["shift_half"]
    moved = BASE["ox"] - half["ox"]
    want = speed(half) * (0.5 * half["fast"] + 0.5)
    assert abs(moved - want) <= speed(half) * half["fast"] * FRAME / 1000 * 2
    assert half["calls"] == []


def test_two_keys(pan):
    """W+D 는 대각선(빠르기는 한 키와 같다), A+D 는 서로 지워 서 있다 — 둘 다 원래 뜻으로 새지 않는다."""
    got = pan["diagonal"]
    dx, dy = got["ox"] - BASE["ox"], got["oy"] - BASE["oy"]
    assert dx < 0 < dy and abs(abs(dx) - dy) < 1e-6
    assert near(math.hypot(dx, dy), speed(got), 1.0)
    opp = pan["opposite"]
    assert (opp["ox"], opp["oy"]) == (BASE["ox"], BASE["oy"])
    assert opp["calls"] == [] and opp["prevented"] == opp["keydowns"]


def test_stops_with_the_image_edge_at_the_middle(pan):
    """이미지 가장자리가 화면 가운데에 오면 더 옮기지 않는다(오래 눌러 이미지를 잃지 않게) — 네 쪽 모두. 이미 넘어가 있으면
    (마우스로 끌어서) 더 멀어지지만 않고, 돌아오는 쪽으로는 움직인다."""
    right = pan["edge_right"]
    assert abs(right["marks"]["stopped"]["ox"] - (BASE["cw"] / 2 - BASE["W"] * BASE["scale"])) < 1e-9
    assert near(right["ox"] - right["marks"]["stopped"]["ox"], speed(right), 0.5)        # A 로는 다시 움직인다
    beyond = pan["beyond"]
    assert beyond["marks"]["after_d"]["ox"] == -4800.0
    assert near(beyond["ox"] - (-4800.0), speed(beyond), 0.5)
    assert abs(pan["edge_up"]["oy"] - BASE["ch"] / 2) < 1e-9
    assert abs(pan["edge_down"]["oy"] - (BASE["ch"] / 2 - BASE["H"] * BASE["scale"])) < 1e-9


def test_keys_keep_their_meaning_when_not_drawing(pan):
    """그리는 중이 아니면 W·A·S·D 는 원래 뜻 그대로다 — 비CCTV·예측 받기·건너뛰기·화면 끌기 모드. 화면은 움직이지 않는다."""
    got = pan["not_drawing"]
    assert got["calls"] == [["setDomain", "noncctv"], ["acceptFinding"], ["skipOrAdvance"], ["setMode", "pan"]]
    assert (got["ox"], got["oy"], got["held"], got["rafPending"]) == (BASE["ox"], BASE["oy"], 0, 0)


def test_the_bug_before_the_fix(pan):
    """대조 — onKeydown 에서 W·A·S·D 를 받는 줄을 빼면(2026-11-20 전의 처리기) 그리는 중에도 원래 뜻으로 샌다. 이 모의가 그
    차이를 실제로 가른다는 확인이다(그래서 위 테스트들이 뜻이 있다)."""
    got = pan["old_handler"]
    assert got["calls"] == [["setDomain", "noncctv"], ["acceptFinding"], ["skipOrAdvance"], ["setMode", "pan"]]
    assert (got["ox"], got["oy"]) == (BASE["ox"], BASE["oy"])


@pytest.mark.parametrize("name", ["close_enter", "close_mouse"])
def test_closing_while_held(pan, name):
    """D 를 누른 채 다각형을 닫으면(Enter·첫 점 누르기) 거기서 서고, 그 뒤의 반복 입력은 뗄 때까지 받아 버린다 — 화면 끌기
    모드로 바뀌지 않는다. 떼고 다시 누른 D 는 원래 뜻(화면 끌기 모드)이다."""
    got = pan[name]
    assert near(BASE["ox"] - got["ox"], speed(got), 0.4)
    assert got["poly"] is None
    assert got["calls"] == ([["polyClose"]] if name == "close_enter" else []) + [["setMode", "pan"]]
    assert (got["held"], got["rafPending"]) == (0, 0)


def test_ctrl_d_is_not_a_pan(pan):
    """Ctrl 과 함께면 받지 않는다(Ctrl 은 누르는 동안 화면 끌기 — 브라우저 단축키도 막지 않는다)."""
    got = pan["ctrl"]
    assert (got["ox"], got["oy"]) == (BASE["ox"], BASE["oy"])
    assert got["calls"] == [] and got["prevented"] == 0


def test_losing_focus_stops(pan):
    """창이 포커스를 잃으면(뗀 것을 받지 못한다) 거기서 선다 — 떼지 않았어도 혼자 계속 가지 않는다."""
    got = pan["blur"]
    assert near(BASE["ox"] - got["ox"], speed(got), 0.3)
    assert (got["held"], got["rafPending"], got["raf"]) == (0, 0, 0)


def test_dashed_line_follows_the_cursor(pan):
    """마우스를 움직이지 않아도 커서 밑의 이미지 자리가 바뀌므로, 마지막 점에서 커서까지의 점선 끝(S.poly.cur)을 다시 센다."""
    got = pan["follow_cursor"]
    cx, cy = got["poly"]["cur"]
    assert abs(cx - (900 - got["ox"]) / BASE["scale"]) < 1e-9
    assert abs(cy - (300 - got["oy"]) / BASE["scale"]) < 1e-9


def test_mouse_pan_drag_moves_along(pan):
    """가운데 버튼·Ctrl 로 끄는 도중이면 끌기의 출발점도 같이 옮긴다 — 다음 마우스 움직임이 키로 옮긴 만큼을 되돌리지 않게."""
    got = pan["drag_pan"]
    assert got["drag"]["ox"] - BASE["ox"] == pytest.approx(got["ox"] - BASE["ox"], abs=1e-9)
    assert got["drag"]["oy"] == BASE["oy"]
    assert got["ox"] < BASE["ox"]


def test_a_stalled_tab_does_not_jump(pan):
    """프레임이 0.5초 멈췄다 돌아와도 한 프레임에 0.1초 만큼만 간다 — 한 번에 크게 뛰지 않는다. 0.2초까지 0.2초어치 + 멈춘 뒤 첫
    프레임 0.1초어치 + 0.7초부터 뗄 때까지 0.28초어치(마지막 프레임 983ms)로 0.58초어치다 — 막지 않으면 0.98초어치."""
    got = pan["stall"]
    moved = BASE["ox"] - got["ox"]
    assert 0.55 * speed(got) <= moved <= 0.62 * speed(got)
