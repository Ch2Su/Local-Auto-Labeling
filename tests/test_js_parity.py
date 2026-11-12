"""화면(JS)과 서버(Python)에 두 번 쓴 식이 같은 답을 내는가 — 문서의 '고치면 둘 다' 를 대신 지킨다.

`app.js`·`capture.js` 는 전역 스크립트라 통째로 불러올 수 없다. 그래서 필요한 선언만 **글자로 떼어** node 로 돌리고, 같은
입력을 Python 함수에도 넣어 견준다. 떼는 규칙은 단순하다 — 맨 왼쪽에서 시작하는 `function 이름(` 부터 맨 왼쪽 `}` 줄까지,
한 줄짜리 `const 이름 = …`. 선언의 모양을 바꿔 못 찾게 되면 그 이름을 대며 실패한다(그때는 이 파일의 규칙을 맞출 것).

node 가 없으면 건너뛴다(PATH 의 node, 없으면 `C:/Program Files/nodejs/node.exe`).
"""

from __future__ import annotations

import calendar
import json
import random
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from autolabel import masks
from autolabel.capture import frame_name, notime_name
from autolabel.yolo_io import Box
from helpers import WEB

W, H = 2560, 1440


def find_node() -> str | None:
    for cand in (shutil.which("node"), r"C:\Program Files\nodejs\node.exe"):
        if cand and Path(cand).exists():
            return cand
    return None


def js_function(src: str, name: str) -> str:
    lines = src.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith(f"function {name}(")), None)
    assert start is not None, f"JS 에서 'function {name}(' 를 찾지 못했습니다 (맨 왼쪽에서 시작해야 한다)"
    end = next((i for i in range(start, len(lines)) if lines[i] == "}"), None)
    assert end is not None, f"JS 함수 {name} 의 끝(맨 왼쪽 '}}')을 찾지 못했습니다"
    return "\n".join(lines[start:end + 1])


def js_const(src: str, name: str) -> str:
    line = next((line for line in src.splitlines() if line.startswith(f"const {name} = ")), None)
    assert line is not None, f"JS 에서 'const {name} = ' 를 찾지 못했습니다 (맨 왼쪽의 한 줄이어야 한다)"
    return line


# ----------------------------------------------------------------- 입력

def cover_cases() -> list[tuple[dict, list[list[float]]]]:
    """(박스 {cx, cy, w, h}, 영역 [x1, y1, x2, y2] …) — 픽셀 격자·성긴 격자·아무 실수를 섞는다."""
    rng = random.Random(20261112)
    coarse = [i / 16 for i in range(17)]
    picks = [rng.random, lambda: rng.choice(coarse), lambda: rng.randrange(0, W + 1) / W]
    out = [({"cx": 0.5, "cy": 0.5, "w": 0.5, "h": 0.5}, [[0.25, 0.25, 0.5, 0.75]]),      # 정확히 반
           ({"cx": 0.5, "cy": 0.5, "w": 0.5, "h": 0.5}, []),
           ({"cx": 0.5, "cy": 0.5, "w": 0.0, "h": 0.5}, [[0.0, 0.0, 1.0, 1.0]])]
    for case in range(900):
        pick = picks[case % 3]
        b = {"cx": 0.2 + 0.6 * rng.random(), "cy": 0.2 + 0.6 * rng.random(),
             "w": 0.02 + 0.3 * rng.random(), "h": 0.02 + 0.3 * rng.random()}
        rects = []
        for _ in range(rng.choice([0, 1, 2, 3, 5, 8, 13, 40])):
            x1, x2 = sorted((pick(), pick()))
            y1, y2 = sorted((pick(), pick()))
            if x2 > x1 and y2 > y1:
                rects.append([x1, y1, x2, y2])
        out.append((b, rects))
    return out


def key_cases() -> list[list[float]]:
    """손으로 그린 영역 꼴 — 화면 좌표가 픽셀 사이에 떨어진다(지금 DB 의 영역이 전부 이 꼴이다)."""
    rng = random.Random(7)
    out = []
    for _ in range(400):
        x1, x2 = sorted(rng.uniform(0, W) for _ in range(2))
        y1, y2 = sorted(rng.uniform(0, H) for _ in range(2))
        r = [x1 / W, y1 / H, x2 / W, y2 / H]
        out += [r, [v + rng.uniform(-1e-9, 1e-9) for v in r], [rng.random() for _ in range(4)]]
    return out


def pixel_key_cases() -> list[list[float]]:
    """픽셀에 딱 맞춘 영역 — `tools/mask_images.py --rect` 가 만드는 꼴."""
    rng = random.Random(11)
    out = []
    for _ in range(400):
        x1, x2 = sorted(rng.sample(range(W + 1), 2))
        y1, y2 = sorted(rng.sample(range(H + 1), 2))
        out.append([x1 / W, y1 / H, x2 / W, y2 / H])
    return out


def py_key(r) -> str:
    (key,) = masks.mask_key([r])
    return ",".join(f"{v:.7f}" for v in key)


def notime_cases() -> list[tuple[str, str, float]]:
    rng = random.Random(3)
    secs = [0, 0.0004, 0.0005, 0.4994, 0.4996, 0.9996, 1.0005, 59.9996, 65.5, 3599.9996, 3600, 86399.9994,
            123456.789, 359999.9996, -3, -0.0004]
    secs += [round(rng.uniform(0, 30000), 3) for _ in range(300)]       # 화면이 주는 꼴(밀리초까지)
    secs += [rng.uniform(0, 30000) for _ in range(300)]
    secs += [k / 30 for k in range(0, 9000, 37)] + [k * 1001 / 30000 for k in range(0, 9000, 41)]   # 30·29.97fps 프레임 시각
    return [("cam-01", "0a1b2c3d", float(s)) for s in secs]


def name_cases() -> list[datetime]:
    rng = random.Random(5)
    out = [datetime(2026, 11, 3, 14, 0, 5, 120000), datetime(2026, 1, 1, 0, 0, 0), datetime(2026, 12, 31, 23, 59, 59, 999000),
           datetime(2028, 2, 29, 9, 5, 7, 7000)]
    out += [datetime(2026, 1, 1) + timedelta(milliseconds=rng.randrange(0, 365 * 86400 * 1000)) for _ in range(200)]
    return out


# ----------------------------------------------------------------- node 로 한 번 돌린다

@pytest.fixture(scope="module")
def js(tmp_path_factory) -> dict:
    node = find_node()
    if node is None:
        pytest.skip("node 가 없어 화면(JS) 쪽을 돌릴 수 없습니다")
    app = (WEB / "app.js").read_text(encoding="utf-8")
    cap = (WEB / "capture.js").read_text(encoding="utf-8")
    script = "\n".join([
        "const S = { masks: [] };",
        js_const(app, "maskFromXyxy"), js_const(app, "maskToXyxy"), js_const(app, "MASK_DROP"), js_const(app, "MASK_MAX"),
        js_function(app, "maskCover"), js_const(app, "xyxyKey"),
        js_function(cap, "capName"), js_function(cap, "capNoTimeName"),
        'const input = JSON.parse(require("fs").readFileSync(process.argv[2], "utf8"));',
        "const out = {",
        "  consts: { MASK_DROP, MASK_MAX },",
        "  cover_same: input.cover.map(([b, rects]) => maskCover(b, rects.map(maskFromXyxy))),",
        "  screen_xyxy: input.cover.map(([b, rects]) => rects.map((r) => maskToXyxy(maskFromXyxy(r)))),",
        "  keys: input.keys.map((r) => xyxyKey(r)),",
        "  pixel: input.pixel.map((r) => { const back = maskToXyxy(maskFromXyxy(r));",
        "    return { key: xyxyKey(r), back, backKey: xyxyKey(back) }; }),",
        "  notime: input.notime.map(([cam, tag, sec]) => capNoTimeName(cam, tag, sec)),",
        "  names: input.names.map(([cam, ms]) => capName(cam, new Date(ms))),",
        "};",
        'require("fs").writeFileSync(process.argv[3], JSON.stringify(out));',
    ])
    d = tmp_path_factory.mktemp("js")
    (d / "run.js").write_text(script, encoding="utf-8")
    data = {
        "cover": cover_cases(),
        "keys": key_cases(),
        "pixel": pixel_key_cases(),
        "notime": notime_cases(),
        "names": [["cam-01", calendar.timegm(t.timetuple()) * 1000 + t.microsecond // 1000] for t in name_cases()],
    }
    (d / "in.json").write_text(json.dumps(data), encoding="utf-8")
    done = subprocess.run([node, str(d / "run.js"), str(d / "in.json"), str(d / "out.json")],
                          capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert done.returncode == 0, f"node 실패:\n{done.stderr}"
    return json.loads((d / "out.json").read_text(encoding="utf-8"))


# ----------------------------------------------------------------- 견주기

def test_constants_match(js):
    """app.js 의 MASK_MAX·MASK_DROP 은 masks.MAX_MASKS·DROP 과 같은 값이어야 한다."""
    assert js["consts"] == {"MASK_DROP": masks.DROP, "MASK_MAX": masks.MAX_MASKS}


def test_mask_cover_is_the_same_formula(js):
    """masks.covered ↔ app.js maskCover — 화면이 다루는 좌표(영역을 {cx, cy, w, h} 로 들고 있다가 xyxy 로 되돌린 값)를
    똑같이 넣으면 끝자리까지 같아야 한다(더하는 순서까지 맞춘 식이다 — masks.covered 머리말)."""
    cases = cover_cases()
    assert len(js["cover_same"]) == len(cases) > 900
    hidden = 0
    for (b, _), rects, got in zip(cases, js["screen_xyxy"], js["cover_same"]):
        want = masks.covered(Box(0, b["cx"], b["cy"], b["w"], b["h"]), [tuple(r) for r in rects])
        assert got == want, (b, rects)
        hidden += want >= masks.DROP
    assert 50 < hidden < len(cases) - 50                # 빠지는 박스와 남는 박스가 둘 다 넉넉히 나오는 입력이다


def test_screen_and_training_agree_on_which_boxes_drop(js):
    """실제 길 — 서버는 DB 의 xyxy 로, 화면은 그것을 {cx, cy, w, h} 로 바꿔 들고 잰다. 변환 오차(끝자리)만큼만 다르다."""
    for (b, rects), got in zip(cover_cases(), js["cover_same"]):
        want = masks.covered(Box(0, b["cx"], b["cy"], b["w"], b["h"]), [tuple(r) for r in rects])
        assert got == pytest.approx(want, abs=1e-9), (b, rects)


def test_same_place_key_is_the_same_rule(js):
    """masks.mask_key ↔ app.js xyxyKey — 템플릿을 더하고 뺄 때 '같은 자리' 를 가르는 값(소수 7자리)."""
    cases = key_cases()
    assert len(js["keys"]) == len(cases)
    for r, got in zip(cases, js["keys"]):
        assert got == py_key(r), r


@pytest.mark.xfail(strict=True, reason="known: 7-digit rounding boundary on pixel-aligned rects (see docstring)")
def test_same_place_key_for_pixel_aligned_rects(js):
    """**알려진 어긋남 (2026-11-12 에 이 테스트로 찾았다 — 아직 고치지 않았다).**

    픽셀에 딱 맞춘 좌표는 2560·1280 폭에서 소수 8번째 자리가 정확히 5 인 값이 많다(예: 1326/2560 = 0.51796875) — 7자리
    반올림의 경계다. 거기서 JS 의 `Math.round(v * 1e7)` 는 늘 올리고 Python 의 `round(v, 7)` 는 실제 값(경계 바로 위나
    아래)대로 가서 두 키가 달라지고, 화면이 영역을 {cx, cy, w, h} 로 들었다 xyxy 로 되돌릴 때 생기는 끝자리 잡음만으로도
    키가 바뀐다. 그러면 '같은 자리' 로 못 알아봐, 같은 템플릿을 두 번 더하면 겹쳐 들어가거나 빼기가 그 영역을 남길 수 있다.
    11-12 실측(무작위 2만 개): 2560×1440 에서 두 키가 다른 영역 22.8%, 화면을 한 번 오가면 키가 바뀌는 영역 JS 6.9%·
    Python 8.4%(1280×720 은 44.5%·14.5%·16.5%, 1920×1080 은 0). 손으로 그린 영역(픽셀 사이에 떨어진다 — 그날 DB 의
    561개 전부)에는 생기지 않았다. 고치면(반올림 대신 허용 오차로 견주는 식 등) 위 xfail 표시를 뗄 것 — strict 라 고쳐지면
    이 테스트가 '뜻밖의 통과' 로 실패해 알려 준다.
    """
    cases = pixel_key_cases()
    assert len(js["pixel"]) == len(cases)
    for r, got in zip(cases, js["pixel"]):
        assert got["key"] == py_key(r), r                       # 화면과 서버가 같은 키
        assert got["backKey"] == got["key"], r                  # 화면을 한 번 오가도 같은 자리 (JS)
        assert py_key(got["back"]) == py_key(r), r              # 〃 (Python)


def test_notime_frame_name_is_the_same(js):
    """capture.notime_name ↔ capture.js capNoTimeName — 화면의 미리 보기 이름과 실제로 넣는 이름."""
    cases = notime_cases()
    assert len(js["notime"]) == len(cases)
    for (cam, tag, sec), got in zip(cases, js["notime"]):
        assert got == notime_name(cam, tag, sec), sec


def test_timed_frame_name_is_the_same(js):
    """capture.frame_name ↔ capture.js capName."""
    for t, got in zip(name_cases(), js["names"]):
        assert got == {"day": f"{t:%Y%m%d}", "name": frame_name("cam-01", t)}, t


# ----------------------------------------------------------------- 떼는 규칙 자체

def test_extraction_rules():
    src = "const A = 1;      // 주석\nfunction f(x) {\n  if (x) {\n    return 1;\n  }\n  return 2;\n}\nfunction g() {\n}\n"
    assert js_const(src, "A") == "const A = 1;      // 주석"
    assert js_function(src, "f").splitlines()[0] == "function f(x) {"
    assert js_function(src, "f").count("\n") == 5
    with pytest.raises(AssertionError, match="function h"):
        js_function(src, "h")
    with pytest.raises(AssertionError, match="const B"):
        js_const(src, "B")
