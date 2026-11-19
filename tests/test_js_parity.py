"""화면(JS)과 서버(Python)에 두 번 쓴 식이 같은 답을 내는가 — 문서의 '고치면 둘 다' 를 대신 지킨다.

`app.js`·`capture.js` 는 전역 스크립트라 통째로 불러올 수 없다. 그래서 필요한 선언만 **글자로 떼어** node 로 돌리고, 같은
입력을 Python 함수에도 넣어 견준다. 떼는 규칙은 단순하다 — 맨 왼쪽에서 시작하는 `function 이름(` 부터 맨 왼쪽 `}` 줄까지,
한 줄짜리 `const 이름 = …`. 선언의 모양을 바꿔 못 찾게 되면 그 이름을 대며 실패한다(그때는 이 파일의 규칙을 맞출 것).

node 는 `find_node` 가 찾는다 — 환경 변수 `NODE_BIN`, PATH 의 node, `C:/Program Files/nodejs/node.exe`, 그것도 없으면 VS Code
(`Code.exe` 를 `ELECTRON_RUN_AS_NODE=1` 로 — node 를 깔지 않은 PC 에서도 화면 테스트가 돈다). 하나도 없으면 건너뛴다.
"""

from __future__ import annotations

import calendar
import functools
import json
import os
import random
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from autolabel import masks
from autolabel.capture import frame_name, notime_name
from autolabel.yolo_io import Box
from helpers import WEB, pixel_rects, random_poly, screen_round_trip

W, H = 2560, 1440


# node 대신 쓸 수 있는 Electron 앱 — ELECTRON_RUN_AS_NODE=1 이면 창을 띄우지 않고 node 처럼 스크립트를 돌린다
_ELECTRON = (r"%LOCALAPPDATA%\Programs\Microsoft VS Code\Code.exe", r"%ProgramFiles%\Microsoft VS Code\Code.exe")


@functools.lru_cache(maxsize=1)
def find_node() -> str | None:
    """화면(JS) 테스트를 돌릴 node 의 경로, 없으면 None(그 테스트들은 건너뛴다).

    차례로: 환경 변수 `NODE_BIN` → PATH 의 node → `C:\\Program Files\\nodejs\\node.exe` → VS Code 의 `Code.exe`.
    **node 가 없는 PC 에서 화면 테스트 38개가 늘 건너뛰어졌다(2026-11-16)** — VS Code 는 Electron 이라
    `ELECTRON_RUN_AS_NODE=1` 이면 node 로 돈다. 이름이 node 가 아닌 것을 고르면(`NODE_BIN` 에 Electron 앱을 적었을 때도) 그
    변수를 이 프로세스의 환경에 넣는다 — 테스트가 띄우는 node 프로세스가 물려받는다(진짜 node 에는 아무 뜻 없는 변수다).
    고른 것이 실제로 스크립트를 돌리는지 한 번 돌려 보고, 안 되면 다음 후보로 넘어간다.
    """
    cands = [os.environ.get("NODE_BIN"), shutil.which("node"), r"C:\Program Files\nodejs\node.exe",
             *(os.path.expandvars(p) for p in _ELECTRON)]
    for cand in cands:
        if not cand or not Path(cand).exists():
            continue
        env = dict(os.environ)
        if not Path(cand).stem.lower().startswith("node"):
            env["ELECTRON_RUN_AS_NODE"] = "1"
        try:
            done = subprocess.run([cand, "-e", "process.stdout.write('ok')"], capture_output=True, text=True,
                                  timeout=60, env=env)
        except (OSError, subprocess.SubprocessError):
            continue
        if done.returncode == 0 and done.stdout.strip() == "ok":
            if "ELECTRON_RUN_AS_NODE" in env:
                os.environ["ELECTRON_RUN_AS_NODE"] = "1"
            return cand
    return None


def js_function(src: str, name: str) -> str:
    lines = src.splitlines()
    heads = (f"function {name}(", f"async function {name}(")
    start = next((i for i, line in enumerate(lines) if line.startswith(heads)), None)
    assert start is not None, f"JS 에서 'function {name}(' 를 찾지 못했습니다 (맨 왼쪽에서 시작해야 한다 — async 는 붙어도 된다)"
    end = next((i for i in range(start, len(lines)) if lines[i] == "}"), None)
    assert end is not None, f"JS 함수 {name} 의 끝(맨 왼쪽 '}}')을 찾지 못했습니다"
    return "\n".join(lines[start:end + 1])


def js_const(src: str, name: str) -> str:
    line = next((line for line in src.splitlines() if line.startswith(f"const {name} = ")), None)
    assert line is not None, f"JS 에서 'const {name} = ' 를 찾지 못했습니다 (맨 왼쪽의 한 줄이어야 한다)"
    return line


def js_const_lines(src: str, name: str) -> str:
    """여러 줄에 걸친 `const 이름 = …;` — 맨 왼쪽 `const 이름 = ` 줄부터 `;` 로 끝나는 첫 줄까지(app.js 의 `esc`)."""
    lines = src.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith(f"const {name} = ")), None)
    assert start is not None, f"JS 에서 'const {name} = ' 를 찾지 못했습니다 (맨 왼쪽에서 시작해야 한다)"
    end = next((i for i in range(start, len(lines)) if lines[i].rstrip().endswith(";")), None)
    assert end is not None, f"JS 의 'const {name}' 이 ';' 로 끝나는 줄을 찾지 못했습니다"
    return "\n".join(lines[start:end + 1])


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


def poly_cover_cases() -> list[tuple[dict, list]]:
    """(박스, 영역 …) — 다각형이 걸린 박스(2026-11-19). 영역은 서버의 꼴(사각형 [x1, y1, x2, y2], 다각형 [[x, y], …]).
    꼬이지 않은 다각형·꼬인 다각형·픽셀에 맞춘 다각형·사각형을 섞고, 박스 가장자리·사각형 위아래에 꼭짓점이나 가로 변이
    딱 걸리는 모양(띠 경계가 겹치는 자리)도 넣는다."""
    rng = random.Random(20261119)
    out = []
    for case in range(700):
        b = {"cx": 0.2 + 0.6 * rng.random(), "cy": 0.2 + 0.6 * rng.random(),
             "w": 0.02 + 0.4 * rng.random(), "h": 0.02 + 0.4 * rng.random()}
        ms: list = []
        for _ in range(rng.choice([1, 1, 2, 3, 5, 8])):
            pick = rng.random()
            if pick < 0.4:
                ms.append([list(p) for p in random_poly(rng, cx=b["cx"] + rng.uniform(-0.2, 0.2),
                                                         cy=b["cy"] + rng.uniform(-0.2, 0.2))])
            elif pick < 0.55:
                ms.append([list(p) for p in random_poly(rng, simple=False, cx=b["cx"], cy=b["cy"])])
            elif pick < 0.75:
                ms.append([list(p) for p in random_poly(rng, grid=(W, H), cx=b["cx"], cy=b["cy"])])
            else:
                x1, x2 = sorted((rng.random(), rng.random()))
                y1, y2 = sorted((rng.random(), rng.random()))
                if x2 > x1 and y2 > y1:
                    ms.append([x1, y1, x2, y2])
        out.append((b, ms))
    # 꼭짓점·변이 박스 경계와 사각형 위아래에 딱 걸리는 모양
    b = {"cx": 0.5, "cy": 0.5, "w": 0.5, "h": 0.5}                    # 박스 [0.25, 0.75]²
    out += [
        (b, [[[0.25, 0.25], [0.75, 0.25], [0.25, 0.75]]]),             # 박스의 정확히 반 — 대각선 아래 삼각형
        (b, [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]]),                   # 박스를 대각선으로 가르는 큰 삼각형
        (b, [[0.25, 0.25, 0.5, 0.75], [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75]]]),   # 사각형과 다각형이 겹침
        (b, [[[0.25, 0.5], [0.75, 0.5], [0.75, 0.6], [0.25, 0.6]]]),   # 가로 변이 사각형처럼 누운 다각형
        (b, [[0.3, 0.5, 0.7, 0.6], [[0.25, 0.5], [0.75, 0.5], [0.5, 0.75]]]),        # 사각형 위 변과 같은 높이의 변
        (b, [[[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]]),       # 8자(꼬인) — 홀짝 규칙
        (b, [[[0.0, 0.0], [0.2, 0.0], [1.0, 0.8], [1.0, 1.0]]]),       # 대각선 띠
        (b, [[[0.6, 0.6], [0.9, 0.6], [0.9, 0.9]]]),                   # 바깥 사각형만 박스에 걸리고 박스 밖
    ]
    return out


def poly_same_pairs() -> list[tuple[list, list, float]]:
    """다각형끼리의 '같은 자리' 쌍 (a, b, 한 좌표에 더한 값) — 꼭짓점 하나의 x 나 y 를 옮긴다. 점 수가 다른 것·사각형과
    견주는 것도 넣는다(늘 다르다 — 그때 값은 nan)."""
    rng = random.Random(29)
    out = []
    for i in range(200):
        a = [list(p) for p in random_poly(rng, grid=(W, H) if i % 2 else None)]
        for d in (*CLEAR_SAME, *CLEAR_DIFFERENT, *(rng.uniform(-2e-7, 2e-7) for _ in range(4))):
            b = [list(p) for p in a]
            b[rng.randrange(len(b))][rng.randrange(2)] += d
            out.append((a, b, d))
        out.append((a, a[:-1], float("nan")))                         # 점 하나가 빠졌다
        out.append((a, list(masks.bbox(a)), float("nan")))             # 바깥 사각형과는 다르다
    sq = [[0.1, 0.2], [0.4, 0.2], [0.4, 0.5], [0.1, 0.5]]
    out.append((sq, [0.1, 0.2, 0.4, 0.5], float("nan")))               # 네모난 다각형도 사각형과는 다르다
    return out


def hand_drawn(rng: random.Random) -> list[float]:
    """손으로 그린 영역 꼴 — 화면 좌표가 픽셀 사이에 떨어진다."""
    x1, x2 = sorted(rng.uniform(0, W) for _ in range(2))
    y1, y2 = sorted(rng.uniform(0, H) for _ in range(2))
    return [x1 / W, y1 / H, x2 / W, y2 / H]


CLEAR_SAME = (0.0, 1e-16, -1e-16, 1e-9, 4e-8, -5e-8, 7e-8)
CLEAR_DIFFERENT = (8e-8, -1e-7, 1e-6, -1e-4)


def same_place_pairs() -> list[tuple[list[float], list[float], float]]:
    """(a, b, 한 좌표에 더한 값) — 뚜렷이 같은 것·뚜렷이 다른 것, 그리고 허용 오차 둘레의 아무 값."""
    rng = random.Random(7)
    # 차가 정확히 허용 오차인 쌍 — '이하' 라 같은 자리다 (0.0 과 SAME 은 빼도 오차가 없다)
    out = [([0.0, 0.0, 0.5, 0.5], [masks.SAME, 0.0, 0.5, 0.5], 0.0),
           ([0.0, masks.SAME, 0.5, 0.5], [0.0, 0.0, 0.5, 0.5], 0.0)]
    bases = [list(r) for r in pixel_rects(W, H, 150)] + [hand_drawn(rng) for _ in range(150)]
    for a in bases:
        for d in (*CLEAR_SAME, *CLEAR_DIFFERENT, *(rng.uniform(-2e-7, 2e-7) for _ in range(6))):
            b = list(a)
            b[rng.randrange(4)] += d
            out.append((a, b, d))
    return out


def template_cases() -> list[dict]:
    """템플릿을 한 장에 적용하는 장면 — saved 는 그 이미지에 이미 있는 영역(DB 의 xyxy), rects 는 템플릿."""
    rng = random.Random(13)
    aligned = [list(r) for r in pixel_rects(W, H, 120)]
    small = [list(r) for r in pixel_rects(1280, 720, 60, seed=17)]
    drawn = [hand_drawn(rng) for _ in range(40)]
    other = [hand_drawn(rng) for _ in range(5)]
    polys = [[list(p) for p in random_poly(rng, grid=(W, H) if i % 2 else None)] for i in range(12)]
    return [
        {"saved": [], "rects": aligned},
        {"saved": [], "rects": small},
        {"saved": [], "rects": drawn},
        # 템플릿의 절반은 전에 화면에서 적용해 저장한 것(좌표가 한 번 오갔다) + 손으로 그린 다른 영역
        {"saved": [list(screen_round_trip(r)) for r in aligned[::2]] + other, "rects": aligned},
        {"saved": other, "rects": aligned[:10] + aligned[:10] + [[v + 1e-9 for v in aligned[0]]]},     # 템플릿 안에서 겹침
        {"saved": [list(r) for r in pixel_rects(W, H, 490, seed=19)], "rects": aligned[:20]},          # 한도를 넘는다
        # 다각형이 섞인 템플릿 (2026-11-19) — 절반은 이미 들어 있다(사각형은 화면을 한 번 오간 값, 다각형은 그대로)
        {"saved": [list(screen_round_trip(r)) for r in aligned[:5]] + polys[::2], "rects": aligned[:10] + polys},
        {"saved": [], "rects": polys + polys[:3]},                                                     # 템플릿 안에서 겹침
    ]


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
def node() -> str:
    found = find_node()
    if found is None:
        pytest.skip("node 가 없어 화면(JS) 쪽을 돌릴 수 없습니다")
    return found


@pytest.fixture(scope="module")
def js(node, tmp_path_factory) -> dict:
    app = (WEB / "app.js").read_text(encoding="utf-8")
    cap = (WEB / "capture.js").read_text(encoding="utf-8")
    script = "\n".join([
        # applyMaskTemplate 이 쓰는 화면 상태·함수의 대역 — 이미지가 떠 있고 영역을 둘 수 있는 장이다
        "const S = { masks: [], cur: { id: 1, width: 2560, height: 1440 }, imgId: 1, maskBlock: null,",
        "            masksTouched: false, msel: -1, maskPoly: true };",
        "const toasts = [];",
        "const toast = (m) => { toasts.push(m); };",
        "const confirm = () => true, pushUndo = () => {}, markDirty = () => {}, refresh = () => {};",
        js_const(app, "maskFromXyxy"), js_const(app, "maskToXyxy"), js_const(app, "MASK_DROP"), js_const(app, "MASK_MAX"),
        js_const(app, "MASK_MAX_PTS"), js_function(app, "ptsBox"), js_function(app, "polyMask"),
        js_function(app, "maskFromWire"), js_function(app, "maskToWire"),
        js_function(app, "maskCover"), js_function(app, "maskSlabs"), js_const(app, "MASK_SAME"),
        js_function(app, "sameRect"), js_function(app, "applyMaskTemplate"),
        js_function(cap, "capName"), js_function(cap, "capNoTimeName"),
        'const input = JSON.parse(require("fs").readFileSync(process.argv[2], "utf8"));',
        "const apply = ({ saved, rects }) => {",
        "  const t = { name: 't', rects, width: 2560, height: 1440 };",
        "  S.masks = saved.map(maskFromWire);                // 이미지를 열었다 (app.js loadImage)",
        "  applyMaskTemplate(t);",
        "  const once = S.masks.length;",
        "  applyMaskTemplate(t);                             // 한 번 더 눌렀다",
        "  const twice = S.masks.length;",
        "  const sent = S.masks.map(maskToWire);             // 저장할 때 서버로 가는 값 (app.js saveBoxes·maskToBody)",
        "  S.masks = sent.map(maskFromWire);                 // 저장한 뒤 다시 열었다",
        "  applyMaskTemplate(t);",
        "  return { once, twice, reopened: S.masks.length, sent };",
        "};",
        "const out = {",
        "  consts: { MASK_DROP, MASK_MAX, MASK_SAME, MASK_MAX_PTS },",
        "  cover_same: input.cover.map(([b, rects]) => maskCover(b, rects.map(maskFromXyxy))),",
        "  screen_xyxy: input.cover.map(([b, rects]) => rects.map((r) => maskToXyxy(maskFromXyxy(r)))),",
        "  pcover: input.pcover.map(([b, ms]) => maskCover(b, ms.map(maskFromWire))),",
        "  pscreen: input.pcover.map(([b, ms]) => ms.map((w) => maskToWire(maskFromWire(w)))),",
        "  same: input.pairs.map(([a, b]) => sameRect(a, b)),",
        "  psame: input.ppairs.map(([a, b]) => sameRect(a, b)),",
        "  pixel: input.pixel.map((r) => { const back = maskToXyxy(maskFromXyxy(r));",
        "    return { back, same: sameRect(r, back) }; }),",
        "  apply: input.apply.map(apply),",
        "  notime: input.notime.map(([cam, tag, sec]) => capNoTimeName(cam, tag, sec)),",
        "  names: input.names.map(([cam, ms]) => capName(cam, new Date(ms))),",
        "};",
        'require("fs").writeFileSync(process.argv[3], JSON.stringify(out));',
    ])
    d = tmp_path_factory.mktemp("js")
    (d / "run.js").write_text(script, encoding="utf-8")
    data = {
        "cover": cover_cases(),
        "pcover": poly_cover_cases(),
        "pairs": [[a, b] for a, b, _ in same_place_pairs()],
        "ppairs": [[a, b] for a, b, _ in poly_same_pairs()],
        "pixel": pixel_rects(W, H, 2000) + pixel_rects(1280, 720, 2000),
        "apply": template_cases(),
        "notime": notime_cases(),
        "names": [["cam-01", calendar.timegm(t.timetuple()) * 1000 + t.microsecond // 1000] for t in name_cases()],
    }
    (d / "in.json").write_text(json.dumps(data), encoding="utf-8")
    done = subprocess.run([node, str(d / "run.js"), str(d / "in.json"), str(d / "out.json")],
                          capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert done.returncode == 0, f"node 실패:\n{done.stderr}"
    return json.loads((d / "out.json").read_text(encoding="utf-8"))


# ----------------------------------------------------------------- 견주기

def test_web_scripts_parse(node):
    """화면 스크립트들이 문법 오류 없이 읽히는가 — 하나라도 깨지면 검수 화면이 통째로 안 뜬다."""
    for name in ("app.js", "gallery.js", "capture.js", "health.js"):
        done = subprocess.run([node, "--check", str(WEB / name)], capture_output=True, text=True, encoding="utf-8")
        assert done.returncode == 0, f"{name}:\n{done.stderr}"


def test_constants_match(js):
    """app.js 의 MASK_MAX·MASK_DROP·MASK_SAME·MASK_MAX_PTS 는 masks.MAX_MASKS·DROP·SAME·MAX_POINTS 와 같은 값이어야 한다."""
    assert js["consts"] == {"MASK_DROP": masks.DROP, "MASK_MAX": masks.MAX_MASKS, "MASK_SAME": masks.SAME,
                            "MASK_MAX_PTS": masks.MAX_POINTS}


def test_polygon_cover_is_the_same_formula(js):
    """다각형이 걸린 박스(2026-11-19) — masks.covered(_slab_union) ↔ app.js maskCover(maskSlabs) 가 끝자리까지 같다. 다각형은
    화면이 받은 꼭짓점을 그대로 들고 있어(사각형처럼 {cx, cy, w, h} 로 오가지 않는다) 서버로 다시 가는 값도 처음과 같다."""
    cases = poly_cover_cases()
    assert len(js["pcover"]) == len(cases) > 700
    hidden = part = 0
    for (b, ms), screen, got in zip(cases, js["pscreen"], js["pcover"]):
        for w, back in zip(ms, screen):
            if masks.is_poly(w):
                assert back == w                                  # 다각형은 화면을 오가도 끝자리까지 그대로
        want = masks.covered(Box(0, b["cx"], b["cy"], b["w"], b["h"]), [masks.as_mask(w) for w in screen])
        assert got == want, (b, ms)
        hidden += want >= masks.DROP
        part += 0.01 <= want < masks.DROP
    assert hidden > 50 and part > 50                              # 빠지는 박스·일부만 가려지는 박스가 둘 다 넉넉하다


def test_same_place_for_polygons_is_the_same_rule(js):
    """다각형의 '같은 자리' — 꼭짓점 수가 같고 순서대로 모든 좌표가 SAME 안. 사각형과 다각형은 늘 다르다. 화면과 서버가 같다."""
    cases = poly_same_pairs()
    assert len(js["psame"]) == len(cases)
    for (a, b, d), got in zip(cases, js["psame"]):
        assert got == masks.same_rect(a, b), (a, b, d)
        if d in CLEAR_SAME:
            assert got is True, (a, d)
        if d in CLEAR_DIFFERENT or d != d:                        # nan — 점 수가 다르거나 사각형과 견줌
            assert got is False, (a, b)


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
    """실제 길 — 서버는 DB 의 xyxy 로, 화면은 그것을 {cx, cy, w, h} 로 바꿔 들고 잰다. 변환 오차(끝자리)만큼만 다르고,
    학습 라벨에서 빠질 박스(DROP 이상)의 판정이 같다(문턱에 끝자리까지 붙은 경우만 빼고 본다)."""
    for (b, rects), got in zip(cover_cases(), js["cover_same"]):
        want = masks.covered(Box(0, b["cx"], b["cy"], b["w"], b["h"]), [tuple(r) for r in rects])
        assert got == pytest.approx(want, abs=1e-9), (b, rects)
        if abs(want - masks.DROP) > 1e-9:
            assert (got >= masks.DROP) == (want >= masks.DROP), (b, rects)


def test_screen_round_trip_is_what_the_python_tests_assume(js):
    """tests/helpers.py 의 screen_round_trip 은 app.js 의 maskFromXyxy·maskToXyxy 와 끝자리까지 같은 계산이다 —
    test_masks·test_store 가 그것으로 '화면을 한 번 오간 값' 을 만든다."""
    cases = pixel_rects(W, H, 2000) + pixel_rects(1280, 720, 2000)
    assert len(js["pixel"]) == len(cases)
    for r, got in zip(cases, js["pixel"]):
        assert tuple(got["back"]) == screen_round_trip(r), r


def test_same_place_is_the_same_rule(js):
    """masks.same_rect ↔ app.js sameRect — 템플릿을 더하고 뺄 때 '같은 자리' 를 가르는 식. 허용 오차 둘레의 값까지 판정이 같다."""
    cases = same_place_pairs()
    assert len(js["same"]) == len(cases)
    seen = set()
    for (a, b, d), got in zip(cases, js["same"]):
        assert got == masks.same_rect(a, b), (a, b, d)
        if d in CLEAR_SAME:
            assert got is True, (a, d)
        if d in CLEAR_DIFFERENT:
            assert got is False, (a, d)
        seen.add(got)
    assert seen == {True, False}


def test_pixel_aligned_rects_stay_the_same_place_on_screen(js):
    """2026-11-12 에 이 파일로 찾은 어긋남 — 소수 7자리로 반올림한 키(xyxyKey·mask_key)를 견주던 때는 픽셀에 딱 맞춘
    좌표(2560·1280 폭)가 반올림 경계에 걸려, 화면을 한 번 오간 것만으로 다른 자리가 됐다(2560×1440 에서 JS 6.9%·Python
    8.4%, 두 키가 서로 다른 영역 22.8%). 지금은 화면도 서버도 전부 같은 자리로 본다."""
    cases = pixel_rects(W, H, 2000) + pixel_rects(1280, 720, 2000)
    moved = 0
    for r, got in zip(cases, js["pixel"]):
        moved += tuple(got["back"]) != r
        assert got["same"] is True, r
        assert masks.same_rect(r, got["back"]), r
    assert moved > 2000                                 # 오가면 끝자리가 정말 달라지는 입력이다


def test_applying_a_template_twice_adds_nothing(js):
    """app.js 의 applyMaskTemplate 을 그대로 돌린다 — 처음 적용한 결과는 서버의 masks.merge 와 같고, 한 번 더 누르거나
    저장한 뒤 다시 열어 눌러도 영역이 늘지 않는다. 한 장 한도를 넘기면 하나도 더하지 않는다."""
    cases = template_cases()
    assert len(js["apply"]) == len(cases)
    for case, got in zip(cases, js["apply"]):
        want = masks.merge(case["saved"], case["rects"])
        if len(want) > masks.MAX_MASKS:
            want = [masks.as_mask(r) for r in case["saved"]]    # 화면은 통째로 거절한다(서버의 여러 장 적용은 그 장을 건너뛴다)
        assert got["once"] == len(want), (len(case["saved"]), len(case["rects"]))
        assert got["twice"] == got["reopened"] == got["once"]
        assert masks.same_rects([masks.as_mask(r) for r in got["sent"]], want)  # 순서까지 같다
    added = [got["once"] - len(case["saved"]) for case, got in zip(cases, js["apply"])]
    assert added == [120, 60, 40, 60, 10, 0, 5 + 6, 12]          # 다각형 템플릿: 사각형 5 + 아직 없는 다각형 6, 겹친 3개는 한 번만


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
    many = "const one = (s) => s;\nconst two = (s) => f(s,\n  (c) => c);\nconst three = 3;\n"
    assert js_const_lines(many, "one") == "const one = (s) => s;"            # 한 줄이어도 그 줄만
    assert js_const_lines(many, "two") == "const two = (s) => f(s,\n  (c) => c);"
    with pytest.raises(AssertionError, match="const four"):
        js_const_lines(many, "four")
