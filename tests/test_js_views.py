"""보기 전환과 키 넘기기 (app.js VIEWS·setView·viewKeydown, 2026-11-15 리뷰 14번) — 화면 코드를 node 로 돌려 본다.

라벨링 말고의 보기(이미지 보기·영상 캡처·데이터 점검)는 app.js 의 `VIEWS` 표 한 곳에 적는다. 전에는 보기를 하나 더할 때마다
setView 의 세 곳·keydown·처음 열 때의 목록·style.css 의 숨김 목록을 따로 고쳤고, keydown 을 빠뜨리면 **숨은 라벨링 이미지에
Space(승인)가 먹었다**. 지금은 표에 `key` 가 없는 보기의 키를 버리는 쪽이 기본이다 — 여기서는 진짜 표에 `key` 없는 보기를
하나 더 넣어, keydown 을 손대지 않아도 그 보기에서 키가 라벨링으로 새지 않는지를 본다.

`VIEWS` 표와 `viewOf`·`setView`·`viewKeydown` 을 app.js 에서 글자로 떼어 **그대로** 돌린다(표는 맨 왼쪽 `const VIEWS = {` 부터
맨 왼쪽 `};` 까지). 그 모양이나 이름을 바꾸면 여기도 맞출 것. node 가 없으면 건너뛴다.
"""

from __future__ import annotations

import re

import pytest

from helpers import WEB, run_node
from test_js_parity import find_node, js_function

NAMES = ("gallery", "work", "capture", "health")


def js_block(src: str, head: str) -> str:
    lines = src.splitlines()
    start = next((i for i, line in enumerate(lines) if line == head), None)
    assert start is not None, f"app.js 에서 '{head}' 줄을 찾지 못했습니다 (맨 왼쪽에서 시작해야 한다)"
    end = next(i for i in range(start, len(lines)) if lines[i] == "};")
    return "\n".join(lines[start:end + 1])


@pytest.fixture(scope="module")
def table() -> str:
    return js_block((WEB / "app.js").read_text(encoding="utf-8"), "const VIEWS = {")


@pytest.fixture(scope="module")
def got(table, tmp_path_factory) -> dict:
    node = find_node()
    if node is None:
        pytest.skip("node 가 없어 화면(JS) 쪽을 돌릴 수 없습니다")
    app = (WEB / "app.js").read_text(encoding="utf-8")
    script = "\n".join([
        "const calls = [], classes = new Set();",
        "const document = { querySelectorAll: () => [],",
        "  body: { classList: { toggle: (name, on) => { if (on) classes.add(name); else classes.delete(name); } } } };",
        "const localStorage = { setItem() {} }, VIEW_KEY = 'view', S = { predOn: false };",
        "const closeMenus = () => {}, markCurrent = () => {}, fetchPreds = () => {};",
        "const resizeCanvas = () => { calls.push('resizeCanvas'); };",
        "let VIEW = 'label';",
        # 보기마다의 진짜 함수 자리 — 불렸는지만 적는다
        *(f"const {n}Show = () => {{ calls.push('{n}Show'); }}, {n}Hide = () => {{ calls.push('{n}Hide'); }}, "
          f"{n}Key = (e) => {{ calls.push('{n}Key:' + e.key); }};" for n in NAMES),
        table,
        # 나중에 누가 보기를 하나 더했다 — 표에 show·hide 만 적고 keydown 은 손대지 않았다
        "VIEWS.brandnew = { show: () => { calls.push('newShow'); }, hide: () => { calls.push('newHide'); } };",
        *(js_function(app, name) for name in ("viewOf", "setView", "viewKeydown")),
        "const press = async (key, mods = {}) => {",
        "  const e = { key, ctrlKey: false, metaKey: false, altKey: false, ...mods, preventDefault() {} };",
        "  const handled = viewKeydown(e);",
        "  await new Promise((done) => setImmediate(done));",
        "  return [handled, VIEW, calls.splice(0)];",
        "};",
        "const go = async (v) => { await setView(v); return [calls.splice(0), [...classes].sort()]; };",
        "(async () => {",
        "  const out = {};",
        "  out.table = Object.fromEntries(Object.entries(VIEWS).map(([n, v]) => [n, { key: !!v.key, vToggle: !!v.vToggle,",
        "    full: !!v.full, show: typeof v.show, hide: typeof v.hide }]));",
        "  out.label_space = await press(' ');",
        "  out.label_ctrl_v = await press('v', { ctrlKey: true });",
        "  out.label_v = await press('v');",
        "  out.gallery_classes = [...classes].sort();",
        "  out.gallery_space = await press(' ');",
        "  out.gallery_v = await press('ㅍ');",
        "  out.to_health = await go('health');",
        "  out.health_space = await press(' ');",
        "  out.health_v = await press('v');",
        "  out.to_capture = await go('capture');",
        "  out.capture_v = await press('V');",
        "  out.to_work = await go('work');",
        "  out.work_space = await press(' ');",
        "  out.work_enter = await press('Enter');",
        "  out.work_v = await press('v');",
        "  out.to_new = await go('brandnew');",
        "  out.new_space = await press(' ');",
        "  out.new_s = await press('s');",
        "  out.new_v = await press('v');",
        "  out.same = await go('brandnew');",
        "  out.to_label = await go('label');",
        "  out.unknown = [viewOf('toString'), viewOf('label'), viewOf(null), viewOf('health') === VIEWS.health];",
        "  console.log(JSON.stringify(out));",
        "})().catch((err) => { console.error((err && err.stack) || err); process.exit(1); });",
    ])
    return run_node(node, script, tmp_path_factory.mktemp("views"))


def test_the_table_lists_every_view_button_and_its_page(table):
    """표의 이름 = 위쪽 줄의 단추(data-view, 라벨링 빼고) = style.css 가 숨기는 화면(`body:not(.view-<이름>)`)."""
    names = re.findall(r"^  (\w+): \{", table, flags=re.M)
    html = (WEB / "index.html").read_text(encoding="utf-8")
    buttons = re.findall(r'class="seg-btn[^"]*" data-view="(\w+)"', html)
    assert buttons == ["label", *names]
    css = (WEB / "style.css").read_text(encoding="utf-8")
    for name in names:
        assert f"body:not(.view-{name}) #{name}" in css, name
    assert "body.view-full .panel-left" in css               # 세 칸을 통째로 쓰는 보기의 숨김 목록은 한 줄뿐이다
    assert not re.search(r"body\.view-(capture|health|work) \.panel-left", css)
    for name in names:                                       # 보기마다 화면(<section id="이름">)이 있다
        assert re.search(rf'<section class="[^"]*" id="{name}">', html), name


def test_the_real_table(got):
    t = got["table"]
    assert {n: (v["key"], v["vToggle"], v["full"]) for n, v in t.items() if n != "brandnew"} == {
        "gallery": (True, True, False), "work": (False, False, True), "capture": (True, False, True),
        "health": (False, False, True)}
    assert all(v["show"] == v["hide"] == "function" for v in t.values())


def test_labeling_gets_its_own_keys_and_v_opens_the_gallery(got):
    assert got["label_space"] == [False, "label", []]              # 라벨링의 Space 는 라벨링 처리기가 받는다(승인)
    assert got["label_ctrl_v"] == [False, "label", []]             # Ctrl+V 는 보기를 바꾸지 않는다
    assert got["label_v"] == [True, "gallery", ["galleryShow"]]
    assert got["gallery_classes"] == ["view-gallery"]
    assert got["gallery_space"] == [True, "gallery", ["galleryKey: "]]     # 이미지 보기의 키는 그쪽으로만
    assert got["gallery_v"] == [True, "label", ["galleryHide", "resizeCanvas"]]    # 한글 입력 상태의 V


def test_a_view_without_a_key_handler_swallows_every_key(got):
    """데이터 점검은 키를 쓰지 않는다 — Space·V 가 숨은 라벨링으로 가지 않는다(handled = true, 아무것도 안 불림)."""
    assert got["to_health"] == [["healthShow"], ["view-full", "view-health"]]
    assert got["health_space"] == [True, "health", []]
    assert got["health_v"] == [True, "health", []]
    assert got["to_capture"] == [["healthHide", "captureShow"], ["view-capture", "view-full"]]
    assert got["capture_v"] == [True, "capture", ["captureKey:V"]]         # 영상 캡처는 V 도 제 키로 받는다


def test_the_work_batch_view_swallows_keys_too(got):
    """작업 배치 보기(2026-11-23)는 키를 쓰지 않는다 — 배치 카드를 보는 동안 Space·Enter 가 숨은 라벨링 이미지를 승인하지 않는다."""
    assert got["to_work"] == [["captureHide", "workShow"], ["view-full", "view-work"]]
    assert got["work_space"] == [True, "work", []]
    assert got["work_enter"] == [True, "work", []]
    assert got["work_v"] == [True, "work", []]


def test_a_view_added_to_the_table_is_safe_without_touching_keydown(got):
    """표에 show·hide 만 적은 새 보기 — keydown 을 고치지 않아도 Space(승인)·S(건너뛰기)·V 가 라벨링으로 새지 않는다."""
    assert got["to_new"] == [["workHide", "newShow"], ["view-brandnew"]]
    assert got["new_space"] == [True, "brandnew", []]
    assert got["new_s"] == [True, "brandnew", []]
    assert got["new_v"] == [True, "brandnew", []]
    assert got["same"] == [[], ["view-brandnew"]]                          # 같은 보기로는 다시 들어가지 않는다
    assert got["to_label"] == [["newHide", "resizeCanvas"], []]
    assert got["unknown"] == [None, None, None, True]                      # 'toString' 같은 이름이 보기로 읽히지 않는다
