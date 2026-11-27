"""필터 칸의 장수 목록 받기 (app.js loadState·loadSplits·listRequest·listStale) — 화면 코드를 node 로 돌려 본다.

2026-11-27, Gemini 리뷰 묶음 8:
- 박스를 저장할 때마다(0.6초 자동 저장) `loadState` 가 필터 칸 목록 다섯(카메라·도메인·분할·태그·작업 배치)까지 다시 받아
  요청 여섯 개가 나갔다. 박스 저장은 그 장수를 바꾸지 않는다 — `loadState({ lists: false })` 는 `/api/state` 하나만 받는다.
- 같은 목록을 연달아 받으면 응답이 뒤바뀌어 올 수 있다. 늦게 온 옛 응답은 버리고, 옛 요청을 기다리던 쪽도 마지막 요청이
  끝난 뒤에 돌아간다(부르는 쪽이 `await loadWorks()` 뒤에 S.works 를 읽는다).

`loadState`·`loadSplits`·`listRequest`·`listStale`·`LIST_LAST` 를 app.js 에서 글자로 떼어 그대로 돌린다. 서버 응답은 테스트가
원하는 순서로 내준다(요청마다 풀어 줄 손잡이를 쥔다).
"""

from __future__ import annotations

import pytest

from helpers import WEB, run_node
from test_js_parity import find_node, js_const, js_function

PARTS = ("listRequest", "listStale", "loadSplits", "loadState")


@pytest.fixture(scope="module")
def got(tmp_path_factory) -> dict:
    node = find_node()
    if node is None:
        pytest.skip("node 가 없어 화면(JS) 쪽을 돌릴 수 없습니다")
    app = (WEB / "app.js").read_text(encoding="utf-8")
    script = "\n".join([
        "const el = { innerHTML: '', disabled: false };",
        "const $ = () => el;",
        "const S = { filterClasses: new Set(), filterSplit: '', jobPoll: null };",
        "const calls = [], pending = [], filled = [];",
        # /api/state 는 바로 답하고, 그 밖의 요청은 테스트가 풀어 줄 때까지 기다린다
        "const api = (url) => { calls.push(url);",
        "  if (url === '/api/state') return Promise.resolve({ classes: [], counts: {}, class_boxes: [] });",
        "  return new Promise((ok, fail) => pending.push({ url, ok, fail })); };",
        "const renderClassList = () => {}, renderClassFilter = () => {}, watchJob = () => {};",
        "const fillCountFilter = (sel, keep, all, items) => { filled.push(items[0].n); return keep; };",
        "const other = (name) => async () => { calls.push(name); };",
        "const loadCameras = other('cameras'), loadDomains = other('domains'), loadTags = other('tags'), loadWorks = other('works');",
        js_const(app, "LIST_LAST"),
        *(js_function(app, name) for name in PARTS),
        "const tick = () => new Promise((done) => setImmediate(done));",
        "const answer = (i, n) => pending[i].ok({ splits: [{ key: 'train', n }], none: 0, none_key: '__none__' });",
        "(async () => {",
        "  const out = {};",
        "  await loadState({ lists: false }); await tick();",
        "  out.light = [...calls]; calls.length = 0;",
        "  await loadState(); await tick();",
        "  out.full = [...calls]; calls.length = 0; pending.length = 0;",
        # 두 번 걸고, 뒤의 것(새 값 2)이 먼저, 앞의 것(옛 값 1)이 나중에 온다
        "  const order = [];",
        "  const first = loadSplits().then(() => order.push('first:' + filled.join(',')));",
        "  const second = loadSplits().then(() => order.push('second:' + filled.join(',')));",
        "  answer(1, 2); await tick();",
        "  out.after_new = [...filled];",
        "  answer(0, 1); await first; await second;",
        "  out.reversed = { filled: [...filled], order };",
        # 앞의 것이 먼저 오면: 뒤의 것이 올 때까지 기다렸다가 돌아간다(그리고 옛 값은 쓰지 않는다)
        "  filled.length = 0; pending.length = 0; const order2 = [];",
        "  const a = loadSplits().then(() => order2.push('first:' + filled.join(',')));",
        "  const b = loadSplits().then(() => order2.push('second:' + filled.join(',')));",
        "  answer(0, 1); await tick();",
        "  out.waiting = { filled: [...filled], returned: order2.length };",
        "  answer(1, 2); await a; await b;",
        "  out.in_order = { filled: [...filled], order: order2 };",
        # 실패한 요청은 조용히 넘어간다(검수는 되어야 한다)
        "  filled.length = 0; pending.length = 0;",
        "  const c = loadSplits(); pending[0].fail(new Error('down')); await c;",
        "  out.failed = [...filled];",
        "  console.log(JSON.stringify(out));",
        "})().catch((err) => { console.error((err && err.stack) || err); process.exit(1); });",
    ])
    return run_node(node, script, tmp_path_factory.mktemp("lists"))


def test_a_box_save_refreshes_only_the_state(got):
    assert got["light"] == ["/api/state"]
    assert sorted(got["full"]) == sorted(["/api/state", "cameras", "domains", "/api/splits", "tags", "works"])


def test_an_old_answer_that_arrives_late_is_dropped(got):
    """뒤에 건 요청의 답(2)이 먼저 그려지고, 늦게 온 옛 답(1)은 그 위에 덮이지 않는다."""
    assert got["after_new"] == [2]
    assert got["reversed"]["filled"] == [2]
    assert sorted(got["reversed"]["order"]) == ["first:2", "second:2"]


def test_a_caller_of_the_old_request_returns_only_after_the_newest_arrived(got):
    """옛 요청의 답이 먼저 와도 그것으로 그리지 않고, 그 호출은 마지막 요청이 들어온 뒤에야 돌아간다."""
    assert got["waiting"] == {"filled": [], "returned": 0}
    assert got["in_order"]["filled"] == [2]
    assert sorted(got["in_order"]["order"]) == ["first:2", "second:2"]


def test_a_failed_request_fills_nothing(got):
    assert got["failed"] == []
