"""영상 캡처 화면이 영상을 비울 때(capture.js capReset, 2026-11-16) — 화면 코드를 node 로 돌려 본다.

시작 시각을 고치면 0.6초 뒤 서버에 남기려고 `C.startPending` 에 들어 둔다. 그 영상을 넣거나 닫으면 `capReset` 이 타이머를 끄는데,
**들어 둔 값은 그대로 남아서** 다음에 다른 보기로 갈 때(`captureHide` → `capFlushStart`) 이미 없어진 세션에 PUT 을 보냈다(404).
`capReset`·`capFlushStart`·`captureHide` 를 capture.js 에서 글자로 떼어 그대로 돌린다. node 가 없으면 건너뛴다.
"""

from __future__ import annotations

import pytest

from helpers import WEB, run_node
from test_js_parity import find_node, js_function


@pytest.fixture(scope="module")
def got(tmp_path_factory) -> dict:
    node = find_node()
    if node is None:
        pytest.skip("node 가 없어 화면(JS) 쪽을 돌릴 수 없습니다")
    src = (WEB / "capture.js").read_text(encoding="utf-8")
    script = "\n".join([
        "const sent = [];",
        "const api = (path, opts) => { sent.push([opts.method, path, opts.body]); return Promise.resolve({}); };",
        "const $ = () => ({ value: '', classList: { add() {}, remove() {} } });",
        "const localStorage = { removeItem() {} }, CAP_CUR_KEY = 'cap';",
        "const capStopPoll = () => {}, capClosePreview = () => {}, capFlushKeep = () => {};",
        "let fired = 0;",
        "const C = { cur: { sid: '20261116_101500_ab12', start: '2026-11-10 12:00:00' }, frames: [], cells: [], focus: 0, anchor: 0,",
        "            keepTimer: null, keepDirty: false, shown: true,",
        "            startTimer: setTimeout(() => { fired++; }, 600000),       // 0.6초 뒤 남기려던 타이머(시험에서는 안 울린다)",
        "            startPending: { sid: '20261116_101500_ab12', start: '2026-11-10 12:00:00' } };",
        *(js_function(src, name) for name in ("capReset", "capFlushStart", "captureHide")),
        "const out = {};",
        "capReset();                                 // 그 영상을 닫았다(또는 넣었다)",
        "out.after_reset = { pending: C.startPending, timer: C.startTimer, cur: C.cur };",
        "captureHide();                              // 그리고 다른 보기로 갔다",
        "out.sent_after_closing = sent.splice(0);",
        # 닫지 않고 보기를 떠나면 고친 시각은 전처럼 바로 남긴다
        "C.cur = { sid: 's2', start: 'x' }; C.startPending = { sid: 's2', start: '2026-11-11 08:00:00' };",
        "captureHide();",
        "out.sent_when_leaving = sent.splice(0);",
        "out.pending_after_leaving = C.startPending;",
        "console.log(JSON.stringify(out));",
        "process.exit(0);",
    ])
    return run_node(node, script, tmp_path_factory.mktemp("capture"))


def test_closing_a_video_forgets_its_unsaved_start_time(got):
    assert got["after_reset"] == {"pending": None, "timer": None, "cur": None}
    assert got["sent_after_closing"] == []              # 없어진 세션에 아무것도 보내지 않는다


def test_leaving_the_view_still_saves_a_start_time_in_hand(got):
    assert got["sent_when_leaving"] == [["PUT", "/api/capture/s2/state", {"start": "2026-11-11 08:00:00"}]]
    assert got["pending_after_leaving"] is None
