"""영상 캡처 화면이 영상을 비울 때(capture.js capReset, 2026-11-16) — 화면 코드를 node 로 돌려 본다.

시작 시각을 고치면 0.6초 뒤 서버에 남기려고 `C.startPending` 에 들어 둔다. 그 영상을 넣거나 닫으면 `capReset` 이 타이머를 끄는데,
**들어 둔 값은 그대로 남아서** 다음에 다른 보기로 갈 때(`captureHide` → `capFlushStart`) 이미 없어진 세션에 PUT 을 보냈다(404).
`capReset`·`capFlushStart`·`captureHide` 를 capture.js 에서 글자로 떼어 그대로 돌린다. node 가 없으면 건너뛴다.

카메라를 고르는 칸의 글(`capCamLabel`·`capRenderCameras`, 2026-11-17)도 여기서 본다 — 키 옆에 장소가 나오는가.
"""

from __future__ import annotations

import pytest

from helpers import WEB, run_node
from test_js_parity import find_node, js_const, js_function


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


@pytest.fixture(scope="module")
def cams(tmp_path_factory) -> dict:
    """카메라를 고르는 칸의 글(2026-11-17 사용자 요청) — 키가 번호뿐이라 장소를 옆에 적는다. `capRenderCameras` 를 그대로 돌려
    `#cap-camera` 에 들어가는 option 을 본다(장소는 지어낸 것 — 진짜 장소는 git 에 넣지 않는다)."""
    node = find_node()
    if node is None:
        pytest.skip("node 가 없어 화면(JS) 쪽을 돌릴 수 없습니다")
    src = (WEB / "capture.js").read_text(encoding="utf-8")
    app = (WEB / "app.js").read_text(encoding="utf-8").splitlines()
    at = next(i for i, line in enumerate(app) if line.startswith("const esc = "))
    script = "\n".join([
        *app[at:at + 2],                                     # esc — 두 줄짜리 const
        "const sel = { innerHTML: '', value: '' };",
        "const $ = () => sel;",
        "const C = { cur: { camera: 'hw-02' }, cameras: [",
        "  { key: 'hw-01', kind: 'cctv', place: '가짜 나들목 북쪽', tz: '+09:00' },",
        "  { key: 'hw-02', kind: 'cctv', place: '가짜 터널 입구', split: 'test', tz: '-05:00' },",
        "  { key: 'vid-01', kind: 'cctv', place: '모름 — 영상 ' + 'a'.repeat(60) + '.mp4', tz: '+09:00' },",
        "  { key: 'x-1', kind: 'screen', place: '<b>굵게</b> & \"따옴표\"', tz: '+09:00' },",
        "  { key: 'x-2', kind: 'phone', place: '', tz: '+09:00' },",
        "  { key: 'x-3', kind: 'phone', place: null },",
        "] };",
        js_const(src, "CAP_PLACE_MAX"),
        *(js_function(src, name) for name in ("capCamLabel", "capRenderCameras")),
        "capRenderCameras();",
        "const opts = [...sel.innerHTML.matchAll(/<option value=\"([^\"]*)\"(?: title=\"([^\"]*)\")?>([^<]*)<\\/option>/g)]",
        "  .map((m) => ({ value: m[1], title: m[2] ?? null, text: m[3] }));",
        "console.log(JSON.stringify({ opts, html: sel.innerHTML, selected: sel.value, max: CAP_PLACE_MAX,",
        "  bare: capCamLabel({ key: 'hw-01', place: ' 가짜 나들목 북쪽 ' }), long: capCamLabel(C.cameras[2]) }));",
        "process.exit(0);",
    ])
    return run_node(node, script, tmp_path_factory.mktemp("capture_cams"))


def test_camera_picker_shows_the_place_next_to_the_key(cams):
    by = {o["value"]: o for o in cams["opts"]}
    assert list(by) == ["", "hw-01", "hw-02", "vid-01", "x-1", "x-2", "x-3"]         # 자동 + 등록 순서 그대로
    assert by[""]["text"] == "자동 — 이 영상용 카메라"
    assert by["hw-01"]["text"] == "hw-01 · 가짜 나들목 북쪽 (cctv)"
    assert by["hw-02"]["text"] == "hw-02 · 가짜 터널 입구 (cctv · test 고정 · UTC-05:00)"   # 꼬리(종류·고정·시간대)는 전과 같다
    assert by["x-2"]["text"] == "x-2 (phone)" and by["x-3"]["text"] == "x-3 (phone)"   # 장소가 없으면 키만(빈 ' · ' 없이)
    assert cams["selected"] == "hw-02"                                               # 고른 카메라는 그대로 고른 채
    assert cams["bare"] == "hw-01 · 가짜 나들목 북쪽"                                 # 꼬리 없이('고칠 것' 칸)


def test_camera_picker_shortens_a_long_place_and_keeps_it_whole_in_the_title(cams):
    opt = next(o for o in cams["opts"] if o["value"] == "vid-01")
    place = "모름 — 영상 " + "a" * 60 + ".mp4"
    shown = cams["long"].removeprefix("vid-01 · ")
    assert len(shown) == cams["max"] and shown.endswith("…") and place.startswith(shown[:-1])
    assert opt["text"] == f"vid-01 · {shown} (cctv)"
    assert opt["title"] == place                                                     # 전체는 title 에


def test_camera_picker_escapes_the_place(cams):
    assert "<b>" not in cams["html"]
    opt = next(o for o in cams["opts"] if o["value"] == "x-1")
    assert opt["text"] == "x-1 · &lt;b&gt;굵게&lt;/b&gt; &amp; &quot;따옴표&quot; (screen)"


def test_closing_a_video_forgets_its_unsaved_start_time(got):
    assert got["after_reset"] == {"pending": None, "timer": None, "cur": None}
    assert got["sent_after_closing"] == []              # 없어진 세션에 아무것도 보내지 않는다


def test_leaving_the_view_still_saves_a_start_time_in_hand(got):
    assert got["sent_when_leaving"] == [["PUT", "/api/capture/s2/state", {"start": "2026-11-11 08:00:00"}]]
    assert got["pending_after_leaving"] is None
