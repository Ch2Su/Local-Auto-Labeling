"""영상 캡처 화면이 영상을 비울 때(capture.js capReset, 2026-11-16) — 화면 코드를 node 로 돌려 본다.

시작 시각을 고치면 0.6초 뒤 서버에 남기려고 `C.startPending` 에 들어 둔다. 그 영상을 넣거나 닫으면 `capReset` 이 타이머를 끄는데,
**들어 둔 값은 그대로 남아서** 다음에 다른 보기로 갈 때(`captureHide` → `capFlushStart`) 이미 없어진 세션에 PUT 을 보냈다(404).
`capReset`·`capFlushStart`·`captureHide` 를 capture.js 에서 글자로 떼어 그대로 돌린다. node 가 없으면 건너뛴다.

카메라를 고르는 칸의 글(app.js `camLabel` 과 capture.js 의 영상의 카메라·고칠 것·지우기 목록, 2026-11-17)도 여기서 본다 —
키 옆에 장소가 나오는가.
"""

from __future__ import annotations

import pytest

from helpers import WEB, run_node
from test_js_parity import find_node, js_const, js_const_lines, js_function


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


CAMS_FUNCS = ("capCamFlags", "capCamNote", "capCamTitle", "capRenderCameras", "capCamEditOpen",
              "capCamDelKeys", "capCamDelCount", "capCamDelRender")


@pytest.fixture(scope="module")
def cams(tmp_path_factory) -> dict:
    """카메라를 고르는 칸의 글(2026-11-17 사용자 요청) — 키가 번호뿐이라 장소를 옆에 적는다. 영상의 카메라(`capRenderCameras`)·
    고칠 것(`capCamEditOpen`)·지우기 목록(`capCamDelRender`)을 그대로 돌려 화면에 들어가는 글을 본다(장소는 지어낸 것 — 진짜
    장소는 git 에 넣지 않는다). 이름을 만드는 `camLabel` 은 app.js 에 있다(라벨링의 카메라 필터·이미지 보기의 '옮기기' 와 함께 쓴다)."""
    node = find_node()
    if node is None:
        pytest.skip("node 가 없어 화면(JS) 쪽을 돌릴 수 없습니다")
    src = (WEB / "capture.js").read_text(encoding="utf-8")
    app = (WEB / "app.js").read_text(encoding="utf-8")
    script = "\n".join([
        js_const_lines(app, "esc"),
        "const els = {};",
        "const $ = (q) => els[q] || (els[q] = { innerHTML: '', value: '', title: '', textContent: '', disabled: false,",
        "  classList: { add() {}, remove() {}, toggle() {} }, querySelectorAll: () => [] });",
        "const toast = () => {}, capCamDelClose = () => {}, capCamEditFill = () => {};",
        "const capLoadList = () => new Promise(() => {});          // 목록을 다시 받는 것은 여기서 보지 않는다",
        "const C = { cur: { camera: 'hw-02' }, cameras: [",
        "  { key: 'hw-01', kind: 'cctv', place: '가짜 나들목 북쪽', tz: '+09:00' },",
        "  { key: 'hw-02', kind: 'cctv', place: '가짜 터널 입구', split: 'test', tz: '-05:00' },",
        "  { key: 'vid-01', kind: 'cctv', place: '모름 — 영상 ' + 'a'.repeat(60) + '.mp4', tz: '+09:00' },",
        "  { key: 'x-1', kind: 'screen', place: '<b>굵게</b> & \"따옴표\"', tz: '+09:00', db: 3 },",
        "  { key: 'x-2', kind: 'phone', place: '', tz: '+09:00' },",
        "  { key: 'x-3', kind: 'phone', place: null },",
        "] };",
        js_const(app, "CAM_PLACE_MAX"),
        js_function(app, "camLabel"),
        *(js_function(src, name) for name in CAMS_FUNCS),
        "capRenderCameras(); capCamEditOpen(); capCamDelRender();",
        "const parse = (html) => [...html.matchAll(/<option value=\"([^\"]*)\"(?: title=\"([^\"]*)\")?>([^<]*)<\\/option>/g)]",
        "  .map((m) => ({ value: m[1], title: m[2] ?? null, text: m[3] }));",
        "const rec = (t) => camLabel({ key: 'vid-09', place: `모름 — 영상 Screen Recording 2026-11-09 at ${t}.mp4` });",
        "console.log(JSON.stringify({ opts: parse($('#cap-camera').innerHTML), html: $('#cap-camera').innerHTML,",
        "  selected: $('#cap-camera').value, max: CAM_PLACE_MAX,",
        "  bare: camLabel({ key: 'hw-01', place: ' 가짜 나들목 북쪽 ' }), long: camLabel(C.cameras[2]),",
        "  counted: [camLabel(C.cameras[0], 12), camLabel(C.cameras[0], 0), camLabel(C.cameras[4], 3)],",
        "  recordings: [rec('14.22.31'), rec('15.40.02')],",
        "  notes: [capCamNote('hw-01'), capCamNote('hw-02'), capCamNote(''), capCamNote('gone')],",
        "  titles: [capCamTitle('vid-01'), capCamTitle(''), capCamTitle('x-3'), capCamTitle('gone')],",
        "  target: parse($('#cc-target').innerHTML), targetHtml: $('#cc-target').innerHTML, targetPick: $('#cc-target').value,",
        "  del: $('#cap-cam-del-list').innerHTML }));",
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


def test_camera_picker_shortens_a_long_place_in_the_middle_and_keeps_it_whole_in_the_title(cams):
    opt = next(o for o in cams["opts"] if o["value"] == "vid-01")
    place = "모름 — 영상 " + "a" * 60 + ".mp4"
    shown = cams["long"].removeprefix("vid-01 · ")
    head, _, tail = shown.partition("…")
    assert len(shown) == cams["max"] and place.startswith(head) and place.endswith(tail) and head and tail
    assert tail.endswith(".mp4")                                                     # 이름 끝이 남는다
    assert opt["text"] == f"vid-01 · {shown} (cctv)"
    assert opt["title"] == place                                                     # 전체는 title 에
    assert cams["titles"] == [place, "", "", ""]                                     # 닫힌 칸의 title — 고른 카메라의 전체 장소


def test_recordings_with_the_same_long_prefix_stay_apart(cams):
    """앞만 남기면 자동 카메라의 '모름 — 영상 Screen Recording 2026-11-09 at 1…' 둘이 똑같이 보였다 (2026-11-17 리뷰)."""
    a, b = cams["recordings"]
    assert a != b and a.endswith("14.22.31.mp4") and b.endswith("15.40.02.mp4")


def test_camera_picker_escapes_the_place(cams):
    assert "<b>" not in cams["html"] and "<b>" not in cams["targetHtml"] and "<b>" not in cams["del"]
    opt = next(o for o in cams["opts"] if o["value"] == "x-1")
    assert opt["text"] == "x-1 · &lt;b&gt;굵게&lt;/b&gt; &amp; &quot;따옴표&quot; (screen)"


def test_the_picked_cameras_fixed_split_and_time_zone_are_said_outside_the_narrow_box(cams):
    """닫힌 칸(왼쪽 칸 300px)은 `키 · 장소` 뒤의 꼬리가 잘린다 — 고른 카메라의 고정 분할·시간대는 시작 시각 안내에도 적는다."""
    assert cams["notes"] == ["", "카메라 <b>hw-02</b> — test 고정 · UTC-05:00<br>", "", ""]
    commit = js_function((WEB / "capture.js").read_text(encoding="utf-8"), "capRenderCommit")
    assert "h += capCamNote(cam);" in commit and "sel.title = capCamTitle(cam);" in commit


def test_the_edit_picker_and_the_delete_list_show_the_place_too(cams):
    by = {o["value"]: o for o in cams["target"]}
    assert list(by) == ["hw-01", "hw-02", "vid-01", "x-1", "x-2", "x-3"]
    assert by["hw-01"] == {"value": "hw-01", "title": "가짜 나들목 북쪽", "text": "hw-01 · 가짜 나들목 북쪽"}
    assert by["x-2"]["text"] == "x-2" and cams["targetPick"] == "hw-02"               # 지금 영상이 고른 카메라를 채운다
    rows = cams["del"].split("</li>")
    assert '<span class="k">hw-01</span><span class="p">가짜 나들목 북쪽</span><span class="m">cctv</span>' in rows[0]
    assert '<span class="m">cctv · test 고정</span>' in rows[1]
    assert '<span class="p">&lt;b&gt;굵게&lt;/b&gt; &amp; &quot;따옴표&quot;</span>' in rows[3]
    assert 'class="busy"' in rows[3] and '<span class="u">DB 3장</span>' in rows[3]   # 이미지가 남은 카메라는 이유와 함께 막는다
    assert '<span class="k">x-2</span><span class="m">phone</span>' in rows[4]         # 장소가 없으면 칸도 없다


def test_the_other_camera_pickers_use_the_same_label(cams):
    """라벨링의 카메라 필터·이미지 보기의 '옮기기' — 꼬리는 이미지 수(0 도 적는다). 전에는 세 곳이 `키 · 장소` 를 따로 이었다."""
    assert cams["counted"] == ["hw-01 · 가짜 나들목 북쪽 (12)", "hw-01 · 가짜 나들목 북쪽 (0)", "x-2 (3)"]
    assert 'camLabel({ key: c.key, place: c.place || "(등록 파일에 없음)" }, c.n)' in (WEB / "app.js").read_text(encoding="utf-8")
    assert "camLabel(c, c.n)" in (WEB / "gallery.js").read_text(encoding="utf-8")
    assert "c.place}" not in (WEB / "gallery.js").read_text(encoding="utf-8")        # 손으로 잇는 사본이 없다


def test_closing_a_video_forgets_its_unsaved_start_time(got):
    assert got["after_reset"] == {"pending": None, "timer": None, "cur": None}
    assert got["sent_after_closing"] == []              # 없어진 세션에 아무것도 보내지 않는다


def test_leaving_the_view_still_saves_a_start_time_in_hand(got):
    assert got["sent_when_leaving"] == [["PUT", "/api/capture/s2/state", {"start": "2026-11-11 08:00:00"}]]
    assert got["pending_after_leaving"] is None
