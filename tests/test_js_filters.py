"""왼쪽 필터를 바꿔 목록을 다시 받기 (app.js reloadList·clearFilters, 2026-11-16) — 화면 코드를 node 로 돌려 본다.

필터 칸을 바꾸면 목록을 첫 묶음부터 다시 받고 첫 장을 연다. **고치던 박스를 먼저 저장하고**, 저장하지 못하면 목록을 바꾸지 않은 채
필터 칸을 받아 둔 목록의 조건(`S.listAt`)으로 되돌린다. 고치기 전에는 목록부터 받고 첫 장을 열 때에야 저장을 봐서, 저장이
거절되면(잘못된 박스 422 등 — 목록 요청은 잘 된다) 목록은 새 조건인데 가운데에는 그 목록에 없는 옛
이미지가 남았다.

`noFilters`·`filterState`·`setFilters`·`listControls`·`setListControls`·`reloadList`·`clearFilters` 를 app.js 에서 글자로 떼어
그대로 돌린다(이름을 바꾸면 `PARTS` 도). 서버·목록 그리기는 대역이다. node 가 없으면 건너뛴다.
"""

from __future__ import annotations

import pytest

from helpers import WEB, run_node
from test_js_parity import find_node, js_function

PARTS = ("noFilters", "filterState", "setFilters", "listControls", "setListControls", "reloadList", "clearFilters")


@pytest.fixture(scope="module")
def got(tmp_path_factory) -> dict:
    node = find_node()
    if node is None:
        pytest.skip("node 가 없어 화면(JS) 쪽을 돌릴 수 없습니다")
    app = (WEB / "app.js").read_text(encoding="utf-8")
    script = "\n".join([
        "const els = {};",
        "const $ = (sel) => els[sel] || (els[sel] = { value: '', on: false,",
        "  classList: { toggle: (name, v) => { if (name === 'on') els[sel].on = !!v; } } });",
        "let saveOk = true;",
        "const log = { lists: [], loaded: [], syncs: 0, bars: [] };",
        "const S = { filterClasses: new Set(), filterEmpty: false, filterCamera: '', filterDomain: '', filterSplit: '',",
        "            filterWork: '', tagF: { inc: [], exc: [], any: false, none: false }, images: [{ id: 11 }], listAt: null };",
        "const tagFilter = { render() {} }, syncClassFilter = () => {}, syncListControls = () => { log.syncs++; };",
        # 목록 위 작업 배치 머리줄 — 필터를 맞출 때마다 다시 그린다(그릴 때의 배치 필터 값을 적는다)
        "const renderWorkBar = () => { log.bars.push(S.filterWork); };",
        "const saveBeforeLeaving = async () => saveOk;",
        "const loadImage = async (id) => { log.loaded.push(id); };",
        *(js_function(app, name) for name in PARTS),
        # 진짜 loadImageList 처럼: 지금 칸의 조건으로 목록을 받고, 받은 조건을 S.listAt 에 적는다
        "const loadImageList = async () => { log.lists.push(listControls()); S.images = [{ id: 42 }]; S.listAt = listControls(); };",
        "const show = () => ({ now: listControls(), selects: ['#filter-camera', '#filter-domain', '#filter-split', '#filter-work']",
        "  .map((s) => [$(s).value, $(s).on]), lists: log.lists.length, loaded: [...log.loaded], images: S.images.map((i) => i.id),",
        "  bar: log.bars[log.bars.length - 1] });",
        # 화면에서 카메라 칸을 바꾼 것과 같다(#filter-camera 의 onchange)
        "const pickCamera = async (key) => { $('#filter-camera').value = key; S.filterCamera = key;",
        "  $('#filter-camera').classList.toggle('on', !!key); await reloadList(); };",
        # 작업 배치 칸을 바꾼 것(#filter-work 의 onchange — 머리줄을 먼저 맞추고 목록을 받는다)
        "const pickWork = async (id) => { $('#filter-work').value = id; S.filterWork = id;",
        "  $('#filter-work').classList.toggle('on', !!id); renderWorkBar(); await reloadList(); };",
        "(async () => {",
        "  const out = {};",
        "  $('#filter-status').value = 'pending'; $('#filter-order').value = 'recent'; $('#chunk-size').value = '100';",
        "  await reloadList();                       // 처음 목록 — 검수 대기, 필터 없음",
        "  out.first = show();",
        "  await pickCamera('cam-02');",
        "  out.picked = show();",
        "  saveOk = false;                           // 고치던 박스를 저장할 수 없게 됐다",
        "  await pickCamera('cam-03');",
        "  out.blocked = show();",
        "  $('#filter-order').value = 'name'; await reloadList();      // 정렬 칸도 같은 길이다",
        "  out.blocked_order = show();",
        "  S.filterClasses.add(4); S.tagF = { inc: [7], exc: [], any: false, none: false };",
        "  clearFilters(); await new Promise((done) => setImmediate(done));",
        "  out.blocked_clear = show();",
        "  saveOk = true;",
        "  clearFilters(); await new Promise((done) => setImmediate(done));",
        "  out.cleared = show();",
        # 작업 배치 (2026-11-23): 배치를 고르면 그 조건으로 받고, 저장을 못 하면 칸과 머리줄이 받아 둔 목록의 배치로 돌아간다
        "  await pickWork('7');",
        "  out.work = show();",
        "  saveOk = false;",
        "  await pickWork('9');",
        "  out.work_blocked = show();",
        "  saveOk = true;",
        "  clearFilters(); await new Promise((done) => setImmediate(done));",
        "  out.work_cleared = show();",
        "  console.log(JSON.stringify(out));",
        "})().catch((err) => { console.error((err && err.stack) || err); process.exit(1); });",
    ])
    return run_node(node, script, tmp_path_factory.mktemp("filters"))


def controls(camera="", order="recent", status="pending", classes=(), inc=(), work=""):
    return {"status": status, "classes": list(classes), "empty": False, "camera": camera, "domain": "", "split": "",
            "work": work, "tagF": {"inc": list(inc), "exc": [], "any": False, "none": False}, "order": order, "chunk": "100"}


def test_a_filter_change_reloads_the_list_and_opens_the_first_image(got):
    assert got["first"]["now"] == controls() and got["first"]["lists"] == 1
    picked = got["picked"]
    assert picked["now"] == controls("cam-02") and picked["selects"][0] == ["cam-02", True]
    assert (picked["lists"], picked["loaded"], picked["images"]) == (2, [42, 42], [42])


def test_when_the_edits_cannot_be_saved_the_list_stays_and_the_controls_go_back(got):
    """저장을 못 하면 목록을 다시 받지 않는다 — 그리고 바꾼 칸을 받아 둔 목록의 조건(cam-02)으로 되돌린다."""
    blocked = got["blocked"]
    assert blocked["lists"] == 2 and blocked["loaded"] == [42, 42]             # 목록도, 이미지도 그대로
    assert blocked["now"] == controls("cam-02") and blocked["selects"][0] == ["cam-02", True]
    assert got["blocked_order"]["now"] == controls("cam-02") and got["blocked_order"]["lists"] == 2    # 정렬 칸도 되돌아온다


def test_clearing_filters_follows_the_same_rule(got):
    """'필터 풀기' 도 같다 — 저장을 못 하면 풀지 않고, 저장이 되면 상태·정렬만 남기고 다 푼다."""
    assert got["blocked_clear"]["now"] == controls("cam-02") and got["blocked_clear"]["lists"] == 2
    cleared = got["cleared"]
    assert cleared["now"] == controls() and cleared["selects"] == [["", False], ["", False], ["", False], ["", False]]
    assert cleared["lists"] == 3


def test_the_work_batch_filter_follows_the_same_rule(got):
    """작업 배치 필터(2026-11-23)도 목록 필터다 — 고르면 그 조건으로 받고, 저장을 못 하면 칸·머리줄이 받아 둔 목록의 배치로
    돌아가고, '필터 풀기' 가 함께 푼다. 필터를 새로 만들고 noFilters·filterState·setFilters 에 안 넣으면 여기서 어긋난다."""
    work = got["work"]
    assert work["now"] == controls(work="7") and work["selects"][3] == ["7", True]
    assert (work["lists"], work["bar"]) == (4, "7")
    blocked = got["work_blocked"]                           # 9 로 바꾸려 했지만 저장을 못 했다 — 목록도 칸도 머리줄도 7 그대로
    assert blocked["now"] == controls(work="7") and blocked["selects"][3] == ["7", True]
    assert (blocked["lists"], blocked["bar"]) == (4, "7")
    cleared = got["work_cleared"]
    assert cleared["now"] == controls() and cleared["selects"][3] == ["", False]
    assert (cleared["lists"], cleared["bar"]) == (5, "")
