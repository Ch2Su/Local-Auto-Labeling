"""작업 배치의 화면 쪽 (app.js approveTarget·setStatus·setStatusNow, work.js workOpen·upExtOk, 2026-11-23) — node 로 돌려 본다.

지키는 것:

- **승인(Space)이 어디로 가는가.** '모아서 한꺼번에'(hold)인 배치의 이미지는 먼저 '검수 끝'(reviewed)으로, 검수 끝인 장에서는
  승인(approved)으로, 그 밖(배치 없음·'바로 추가' 배치)은 지금까지처럼 곧바로 승인으로. **이미 승인된 장은 어느 배치에서든
  승인 그대로다** — 승인된 장에서 Space 로 넘기는 것이 그 장을 데이터셋에서 빼면 안 된다(처음 쓴 식이 그랬다).
- **'검수 끝' 도 사람이 봤다는 표시다.** 승인 전 확인(사진이 뜨기 전·그리던 다각형·겹친 박스·자동 라벨링이 그새 넣은 박스·
  휴지통)을 검수 끝에도 건다 — 검수 끝은 나중에 보지 않고 한꺼번에 승인으로 올라간다. 건너뛰기·검수 대기로는 그대로다.
- **배치 열기**(`workOpen`)는 필터를 그 배치 하나로 바꾸고 라벨링으로 간다. 고치던 박스를 저장하지 못하면 아무것도 바꾸지 않고,
  목록을 못 받으면 필터를 되돌린다(reloadList·hcOpenFiltered 와 같은 약속).
- 화면이 넣기 전에 거르는 형식(`upExtOk`)이 서버(`upload.clean_name`)가 받는 것과 같다.

함수는 test_js_parity.py 와 같은 규칙으로 글자로 뗀다 — 이름이나 모양을 바꾸면 `APP_PARTS`·`WORK_PARTS` 도 맞출 것.
node 가 없으면 건너뛴다.
"""

from __future__ import annotations

import json

import pytest

from autolabel.upload import UploadError, clean_name
from helpers import WEB, run_node
from test_js_parity import find_node, js_function

APP_PARTS = ("approveTarget", "approveText", "workLabel", "workTodo", "setStatus", "setStatusNow", "nextAfter")
OPEN_APP_PARTS = ("noFilters", "filterState", "setFilters")
WORK_PARTS = ("workFirstStatus", "workOpen", "upExtOk")

HOLD = {"id": 7, "kind": "scan", "name": None, "hold": True}
DIRECT = {"id": 8, "kind": "upload", "name": "묶음", "hold": False}


@pytest.fixture(scope="module")
def node() -> str:
    found = find_node()
    if found is None:
        pytest.skip("node 가 없어 화면(JS) 쪽을 돌릴 수 없습니다")
    return found


# ----------------------------------------------------------------- 승인이 가는 곳

@pytest.fixture(scope="module")
def approve(node, tmp_path_factory) -> dict:
    app = (WEB / "app.js").read_text(encoding="utf-8")
    cases = {
        "none_pending": {"work": None, "status": "pending"},
        "none_unlabeled": {"work": None, "status": "unlabeled"},
        "direct_pending": {"work": DIRECT, "status": "pending"},
        "direct_reviewed": {"work": DIRECT, "status": "reviewed"},        # 방식을 끄기 전에 검수 끝이 된 장
        "hold_pending": {"work": HOLD, "status": "pending"},
        "hold_unlabeled": {"work": HOLD, "status": "unlabeled"},          # 손으로 그린 장
        "hold_skipped": {"work": HOLD, "status": "skipped"},
        "hold_reviewed": {"work": HOLD, "status": "reviewed"},
        "hold_approved": {"work": HOLD, "status": "approved"},
        # 승인 전 확인 — 검수 끝으로 가는 길에도 걸리는가
        "hold_no_photo": {"work": HOLD, "status": "pending", "imgId": None},
        "hold_poly": {"work": HOLD, "status": "pending", "poly": True},
        "hold_trashed": {"work": HOLD, "status": "trashed"},
        "hold_overlap_no": {"work": HOLD, "status": "pending", "overlaps": True, "confirm": False},
        "hold_overlap_yes": {"work": HOLD, "status": "pending", "overlaps": True, "confirm": True},
        "hold_predict_changed": {"work": HOLD, "status": "pending", "job": "predict", "serverBoxes": 3},
        "hold_predict_same": {"work": HOLD, "status": "pending", "job": "predict", "serverBoxes": 2},
        "hold_save_fails": {"work": HOLD, "status": "pending", "saveOk": False},
        # 승인이 아닌 것은 그대로
        "hold_skip": {"work": HOLD, "status": "pending", "to": "skipped", "imgId": None},
        "hold_back": {"work": HOLD, "status": "reviewed", "to": "pending"},
        "hold_next_only": {"work": HOLD, "status": "reviewed", "to": None},
        "hold_trash": {"work": HOLD, "status": "pending", "to": "trashed", "poly": True},
    }
    script = "\n".join([
        f"const CASES = {json.dumps(cases, ensure_ascii=False)};",
        "let S, log, sc;",
        "const navBusy = () => false;",
        "const toast = (msg, bad) => { log.toasts.push(String(msg)); };",
        "const confirm = (msg) => { log.asked++; return !!sc.confirm; };",
        "const saveBeforeLeaving = async () => sc.saveOk !== false;",
        "const selfOverlaps = () => (sc.overlaps ? [[0, 1, 0.9]] : []);",
        "const clsName = (c) => `cls${c}`;",
        "const sameBoxes = (a, b) => a.length === b.length;",
        "const renderStatusButtons = () => { log.rendered.push(S.cur.status); };",
        "const loadState = async () => {}, loadImageList = async () => {}, openChunkWith = async () => {};",
        "const keepCurrentRow = async (fn) => fn();",
        "const loadImage = async (id) => { log.loaded.push(id); };",
        "const advanceFrom = async (id) => { log.advanced.push(id); };",
        "const api = async (path, opts) => {",
        "  if (opts && opts.method === 'POST') { log.posts.push([path, opts.body.status]); return { ok: true }; }",
        "  return { boxes: new Array(sc.serverBoxes ?? 2).fill({}) };     // 자동 라벨링 중의 '서버 박스 다시 보기'",
        "};",
        *(js_function(app, name) for name in APP_PARTS),
        "(async () => {",
        "  const out = {};",
        "  for (const [name, c] of Object.entries(CASES)) {",
        "    sc = c;",
        "    log = { posts: [], toasts: [], asked: 0, rendered: [], loaded: [], advanced: [] };",
        "    S = { cur: { id: 5, status: c.status, work: c.work }, imgId: 'imgId' in c ? c.imgId : 5, poly: c.poly ? {} : null,",
        "          boxes: [{ cls: 0 }, { cls: 1 }], images: [{ id: 5 }, { id: 6 }], jobKind: c.job || null, statusBusy: false };",
        "    const target = approveTarget(S.cur), text = approveText(S.cur);",
        "    await setStatus('to' in c ? c.to : 'approved');",
        "    out[name] = { target, text, ...log, status: S.cur.status, busy: S.statusBusy };",
        "  }",
        "  out.labels = [workLabel({ id: 7, name: null, kind_name: '영상 캡처', kind: 'capture' }),",
        "                workLabel({ id: 8, name: '묶음', kind_name: '이미지 올리기', kind: 'upload' }),",
        "                workTodo({ counts: { unlabeled: 1, pending: 2, reviewed: 3, approved: 9, skipped: 4, trashed: 5 } }),",
        "                workTodo({ counts: { approved: 9, skipped: 4, trashed: 5 } }), approveTarget(null), approveText(null)];",
        "  console.log(JSON.stringify(out));",
        "})().catch((err) => { console.error((err && err.stack) || err); process.exit(1); });",
    ])
    return run_node(node, script, tmp_path_factory.mktemp("approve"))


def test_without_a_holding_batch_space_approves_as_before(approve):
    for name in ("none_pending", "none_unlabeled", "direct_pending", "direct_reviewed"):
        got = approve[name]
        assert got["target"] == "approved" and got["posts"] == [["/api/images/5/status", "approved"]], name
        assert got["status"] == "approved" and got["advanced"] == [5] and got["busy"] is False, name
    assert approve["none_pending"]["text"] == "승인 후 다음"
    assert approve["direct_reviewed"]["text"] == "데이터셋에 추가 후 다음"


def test_in_a_holding_batch_space_marks_reviewed_first(approve):
    for name in ("hold_pending", "hold_unlabeled", "hold_skipped"):
        got = approve[name]
        assert (got["target"], got["text"]) == ("reviewed", "검수 끝 후 다음"), name
        assert got["posts"] == [["/api/images/5/status", "reviewed"]] and got["status"] == "reviewed", name
        assert got["rendered"] == ["reviewed"] and got["advanced"] == [5], name      # 단추도 새 상태를 따른다


def test_a_reviewed_image_goes_to_the_dataset_and_an_approved_one_stays_there(approve):
    got = approve["hold_reviewed"]                               # 검수 끝인 장에서 한 번 더 — 그 장만 데이터셋에
    assert (got["target"], got["text"], got["posts"]) == ("approved", "데이터셋에 추가 후 다음", [["/api/images/5/status", "approved"]])
    got = approve["hold_approved"]                               # 이미 승인된 장 — '검수 끝' 으로 내려가면 데이터셋에서 빠진다
    assert (got["target"], got["text"], got["posts"]) == ("approved", "승인 후 다음", [["/api/images/5/status", "approved"]])
    assert got["status"] == "approved"


def test_the_checks_before_approval_also_guard_reviewed(approve):
    """사진이 아직 안 떴거나, 다각형을 그리는 중이거나, 휴지통이면 '검수 끝' 으로도 보내지 않는다."""
    for name, word in (("hold_no_photo", "사진이 아직 화면에 없어"), ("hold_poly", "그리던 다각형"), ("hold_trashed", "휴지통")):
        got = approve[name]
        assert got["posts"] == [] and got["advanced"] == [] and word in got["toasts"][0], name
    assert approve["hold_no_photo"]["status"] == "pending"
    no, yes = approve["hold_overlap_no"], approve["hold_overlap_yes"]         # 겹친 박스는 한 번 묻는다
    assert (no["asked"], no["posts"], no["status"]) == (1, [], "pending")
    assert (yes["asked"], yes["posts"]) == (1, [["/api/images/5/status", "reviewed"]])
    changed = approve["hold_predict_changed"]                    # 자동 라벨링이 그새 박스를 넣었다 — 다시 읽고 멈춘다
    assert changed["posts"] == [] and changed["loaded"] == [5] and "자동 라벨링" in changed["toasts"][0]
    assert approve["hold_predict_same"]["posts"] == [["/api/images/5/status", "reviewed"]]
    assert approve["hold_save_fails"]["posts"] == []             # 저장을 못 했으면 상태도 바꾸지 않는다


def test_other_status_changes_are_left_alone(approve):
    """건너뛰기·검수 대기로·휴지통은 배치 방식과 상관없고, 승인 전 확인도 걸리지 않는다(사진이 안 떠도 건너뛸 수 있다)."""
    assert approve["hold_skip"]["posts"] == [["/api/images/5/status", "skipped"]] and approve["hold_skip"]["asked"] == 0
    assert approve["hold_back"]["posts"] == [["/api/images/5/status", "pending"]]
    assert approve["hold_trash"]["posts"] == [["/api/images/5/status", "trashed"]]
    nxt = approve["hold_next_only"]                              # 상태를 바꾸지 않는 '다음'
    assert nxt["posts"] == [] and nxt["advanced"] == [5] and nxt["status"] == "reviewed"


def test_labels_and_what_is_left(approve):
    assert approve["labels"] == ["작업 7 · 영상 캡처", "작업 8 · 묶음", 6, 0, "approved", "승인 후 다음"]


# ----------------------------------------------------------------- 배치 열기

@pytest.fixture(scope="module")
def opened(node, tmp_path_factory) -> dict:
    app = (WEB / "app.js").read_text(encoding="utf-8")
    work = (WEB / "work.js").read_text(encoding="utf-8")
    works = [
        {"id": 7, "counts": {"unlabeled": 3, "pending": 5, "reviewed": 1, "approved": 2}},
        {"id": 8, "counts": {"unlabeled": 4, "pending": 0, "reviewed": 0, "approved": 0}},
        {"id": 9, "counts": {"unlabeled": 0, "pending": 0, "reviewed": 2, "approved": 6}},
        {"id": 10, "counts": {"unlabeled": 0, "pending": 0, "reviewed": 0, "approved": 6}},
    ]
    cases = {
        "pending_first": {"id": 7},
        "unlabeled_next": {"id": 8},
        "reviewed_next": {"id": 9},
        "all_done": {"id": 10},
        "a_chip": {"id": 7, "status": "approved"},
        "no_batch": {"id": "__none__", "status": "pending"},
        "from_capture": {"id": 7, "view": "capture"},
        "from_label": {"id": 7, "view": "label"},
        "left_meanwhile": {"id": 7, "leave": True},
        "save_fails": {"id": 7, "saveOk": False},
        "list_fails": {"id": 7, "listFails": True},
        "empty_list": {"id": 7, "empty": True},
        "busy": {"id": 7, "navBusy": True},
        "unknown_batch": {"id": 42},
    }
    script = "\n".join([
        f"const CASES = {json.dumps(cases)}, WORKS = {json.dumps(works)};",
        "let S, W, VIEW, log, sc, els;",
        "const $ = (sel) => els[sel] || (els[sel] = { value: '', on: false,",
        "  classList: { toggle: (name, v) => { if (name === 'on') els[sel].on = !!v; } } });",
        "const tagFilter = { render() {} }, syncClassFilter = () => {}, renderWorkBar = () => { log.bars.push(S.filterWork); };",
        "const syncListControls = () => { log.syncs++; };",
        "const navBusy = () => !!sc.navBusy;",
        "const saveBeforeLeaving = async () => sc.saveOk !== false;",
        "const toast = (msg, bad) => { log.toasts.push(String(msg)); };",
        "const loadWorks = async () => { log.reloads++; };",
        "const loadImage = async (id) => { log.loaded.push(id); };",
        "const setView = async (v) => { log.views.push(v); VIEW = v; };",
        *(js_function(app, name) for name in OPEN_APP_PARTS),
        "const loadImageList = async () => {",
        "  log.listed.push(filterState());",
        "  if (sc.leave) VIEW = 'health';                         // 받는 사이 다른 보기로 갔다",
        "  if (sc.listFails) throw new Error('서버가 꺼져 있습니다');",
        "  S.images = sc.empty ? [] : [{ id: 42 }];",
        "};",
        *(js_function(work, name) for name in WORK_PARTS),
        "(async () => {",
        "  const out = {};",
        "  for (const [name, c] of Object.entries(CASES)) {",
        "    sc = c; els = {};",
        "    log = { listed: [], loaded: [], views: [], toasts: [], bars: [], syncs: 0, reloads: 0 };",
        "    W = { opening: false }; VIEW = c.view || 'work';",
        "    S = { works: WORKS, workNoneKey: '__none__', filterClasses: new Set([3]), filterEmpty: false, filterCamera: 'cam-01',",
        "          filterDomain: '', filterSplit: '', filterWork: '', tagF: { inc: [], exc: [], any: false, none: false },",
        "          images: [{ id: 11 }] };",
        "    $('#filter-status').value = 'approved'; $('#filter-camera').value = 'cam-01';",
        "    await workOpen(c.id, c.status || null);",
        "    out[name] = { ...log, now: filterState(), view: VIEW, opening: W.opening,",
        "                  select: [$('#filter-work').value, $('#filter-work').on] };",
        "  }",
        "  console.log(JSON.stringify(out));",
        "})().catch((err) => { console.error((err && err.stack) || err); process.exit(1); });",
    ])
    return run_node(node, script, tmp_path_factory.mktemp("workopen"))


def want(work: str, status: str) -> dict:
    return {"status": status, "classes": [], "empty": False, "camera": "", "domain": "", "split": "", "work": work,
            "tagF": {"inc": [], "exc": [], "any": False, "none": False}}


BEFORE = {"status": "approved", "classes": [3], "empty": False, "camera": "cam-01", "domain": "", "split": "", "work": "",
          "tagF": {"inc": [], "exc": [], "any": False, "none": False}}


def test_opening_a_batch_filters_the_list_to_it_and_goes_to_labeling(opened):
    got = opened["pending_first"]                                # 할 일이 있는 상태부터 — 검수 대기
    assert got["listed"] == [want("7", "pending")] and got["now"] == want("7", "pending")    # 다른 필터는 푼다
    assert (got["loaded"], got["views"], got["view"], got["select"]) == ([42], ["label"], "label", ["7", True])
    assert got["bars"][-1] == "7" and got["toasts"] == [] and got["opening"] is False
    assert opened["unlabeled_next"]["now"]["status"] == "unlabeled"
    assert opened["reviewed_next"]["now"]["status"] == "reviewed"
    assert opened["all_done"]["now"]["status"] == "live"         # 할 일이 없으면 휴지통 빼고 전부
    assert opened["a_chip"]["now"] == want("7", "approved")      # 상태 칩을 눌렀으면 그 상태로
    assert opened["no_batch"]["now"] == want("__none__", "pending") and opened["no_batch"]["reloads"] == 0
    assert opened["empty_list"]["loaded"] == [] and opened["empty_list"]["views"] == ["label"]


def test_it_comes_from_wherever_it_was_opened_but_does_not_drag_you_back(opened):
    assert opened["from_capture"]["views"] == ["label"]          # 영상 캡처의 '작업 N 열기'
    assert opened["from_label"]["views"] == []                   # 이미 라벨링이면 보기는 그대로
    left = opened["left_meanwhile"]                              # 받는 사이 다른 보기로 갔다 — 끌고 오지 않는다
    assert (left["views"], left["view"], left["now"]["work"]) == ([], "health", "7")


def test_nothing_changes_when_edits_cannot_be_saved_or_another_step_is_running(opened):
    for name in ("save_fails", "busy"):
        got = opened[name]
        assert (got["now"], got["listed"], got["views"], got["view"]) == (BEFORE, [], [], "work"), name
        assert got["opening"] is False


def test_filters_go_back_when_the_list_cannot_be_fetched(opened):
    got = opened["list_fails"]
    assert got["listed"] == [want("7", "pending")]               # 물을 때는 새 조건이었다
    assert got["now"] == BEFORE and got["select"] == ["", False] and got["bars"][-1] == ""
    assert (got["views"], got["view"], got["loaded"], got["syncs"]) == ([], "work", [], 1)
    assert got["toasts"] == ["목록을 받지 못했습니다: 서버가 꺼져 있습니다"] and got["opening"] is False


def test_a_batch_that_was_just_made_is_fetched_once(opened):
    """방금 넣은 배치는 받아 둔 목록에 아직 없다 — 한 번 받고 연다(필터 칸과 머리줄이 그 배치를 알아야 한다)."""
    got = opened["unknown_batch"]
    assert got["reloads"] == 1 and got["now"] == want("42", "live") and got["views"] == ["label"]
    assert opened["pending_first"]["reloads"] == 0


# ----------------------------------------------------------------- 형식 거르기 (화면 ↔ 서버)

NAMES = ["a.jpg", "A.JPG", "b.jpeg", "c.png", "d.bmp", "e.webp", "f.tif", "g.tiff", "h.avif", "i.heic", "j.gif", "k.txt",
         "noext", "", ".jpg", "a.jpg.exe", "dir/sub/x.png", "dir\\sub\\y.PNG", "movie.mp4", "trail.jpg. ", "a.", "x.svg",
         "한글 이름.jpg", "two.dots.jpeg", "..\\..\\evil.jpg", "a.jpg/"]


def test_the_page_filters_the_same_formats_the_server_takes(node, tmp_path):
    """화면은 넣기 전에 이미지가 아닌 파일을 뺀다(`upExtOk`) — 서버가 받는 것(`upload.clean_name`)과 어긋나면 받을 파일을
    화면이 빼거나, 화면이 넣은 파일을 서버가 하나씩 거절한다."""
    work = (WEB / "work.js").read_text(encoding="utf-8")
    script = "\n".join([
        "const W = { limits: null };",
        "const WK_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff', '.avif'];",
        js_function(work, "upExtOk"),
        f"console.log(JSON.stringify({json.dumps(NAMES, ensure_ascii=False)}.map(upExtOk)));",
    ])
    got = run_node(node, script, tmp_path)

    def server_takes(name: str) -> bool:
        try:
            clean_name(name)
        except UploadError:
            return False
        return True

    assert got == [server_takes(n) for n in NAMES]
    assert got[:9] == [True] * 9 and got[9:14] == [False] * 5       # 받는 아홉 가지와, heic·gif·txt·확장자 없음·빈 이름


def test_the_fallback_list_of_formats_matches_the_server(node):
    """서버가 한도를 알려 주기 전에 쓰는 화면의 기본 형식 목록(WK_EXTS)이 yolo_io.IMG_EXTS 와 같다."""
    import re

    from autolabel.yolo_io import IMG_EXTS

    work = (WEB / "work.js").read_text(encoding="utf-8")
    line = next(ln for ln in work.splitlines() if ln.startswith("const WK_EXTS = "))
    assert sorted(re.findall(r'"(\.\w+)"', line)) == sorted(IMG_EXTS)
    html = (WEB / "index.html").read_text(encoding="utf-8")
    accept = re.search(r'<input id="wk-files"[^>]*accept="([^"]+)"', html, flags=re.S).group(1)
    assert sorted(a for a in accept.split(",") if a.startswith(".")) == sorted(IMG_EXTS)
