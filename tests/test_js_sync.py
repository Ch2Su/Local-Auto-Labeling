"""이미지를 바꾸는 작업이 끝난 뒤 지금 이미지 맞추기 (app.js syncCurrentAfterJob) — 화면 코드를 node 로 돌려 본다.

자동 라벨링·휴지통 비우기·정리(`IMAGE_JOBS`)가 끝나면 지금 이미지를 서버와 맞춘다: 없어졌으면 비우고, 박스·상태가 다르면
다시 열고, 같으면 다시 열지 않는다(되돌리기·확대를 지킨다). **같을 때도 이름·경로는 맞춘다 (2026-11-27, Gemini 리뷰 묶음 7)** —
정리가 승인 이미지의 파일을 `labeled/images/` 로 옮기거나 같은 이름을 피해 번호를 붙이면 박스·상태는 그대로인데 이름이
바뀐다. 고치기 전에는 그냥 돌아가 위의 이름과 `S.cur.path` 가 옛 값으로 남았다.

`syncCurrentAfterJob`·`sameBoxes`·`CUR_META_KEYS` 를 app.js 에서 글자로 떼어 그대로 돌린다(이름을 바꾸면 `PARTS` 도).
"""

from __future__ import annotations

import pytest

from helpers import WEB, run_node
from test_js_parity import find_node, js_const, js_function

PARTS = ("sameBoxes", "syncCurrentAfterJob")


@pytest.fixture(scope="module")
def got(tmp_path_factory) -> dict:
    node = find_node()
    if node is None:
        pytest.skip("node 가 없어 화면(JS) 쪽을 돌릴 수 없습니다")
    app = (WEB / "app.js").read_text(encoding="utf-8")
    script = "\n".join([
        "const box = { cls: 2, cx: 0.5, cy: 0.5, w: 0.2, h: 0.2, conf: null, origin: 'human' };",
        "const base = { id: 7, status: 'approved', name: 'a.jpg', path: 'D:/u/cam/a.jpg', camera: 'cam', captured_at: null,",
        "               camera_place: null, grp: null, split: null, work: null, width: 10, height: 10, boxes: [box] };",
        "let server = null, S = null;",
        "const log = { loaded: [], metas: [], templates: 0, cleared: 0 };",
        "const fresh = () => { S = { cur: { ...base, undo: 'kept' }, boxes: [{ ...box, cls0: 2 }], dirty: false, images: [{ id: 7 }] };",
        "  log.loaded = []; log.metas = []; log.templates = 0; log.cleared = 0; };",
        "const api = async () => { if (!server) { const e = new Error('gone'); e.status = 404; throw e; } return server; };",
        "const flushSave = async () => {}, toast = () => {};",
        "const clearCurrent = () => { log.cleared++; S.cur = null; S.images = []; };",
        "const loadImage = async (id) => { log.loaded.push(id); };",
        "const renderMaskTemplates = () => { log.templates++; };",
        "const renderImgMeta = (m) => { log.metas.push([m.name, m.status]); };",
        js_const(app, "CUR_META_KEYS"),
        *(js_function(app, name) for name in PARTS),
        "const show = () => ({ cur: S.cur && { name: S.cur.name, path: S.cur.path, undo: S.cur.undo, status: S.cur.status },",
        "  loaded: [...log.loaded], metas: [...log.metas], templates: log.templates, cleared: log.cleared });",
        "(async () => {",
        "  const out = {};",
        "  fresh(); server = { ...base };",
        "  await syncCurrentAfterJob(); out.same = show();",
        "  fresh(); server = { ...base, name: 'a_2.jpg', path: 'D:/l/images/cam/a_2.jpg' };       // 정리가 옮기고 번호를 붙였다",
        "  await syncCurrentAfterJob(); out.moved = show();",
        "  fresh(); server = { ...base, boxes: [box, { ...box, cls: 3 }] };                       // 자동 라벨링이 박스를 넣었다",
        "  await syncCurrentAfterJob(); out.boxes = show();",
        "  fresh(); server = { ...base, status: 'pending' };",
        "  await syncCurrentAfterJob(); out.status = show();",
        "  fresh(); S.dirty = true; server = { ...base, name: 'a_2.jpg', boxes: [] };              // 저장하지 못한 수정이 남았다",
        "  await syncCurrentAfterJob(); out.dirty = show();",
        "  fresh(); server = null;                                                                 // 휴지통 비우기가 지웠다",
        "  await syncCurrentAfterJob(); out.gone = show();",
        "  console.log(JSON.stringify(out));",
        "})().catch((err) => { console.error((err && err.stack) || err); process.exit(1); });",
    ])
    return run_node(node, script, tmp_path_factory.mktemp("sync"))


OLD = {"name": "a.jpg", "path": "D:/u/cam/a.jpg", "undo": "kept", "status": "approved"}


def test_nothing_changed_touches_nothing(got):
    assert got["same"] == {"cur": OLD, "loaded": [], "metas": [], "templates": 0, "cleared": 0}


def test_a_file_moved_by_tidy_updates_the_name_without_reopening(got):
    """박스·상태가 같으면 다시 열지 않는다(되돌리기·확대가 남는다) — 이름·경로만 새 값으로, 정보 줄을 다시 그린다."""
    moved = got["moved"]
    assert moved["loaded"] == [] and moved["cur"]["undo"] == "kept"
    assert (moved["cur"]["name"], moved["cur"]["path"]) == ("a_2.jpg", "D:/l/images/cam/a_2.jpg")
    assert moved["metas"] == [["a_2.jpg", "approved"]] and moved["templates"] == 1


def test_changed_boxes_or_status_reopen_the_image(got):
    assert got["boxes"]["loaded"] == [7] and got["status"]["loaded"] == [7]
    assert got["boxes"]["metas"] == [] and got["status"]["metas"] == []           # 다시 여는 쪽이 그린다


def test_unsaved_edits_are_left_alone_and_a_deleted_image_is_cleared(got):
    assert got["dirty"] == {"cur": OLD, "loaded": [], "metas": [], "templates": 0, "cleared": 0}
    assert got["gone"]["cleared"] == 1 and got["gone"]["cur"] is None
