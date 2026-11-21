"""DB 저장 경로 (store.py) — 라벨은 DB 가 작업물 그 자체라, 여기가 틀리면 조용히 잃는다."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from autolabel.masks import MAX_MASKS
from helpers import CLASSES, box, keys, pixel_rects, screen_round_trip

CAR, PICKUP, CONE = 2, 7, 9
RECT_A = (0.0, 0.0, 0.125, 0.125)
RECT_B = (0.5, 0.5, 0.75, 0.75)


def history(store, image_id):
    """오래된 것부터."""
    return store.box_history(image_id)[::-1]


def raw_history(store, image_id):
    with store.connect() as con:
        return [dict(r) for r in con.execute("SELECT * FROM box_history WHERE image_id=? ORDER BY id", (image_id,))]


def count(store, table: str) -> int:
    with store.connect() as con:
        return int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def box_row_ids(store, image_id) -> list[int]:
    with store.connect() as con:
        return [int(r[0]) for r in con.execute("SELECT id FROM boxes WHERE image_id=? ORDER BY id", (image_id,))]


def row(store, image_id) -> dict:
    return store.get_image(image_id)


# ----------------------------------------------------------------- 함정 11·12, 재스캔

def test_image_ids_are_never_reused(store, add_image):
    """함정 11 — id 가 다시 쓰이면 브라우저가 캐시한 옛 사진을 새 이미지 자리에 보여 준다."""
    add_image("a.jpg")
    last = add_image("b.jpg")
    store.delete_images([last])
    assert add_image("c.jpg") > last


def test_tag_ids_are_never_reused(store):
    store.create_tag("one")
    last = store.create_tag("two")
    store.delete_tag(last)
    assert store.create_tag("three") > last


def test_scan_does_not_drop_classes_added_in_the_ui(store):
    """함정 12 — 검수 UI 로 추가한 클래스는 DB 에만 있다. 설정의 목록으로 다시 쓸 때 지워지면 안 된다."""
    extra = store.add_class("forklift")
    assert extra == len(CLASSES)
    assert store.add_class("forklift") == extra                 # 같은 이름은 같은 번호
    store.set_classes(CLASSES)
    assert store.get_classes() == [*CLASSES, "forklift"]
    store.set_classes(CLASSES, drop_extra=True)
    assert store.get_classes() == CLASSES


def test_rescan_keeps_reviewed_labels_and_does_not_mix_up_boxes(store, add_image, tmp_path):
    """재스캔은 이미 등록된 이미지의 박스·상태를 덮지 않는다. 예전 `ON CONFLICT DO UPDATE` + lastrowid 는 겹친
    이미지의 박스를 **직전에 넣은 다른 이미지**에 붙였다."""
    old = add_image("old.jpg", status="approved", boxes=[box(cls=0)], camera="cam-01")
    old_path = row(store, old)["path"]
    new_path = str(tmp_path / "img" / "new.jpg")
    store.ingest_batch([
        {"path": new_path, "source": "legacy", "width": 10, "height": 10, "status": "unlabeled", "boxes": [box(cls=1)]},
        {"path": old_path, "source": "collected", "width": 1280, "height": 720, "status": "unlabeled",
         "boxes": [box(cls=5), box(cls=5, cx=0.25)], "camera": "cam-99"},
    ])
    assert keys(store.get_boxes(old)) == keys([box(cls=0)])
    assert (row(store, old)["status"], row(store, old)["camera"]) == ("approved", "cam-01")
    with store.connect() as con:
        new = int(con.execute("SELECT id FROM images WHERE path=?", (new_path,)).fetchone()[0])
    assert [b.cls for b in store.get_boxes(new)] == [1]


def test_background_images_are_part_of_the_split_input(store, add_image):
    empty = add_image("empty.jpg")
    full = add_image("full.jpg", boxes=[box(cls=CAR), box(cls=CAR, cx=0.25), box(cls=0)])
    add_image("pending.jpg", status="pending")
    got = {it["id"]: it["counts"] for it in store.approved_class_counts()}
    assert got == {empty: {}, full: {CAR: 2, 0: 1}}


# ----------------------------------------------------------------- 박스 저장과 변경 이력

def test_first_save_on_an_empty_image_leaves_no_history(store, add_image):
    i = add_image("a.jpg", status="unlabeled")
    assert store.set_boxes(i, [box()]) == (0, 1)
    assert store.history_count(i) == 0


def test_every_changed_save_is_recorded_with_before_and_after(store, add_image):
    a, b = box(cx=0.25), box(cx=0.75, conf=None, origin="human")
    i = add_image("a.jpg", boxes=[a])
    assert store.set_boxes(i, [a, b]) == (1, 2)
    assert store.set_boxes(i, [b], source="restore") == (2, 1)
    h = history(store, i)
    assert [(e["source"], e["n_before"], e["n_after"]) for e in h] == [("ui", 1, 2), ("restore", 2, 1)]
    assert keys(h[0]["before"]) == keys([a])
    assert keys(h[0]["after"]) == keys(h[1]["before"]) == keys([a, b])
    assert keys(h[1]["after"]) == keys(store.get_boxes(i)) == keys([b])
    assert h[0]["masks_before"] is None and h[0]["masks_after"] is None        # 박스만 바뀐 줄


def test_saving_the_same_boxes_is_not_a_change(store, add_image):
    a = box()
    i = add_image("a.jpg", boxes=[a])
    add_image("later.jpg", boxes=[box()])               # 뒤에 다른 박스 행이 있어야 '다시 넣은 행' 이 새 id 를 받는다
    ids = box_row_ids(store, i)
    store.set_boxes(i, [a])
    store.set_boxes(i, [box(cx=a.cx + 1e-9)])           # 부동소수 잡음
    assert store.history_count(i) == 0
    assert box_row_ids(store, i) == ids                 # 행을 지우고 다시 넣지도 않는다


def test_invalid_boxes_are_not_saved(store, add_image):
    i = add_image("a.jpg", status="unlabeled")
    assert store.set_boxes(i, [box(), box(w=0.0), box(cx=float("nan"))]) == (0, 1)


def test_history_lists_are_stored_once(store, add_image):
    """2026-11-11 — 한 줄의 '전' 은 앞 줄의 '후' 와 같은 목록이다. 목록은 hist_lists 에 한 번만 둔다."""
    a, b, c = [box(cx=0.25)], [box(cx=0.5)], [box(cx=0.75)]
    i = add_image("a.jpg", boxes=a)
    store.set_boxes(i, b)
    store.set_boxes(i, c)
    store.set_boxes(i, a)                               # 처음 목록으로 돌아온다 — 새 목록이 아니다
    rows = raw_history(store, i)
    assert all(r[k].startswith("#") for r in rows for k in ("before", "after"))
    assert rows[1]["before"] == rows[0]["after"]
    assert rows[2]["after"] == rows[0]["before"]
    assert count(store, "hist_lists") == 3


def test_history_written_before_the_change_is_still_readable(store, add_image):
    """11-11 전 줄은 JSON 글자를 그대로 갖고 있다 — 새 줄('#id')과 섞여 있어도 읽는다."""
    i = add_image("a.jpg", boxes=[box(cx=0.25)])
    old = json.dumps([[CAR, 0.5, 0.5, 0.25, 0.25, 0.875, "auto"]])
    with store.connect() as con:
        con.execute("INSERT INTO box_history(image_id, at, source, batch, before, after, n_before, n_after)"
                    " VALUES (?, '2026-11-10T00:00:00+00:00', 'ui', NULL, ?, '[]', 1, 0)", (i, old))
    store.set_boxes(i, [box(cx=0.75)])
    h = history(store, i)
    assert keys(h[0]["before"]) == keys([box()]) and h[0]["after"] == []
    assert keys(h[1]["after"]) == keys([box(cx=0.75)])
    assert keys(store.history_entry(h[0]["id"])["before"]) == keys([box()])


def test_deleting_an_image_removes_its_rows_and_unreferenced_lists(store, add_image):
    same = [box(cx=0.25)]
    i = add_image("a.jpg", boxes=same)
    j = add_image("b.jpg", boxes=same)
    for image_id in (i, j):
        store.set_boxes(image_id, [box(cx=0.75)], masks=[RECT_A])
    assert count(store, "hist_lists") == 4              # 박스 전·후, 영역 전([])·후 — 두 이미지가 함께 쓴다
    store.delete_images([i])
    assert (store.get_boxes(i), store.get_masks(i), store.history_count(i)) == ([], [], 0)
    assert count(store, "hist_lists") == 4              # j 가 아직 가리킨다
    assert keys(history(store, j)[0]["before"]) == keys(same)
    store.delete_images([j])
    assert count(store, "hist_lists") == 0


# ----------------------------------------------------------------- 가릴 영역

def test_mask_only_save_is_recorded_and_keeps_box_rows(store, add_image):
    a = box()
    i = add_image("a.jpg", boxes=[a])
    add_image("later.jpg", boxes=[box()])
    ids = box_row_ids(store, i)
    store.set_boxes(i, [a], masks=[RECT_A])
    assert store.get_masks(i) == [RECT_A]
    assert box_row_ids(store, i) == ids                 # 영역만 바꾼 저장은 박스 행을 다시 쓰지 않는다 (2026-11-10)
    (e,) = history(store, i)
    assert (e["masks_before"], e["masks_after"]) == ([], [list(RECT_A)])
    assert keys(e["before"]) == keys(e["after"]) == keys([a])


def test_masks_stay_when_a_save_does_not_send_them(store, add_image):
    i = add_image("a.jpg", boxes=[box()])
    store.set_boxes(i, [box()], masks=[RECT_A, RECT_B])
    store.set_boxes(i, [box(cx=0.25)])                  # masks=None — 영역을 손대지 않은 화면의 저장
    assert store.get_masks(i) == [RECT_A, RECT_B]
    store.set_boxes(i, [box(cx=0.25)], masks=[])        # 빈 목록은 '전부 지움'
    assert store.get_masks(i) == []


def test_set_masks_respects_expect(store, add_image):
    i = add_image("a.jpg", boxes=[box()])
    stale = (row(store, i)["status"], row(store, i)["updated_at"])
    store.set_status(i, "pending")
    assert store.set_masks(i, [RECT_A], expect=stale) is False
    assert store.get_masks(i) == []
    assert store.set_masks(i, [RECT_A]) is True
    assert history(store, i)[-1]["source"] == "mask"


def test_template_is_copied_and_reverted_even_after_boxes_were_reviewed(store, add_image):
    """템플릿을 넣은 뒤 박스를 검수한 장도 영역은 묶음째 되돌릴 수 있어야 한다 — 그때 박스는 지금 것을 둔다."""
    i = add_image("a.jpg", boxes=[box()])
    j = add_image("b.jpg", boxes=[box()])
    k = add_image("c.jpg", boxes=[box()])
    store.set_masks(j, [RECT_B])
    out = store.apply_mask_template([i, j, k, i, 99999], [RECT_A, RECT_B], mode="add", batch="tpl")
    assert out == {"changed": 3, "unchanged": 0, "blocked": 0, "full": 0, "missing": 1}
    assert store.get_masks(j) == [RECT_B, RECT_A]       # 이미 있던 같은 자리는 건너뛴다

    reviewed = [box(cx=0.25, conf=None, origin="human")]
    store.set_boxes(i, reviewed)                        # i: 박스만 검수
    store.set_boxes(k, [box()], masks=[RECT_A])         # k: 영역을 손으로 고쳤다 — 되돌리지 않는다
    assert store.revert_batch("tpl") == (2, 1)
    assert store.get_masks(i) == [] and keys(store.get_boxes(i)) == keys(reviewed)
    assert store.get_masks(j) == [RECT_B]
    assert store.get_masks(k) == [RECT_A]


def test_template_revert_works_with_any_coordinates(store, add_image):
    """이력의 영역은 7자리로 반올림돼 있다 — DB 의 지금 영역(반올림 전)과 같은 자리로 봐야 되돌릴 수 있다."""
    rects = [(1 / 3, 1 / 7, 2 / 3, 5 / 7), (1326 / 2560, 100 / 1440, 1900 / 2560, 771 / 1440)]
    i = add_image("a.jpg", boxes=[box()])
    assert store.apply_mask_template([i], rects, mode="add", batch="t")["changed"] == 1
    assert store.get_masks(i) == rects                  # DB 에는 반올림하지 않은 값이 들어간다
    assert store.revert_batch("t") == (1, 0)
    assert store.get_masks(i) == []


def test_template_applied_through_the_screen_is_still_recognised(store, add_image):
    """화면에서 한 장에 적용하면 좌표가 {cx, cy, w, h} 를 한 번 오간 값으로 저장된다. 그 뒤의 여러 장 적용·빼기와 '바뀌었나'
    판정이 그 영역을 같은 자리로 알아봐야 한다 (2026-11-12 — 픽셀에 맞춘 2560 폭 좌표의 8.4% 가 다른 자리가 됐다)."""
    rects = pixel_rects(2560, 1440, 300)
    i = add_image("a.jpg", boxes=[box()])
    on_screen = [screen_round_trip(r) for r in rects]
    assert on_screen != rects
    store.set_boxes(i, [box()], masks=on_screen)                        # 화면의 저장

    out = store.apply_mask_template([i], rects, mode="add", batch="again")
    assert out["unchanged"] == 1 and len(store.get_masks(i)) == len(rects)      # 겹쳐 들어가지 않는다
    n = store.history_count(i)
    store.set_boxes(i, [box()], masks=[screen_round_trip(m) for m in store.get_masks(i)])    # 영역은 그대로인 저장
    assert store.history_count(i) == n                                  # 바뀐 것이 없다 — 이력도 남지 않는다
    out = store.apply_mask_template([i], rects, mode="remove", batch="off")
    assert out["changed"] == 1 and store.get_masks(i) == []             # 빼기가 하나도 남기지 않는다
    assert store.revert_batch("off") == (1, 0) and len(store.get_masks(i)) == len(rects)


def test_template_is_not_added_to_blocked_images_but_can_be_removed(store, add_image):
    """test 이미지에는 영역을 더하지 않는다(규칙은 서버가 넘긴다). 지우기만은 받는다."""
    is_test = lambda r: r["split"] == "test"
    t = add_image("t.jpg", split="test")
    n = add_image("n.jpg", split="train")
    out = store.apply_mask_template([t, n], [RECT_A], mode="add", batch="b1", blocked=is_test)
    assert (out["changed"], out["blocked"]) == (1, 1)
    assert store.get_masks(t) == []

    store.apply_mask_template([t], [RECT_A], mode="add", batch="b2")           # (규칙 없이 들어간 옛 영역)
    out = store.apply_mask_template([t, n], [RECT_A], mode="remove", batch="b3", blocked=is_test)
    assert out["changed"] == 2
    assert store.get_masks(t) == store.get_masks(n) == []
    # 빼기를 되돌리면 영역이 새로 생긴다 — test 는 건너뛴다
    assert store.revert_batch("b3", mask_blocked=is_test) == (1, 1)
    assert (store.get_masks(t), store.get_masks(n)) == ([], [RECT_A])


POLY = ((0.1, 0.1), (0.4, 0.15), (0.2, 0.45))          # 다각형 영역 (2026-11-19)


def test_polygon_masks_are_saved_exactly_and_recorded(store, add_image):
    i = add_image("a.jpg", boxes=[box()])
    store.set_boxes(i, [box()], masks=[RECT_A, POLY])
    assert store.get_masks(i) == [RECT_A, POLY]
    assert store.masks_by_image()[i] == [RECT_A, POLY]
    with store.connect() as con:
        rows = [tuple(r) for r in con.execute("SELECT x1, y1, x2, y2, poly FROM masks WHERE image_id=? ORDER BY id", (i,))]
        old_read = [tuple(r) for r in con.execute("SELECT x1, y1, x2, y2 FROM masks WHERE image_id=? ORDER BY id", (i,))]
    assert rows[0][4] is None and json.loads(rows[1][4]) == [list(p) for p in POLY]
    assert old_read[1] == (0.1, 0.1, 0.4, 0.45)         # poly 를 모르는 옛 코드는 바깥 사각형을 읽는다(그 자리 둘레를 가린다)
    (e,) = history(store, i)
    assert (e["masks_before"], e["masks_after"]) == ([], [list(RECT_A), [list(p) for p in POLY]])

    n = store.history_count(i)
    store.set_boxes(i, [box()], masks=[list(RECT_A), [list(p) for p in POLY]])     # 화면이 보낸 같은 영역 — 바뀐 것이 없다
    assert store.history_count(i) == n
    moved = (POLY[0], (0.45, 0.15), POLY[2])                                        # 꼭짓점 하나를 옮겼다
    store.set_boxes(i, [box()], masks=[RECT_A, moved])
    assert store.history_count(i) == n + 1 and store.get_masks(i) == [RECT_A, moved]
    assert history(store, i)[-1]["masks_before"][1] == [list(p) for p in POLY]


def test_old_db_gets_the_poly_column_and_keeps_its_rects(cfg, store, add_image):
    """다각형을 더하기 전(2026-11-19 전)의 DB — 새 코드가 열 때 poly 열을 더하고(사용자 결정), 있던 사각형은 그대로 읽힌다."""
    from autolabel.store import Store

    i = add_image("a.jpg", boxes=[box()])
    with store.connect() as con:
        con.execute("ALTER TABLE masks DROP COLUMN poly")
        con.execute("INSERT INTO masks(image_id, x1, y1, x2, y2) VALUES (?,?,?,?,?)", (i, *RECT_A))   # 옛 코드의 쓰기
    again = Store(cfg.db)
    with again.connect() as con:
        assert [r["name"] for r in con.execute("PRAGMA table_info(masks)")][-1] == "poly"
    assert again.get_masks(i) == [RECT_A]
    again.set_masks(i, [RECT_A, POLY])
    assert again.get_masks(i) == [RECT_A, POLY]
    Store(cfg.db)                                       # 이미 있는 열은 다시 더하지 않는다
    assert again.get_masks(i) == [RECT_A, POLY]


def test_polygon_templates_add_remove_and_revert(store, add_image):
    i = add_image("a.jpg", boxes=[box()])
    tid, _ = store.save_mask_template("p", [RECT_A, POLY], width=1280, height=720)
    tpl = store.get_mask_template(tid)
    assert tpl["rects"] == [list(RECT_A), [list(p) for p in POLY]]
    out = store.apply_mask_template([i], tpl["rects"], mode="add", batch="t1")
    assert out["changed"] == 1 and store.get_masks(i) == [RECT_A, POLY]
    assert store.apply_mask_template([i], tpl["rects"], mode="add", batch="t2")["unchanged"] == 1   # 같은 자리는 건너뛴다
    reviewed = [box(cx=0.3, conf=None, origin="human")]
    store.set_boxes(i, reviewed)                        # 박스만 검수 — 영역은 묶음째 되돌릴 수 있다
    assert store.revert_batch("t1") == (1, 0)
    assert store.get_masks(i) == [] and keys(store.get_boxes(i)) == keys(reviewed)

    store.set_boxes(i, reviewed, masks=[[list(p) for p in POLY], RECT_B])
    out = store.apply_mask_template([i], [POLY, RECT_A], mode="remove", batch="t3")
    assert out["changed"] == 1 and store.get_masks(i) == [RECT_B]     # 같은 자리의 다각형만 뺀다
    assert store.revert_batch("t3") == (1, 0) and store.get_masks(i) == [POLY, RECT_B]


def test_polygons_with_long_decimals_survive_the_rounded_history(store, add_image):
    """이력은 7자리로 반올림한다 — 되읽은 다각형이 DB 의 다각형(반올림 전)과 같은 자리로 보여야 묶음째 되돌린다."""
    poly = ((1 / 3, 1 / 7), (1326 / 2560, 100 / 1440), (1900 / 2560, 771 / 1440), (2 / 3, 5 / 7))
    i = add_image("a.jpg", boxes=[box()])
    assert store.apply_mask_template([i], [poly], mode="add", batch="t")["changed"] == 1
    assert store.get_masks(i) == [poly]                 # DB 에는 반올림하지 않은 값
    assert store.revert_batch("t") == (1, 0) and store.get_masks(i) == []


def test_template_does_not_push_an_image_over_the_limit(store, add_image):
    i = add_image("a.jpg")
    side = 1 / 32
    full = [(c * side, r * side, (c + 0.5) * side, (r + 0.5) * side) for r in range(20) for c in range(25)]
    assert len(full) == MAX_MASKS
    store.set_masks(i, full)
    out = store.apply_mask_template([i], [(0.9, 0.9, 1.0, 1.0)], mode="add", batch="b")
    assert out["full"] == 1 and len(store.get_masks(i)) == MAX_MASKS


# ----------------------------------------------------------------- 자동 라벨링이 사람의 손을 덮지 않게

def test_prediction_is_applied_only_if_nobody_touched_the_image(store, add_image):
    i = add_image("a.jpg", status="unlabeled")
    seen = (row(store, i)["status"], row(store, i)["updated_at"])
    assert store.apply_prediction(i, [box()], expect=seen, uncertainty=0.5, model_version="m") is True
    assert (row(store, i)["status"], row(store, i)["model_version"]) == ("pending", "m")
    assert keys(store.get_boxes(i)) == keys([box()])

    # 작업이 도는 사이 사람이 승인했다 — 시작할 때 본 값과 다르면 아무것도 쓰지 않는다
    seen = (row(store, i)["status"], row(store, i)["updated_at"])
    store.set_status(i, "approved")
    assert store.apply_prediction(i, [box(cx=0.25)], expect=seen, uncertainty=0.5, model_version="m2") is False
    assert (row(store, i)["status"], row(store, i)["model_version"]) == ("approved", "m")
    assert keys(store.get_boxes(i)) == keys([box()])
    assert store.apply_prediction(99999, [box()], expect=seen, uncertainty=0.5, model_version="m") is False


def test_prediction_over_existing_boxes_is_recorded(store, add_image):
    i = add_image("a.jpg", status="pending", boxes=[box()])
    seen = (row(store, i)["status"], row(store, i)["updated_at"])
    assert store.apply_prediction(i, [box(cx=0.25)], expect=seen, uncertainty=0.5, model_version="m")
    (e,) = history(store, i)
    assert e["source"] == "predict" and keys(e["before"]) == keys([box()])


# ----------------------------------------------------------------- 같은 자리 일괄 수정과 되돌리기

def test_bulk_class_change_and_revert_skip_images_edited_since(store, add_image):
    original = box(cls=CAR, conf=0.75, origin="auto")
    i = add_image("a.jpg", boxes=[original, box(cls=0, cx=0.25)])
    j = add_image("b.jpg", boxes=[original])
    bi, bj = box_row_ids(store, i)[0], box_row_ids(store, j)[0]

    assert store.set_class_bulk({i: [bi], j: [bj]}, PICKUP, batch="bulk-1") == 2
    assert keys(store.get_boxes(j)) == keys([box(cls=PICKUP, conf=None, origin="human")])    # 사람이 정한 것
    assert [b.cls for b in store.get_boxes(i)] == [PICKUP, 0]
    assert box_row_ids(store, i)[0] == bi                                                    # 행은 그대로
    assert [e["source"] for e in history(store, i)] == ["bulk"]

    edited = [box(cls=PICKUP, cx=0.75, conf=None, origin="human")]
    store.set_boxes(j, edited)                          # j 는 그 뒤에 또 고쳤다
    assert store.revert_batch("bulk-1") == (1, 1)
    assert keys(store.get_boxes(i)) == keys([original, box(cls=0, cx=0.25)])                 # conf·origin 까지 원래대로
    assert keys(store.get_boxes(j)) == keys(edited)
    assert history(store, i)[-1]["source"] == "revert"


def test_bulk_change_only_touches_boxes_of_that_image(store, add_image):
    """2026-11-05 — 다른 이미지의 박스 id 가 섞이면 그 이미지가 이력 없이 바뀌었다."""
    i = add_image("a.jpg", boxes=[box(cls=CAR)])
    j = add_image("b.jpg", boxes=[box(cls=CAR)])
    (bi,), (bj,) = box_row_ids(store, i), box_row_ids(store, j)
    assert store.set_class_bulk({i: [bi, bj]}, PICKUP, batch="x") == 1          # i 의 목록에 j 의 박스 id 가 섞였다
    assert [b.cls for b in store.get_boxes(i)] == [PICKUP]
    assert [b.cls for b in store.get_boxes(j)] == [CAR]
    assert (store.history_count(i), store.history_count(j)) == (1, 0)
    assert store.set_class_bulk({i: [bj]}, CAR, batch="y") == 0                 # 제 박스가 하나도 없으면 아무 일도 없다


def test_revert_refuses_to_bring_back_a_class_that_no_longer_exists(store, add_image):
    """클래스를 합쳐 뺀 묶음(tools/merge_class.py)은 그 클래스를 되살리기 전에는 되돌리지 않는다 — 하나도 안 바꾼다."""
    i = add_image("a.jpg", boxes=[box(cls=CONE)])
    j = add_image("b.jpg", boxes=[box(cls=CAR)])
    store.set_class_bulk({i: box_row_ids(store, i), j: box_row_ids(store, j)}, 0, batch="merge")
    store.set_classes(CLASSES[:CONE], drop_extra=True)
    with pytest.raises(ValueError):
        store.revert_batch("merge")
    assert [b.cls for b in store.get_boxes(i)] == [b.cls for b in store.get_boxes(j)] == [0]
    store.add_class(CLASSES[CONE])
    assert store.revert_batch("merge") == (2, 0)
    assert [b.cls for b in store.get_boxes(i)] == [CONE]


def test_reverts_do_not_push_bulk_batches_out_of_the_list(store, add_image):
    """변경 기록의 '일괄 수정' 목록은 일괄 수정을 limit 건 싣는다 — 되돌리기 줄이 그 자리를 차지하지 않는다.

    2026-11-16 에 고침: 일괄 수정(bulk)과 되돌리기(revert) 줄을 함께 limit 줄로 잘라, 여러 건을 되돌린 뒤에는 화면(bulk 만 그린다)에
    그만큼 덜 보였다 — 여기처럼 최근 줄이 전부 되돌리기면 한 건도 안 보였다.
    """
    ids = [add_image(f"b{k}.jpg", boxes=[box()]) for k in range(8)]
    for k, i in enumerate(ids):                         # 묶음 여덟 — 이미지 하나씩에 영역을 더한다
        assert store.apply_mask_template([i], [RECT_A], mode="add", batch=f"b{k}")["changed"] == 1
    for k in range(5):                                  # 그 가운데 다섯을 되돌렸다(가장 새 줄 다섯이 전부 revert 다)
        assert store.revert_batch(f"b{k}") == (1, 0)

    rows = store.batches(limit=4)
    bulk = [r["batch"] for r in rows if r["source"] == "bulk"]
    undone = {r["batch"] for r in rows if r["source"] == "revert"}
    assert bulk == ["b7", "b6", "b5", "b4"]             # 일괄 수정 넷 — 최근 것부터
    assert "취소: b4" in undone                         # 그 묶음을 되돌린 줄도 함께 온다(화면이 '되돌림' 으로 표시한다)
    assert all(r["first_id"] > rows[-1]["first_id"] or r is rows[-1] for r in rows)
    assert [r["batch"] for r in store.batches(limit=30) if r["source"] == "bulk"] == [f"b{k}" for k in range(7, -1, -1)]
    assert store.batches(limit=0) == [r for r in store.batches(limit=30) if r["source"] != "bulk"][:5]


def test_backup_daily_once_a_day_keeps_latest(store):
    """하루 사본 — 그날 것이 있으면 건너뛰고, 이 함수가 만든 사본만 최근 keep 개 남긴다(2026-11-21 부터 now 를 받는다)."""
    d = store.db_path.parent / "backups"
    d.mkdir()
    (d / "labeling_camrename_20261101_000000.db").write_bytes(b"")     # 카메라 키 바꾸기 전 사본 — 지우지 않는다
    for day in (17, 18, 19, 20):
        assert store.backup_daily(3, now=datetime(2026, 11, day, 9, 0, 0)) is not None
        assert store.backup_daily(3, now=datetime(2026, 11, day, 23, 0, 0)) is None      # 같은 날 두 번째
    assert sorted(p.name for p in d.glob("*.db")) == [
        "labeling_20261118_090000.db", "labeling_20261119_090000.db", "labeling_20261120_090000.db",
        "labeling_camrename_20261101_000000.db"]
    assert store.backup_daily(0, now=datetime(2026, 11, 21, 9, 0, 0)) is None
