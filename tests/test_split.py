"""train / val / test 배정 (split.py) — 한 번 정한 배정이 움직이면 그때까지 잰 test 점수를 쓸 수 없게 된다."""

from __future__ import annotations

import pytest

from autolabel.split import ensure_splits
from helpers import box, far_hashes, plain_name

CAMERAS = """\
cameras:
  cam-01:
    kind: cctv
    place: 시험 장소 A
    split:
  cam-02:
    kind: cctv
    place: 시험 장소 B
    split: test
  cam-03:
    kind: cctv
    place: 시험 장소 C
    split: train
"""


def db_splits(store) -> dict[int, str | None]:
    with store.connect() as con:
        return {int(r["id"]): r["split"] for r in con.execute("SELECT id, split FROM images")}


def db_groups(store) -> dict[int, str | None]:
    with store.connect() as con:
        return {int(r["id"]): r["grp"] for r in con.execute("SELECT id, grp FROM images")}


@pytest.fixture
def singles(add_image):
    """서로 이어지지 않는 이미지 n 장 (이름 규칙에 안 걸리고 dHash 가 멀다)."""

    def make(n: int, seed: int = 1, start: int = 0) -> list[int]:
        return [add_image(plain_name(start + k), boxes=[box(cls=k % 4)], dhash=h)
                for k, h in enumerate(far_hashes(n, seed))]

    return make


def test_nothing_to_split(cfg, store):
    assign, report = ensure_splits(cfg, store)
    assert assign == {} and report.total == 0


def test_a_scene_never_spans_two_splits(cfg, store, add_image, singles):
    hashes = iter(far_hashes(36, seed=7))
    series = {name: [add_image(f"{name}_{n:04d}.jpg", boxes=[box(cls=n % 3)], dhash=next(hashes)) for n in range(1, 13)]
              for name in ("alpha", "beta", "gamma")}
    loose = singles(60, seed=8)
    assign, report = ensure_splits(cfg, store)

    for ids in series.values():
        assert len({assign[i] for i in ids}) == 1
    assert set(assign) == {i for ids in series.values() for i in ids} | set(loose)
    assert set(assign.values()) == {"train", "val", "test"}
    assert report.counts["train"] > report.counts["val"] >= report.counts["test"] > 0
    assert report.moved == {} and report.mixed == {}
    assert report.assigned == len(assign) and report.biggest_group == 12
    assert db_splits(store) == assign                   # DB 에 고정된다


def test_pending_images_are_not_assigned(cfg, store, add_image, singles):
    singles(10)
    waiting = add_image("waiting.jpg", status="pending", dhash="00000000000000ff")
    assign, _ = ensure_splits(cfg, store)
    assert waiting not in assign and db_splits(store)[waiting] is None


def test_running_again_changes_nothing(cfg, store, singles):
    singles(40)
    first, _ = ensure_splits(cfg, store)
    second, report = ensure_splits(cfg, store)
    assert second == first
    assert report.assigned == 0 and report.moved == {}


def test_new_images_follow_their_scene_and_old_ones_stay(cfg, store, add_image, singles):
    hashes = far_hashes(12, seed=3)
    scene = [add_image(f"alpha_{n:04d}.jpg", dhash=hashes[n]) for n in range(10)]
    singles(40, seed=4)
    first, _ = ensure_splits(cfg, store)

    late = add_image("alpha_0500.jpg", dhash=hashes[10])        # 같은 계열의 새 프레임
    second, report = ensure_splits(cfg, store)
    assert second[late] == first[scene[0]]
    assert {i: second[i] for i in first} == first
    assert report.assigned == 1 and report.moved == {}


def test_a_new_image_bridging_train_and_test_moves_nobody(cfg, store, add_image, singles):
    """2026-10-25 — 새 이미지가 train 묶음과 test 묶음을 dHash 로 이어 붙이면, 예전에는 소수 쪽 기존 이미지가 다수결로
    옮겨졌다(test 기준선이 바뀌거나 학습한 이미지가 test 로 들어간다). 이제는 옮기지 않고 알린다."""
    singles(40)
    first, _ = ensure_splits(cfg, store)
    in_train = next(i for i, s in first.items() if s == "train")
    in_test = next(i for i, s in first.items() if s == "test")
    # 두 기존 이미지는 거리 4 라 서로 이어지지 않는다. 새 이미지는 양쪽과 거리 2 다
    store.set_dhashes([(in_train, "0000000000000000"), (in_test, "000000000000000f")])
    bridge = add_image("bridge.jpg", dhash="0000000000000003")

    second, report = ensure_splits(cfg, store)
    assert report.moved == {}
    assert {i: second[i] for i in first} == first
    assert second[in_train] == "train" and second[in_test] == "test"
    assert list(report.mixed.values()) == [{"train": 1, "test": 1}]
    assert second[bridge] in ("train", "test") and report.assigned == 1
    groups = db_groups(store)
    assert groups[in_train] == groups[in_test] == groups[bridge]
    assert "두 분할에 걸칩니다" in report.summary()


def test_dropping_a_split_from_the_ratios_stops_before_writing(cfg, store, singles):
    """2026-11-05 — test_ratio 를 0 으로 바꾸면 test 이미지가 경고 없이 train·val 로 흡수됐다."""
    singles(40)
    ensure_splits(cfg, store)
    before = db_splits(store)
    assert "test" in before.values()
    cfg.raw["train"]["test_ratio"] = 0
    with pytest.raises(RuntimeError):
        ensure_splits(cfg, store)
    assert db_splits(store) == before


def test_without_test_ratio_only_train_and_val(cfg, store, singles):
    cfg.raw["train"]["test_ratio"] = 0
    singles(30)
    assign, _ = ensure_splits(cfg, store)
    assert set(assign.values()) == {"train", "val"}


def test_boxes_under_a_masking_region_do_not_steer_the_split(cfg, store, add_image):
    """가릴 영역이 반 넘게 덮은 박스는 라벨 파일에 없다 — 계층화가 그 박스로 클래스 비중을 맞추면 안 된다 (2026-11-25,
    Gemini 리뷰 묶음 6).

    네 장을 반반으로 나눈다. `decoy` 에는 차 박스 20개가 있지만 전부 영역 아래라 라벨에는 하나도 없고, 라벨에 남는 차는
    `one`·`two` 의 하나씩뿐이다. 고치기 전에는 DB 의 박스(22개)로 비중을 맞춰 decoy 를 한쪽에, one·two 를 **함께** 반대쪽에
    넣었다 — 한 분할에는 차 라벨이 하나도 없었다(실제 DB 의 사본에서는 val 에 사람 라벨이 433개 중 6개였다)."""
    from autolabel import masks

    cfg.raw["train"].update(val_ratio=0.5, test_ratio=0)
    hashes = far_hashes(4, seed=5)
    decoy = add_image(plain_name(0), dhash=hashes[0],
                      boxes=[box(cls=2, cx=0.05 + 0.045 * k, cy=0.5, w=0.04, h=0.04) for k in range(20)])
    add_image(plain_name(1), dhash=hashes[1], boxes=[box(cls=0)])
    one = add_image(plain_name(2), dhash=hashes[2], boxes=[box(cls=2)])
    two = add_image(plain_name(3), dhash=hashes[3], boxes=[box(cls=2)])
    store.set_masks(decoy, [(0.0, 0.0, 1.0, 1.0)])

    assign, report = ensure_splits(cfg, store)
    assert sorted(report.counts.values()) == [2, 2]
    assert assign[one] != assign[two]                       # 라벨에 남는 차가 두 분할에 하나씩
    rects = store.masks_by_image()
    cars = {"train": 0, "val": 0}
    for im, boxes in store.approved_with_boxes():           # 라벨 파일에 들어갈 박스로 따로 센다(학습 세트와 같은 함수)
        keep, _ = masks.split_boxes(boxes, rects.get(im["id"], []))
        cars[assign[im["id"]]] += sum(1 for b in keep if b.cls == 2)
    assert cars == {"train": 1, "val": 1}
    # 보고의 클래스 비중도 라벨 기준이다 — 차는 반반(전에는 DB 박스로 20/22 대 2/22)
    assert report.shares["train"]["car"] == report.shares["val"]["car"] == 0.5


# ----------------------------------------------------------------- 카메라

def test_a_camera_goes_whole_and_pins_are_obeyed(cfg, store, add_image, singles):
    cfg.cameras.write_text(CAMERAS, encoding="utf-8")
    hashes = iter(far_hashes(30, seed=5))
    free = [add_image(f"a{plain_name(k)}", camera="cam-01", boxes=[box()], dhash=next(hashes)) for k in range(9)]
    held_out = [add_image(f"b{plain_name(k)}", camera="cam-02", boxes=[box()], dhash=next(hashes)) for k in range(6)]
    singles(50, seed=6)
    assign, report = ensure_splits(cfg, store)

    assert len({assign[i] for i in free}) == 1
    assert {assign[i] for i in held_out} == {"test"}
    assert report.cameras["cam-02"] == {"test": 6}
    assert len(report.cameras["cam-01"]) == 1


def test_a_pin_does_not_move_images_that_were_already_assigned(cfg, store, add_image, singles):
    cfg.cameras.write_text(CAMERAS, encoding="utf-8")
    hashes = far_hashes(4, seed=9)
    old = add_image("old.jpg", camera="cam-02", dhash=hashes[0], split="train")     # 고정을 적기 전에 train 이 됐다
    singles(30, seed=10)
    new = add_image("new.jpg", camera="cam-02", dhash=hashes[1])
    assign, report = ensure_splits(cfg, store)
    assert assign[old] == "train" and assign[new] == "test"
    assert report.moved == {} and list(report.pin_kept.values()) == [{"train": 1}]


def test_conflicting_pins_in_one_scene_stop_before_writing(cfg, store, add_image, singles):
    cfg.cameras.write_text(CAMERAS, encoding="utf-8")
    singles(20)
    add_image("x.jpg", camera="cam-02", dhash="00000000000000ff")       # test 고정
    add_image("y.jpg", camera="cam-03", dhash="00000000000000ff")       # train 고정인데 화면이 같다
    with pytest.raises(RuntimeError):
        ensure_splits(cfg, store)
    assert set(db_splits(store).values()) == {None}
    assert set(db_groups(store).values()) == {None}


def test_a_broken_camera_file_stops_the_split(cfg, store, add_image, singles):
    cfg.cameras.write_text("cameras:\n  cam-01:\n    kind: nonsense\n", encoding="utf-8")
    singles(10)
    add_image("x.jpg", camera="cam-01", dhash="00000000000000ff")
    with pytest.raises(RuntimeError):
        ensure_splits(cfg, store)
    assert set(db_splits(store).values()) == {None}
