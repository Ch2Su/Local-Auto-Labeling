"""출처 기록 (provenance.py) — 촬영 시각 읽기와 cameras.yaml 을 글자로 고치는 길.

cameras.yaml 은 사람이 손으로 쓰는 파일이라(주석·순서·`crop` 같은 다른 속성) 통째로 다시 쓰지 않는다. 여기의 장소 이름은
전부 지어낸 것이다(실제 장소 이름은 git 에 넣지 않는다 — CLAUDE.md).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from autolabel.capture import frame_name, notime_name
from autolabel.provenance import (CameraExists, CameraMissing, add_camera, capture_time, edit_camera, load_cameras,
                                  normalize_tz, parse_tz, remove_cameras, set_splits, tz_text)

KST = parse_tz("+09:00")

SAMPLE = """\
# 카메라 등록 (시험용)
cameras:
  # 손으로 쓴 설명
  cam-01:
    kind: cctv
    place: 시험 장소 A
    mount: pole
    split:

  cam-02:
    kind: screen
    place: 시험 장소 B   # 줄 끝 주석
    crop: [0, 0, 100, 100]
    split: test
"""


@pytest.fixture(params=["\n", "\r\n"], ids=["lf", "crlf"])
def reg(request, tmp_path) -> Path:
    """손으로 쓴 등록 파일 — 줄끝이 LF 인 것과 CRLF 인 것."""
    p = tmp_path / "data" / "cameras.yaml"
    p.parent.mkdir()
    p.write_bytes(SAMPLE.replace("\n", request.param).encode("utf-8"))
    return p


def only_one_newline_style(data: bytes) -> bool:
    return data.count(b"\n") == data.count(b"\r\n") or b"\r" not in data


# ----------------------------------------------------------------- 촬영 시각

@pytest.mark.parametrize("name, want", [
    ("cam-01_20261103_140005_120.jpg", "2026-11-03T14:00:05.120+09:00"),       # extract_frames·영상 캡처가 짓는 이름
    ("cam-01_20261103_140005.jpg", "2026-11-03T14:00:05+09:00"),
    ("스크린샷 2026-06-08 165037.png", "2026-06-08T16:50:37+09:00"),
    ("Screenshot_20261103-141500.png", "2026-11-03T14:15:00+09:00"),
    ("cam-01_20261103_140005_120__ab12cd34.jpg", "2026-11-03T14:00:05.120+09:00"),      # tidy 꼬리표
    ("cam-01_20261103_140005_jpg.rf.0a1b2c3d.jpg", "2026-11-03T14:00:05+09:00"),        # 로보플로 꼬리표
    ("x_20261399_120000.jpg", None),                    # 13월 99일 — 날짜가 아닌 숫자
    ("holiday.jpg", None),
])
def test_capture_time_from_name(tmp_path, name, want):
    assert capture_time(tmp_path / name, KST) == want


def test_scan_reads_back_the_time_that_capture_writes():
    t = datetime(2026, 11, 3, 23, 59, 59, 7000)
    assert capture_time(Path(frame_name("cam-01", t)), KST) == "2026-11-03T23:59:59.007+09:00"


@pytest.mark.parametrize("camera, tag", [("vid-01", "0a1b2c3d"), ("cam-20261103", "20251231"), ("vid-01", "20261103")])
def test_names_without_a_time_are_not_read_as_a_time(tmp_path, camera, tag):
    """촬영 시각을 모르는 프레임의 이름은 scan 이 시각으로 읽으면 안 된다 — 엉뚱한 captured_at 이 들어간다."""
    for sec in (0, 65.5, 86399.999, 140005.12):
        assert capture_time(tmp_path / notime_name(camera, tag, sec), KST) is None


# ----------------------------------------------------------------- 시간대

@pytest.mark.parametrize("text, want", [
    ("+09:00", "+09:00"), ("+9", "+09:00"), ("+0930", "+09:30"), ("-5", "-05:00"),
    ("KST", "+09:00"), ("kst", "+09:00"), ("JST", "+09:00"), ("UTC", "+00:00"), ("Z", "+00:00"),
    ("UTC+9", "+09:00"), ("GMT-05:00", "-05:00"), ("Asia/Seoul", "+09:00"),
    ("America/New_York", None),
])
def test_normalize_tz(text, want):
    assert normalize_tz(text) == want


@pytest.mark.parametrize("text", ["CST", "IST", "EST", "PST", "+15", "+14:30", "+9:60"])
def test_ambiguous_or_out_of_range_tz_is_refused(text):
    with pytest.raises(ValueError):
        normalize_tz(text)


def test_tz_text_round_trip():
    for text in ("+09:00", "-05:30", "+00:00"):
        assert tz_text(parse_tz(text)) == text


# ----------------------------------------------------------------- 등록 파일 읽기

def test_load(reg):
    cams = load_cameras(reg)
    assert set(cams) == {"cam-01", "cam-02"}
    assert (cams["cam-01"].kind, cams["cam-01"].mount, cams["cam-01"].split) == ("cctv", "pole", None)
    assert (cams["cam-02"].mount, cams["cam-02"].split, tz_text(cams["cam-02"].tz)) == ("unknown", "test", "+09:00")


def test_load_missing_file_is_empty(tmp_path):
    assert load_cameras(tmp_path / "none.yaml") == {}


@pytest.mark.parametrize("text", [
    "cameras: [unclosed",                               # YAML 이 아니다
    "- a\n- b\n",                                       # 맨 위가 사전이 아니다
    "cameras:\n  - cam-01\n",                           # 항목이 사전이 아니다
    "cameras:\n  cam-01: cctv\n",                       # 속성이 사전이 아니다
    "cameras:\n  Cam_01:\n    kind: cctv\n    place: 시험 장소\n",        # 키 꼴
    "cameras:\n  cam-01:\n    kind: cctv\n",            # 장소 없음
    "cameras:\n  cam-01:\n    kind: cctv\n    place: 시험 장소\n    split: holdout\n",
    "cameras:\n  cam-01:\n    kind: cctv\n    place: 시험 장소\n    tz: CST\n",
])
def test_every_kind_of_broken_file_is_a_value_error(tmp_path, text):
    """ValueError 만 받는 쪽(검수 서버의 카메라 목록, 분할)을 다른 예외가 뚫으면 500 이 된다 (2026-11-05)."""
    p = tmp_path / "cameras.yaml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_cameras(p)


def test_all_problems_are_reported_at_once(tmp_path):
    p = tmp_path / "cameras.yaml"
    p.write_text("cameras:\n  cam-01:\n    kind: nonsense\n    place: 시험 장소\n  cam-02:\n    kind: cctv\n",
                 encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        load_cameras(p)
    assert "cam-01" in str(exc.value) and "cam-02" in str(exc.value)


# ----------------------------------------------------------------- 더하기·빼기

def test_add_then_remove_restores_the_file_byte_for_byte(reg, tmp_path):
    original = reg.read_bytes()
    backups = tmp_path / "backups"
    cam = add_camera(reg, "cam-03", {"kind": "cctv", "place": "시험 장소 C", "mount": "wall", "tz": "KST",
                                     "note": "메모: 콜론 # 과 샵"},
                     backup_dir=backups, comment="시험에서 등록")
    assert (cam.key, cam.mount, cam.split, tz_text(cam.tz)) == ("cam-03", "wall", None, "+09:00")

    added = reg.read_bytes()
    assert added.startswith(original.rstrip())          # 있던 줄은 그대로, 끝에 덧붙였다
    assert only_one_newline_style(added)
    after = load_cameras(reg)
    assert set(after) == {"cam-01", "cam-02", "cam-03"}
    assert after["cam-02"].split == "test"
    assert b"crop: [0, 0, 100, 100]" in added and "# 줄 끝 주석".encode() in added
    assert [p.read_bytes() for p in backups.glob("cameras_*.yaml")] == [original]      # 쓰기 전 파일을 남긴다

    assert remove_cameras(reg, ["cam-03"], backup_dir=backups) == ["cam-03"]
    assert reg.read_bytes() == original
    assert not list(reg.parent.glob("*.tmp"))


def test_add_to_a_missing_file_creates_it(tmp_path):
    p = tmp_path / "data" / "cameras.yaml"
    add_camera(p, "cam-01", {"kind": "cctv", "place": "시험 장소 A"})
    add_camera(p, "cam-02", {"kind": "phone", "place": "시험 장소 B", "split": "test"})
    cams = load_cameras(p)
    assert [(c.key, c.kind, c.split) for c in cams.values()] == [("cam-01", "cctv", None), ("cam-02", "phone", "test")]


@pytest.mark.parametrize("key, attrs", [
    ("cam-01", {"kind": "cctv", "place": "시험 장소"}),              # 이미 있는 키
    ("Cam 03", {"kind": "cctv", "place": "시험 장소"}),              # 키 꼴 (폴더 이름이 된다)
    ("cam-03", {"kind": "tv", "place": "시험 장소"}),
    ("cam-03", {"kind": "cctv", "place": "  "}),
    ("cam-03", {"kind": "cctv", "place": "시험 장소", "mount": "ceiling"}),
    ("cam-03", {"kind": "cctv", "place": "시험 장소", "split": "holdout"}),
    ("cam-03", {"kind": "cctv", "place": "시험 장소", "tz": "CST"}),
    ("cam-03", {"kind": "cctv", "place": "두 줄\n장소"}),
])
def test_a_refused_add_writes_nothing(reg, tmp_path, key, attrs):
    original = reg.read_bytes()
    with pytest.raises(ValueError):
        add_camera(reg, key, attrs, backup_dir=tmp_path / "backups")
    assert reg.read_bytes() == original
    assert not (tmp_path / "backups").exists()


def test_add_existing_key_is_its_own_error(reg):
    with pytest.raises(CameraExists):
        add_camera(reg, "cam-02", {"kind": "cctv", "place": "시험 장소"})


def test_remove_takes_the_camera_with_its_comment_and_leaves_the_rest(reg):
    remove_cameras(reg, ["cam-01"])
    text = reg.read_bytes().decode("utf-8")
    assert set(load_cameras(reg)) == {"cam-02"}
    assert "# 손으로 쓴 설명" not in text               # 키 바로 위에 같은 들여쓰기로 붙은 주석은 그 카메라의 것
    assert "# 카메라 등록 (시험용)" in text and "crop: [0, 0, 100, 100]" in text
    assert only_one_newline_style(reg.read_bytes())


def test_remove_with_an_unknown_key_removes_nothing(reg):
    original = reg.read_bytes()
    with pytest.raises(CameraMissing):
        remove_cameras(reg, ["cam-01", "cam-77"])
    assert reg.read_bytes() == original


def test_a_broken_file_is_never_touched(tmp_path):
    p = tmp_path / "cameras.yaml"
    p.write_text("cameras:\n  cam-01:\n    kind: nonsense\n", encoding="utf-8")
    original = p.read_bytes()
    for call in (lambda: add_camera(p, "cam-02", {"kind": "cctv", "place": "시험 장소"}),
                 lambda: remove_cameras(p, ["cam-01"]),
                 lambda: edit_camera(p, "cam-01", {"kind": "cctv"})):
        with pytest.raises(ValueError):
            call()
        assert p.read_bytes() == original


# ----------------------------------------------------------------- 고치기

def changed_lines(before: bytes, after: bytes) -> tuple[list[str], list[str]]:
    a, b = before.decode("utf-8").splitlines(), after.decode("utf-8").splitlines()
    return [x for x in a if x not in b], [x for x in b if x not in a]


def test_edit_changes_only_that_line_and_keeps_the_comment(reg, tmp_path):
    original = reg.read_bytes()
    cam, changed = edit_camera(reg, "cam-02", {"place": "시험 장소 D", "kind": "screen"}, backup_dir=tmp_path / "bk")
    assert (cam.place, changed) == ("시험 장소 D", ["place"])          # 뜻이 같은 칸(kind)은 건드리지 않는다
    gone, new = changed_lines(original, reg.read_bytes())
    assert gone == ["    place: 시험 장소 B   # 줄 끝 주석"]
    assert len(new) == 1 and "시험 장소 D" in new[0] and new[0].rstrip().endswith("# 줄 끝 주석")
    assert only_one_newline_style(reg.read_bytes())
    assert len(list((tmp_path / "bk").glob("cameras_*.yaml"))) == 1


def test_edit_adds_a_missing_field_and_clears_one(reg):
    original = reg.read_bytes()
    cam, changed = edit_camera(reg, "cam-02", {"mount": "wall", "split": ""})
    assert (cam.mount, cam.split, sorted(changed)) == ("wall", None, ["mount", "split"])
    gone, new = changed_lines(original, reg.read_bytes())
    assert gone == ["    split: test"]
    assert sorted(x.strip() for x in new) == ["mount: wall"]            # 비운 split 은 'split:' — cam-01 에 같은 줄이 있다
    assert load_cameras(reg)["cam-01"] == load_cameras_from(original, reg)["cam-01"]


def load_cameras_from(data: bytes, near: Path):
    p = near.with_name("original.yaml")
    p.write_bytes(data)
    try:
        return load_cameras(p)
    finally:
        p.unlink()


def test_edit_with_nothing_to_change_writes_nothing(reg, tmp_path):
    original = reg.read_bytes()
    cam, changed = edit_camera(reg, "cam-01", {"mount": "pole", "tz": "KST", "split": ""}, backup_dir=tmp_path / "bk")
    assert changed == [] and cam.key == "cam-01"
    assert reg.read_bytes() == original and not (tmp_path / "bk").exists()


def test_dry_run_checks_but_does_not_write(reg):
    original = reg.read_bytes()
    cam, changed = edit_camera(reg, "cam-01", {"mount": "wall"}, new_key="cam-09", dry_run=True)
    assert (cam.key, cam.mount, changed) == ("cam-09", "wall", ["mount", "key"])
    assert reg.read_bytes() == original


def test_rename_key_changes_only_the_key_line(reg):
    original = reg.read_bytes()
    cam, changed = edit_camera(reg, "cam-01", {}, new_key="cam-09")
    assert (cam.key, changed) == ("cam-09", ["key"])
    gone, new = changed_lines(original, reg.read_bytes())
    assert (gone, new) == (["  cam-01:"], ["  cam-09:"])
    assert list(load_cameras(reg)) == ["cam-09", "cam-02"]              # 순서도 그대로


@pytest.mark.parametrize("values, new_key, error", [
    ({}, "cam-02", CameraExists),
    ({}, "Cam 9", ValueError),
    ({"kind": "tv"}, None, ValueError),
    ({"place": ""}, None, ValueError),
    ({"split": "holdout"}, None, ValueError),
    ({"tz": "EST"}, None, ValueError),
])
def test_a_refused_edit_writes_nothing(reg, values, new_key, error):
    original = reg.read_bytes()
    with pytest.raises(error):
        edit_camera(reg, "cam-01", values, new_key=new_key)
    assert reg.read_bytes() == original


def test_edit_unknown_camera(reg):
    with pytest.raises(CameraMissing):
        edit_camera(reg, "cam-77", {"mount": "wall"})


# ----------------------------------------------------------------- 여러 카메라의 고정 분할 한 번에 (2026-11-23)

MANY = SAMPLE + """\
  # 도구가 등록한 공개 데이터셋
  ds-0001:
    kind: dataset
    place: 공개 데이터 영상 1
    split:   # 아직 안 정함
    note: 메모
  ds-0002:
    kind: dataset
    place: 공개 데이터 영상 2
    note: split 줄이 없다
  ds-0003:
    kind: dataset
    place: 공개 데이터 영상 3
    split: train
"""


@pytest.fixture(params=["\n", "\r\n"], ids=["lf", "crlf"])
def many(request, tmp_path) -> Path:
    p = tmp_path / "data" / "cameras.yaml"
    p.parent.mkdir()
    p.write_bytes(MANY.replace("\n", request.param).encode("utf-8"))
    return p


def splits_of(path: Path) -> dict:
    return {k: c.split for k, c in load_cameras(path).items()}


def test_set_splits_changes_only_those_split_lines_with_one_backup(many, tmp_path):
    original = many.read_bytes()
    changed = set_splits(many, {"ds-0001": "train", "ds-0002": "train", "ds-0003": "train"}, backup_dir=tmp_path / "bk")
    assert changed == ["ds-0001", "ds-0002"]                    # 이미 train 인 ds-0003 은 건드리지 않는다
    after = many.read_bytes()
    assert splits_of(many) == {"cam-01": None, "cam-02": "test", "ds-0001": "train", "ds-0002": "train", "ds-0003": "train"}
    gone, new = changed_lines(original, after)
    assert gone == ["    split:   # 아직 안 정함"]
    assert new == ["    split: train  # 아직 안 정함"]           # 줄 끝 주석은 남는다 (ds-0002 의 새 줄은 ds-0003 의 줄과 같은 글자)
    lines = after.decode("utf-8").splitlines()
    at = lines.index("    place: 공개 데이터 영상 2")
    assert lines[at + 1:at + 3] == ["    split: train", "    note: split 줄이 없다"]      # 없던 칸은 note 앞에
    assert len(lines) == len(original.decode("utf-8").splitlines()) + 1
    assert only_one_newline_style(after)
    backups = list((tmp_path / "bk").glob("cameras_*.yaml"))
    assert len(backups) == 1 and backups[0].read_bytes() == original       # 카메라 수와 상관없이 쓰기 전 파일 하나


def test_set_splits_writes_what_edit_camera_would(many, tmp_path):
    one_by_one = tmp_path / "one.yaml"
    one_by_one.write_bytes(many.read_bytes())
    set_splits(many, {"ds-0001": "train", "ds-0002": "val", "cam-02": None})
    for key, value in (("ds-0001", "train"), ("ds-0002", "val"), ("cam-02", "")):
        edit_camera(one_by_one, key, {"split": value})
    assert many.read_bytes() == one_by_one.read_bytes()
    assert splits_of(many)["cam-02"] is None and b"crop: [0, 0, 100, 100]" in many.read_bytes()


def test_set_splits_with_nothing_to_change_writes_nothing(many, tmp_path):
    original = many.read_bytes()
    assert set_splits(many, {"ds-0003": "train", "cam-01": None, "cam-02": "test"}, backup_dir=tmp_path / "bk") == []
    assert set_splits(many, {}, backup_dir=tmp_path / "bk") == []
    assert many.read_bytes() == original and not (tmp_path / "bk").exists()


def test_set_splits_dry_run_checks_but_does_not_write(many, tmp_path):
    original = many.read_bytes()
    assert set_splits(many, {"ds-0001": "train", "cam-01": "val"}, backup_dir=tmp_path / "bk", dry_run=True) == [
        "ds-0001", "cam-01"]
    assert many.read_bytes() == original and not (tmp_path / "bk").exists()


@pytest.mark.parametrize("splits, error, said", [
    ({"ds-0001": "train", "ds-9999": "train"}, CameraMissing, "ds-9999"),      # 하나라도 없는 키면 하나도 안 고친다
    ({"ds-0001": "train", "ds-0002": "holdout"}, ValueError, r"^split 은 비우거나"),   # 줄을 고치기 전에 값부터 본다
    ({"ds-0001": 1}, ValueError, "글자로"),
])
def test_a_refused_set_splits_writes_nothing(many, splits, error, said):
    original = many.read_bytes()
    with pytest.raises(error, match=said):
        set_splits(many, splits)
    assert many.read_bytes() == original


def test_set_splits_writes_nothing_if_anything_else_would_change(many, monkeypatch):
    """다시 읽어 확인하는 그물 — 줄을 고치는 코드가 다른 카메라의 값까지 건드리면(버그) 아무것도 쓰지 않는다."""
    from autolabel import provenance

    real = provenance._edit_camera_text

    def clumsy(old, key, edits, new_key):
        return real(old, key, edits, new_key).replace("mount: pole", "mount: wall")

    monkeypatch.setattr(provenance, "_edit_camera_text", clumsy)
    original = many.read_bytes()
    with pytest.raises(ValueError, match="기대와 달라"):
        set_splits(many, {"ds-0001": "train"})
    assert many.read_bytes() == original


def test_set_splits_stops_whole_on_an_entry_it_cannot_edit(tmp_path):
    """한 줄 꼴 항목은 글자로 못 고친다 — 그 하나 때문에 다른 카메라만 고쳐 두지 않는다."""
    p = tmp_path / "cameras.yaml"
    p.write_text("cameras:\n  cam-01:\n    kind: cctv\n    place: 가\n    split:\n"
                 "  flow-1: {kind: cctv, place: 나}\n", "utf-8")
    original = p.read_bytes()
    with pytest.raises(ValueError):
        set_splits(p, {"cam-01": "train", "flow-1": "train"})
    assert p.read_bytes() == original and not list(tmp_path.glob("*.tmp"))
    assert set_splits(p, {"cam-01": "train"}) == ["cam-01"] and splits_of(p) == {"cam-01": "train", "flow-1": None}


def test_hand_written_number_keys_can_be_pinned_edited_and_removed(tmp_path):
    """2026-11-23 Gemini 리뷰 — 손으로 따옴표 없이 쓴 숫자 키(`101:`)는 YAML 이 수로 읽는다. `load_cameras` 는 글자로 바꾸는데
    고친 뒤 다시 읽어 견주는 쪽은 그러지 않아, 고정·고치기·빼기가 '기대와 다르다' 로 거절됐다(쓰지 않는 쪽으로만 틀렸다)."""
    p = tmp_path / "cameras.yaml"
    p.write_text("cameras:\n  101:\n    kind: cctv\n    place: 가\n    split:\n"
                 "  hw-01:\n    kind: cctv\n    place: 나\n    split:\n", "utf-8")
    assert set(load_cameras(p)) == {"101", "hw-01"}
    assert set_splits(p, {"101": "train"}) == ["101"] and splits_of(p) == {"101": "train", "hw-01": None}
    cam, _ = edit_camera(p, "101", {"place": "다"})
    assert cam.place == "다" and load_cameras(p)["hw-01"].place == "나"
    assert remove_cameras(p, ["101"]) == ["101"] and set(load_cameras(p)) == {"hw-01"}
    assert p.read_text("utf-8") == "cameras:\n  hw-01:\n    kind: cctv\n    place: 나\n    split:\n"


def test_keys_that_read_the_same_as_text_stop_the_edit(tmp_path):
    """글자로 바꾸면 같아지는 키 둘(`101:` 과 `'101':`)은 무엇을 견줄지 모르니 아무것도 쓰지 않는다."""
    p = tmp_path / "cameras.yaml"
    p.write_text("cameras:\n  101:\n    kind: cctv\n    place: 가\n  '101':\n    kind: cctv\n    place: 나\n"
                 "  hw-01:\n    kind: cctv\n    place: 다\n    split:\n", "utf-8")
    original = p.read_bytes()
    with pytest.raises(ValueError):
        set_splits(p, {"hw-01": "train"})
    assert p.read_bytes() == original and not list(tmp_path.glob("*.tmp"))
