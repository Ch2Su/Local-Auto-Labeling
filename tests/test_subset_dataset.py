"""tools/build_subset_dataset.py — 조건에 맞는 이미지만 골라 실험용 학습 세트를 만든다.

고르는 기준이 틀리면 에러 없이 다른 데이터로 학습하게 된다. 원본 DB 를 건드리지 않는 것도 여기서 본다.
"""

from __future__ import annotations

import hashlib
import sys
import tarfile
from pathlib import Path

import pytest
import yaml
from PIL import Image

from autolabel.masks import GRAY
from autolabel.store import Store
from helpers import CLASSES, ROOT, box

sys.path.insert(0, str(ROOT / "tools"))
import build_subset_dataset as bsd  # noqa: E402

RED = (200, 10, 10)
CAR, PERSON, RESPONDER = CLASSES.index("car"), CLASSES.index("person"), CLASSES.index("responder")


@pytest.fixture
def source(tmp_path):
    """작은 원본 프로젝트 — DB 와 진짜 그림 파일. {이름: id} 와 DB 경로를 돌려준다."""
    root = tmp_path / "source"
    (root / "img").mkdir(parents=True)
    store = Store(root / "labeling.db")
    store.set_classes(CLASSES)
    ids: dict[str, int] = {}

    def add(name, *, domain="cctv", status="approved", split="train", boxes=(), grp="g1"):
        path = root / "img" / f"{name}.jpg"
        Image.new("RGB", (64, 48), RED).save(path, quality=95)
        store.ingest_batch([{"path": str(path), "source": "seed", "width": 64, "height": 48, "status": status,
                             "boxes": list(boxes)}])
        with store.connect() as con:
            i = int(con.execute("SELECT id FROM images WHERE path=?", (str(path),)).fetchone()[0])
        store.set_domain([i], domain)
        if split:
            store.set_splits([(i, split)])
        if grp:
            store.set_groups([(i, grp)])
        ids[name] = i
        return i

    # a: 위 절반을 가린다 — 그 안의 박스는 학습 라벨에서 빠지고 아래 박스는 남는다
    a = add("a", boxes=[box(cls=CAR, cx=0.5, cy=0.25, w=0.25, h=0.25), box(cls=PERSON, cx=0.5, cy=0.75, w=0.25, h=0.25)])
    store.set_masks(a, [(0.0, 0.0, 1.0, 0.5)])
    add("b", split="val", boxes=[box(cls=PERSON)], grp="g2")
    c = add("c", split="test", boxes=[box(cls=CAR, cx=0.5, cy=0.25, w=0.25, h=0.25)], grp="g3")
    store.set_masks(c, [(0.0, 0.0, 1.0, 0.5)])             # test 는 칠하지도, 박스를 빼지도 않는다
    add("has_responder", boxes=[box(cls=CAR), box(cls=RESPONDER, cx=0.25)], grp="g4")
    add("not_cctv", domain="noncctv", boxes=[box(cls=CAR)], grp="g5")
    add("pending", status="pending", boxes=[box(cls=CAR)], grp="g6")
    plaza = add("plaza", boxes=[box(cls=PERSON)], grp="g7")
    store.set_image_tag([plaza], store.create_tag("plaza"), True)
    return {"db": root / "labeling.db", "root": root, "ids": ids, "store": store}


def digest(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def stems(folder: Path) -> set[str]:
    """세트 안의 이름에서 경로 해시 꼬리표를 뗀 것."""
    return {p.stem.rsplit("__", 1)[0] for p in folder.iterdir()} if folder.exists() else set()


def labels(folder: Path, name: str) -> list[list[str]]:
    (path,) = [p for p in folder.iterdir() if p.stem.rsplit("__", 1)[0] == name]
    return [line.split() for line in path.read_text(encoding="utf-8").splitlines()]


def test_builds_only_what_matches_and_keeps_the_source_untouched(source, tmp_path):
    before = {p: digest(p) for p in [source["db"], *sorted((source["root"] / "img").iterdir())]}
    out = tmp_path / "exp"
    info = bsd.build(source["db"], out)
    ds = out / "dataset"

    assert (info["train"], info["val"], info["test"]) == (2, 1, 1)
    assert stems(ds / "images" / "train") == {"a", "plaza"}
    assert stems(ds / "images" / "val") == {"b"} and stems(ds / "images" / "test") == {"c"}
    spec = yaml.safe_load((ds / "data.yaml").read_text(encoding="utf-8"))
    assert list(spec["names"].values()) == CLASSES[:8]             # responder·cone 은 목록에서도 빠진다
    assert Path(spec["path"]) == ds and spec["test"] == "images/test"

    # 가릴 영역: train 은 칠하고 반 넘게 가려진 박스를 뺀다, test 는 칠하지 않고(실제 화면) 박스만 뺀다 — 그 자리는 ignore/
    assert [row[0] for row in labels(ds / "labels" / "train", "a")] == [str(PERSON)]
    assert (info["masked"], info["mask_dropped"], info["masked_test"]) == (1, 2, 1)
    (painted,) = [p for p in (ds / "images" / "train").iterdir() if p.stem.startswith("a__")]
    with Image.open(painted) as im:
        assert all(abs(v - GRAY) <= 6 for v in im.getpixel((32, 6))) and im.getpixel((32, 40))[0] > 150
    (kept,) = (ds / "images" / "test").iterdir()
    assert kept.read_bytes() == (source["root"] / "img" / "c.jpg").read_bytes()
    assert labels(ds / "labels" / "test", "c") == []                     # 2026-11-24 — 전에는 박스 그대로였다
    assert (ds / "ignore" / "test" / f"{kept.stem}.png").is_file()

    # 사본에는 고른 이미지만, 원본은 바이트까지 그대로
    copy = Store(out / "labeling.db")
    assert copy.counts()["total"] == 4 and copy.get_classes() == CLASSES[:8]
    assert {p: digest(p) for p in before} == before
    assert (out / "subset_info.json").exists() and not (out / "dataset.tar").exists()


def test_filters(source):
    con = bsd.open_readonly(source["db"])
    ids = source["ids"]
    pick = lambda **kw: set(bsd.select_ids(con, **kw))
    try:
        assert pick() == {ids[n] for n in ("a", "b", "c", "has_responder", "plaza")}             # 승인 + cctv
        assert pick(drop_cls=[8, 9]) == {ids[n] for n in ("a", "b", "c", "plaza")}
        assert pick(domain="any") == pick() | {ids["not_cctv"]}
        assert pick(domain="noncctv") == {ids["not_cctv"]}
        assert pick(tags=["plaza"]) == {ids["plaza"]}
        assert pick(not_tags=["plaza", "no-such-tag"]) == pick() - {ids["plaza"]}
        assert ids["pending"] not in pick(domain="any")
    finally:
        con.close()


def test_tag_filter_reaches_the_dataset(source, tmp_path):
    info = bsd.build(source["db"], tmp_path / "exp", not_tags=["plaza"])
    assert info["train"] == 1 and stems(tmp_path / "exp" / "dataset" / "images" / "train") == {"a"}


def test_dropped_classes_must_be_the_last_ones():
    assert bsd.split_classes(CLASSES, ["cone", "responder"]) == (CLASSES[:8], [8, 9])
    assert bsd.split_classes(CLASSES, []) == (CLASSES, [])
    with pytest.raises(ValueError):
        bsd.split_classes(CLASSES, ["responder"])           # cone 이 뒤에 남는다 — 번호가 밀린다
    with pytest.raises(ValueError):
        bsd.split_classes(CLASSES, ["forklift"])


def test_unassigned_images_stop_the_build(source, tmp_path):
    """여기서 새로 나누면 원본에서 잰 점수와 견줄 수 없다."""
    with source["store"].connect() as con:
        con.execute("UPDATE images SET split=NULL WHERE id=?", (source["ids"]["b"],))
    with pytest.raises(RuntimeError, match="분할"):
        bsd.build(source["db"], tmp_path / "exp")


def test_test_only_set_for_unseen_data(source, tmp_path):
    with source["store"].connect() as con:
        con.execute("UPDATE images SET split=NULL, grp=NULL")           # 아직 나눈 적 없는 새 데이터
    out = tmp_path / "exp"
    info = bsd.build(source["db"], out, force_split="test", tar=True)
    ds = out / "dataset"
    assert (info["train"], info["val"], info["test"]) == (0, 0, 4)
    assert stems(ds / "images" / "test") == {"a", "b", "c", "plaza"}
    # 가릴 영역은 채점에서 뺄 자리 — 그림은 원본 그대로, 반 넘게 가려진 박스는 빼고 그 자리를 ignore/ 그림으로(2026-11-24.
    # 전에는 박스까지 그대로라 그 자리의 물체를 찾은 예측이 전부 헛 박스로 세어졌다)
    assert [row[0] for row in labels(ds / "labels" / "test", "a")] == [str(PERSON)]
    (raw,) = [p for p in (ds / "images" / "test").iterdir() if p.stem.startswith("a__")]
    assert raw.read_bytes() == (source["root"] / "img" / "a.jpg").read_bytes()
    assert {p.stem for p in (ds / "ignore" / "test").iterdir()} == {raw.stem} | {
        p.stem for p in (ds / "images" / "test").iterdir() if p.stem.startswith("c__")}
    assert (info["masked"], info["mask_dropped"]) == (2, 2)
    spec = yaml.safe_load((ds / "data.yaml").read_text(encoding="utf-8"))
    assert spec["train"] == spec["val"] == spec["test"] == "images/test"
    assert (out / "dataset.tar").stat().st_size == info["tar_bytes"] > 0
    with tarfile.open(out / "dataset.tar") as tf:                     # pod 로 보내는 묶음에도 그 자리 그림이 들어간다
        assert any(n.startswith("dataset/ignore/test/a__") for n in tf.getnames())


def test_test_only_set_can_paint_masks(source, tmp_path):
    """가릴 영역이 '라벨하지 않기로 한 물체' 를 덮은 데이터는, 칠하지 않으면 그 물체를 찾은 예측이 헛 박스로 세어진다."""
    out = tmp_path / "exp"
    info = bsd.build(source["db"], out, force_split="test", paint_masks=True)
    ds = out / "dataset"
    assert (info["test"], info["masked"], info["mask_dropped"]) == (4, 2, 2)
    assert [row[0] for row in labels(ds / "labels" / "test", "a")] == [str(PERSON)]
    assert labels(ds / "labels" / "test", "c") == []
    (painted,) = [p for p in (ds / "images" / "test").iterdir() if p.stem.startswith("a__")]
    with Image.open(painted) as im:
        assert all(abs(v - GRAY) <= 6 for v in im.getpixel((32, 6))) and im.getpixel((32, 40))[0] > 150
    assert digest(source["root"] / "img" / "a.jpg") != digest(painted)             # 원본이 아니라 칠한 사본이다
    with pytest.raises(ValueError):
        bsd.build(source["db"], tmp_path / "exp2", paint_masks=True)               # 평소 세트는 train·val 만 칠한다


def test_never_writes_into_an_existing_folder_or_next_to_the_source(source, tmp_path):
    out = tmp_path / "exp"
    out.mkdir()
    (out / "keep.txt").write_text("x", encoding="utf-8")
    with pytest.raises(RuntimeError, match="이미 파일"):
        bsd.build(source["db"], out)
    assert [p.name for p in out.iterdir()] == ["keep.txt"]
    with pytest.raises(RuntimeError, match="원본 DB"):
        bsd.build(source["db"], source["root"] / "exp")


def test_path_map_points_the_copy_at_moved_files(source, tmp_path):
    """옛 DB 는 옮기기 전의 절대경로를 들고 있다."""
    moved = tmp_path / "moved"
    (source["root"] / "img").rename(moved)
    with pytest.raises(RuntimeError, match="파일이 없는"):
        bsd.build(source["db"], tmp_path / "exp1")
    info = bsd.build(source["db"], tmp_path / "exp2", path_map=[(str(source["root"] / "img"), str(moved))])
    assert info["train"] == 2
    con = bsd.open_readonly(source["db"])
    try:
        assert all(str(source["root"] / "img") in r[0] for r in con.execute("SELECT path FROM images"))    # 원본 DB 는 그대로
    finally:
        con.close()
