"""tools/build_vehicle_crops.py — VehicleRefiner 크롭 세트 (2026-11-28, docs/vehicle-refiner.md '2. 데이터').

본다: 가릴 영역이 반 넘게 덮은 박스는 빼고 센다 · 원본 DB 는 그대로 · EXIF 로 돌려 보는 사진은 화면 기준 픽셀로 자른다 ·
그림 크기가 DB 와 다르면 건너뛴다 · 기준 세트의 test 에 든 이미지의 그룹(같은 카메라)은 통째로 `fold=test` · 같은 자리 짝은 한
그룹 · AI Hub(`ah-*`)는 `local_only=1` · fold 배정은 그룹을 쪼개지 않고 드문 클래스를 고르게 · 이 도구가 쓰지 않은 폴더에는
쓰지 않고, 쓴 폴더는 지난 결과를 지우고 다시 쓴다 · 보기만이면 아무것도 안 쓴다.

`main()` 을 `sys.argv` 로 돌린다. 그림은 작은 진짜 JPEG 이고 설정·DB 는 임시 폴더다(모델·GPU 없음).
"""

from __future__ import annotations

import collections
import csv
import json
import sys
from pathlib import Path

import pytest
import yaml
from PIL import Image

from helpers import ROOT, box

sys.path.insert(0, str(ROOT / "tools"))
import build_vehicle_crops as bvc  # noqa: E402

ORIENTATION = 0x0112
CAR, BUS, TRUCK, PICKUP, RESPONDER, PERSON = 2, 4, 5, 7, 8, 0      # helpers.CLASSES 의 번호
RED = (220, 30, 30)


@pytest.fixture
def run(cfg, monkeypatch, capsys):
    cfg.path.write_text(yaml.safe_dump(cfg.raw, allow_unicode=True), encoding="utf-8")

    def go(*argv: str) -> tuple[int | str, str]:
        monkeypatch.setattr(sys, "argv", ["build_vehicle_crops.py", "--config", str(cfg.path), *argv])
        try:
            code = bvc.main()
        except SystemExit as exc:
            code = exc.code
        return code, capsys.readouterr().out

    return go


def put(store, path: Path, size: tuple[int, int], boxes, *, camera: str, source: str = "collected",
        db_size: tuple[int, int] | None = None, red: tuple[int, int, int, int] | None = None) -> int:
    """size 크기의 회색 JPEG(red 가 있으면 그 픽셀 사각형을 빨갛게)을 쓰고 승인 행을 넣는다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", size, (120, 120, 120))
    if red:
        im.paste(RED, red)
    im.save(path, "JPEG", quality=95)
    w, h = db_size or size
    store.ingest_batch([{"path": str(path), "source": source, "width": w, "height": h, "status": "approved",
                         "boxes": list(boxes), "camera": camera}])
    with store.connect() as con:
        return int(con.execute("SELECT id FROM images WHERE path=?", (str(path),)).fetchone()[0])


def manifest(out: Path) -> list[dict]:
    with open(out / "manifest.csv", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_masked_box_is_left_out_and_counted_and_db_is_untouched(cfg, store, run, tmp_path):
    """오른쪽 반을 덮는 영역 — 그 안의 car 는 빠지고(가려짐 1), 왼쪽 pickup 과 사람(고른 클래스 아님)은 그대로다."""
    i = put(store, tmp_path / "img" / "cam-a" / "a.jpg", (200, 100),
            [box(PICKUP, 0.25, 0.5, 0.2, 0.4), box(CAR, 0.75, 0.5, 0.2, 0.4), box(PERSON, 0.1, 0.5, 0.05, 0.2)],
            camera="cam-a")
    store.set_masks(i, [(0.5, 0.0, 1.0, 1.0)])
    before = cfg.db.read_bytes()
    out = tmp_path / "crops"
    code, text = run("--out", str(out), "--folds", "2", "--apply")
    assert code == 0, text
    rows = manifest(out)
    assert [(r["cls"], r["box_idx"]) for r in rows] == [("pickup", "0")]
    info = json.loads((out / bvc.MARKER).read_text(encoding="utf-8"))
    assert info["classes"]["car"]["masked_out"] == 1 and info["classes"]["car"]["crops"] == 0
    assert cfg.db.read_bytes() == before                     # 원본 DB 는 읽기만
    assert (out / rows[0]["crop"]).exists()


def test_rotated_photo_is_cropped_in_screen_pixels(cfg, store, run, tmp_path):
    """화면 60×100 그림(빨간 자리 x 10–30·y 40–80)을 Orientation 6 으로 저장 — 파일은 원시 100×60 이다. 박스는 화면 기준."""
    screen = Image.new("RGB", (60, 100), (120, 120, 120))
    screen.paste(RED, (10, 40, 30, 80))
    raw = screen.transpose(Image.Transpose.ROTATE_90)          # Orientation 6 을 적용하면 다시 screen 이 된다
    path = tmp_path / "img" / "cam-r" / "r.jpg"
    path.parent.mkdir(parents=True)
    exif = Image.Exif()
    exif[ORIENTATION] = 6
    raw.save(path, "JPEG", quality=95, exif=exif)
    store.ingest_batch([{"path": str(path), "source": "collected", "width": 100, "height": 60, "status": "approved",
                         "boxes": [box(CAR, 20 / 60, 60 / 100, 20 / 60, 40 / 100)], "camera": "cam-r"}])
    out = tmp_path / "crops"
    code, text = run("--out", str(out), "--folds", "2", "--pad", "0", "--apply")
    assert code == 0, text
    (r,) = manifest(out)
    assert (int(r["img_w"]), int(r["img_h"])) == (60, 100)
    assert (int(r["crop_x1"]), int(r["crop_y1"]), int(r["crop_x2"]), int(r["crop_y2"])) == (10, 40, 30, 80)
    with Image.open(out / r["crop"]) as c:
        assert c.size == (20, 40)
        red, green, blue = c.convert("RGB").resize((1, 1), Image.Resampling.BOX).getpixel((0, 0))
    assert red > 180 and green < 80 and blue < 80               # 원시 좌표로 잘랐다면 회색이 섞인다


def test_pad_widens_inside_the_image(cfg, store, run, tmp_path):
    """박스 크기의 15% 를 사방으로 넓히되 그림 밖으로는 안 나간다(왼쪽 가장자리 박스는 0 에서 멈추고 edge=1)."""
    put(store, tmp_path / "img" / "cam-p" / "p.jpg", (200, 100),
        [box(TRUCK, 0.5, 0.5, 0.2, 0.4), box(BUS, 0.05, 0.5, 0.1, 0.4)], camera="cam-p")
    out = tmp_path / "crops"
    code, text = run("--out", str(out), "--folds", "2", "--apply")
    assert code == 0, text
    rows = {r["cls"]: r for r in manifest(out)}
    t = rows["truck"]
    assert (int(t["crop_x1"]), int(t["crop_y1"]), int(t["crop_x2"]), int(t["crop_y2"])) == (74, 24, 126, 76)
    assert t["edge"] == "0" and float(t["long_px"]) == 40.0
    b = rows["bus"]
    assert int(b["crop_x1"]) == 0 and b["edge"] == "1"


def test_size_mismatch_is_skipped(cfg, store, run, tmp_path):
    put(store, tmp_path / "img" / "cam-s" / "s.jpg", (100, 60), [box(CAR)], camera="cam-s", db_size=(200, 100))
    out = tmp_path / "crops"
    code, text = run("--out", str(out), "--folds", "2", "--apply")
    assert code == 0 and "[건너뜀]" in text and "labeler.py check" in text
    assert manifest(out) == []


def test_ref_test_groups_go_to_test_and_pairs_share_a_group(cfg, store, run, tmp_path):
    """cam-t 의 한 장이 기준 세트의 test 에 있다 → cam-t 의 다른 장(세트에 없는 new)까지 fold=test.
    같은 자리 짝 ah-0001·ah-0002 는 한 그룹·한 fold 이고 local_only=1."""
    t1 = put(store, tmp_path / "img" / "cam-t" / "t1.jpg", (64, 64), [box(CAR)], camera="cam-t")
    t2 = put(store, tmp_path / "img" / "cam-t" / "t2.jpg", (64, 64), [box(PICKUP)], camera="cam-t")
    a1 = put(store, tmp_path / "img" / "ah-0001" / "x.jpg", (64, 64), [box(CAR)], camera="ah-0001", source="aihub_pm")
    a2 = put(store, tmp_path / "img" / "ah-0002" / "y.jpg", (64, 64), [box(TRUCK)], camera="ah-0002", source="aihub_pm")
    others = [put(store, tmp_path / "img" / f"cam-{k}" / "o.jpg", (64, 64), [box(CAR)], camera=f"cam-{k}")
              for k in range(4)]
    ref = tmp_path / "ref"
    for split, name in (("test", "t1"), ("train", "x"), ("val", "o")):
        (ref / "images" / split).mkdir(parents=True, exist_ok=True)
        (ref / "images" / split / f"{name}__abcd1234.jpg").write_bytes(b"")
    pairs = tmp_path / "pairs.json"
    pairs.write_text(json.dumps([{"why": "same_camera", "a": ["ah-0001", a1, "approved"],
                                  "b": ["ah-0002", a2, "approved"]}]), encoding="utf-8")
    out = tmp_path / "crops"
    code, text = run("--out", str(out), "--folds", "2", "--ref-dataset", str(ref), "--same-place", str(pairs), "--apply")
    assert code == 0, text
    rows = {int(r["image_id"]): r for r in manifest(out)}
    assert rows[t1]["fold"] == rows[t2]["fold"] == "test"
    assert rows[t1]["split_ref"] == "test" and rows[t2]["split_ref"] == "new"
    assert rows[a1]["split_ref"] == "train" and rows[a2]["split_ref"] == "new"
    assert rows[a1]["group"] == rows[a2]["group"] and rows[a1]["fold"] == rows[a2]["fold"] != "test"
    assert rows[a1]["local_only"] == rows[a2]["local_only"] == "1"
    assert all(rows[o]["local_only"] == "0" and rows[o]["fold"] in ("0", "1") for o in others)
    assert {rows[o]["fold"] for o in others} == {"0", "1"}


def test_assign_folds_keeps_groups_whole_and_spreads_rare_classes():
    """responder 를 하나씩 가진 그룹 다섯 → fold 다섯에 하나씩. 큰 car 그룹들은 크롭 수가 고르게. 같은 seed 면 같은 결과."""
    counts = {g: collections.Counter({"responder": 1, "car": 2}) for g in range(5)}
    counts.update({10 + g: collections.Counter({"car": 10 + g}) for g in range(10)})
    a = bvc.assign_folds(counts, 5, seed=3)
    assert a == bvc.assign_folds(counts, 5, seed=3)
    assert set(a) == set(counts)                                       # 그룹마다 fold 하나
    assert sorted(a[g] for g in range(5)) == [0, 1, 2, 3, 4]
    sizes = collections.Counter()
    for g, f in a.items():
        sizes[f] += sum(counts[g].values())
    assert max(sizes.values()) - min(sizes.values()) <= 12             # 합 155 → 31 안팎
    with pytest.raises(ValueError):
        bvc.assign_folds(counts, 1)


def test_out_folder_rules(cfg, store, run, tmp_path):
    """이 도구가 쓰지 않은 파일이 있으면 멈춘다. 표식이 있는 폴더는 지난 크롭을 지우고 다시 쓴다. 보기만이면 안 만든다."""
    i = put(store, tmp_path / "img" / "cam-o" / "o.jpg", (64, 64), [box(CAR)], camera="cam-o")
    stray = tmp_path / "stray"
    stray.mkdir()
    (stray / "keep.txt").write_text("사람이 둔 파일", encoding="utf-8")
    code, text = run("--out", str(stray), "--folds", "2", "--apply")
    assert code != 0 and (stray / "keep.txt").exists()

    out = tmp_path / "crops"
    code, _ = run("--out", str(out), "--folds", "2")
    assert code == 0 and not out.exists()                               # 보기만
    code, _ = run("--out", str(out), "--folds", "2", "--apply")
    assert code == 0
    old = out / "crops" / "car" / "999999_0.jpg"
    old.write_bytes(b"old")
    store.set_masks(i, [(0.0, 0.0, 1.0, 1.0)])                          # 이제 그 car 는 가려져 빠진다
    code, text = run("--out", str(out), "--folds", "2", "--apply")
    assert code == 0, text
    assert manifest(out) == [] and not old.exists()
