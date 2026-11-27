"""tools/crop_images.py — 화면 기준 크기와 정리된 이미지의 라벨 파일 (2026-11-27, Gemini 리뷰 묶음 8).

DB 의 width·height 는 EXIF 회전 전 원시 크기이고, 박스·가릴 영역의 정규화 좌표와 OpenCV 가 읽는 배열은 회전을 적용한
화면 기준이다. 고치기 전에는 원시 크기로 재서, 90도 돌려 보는 사진(가로·세로가 뒤바뀐다)의 보기는 틀린 박스를 세고
`--apply` 는 '읽기·크기' 로 멈췄다.

`main()` 을 `sys.argv` 로 돌린다. 그림은 작은 진짜 JPEG 이고 설정·DB 는 임시 폴더다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml
from PIL import Image

from helpers import ROOT, box, keys

sys.path.insert(0, str(ROOT / "tools"))
import crop_images as ci  # noqa: E402

ORIENTATION = 0x0112


@pytest.fixture
def run(cfg, monkeypatch, capsys):
    cfg.path.write_text(yaml.safe_dump(cfg.raw, allow_unicode=True), encoding="utf-8")

    def go(*argv: str) -> tuple[int | str, str]:
        monkeypatch.setattr(sys, "argv", ["crop_images.py", "--config", str(cfg.path), "--camera", "cam-1", *argv])
        try:
            code = ci.main()
        except SystemExit as exc:
            code = exc.code
        return code, capsys.readouterr().out

    return go


def put(store, path: Path, raw: tuple[int, int], boxes, *, orientation: int | None = None, status="approved") -> int:
    """raw = 파일에 든 (가로, 세로). orientation 6 이면 화면에서는 90도 돌아 (세로, 가로)로 보인다. DB 에는 원시 크기를 적는다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    exif = Image.Exif()
    if orientation:
        exif[ORIENTATION] = orientation
    Image.new("RGB", raw, (90, 120, 150)).save(path, "JPEG", quality=90, exif=exif)
    store.ingest_batch([{"path": str(path), "source": "collected", "width": raw[0], "height": raw[1], "status": status,
                         "boxes": list(boxes), "camera": "cam-1"}])
    with store.connect() as con:
        return int(con.execute("SELECT id FROM images WHERE path=?", (str(path),)).fetchone()[0])


def test_a_rotated_photo_is_cropped_in_screen_pixels(cfg, store, run):
    """원시 100×60, Orientation 6 → 화면 60×100. 박스는 화면 x 15~45·y 15~35 이고, 화면의 위 절반(60×50)만 남긴다."""
    i = put(store, cfg.unlabeled / "cam-1" / "r.jpg", (100, 60), [box(2, 0.5, 0.25, 0.5, 0.2), box(0, 0.5, 0.9, 0.2, 0.1)],
            orientation=6)
    code, out = run("--ids", str(i), "--box", "0,0,60,50")
    assert code == 0 and "60×100 → 60×50" in out and "박스 2 → 1" in out            # 아래쪽 사람(y 85~95)만 잃는다
    assert Path(store.get_image(i)["path"]).name == "r.jpg"                          # 보기만 — 그대로다

    code, out = run("--ids", str(i), "--box", "0,0,60,50", "--apply")
    assert code == 0 and "잘라 바꿈 1장" in out and "경고" not in out
    im = store.get_image(i)
    assert (Path(im["path"]).name, im["width"], im["height"], im["status"]) == ("r_c0-0-60-50.jpg", 60, 50, "approved")
    with Image.open(im["path"]) as new:
        assert new.size == (60, 50) and new.getexif().get(ORIENTATION, 1) == 1       # 돌려 놓은 그림 — 회전 표시는 없다
    assert keys(store.get_boxes(i)) == keys([box(2, 0.5, 0.5, 0.5, 0.4)])


def test_a_box_outside_the_raw_size_is_refused_by_the_screen_size(cfg, store, run):
    """영역이 화면 밖인지도 화면 크기로 본다 — 원시 가로(100) 안이어도 화면 가로(60)를 넘으면 멈춘다."""
    i = put(store, cfg.unlabeled / "cam-1" / "r.jpg", (100, 60), [box()], orientation=6)
    code, _ = run("--ids", str(i), "--box", "0,0,80,50")
    assert "밖으로 나간다" in str(code) and "60×100" in str(code)


def test_an_upright_photo_is_cropped_as_before(cfg, store, run):
    i = put(store, cfg.unlabeled / "cam-1" / "u.jpg", (100, 60), [box(2, 0.25, 0.5, 0.2, 0.5)])
    code, out = run("--ids", str(i), "--box", "0,0,50,60", "--apply")
    assert code == 0 and "100×60 → 50×60" in out
    assert keys(store.get_boxes(i)) == keys([box(2, 0.5, 0.5, 0.4, 0.5)])


def test_a_tidied_image_keeps_its_label_file_in_step(cfg, store, run):
    """정리된 이미지(labeled/images/)를 자르면 옆의 라벨 파일도 새 이름·새 박스로 — 옛 것은 옛 그림과 함께 보관 폴더로."""
    img = cfg.labeled / "images" / "cam-1" / "t.jpg"
    i = put(store, img, (100, 60), [box(2, 0.25, 0.5, 0.2, 0.5)])
    labels = cfg.labeled / "labels" / "cam-1"
    labels.mkdir(parents=True)
    (labels / "t.txt").write_text("2 0.250000 0.500000 0.200000 0.500000\n", encoding="utf-8")

    code, out = run("--ids", str(i), "--box", "0,0,50,60", "--apply")
    assert code == 0, out
    assert sorted(p.name for p in labels.iterdir()) == ["t_c0-0-50-60.txt"]
    assert (labels / "t_c0-0-50-60.txt").read_text(encoding="utf-8") == "2 0.500000 0.500000 0.400000 0.500000\n"
    kept = next((cfg.db.parent / "backups").glob("crop_*"))
    assert sorted(p.name for p in kept.iterdir()) == ["manifest.tsv", "t.jpg", "t.txt"]
