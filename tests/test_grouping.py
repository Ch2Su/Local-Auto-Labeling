"""같은 장면 묶기 (grouping.py) — 여기가 틀리면 같은 장면이 train 과 test 로 갈려 점수가 부풀려진다."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from autolabel.grouping import BURST_SEC, DHASH_MAX, base_name, dhash, group_images, name_series


# ----------------------------------------------------------------- 이름 꼬리표

@pytest.mark.parametrize("name, plain", [
    # tidy 가 이름이 겹칠 때 붙이는 `__<sha1 8자>`(+ `_2`) — 2026-10-25: 해시 끝 숫자 '59' 때문에 연번으로 잡혔다
    ("maxresdefault__7bfb8759.jpg", "maxresdefault.jpg"),
    ("cctv01_000100__ab12cd34.jpg", "cctv01_000100.jpg"),
    ("cctv01_000100__ab12cd34_2.jpg", "cctv01_000100.jpg"),
    ("objects365_v1_00420917__ab12cd34.jpg", "objects365_v1_00420917.jpg"),
    # 로보플로 꼬리표 — 두 번 내보낸 `_png_png.rf.<해시>` 까지 (2026-11-05: 하나만 떼어 452장이 규칙을 놓쳤다)
    ("ala_2_016_jpg.rf.0a1b2c3d4e5f.jpg", "ala_2_016.jpg"),
    ("duque_de_caxias-1000_png_png.rf.0a1b2c3d4e5f.jpg", "duque_de_caxias-1000.png"),
    ("frame_0007.rf.0a1b2c3d4e5f.jpg", "frame_0007.jpg"),
])
def test_suffixes_do_not_change_the_series(name, plain):
    assert base_name(name) == base_name(plain) == Path(plain).stem
    assert name_series(name) == name_series(plain)


def test_tidy_hash_is_not_read_as_a_serial_number():
    assert name_series("maxresdefault__7bfb8759.jpg") is None
    assert name_series("cctv01_000100__ab12cd34.jpg") == ("serial cctv01", 100)


def test_public_dataset_photos_are_not_a_series():
    # 번호만 이어졌을 뿐 서로 다른 사진이다 (2026-10-03)
    assert name_series("objects365_v1_00420917.jpg") is None


def test_numeric_names_need_same_digits_and_size():
    a = name_series("000123.jpg", (640, 640))
    assert a[1] == 123
    assert a[0] == name_series("000999.jpg", (640, 640))[0]
    assert a[0] != name_series("000000000123.jpg", (640, 640))[0]      # 자릿수가 다르다
    assert a[0] != name_series("000123.jpg", (2992, 2992))[0]          # 크기가 다르다


@pytest.mark.parametrize("name", [
    "스크린샷 2026-06-08 165037.png",
    "Screenshot 2026-06-08 165037.png",
    "screenshot-2026-06-08-165037.png",
])
def test_screenshots_korean_and_english(name):
    assert name_series(name) == ("shot 2026-06-08", 16 * 3600 + 50 * 60 + 37)


def test_single_trailing_digit_is_not_a_serial():
    assert name_series("photo_7.jpg") is None


# ----------------------------------------------------------------- 묶기

def _item(i: int, name: str, *, camera: str | None = None, dh: str | None = None) -> dict:
    return {"id": i, "path": f"/x/{name}", "dhash": dh, "width": 1280, "height": 720, "camera": camera}


def test_same_camera_is_one_group_whatever_the_names():
    g = group_images([_item(5, "a.jpg", camera="cam-01"), _item(9, "zzz.jpg", camera="cam-01"),
                      _item(7, "b.jpg", camera="cam-02")])
    assert g[5] == g[9] == "g000005"          # 묶음 키는 가장 작은 id
    assert g[7] == "g000007"


def test_serial_names_join_only_inside_one_camera():
    # 카메라가 달라도 같은 꼴의 이름(frame_0001)을 만든다 — 이름으로는 카메라를 넘어 잇지 않는다 (2026-11-05)
    g = group_images([_item(1, "frame_0001.jpg", camera="cam-01"), _item(2, "frame_0002.jpg", camera="cam-02")])
    assert g[1] != g[2]
    g = group_images([_item(1, "frame_0001.jpg"), _item(2, "frame_0900.jpg")])       # 출처 없는 legacy 는 계열째
    assert g[1] == g[2]


def test_screenshots_join_only_within_burst():
    def shot(i, hhmmss):
        return _item(i, f"스크린샷 2026-06-08 {hhmmss}.png")

    g = group_images([shot(1, "165000"), shot(2, f"1650{BURST_SEC:02d}"), shot(3, "165200")])
    assert g[1] == g[2]
    assert g[3] != g[1]


def test_dhash_joins_up_to_max_distance():
    a, near, far = "0000000000000000", "0000000000000003", "00000000000000f0"
    assert bin(int(near, 16)).count("1") == DHASH_MAX
    g = group_images([_item(1, "a.jpg", dh=a), _item(2, "b.jpg", dh=near), _item(3, "c.jpg", dh=far)])
    assert g[1] == g[2]
    assert g[3] != g[1]


def test_dhash_joins_across_cameras():
    # 카메라가 다른 이미지를 잇는 길은 dHash 뿐이다
    g = group_images([_item(1, "a.jpg", camera="cam-01", dh="00000000000000ff"),
                      _item(2, "b.jpg", camera="cam-02", dh="00000000000000ff")])
    assert g[1] == g[2]


# ----------------------------------------------------------------- dHash 와 EXIF 회전

def _hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def test_dhash_follows_exif_rotation(tmp_path):
    """회전 태그로 저장한 사진과 돌려서 저장한 같은 사진은 같은 해시여야 한다 (2026-10-25)."""
    w, h = 320, 240
    img = Image.new("L", (w, h))
    # 위 절반은 왼쪽→오른쪽으로 밝아지고 아래 절반은 어두워진다 — 돌리면 해시가 크게 달라지는 무늬
    img.putdata([(x * 255 // (w - 1)) if y < h // 2 else 255 - (x * 255 // (w - 1))
                 for y in range(h) for x in range(w)])
    img = img.convert("RGB")
    upright = tmp_path / "upright.jpg"
    img.save(upright, quality=95)

    turned = img.transpose(Image.ROTATE_90)             # 반시계 90도로 돌린 픽셀
    exif = Image.Exif()
    exif[0x0112] = 6                                    # Orientation 6 = '시계 90도 돌려서 볼 것' → 화면은 img 와 같다
    tagged = tmp_path / "tagged.jpg"
    turned.save(tagged, quality=95, exif=exif)
    raw = tmp_path / "raw.jpg"                          # 같은 픽셀인데 태그가 없다 — 화면이 정말 다르다
    turned.save(raw, quality=95)

    a, b, c = dhash(upright), dhash(tagged), dhash(raw)
    assert _hamming(a, b) <= DHASH_MAX
    assert _hamming(a, c) > 8


def test_dhash_of_unreadable_file_is_none(tmp_path):
    p = tmp_path / "broken.jpg"
    p.write_bytes(b"not an image")
    assert dhash(p) is None
