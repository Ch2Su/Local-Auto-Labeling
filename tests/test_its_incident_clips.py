"""tools/its_incident_clips.py — 돌발 지점 근처 카메라를 고르는 규칙(네트워크·키는 쓰지 않는다)."""
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import its_incident_clips as T  # noqa: E402

DEG_PER_M = 1 / 110_540            # 위도 1m


def cam(i, y, road="ex", x=127.0):
    return {"id": str(i), "road": road, "name": f"cam{i}", "x": x, "y": y, "url": f"https://h/{i}/tok"}


def ev(y, typ="공사", road="고속도로", start="20261124100000", end="", x=127.0):
    return {"road": road, "type": typ, "detail": "", "start": start, "end": end, "lanes": "", "message": "m",
            "road_name": "", "x": x, "y": y}


def test_cam_id_is_first_path_segment():
    assert T.cam_id("https://cctvsec.example/2514/AbC+/x==") == "2514"
    assert T.cam_id("") == "" and T.cam_id(None) == ""


def test_meters_matches_latitude_scale():
    assert T.meters(127, 36, 127, 36 + 500 * DEG_PER_M) == pytest.approx(500, abs=0.5)


def test_nearest_camera_inside_radius_only():
    cams = [cam(1, 36 + 300 * DEG_PER_M), cam(2, 36 + 100 * DEG_PER_M), cam(3, 36 + 900 * DEG_PER_M)]
    got = T.choose([ev(36.0)], cams, radius=500)
    assert [(p["cam"]["id"], p["dist"]) for p in got] == [("2", 100)]
    assert [p["cam"]["id"] for p in T.choose([ev(36.0)], cams, radius=500, per_event=5)] == ["2", "1"]
    assert T.choose([ev(36.0)], [cam(3, 36 + 900 * DEG_PER_M)], radius=500) == []


def test_radius_wider_than_one_grid_cell():
    far = cam(9, 36 + 4000 * DEG_PER_M)          # 칸(약 2km) 둘 건너
    assert [p["cam"]["id"] for p in T.choose([ev(36.0)], [far], radius=5000)] == ["9"]


def test_camera_must_be_on_the_same_kind_of_road():
    cams = [cam(1, 36.0, road="its"), cam(2, 36 + 200 * DEG_PER_M, road="ex")]
    assert [p["cam"]["id"] for p in T.choose([ev(36.0, road="고속도로")], cams)] == ["2"]
    assert [p["cam"]["id"] for p in T.choose([ev(36.0, road="국도")], cams)] == ["1"]
    assert T.choose([ev(36.0, road="시군도")], cams) == []


def test_rare_types_first_then_recent_and_camera_once():
    cams = [cam(1, 36.0), cam(2, 37.0)]
    events = [ev(36.0, "공사", start="20261124120000"), ev(36.0, "교통사고", start="20261124090000"),
              ev(37.0, "공사", start="20261124080000"), ev(37.0, "공사", start="20261124110000")]
    got = T.choose(events, cams)
    assert [(p["event"]["type"], p["cam"]["id"]) for p in got] == [("교통사고", "1"), ("공사", "2")]
    assert got[1]["event"]["start"] == "20261124110000"          # 같은 유형은 최근 것이 카메라를 가져간다


def test_skips_unwanted_types_ended_events_and_respects_limit():
    cams = [cam(i, 36 + i) for i in range(5)]
    events = [ev(36.0, "재난"), ev(37.0, "공사", end="20261124090000"), ev(38.0, "공사", end="20261124230000"),
              ev(39.0, "공사"), ev(40.0, "공사")]
    got = T.choose(events, cams, now="20261124120000")
    assert sorted(p["cam"]["id"] for p in got) == ["2", "3", "4"]
    assert len(T.choose(events, cams, now="20261124120000", limit=2)) == 2


def test_skips_events_with_no_scene_at_the_point():
    cams = [cam(i, 36 + i) for i in range(4)]
    events = [dict(ev(36.0, "기타돌발"), message="", detail="차량증가/정체"),
              dict(ev(37.0, "공사"), message="(차로교대)제설 예비살포 이동 작업중"),
              dict(ev(38.0, "공사"), message="(갓길) 잡목제거 작업중"), dict(ev(39.0, "기타돌발"), message="(갓길)고장차 처리중")]
    assert sorted(p["cam"]["id"] for p in T.choose(events, cams)) == ["2", "3"]
    assert len(T.choose(events, cams, skip="")) == 4


def test_parse_drops_rows_without_coords_or_url():
    cams = T.parse_cams({"response": {"data": [
        {"cctvurl": "https://h/7/t", "coordx": "127.1", "coordy": "36.2", "cctvname": "n"},
        {"cctvurl": "", "coordx": "127.1", "coordy": "36.2"}, {"cctvurl": "https://h/8/t", "coordx": "", "coordy": None}]}}, "ex")
    assert [(c["id"], c["road"], c["x"]) for c in cams] == [("7", "ex", 127.1)]
    evs = T.parse_events({"body": {"items": [
        {"type": "고속도로", "eventType": "공사", "coordX": "127.5", "coordY": "34.9", "startDate": "20261124075753",
         "message": " (2차로)작업중 "}, {"type": "국도", "eventType": "공사", "coordX": None, "coordY": "1"}]}})
    assert len(evs) == 1 and evs[0]["message"] == "(2차로)작업중" and evs[0]["y"] == 34.9
    assert T.parse_cams({"response": {"data": {"cctvurl": "https://h/1/t", "coordx": 1, "coordy": 2}}}, "its")[0]["id"] == "1"


def test_clip_name_carries_camera_and_time():
    assert T.clip_name("2514", datetime(2026, 11, 24, 15, 4, 5)) == "its-2514_20261124_150405.mp4"


def test_scrub_removes_key(monkeypatch):
    monkeypatch.setattr(T, "_KEY", "abc123KEY")
    assert "abc123KEY" not in T.scrub("https://x?apiKey=abc123KEY&type=ex") and "***" in T.scrub("abc123KEY")


def test_download_rejects_non_mp4(tmp_path, monkeypatch):
    class R:
        def __init__(self, data): self.data, self.i = data, 0
        def read(self, n): b = self.data[self.i:self.i + n]; self.i += n; return b
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(T.urllib.request, "urlopen", lambda req, timeout=0: R(b"<html>not found</html>"))
    with pytest.raises(ValueError):
        T.download("https://h/1/t", tmp_path / "a.mp4")
    assert list(tmp_path.iterdir()) == []
    monkeypatch.setattr(T.urllib.request, "urlopen", lambda req, timeout=0: R(b"\x00\x00\x00\x20ftypisom" + b"x" * 100))
    assert T.download("https://h/1/t", tmp_path / "b.mp4") == 112 and (tmp_path / "b.mp4").is_file()
