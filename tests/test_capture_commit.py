"""영상 캡처의 '넣기'(capture.CaptureManager.commit) — 뽑아 둔 프레임을 unlabeled/<카메라>/ 로 옮기는 길 (2026-11-16).

영상을 실제로 뽑지는 않는다(OpenCV·영상 파일 없이) — 다 뽑은 세션 폴더(`session.json` + `frames/NNNNN.jpg`)를 손으로 만들어
`CaptureManager` 가 읽게 한다. 여기서 지키는 것:

- **넣는 도중 서버가 꺼지면** 다시 켰을 때 그 세션이 '넣는 도중 꺼졌다' 고 알린다. 고치기 전에는 `committing` 상태를 메모리에서만
  바꿔서, 다시 켜면 세션이 넣기 전 상태로 읽히고 안내(`_load` 의 committing 분기)가 한 번도 나오지 않았다 — 프레임 일부는 이미
  unlabeled 로 가 있는데 화면에는 아무 말이 없다.
- 넣기가 끝나면 고른 프레임만 옮기고 세션 폴더를 지운다. 같은 이름이 이미 있으면 덮지 않는다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from autolabel import capture
from autolabel.capture import CaptureManager

SID = "20261116_101500_ab12"
CAMERAS = """\
cameras:
  cam-01:
    kind: cctv
    place: 시험 장소 A
"""


class Killed(BaseException):
    """서버 프로세스가 죽은 것을 흉내 낸다 — commit 이 잡는 OSError 가 아니라서 그대로 빠져나간다."""


@pytest.fixture
def session(cfg) -> Path:
    """다 뽑은 세션 하나: 프레임 4장(0·1·2·3초), 상태 done."""
    cfg.cameras.write_text(CAMERAS, encoding="utf-8")
    d = cfg.db.parent / "capture" / SID
    (d / "frames").mkdir(parents=True)
    for i in range(4):
        Image.new("RGB", (64, 36), (40 * i, 90, 120)).save(d / "frames" / f"{i:05d}.jpg")
    data = {"created": "2026-11-16T10:15:00", "video": {"name": "clip.mp4", "file": "video.mp4", "sha1": "ab" * 20},
            "state": "done", "error": None, "note": None, "settings": {"every": 1.0},
            "frames": [{"i": i, "t": float(i), "c": 10.0, "s": None, "k": True} for i in range(4)],
            "run": 1, "pos": 4.0, "camera": "cam-01", "start": None, "start_guess": None, "start_mtime": None}
    (d / "session.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return d


def manager(cfg, scans: list | None = None, asked: list | None = None) -> CaptureManager:
    def ingest(mine=None, note=None):
        if scans is not None:
            scans.append(sorted(p.name for p in cfg.unlabeled.rglob("*.jpg")))
        if asked is not None:      # 넣기가 스캔에 넘긴 것 — 이번 영상의 파일(작업 배치에 들 것)과 영상 이름
            asked.append((sorted(p.name for p in mine), note))
        raise RuntimeError("시험에서는 스캔하지 않는다")      # commit 은 scan_error 로 적고 계속한다(파일은 이미 옮겼다)

    return CaptureManager(cfg, ingest=ingest)


def test_commit_moves_only_the_chosen_frames_and_closes_the_session(cfg, session):
    scans: list = []
    asked: list = []
    man = manager(cfg, scans, asked)
    out = man.commit(SID, "cam-01", "2026-11-10 12:00:00", [0, 2, 3])
    names = ["cam-01_20261110_120000_000.jpg", "cam-01_20261110_120002_000.jpg", "cam-01_20261110_120003_000.jpg"]
    assert (out["moved"], out["exists"], out["missing"], out["camera"], out["auto"]) == (3, 0, 0, "cam-01", None)
    assert sorted(p.name for p in (cfg.unlabeled / "cam-01" / "20261110").iterdir()) == names
    assert scans == [names] and "scan_error" in out          # 옮긴 뒤에 스캔을 불렀다
    assert asked == [(names, "clip.mp4")]                    # 이번에 넣은 파일과 영상 이름을 스캔에 넘긴다(작업 배치, 2026-11-23)
    assert not session.exists() and SID not in man.sessions  # 넣으면 세션 폴더째 지운다(안 고른 프레임도)


def test_a_commit_cut_short_by_a_dead_server_says_so_after_restart(cfg, session, monkeypatch):
    """프레임 둘을 옮긴 뒤 서버가 죽었다 — 다시 켠 서버는 그 세션을 '넣는 도중 꺼졌다' 로 읽는다."""
    man = manager(cfg)
    real, moved = capture._rename, []

    def dying(src, dst):
        if len(moved) == 2:
            raise Killed()
        real(src, dst)
        moved.append(dst)

    monkeypatch.setattr(capture, "_rename", dying)
    with pytest.raises(Killed):
        man.commit(SID, "cam-01", "2026-11-10 12:00:00", [0, 1, 2, 3])
    assert len(moved) == 2
    assert json.loads((session / "session.json").read_text(encoding="utf-8"))["state"] == "committing"

    again = manager(cfg)                                     # 서버를 다시 켰다
    s = again.sessions[SID]
    assert s.state == "stopped"
    assert "넣는 도중 서버가 꺼졌습니다" in s.note and "폴더 스캔" in s.note
    # 남은 프레임은 다시 넣을 수 있다 — 이미 옮긴 둘은 같은 이름이라 건너뛴다(덮지 않는다)
    monkeypatch.setattr(capture, "_rename", real)
    asked: list = []
    again.ingest = manager(cfg, asked=asked).ingest
    out = again.commit(SID, "cam-01", "2026-11-10 12:00:00", [0, 1, 2, 3])
    assert (out["moved"], out["exists"], out["missing"]) == (2, 2, 0)
    assert len(list((cfg.unlabeled / "cam-01" / "20261110").iterdir())) == 4
    # 앞에서 옮기기만 하고 등록하지 못한 둘도 '이번 영상의 파일' 로 넘긴다 — 같은 작업 배치에 들게 (2026-11-23)
    assert len(asked) == 1 and len(asked[0][0]) == 4


def test_a_video_upload_stops_as_soon_as_more_arrives_than_was_announced(cfg, store, monkeypatch):
    """영상 올리기(server capture_upload) — 알려 준 크기보다 많이 오면 그 자리에서 그만 받고 세션을 지운다 (2026-11-25, Gemini
    리뷰 묶음 6). 고치기 전에는 오는 대로 끝까지 디스크에 쓴 뒤에야 크기를 견줬다."""
    import asyncio

    from fastapi import HTTPException

    from autolabel.server import create_app
    from helpers import StreamBody, post_endpoint

    cfg.cameras.write_text(CAMERAS, encoding="utf-8")
    app = create_app(cfg, label="시험", bind_host="127.0.0.1")
    fn = post_endpoint(app, "/api/capture/upload")
    body = StreamBody([b"x" * 60] * 50)                      # 100바이트라고 했는데 3,000바이트가 온다
    with pytest.raises(HTTPException) as err:
        asyncio.run(fn(body, name="clip.mp4", size=100))
    assert err.value.status_code == 400 and "많이 옵니다" in err.value.detail
    assert body.taken == 2                                   # 둘째 조각에서 넘었다 — 나머지 48조각은 받지 않았다
    root = cfg.db.parent / "capture"
    assert not root.exists() or not list(root.iterdir())     # 받던 세션 폴더는 지웠다


def test_a_video_upload_of_unknown_size_stops_when_the_disk_runs_low(cfg, store, monkeypatch):
    """크기를 알려 주지 않은 올리기는 시작할 때 디스크를 볼 수 없다 — 쓰는 동안 남은 공간이 한도 아래로 내려가면 멈춘다."""
    import asyncio
    from collections import namedtuple

    from fastapi import HTTPException

    from autolabel import server
    from helpers import StreamBody, post_endpoint

    cfg.cameras.write_text(CAMERAS, encoding="utf-8")
    app = server.create_app(cfg, label="시험", bind_host="127.0.0.1")
    fn = post_endpoint(app, "/api/capture/upload")
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(server.shutil, "disk_usage", lambda p: usage(10 ** 12, 10 ** 12, server.CAPTURE_MIN_FREE - 1))
    body = StreamBody([b"x" * (8 << 20)] * 5)                # 8MB 조각 다섯 — 첫 조각을 쓰고 나서 디스크를 본다
    with pytest.raises(HTTPException) as err:
        asyncio.run(fn(body, name="clip.mp4", size=None))
    assert err.value.status_code == 507 and body.taken == 1
    root = cfg.db.parent / "capture"
    assert not root.exists() or not list(root.iterdir())


def test_a_commit_refused_before_moving_leaves_the_saved_state_alone(cfg, session):
    """옮기기 전에 거절된 넣기(없는 카메라)는 상태를 건드리지 않는다 — 파일의 상태도 done 그대로."""
    man = manager(cfg)
    with pytest.raises(capture.CaptureError) as err:
        man.commit(SID, "cam-99", "2026-11-10 12:00:00", [0])
    assert err.value.status == 400
    assert json.loads((session / "session.json").read_text(encoding="utf-8"))["state"] == "done"
    assert man.sessions[SID].state == "done"
