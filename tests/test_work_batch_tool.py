"""tools/work_batch.py — 이미 있는 이미지를 작업 배치 하나로 묶는다 (2026-11-23, 검수 중이던 AI Hub 100장을 배치로).

실제 DB 를 바꾸는 도구라 지키는 것:

- **보기만 할 때는 아무것도 쓰지 않는다** — 옛 DB(작업 배치 열이 없다)에 열을 더하지도, 사본을 뜨지도, 배치를 만들지도 않는다.
- `--apply` 는 `work_batches` 한 줄과 고른 이미지의 `work_batch` 만 쓴다: 상태·박스·`updated_at` 은 그대로다.
- 이미 다른 배치에 든 이미지는 `--move` 없이는 건드리지 않는다. `--dissolve` 는 배치만 지우고 이미지는 남긴다.

`main()` 을 `sys.argv` 로 돌린다. 설정은 임시 폴더의 yaml 이다.
"""

from __future__ import annotations

import sqlite3
import sys

import pytest
import yaml

from autolabel.store import Store
from helpers import ROOT, box
from test_workbatch import old_db

sys.path.insert(0, str(ROOT / "tools"))
import work_batch as wb  # noqa: E402


@pytest.fixture
def run(cfg, monkeypatch, capsys):
    cfg.path.write_text(yaml.safe_dump(cfg.raw, allow_unicode=True), encoding="utf-8")

    def go(*argv: str) -> tuple[int | str, str]:
        monkeypatch.setattr(sys, "argv", ["work_batch.py", "--config", str(cfg.path), *argv])
        try:
            code = wb.main()
        except SystemExit as exc:           # sys.exit("이유") — 멈춘 이유를 돌려준다
            code = exc.code
        return code, capsys.readouterr().out

    return go


@pytest.fixture
def data(store, add_image):
    """AI Hub 꼴 여섯 장(카메라 ah-…·출처 aihub_pm, 상태가 섞여 있다)과 직접 모은 두 장."""
    ids = {}
    for i, status in enumerate(["approved", "approved", "pending", "pending", "pending", "skipped"]):
        ids[f"ah{i}"] = add_image(f"ah{i}.jpg", status=status, boxes=[box()], camera=f"ah-{i + 1:04d}")
    ids["own0"] = add_image("own0.jpg", status="approved", camera="cam-01")
    ids["own1"] = add_image("own1.jpg", status="pending", camera="cam-01")
    with store.connect() as con:
        con.execute("UPDATE images SET source='aihub_pm' WHERE camera LIKE 'ah-%'")
    return ids


def snapshot(store) -> list[tuple]:
    with store.connect() as con:
        return [tuple(r) for r in con.execute(
            "SELECT id, status, camera, split, updated_at, (SELECT COUNT(*) FROM boxes b WHERE b.image_id = images.id)"
            " FROM images ORDER BY id")]


def batch_of(store) -> dict[int, int | None]:
    with store.connect() as con:
        return {int(r[0]): r[1] for r in con.execute("SELECT id, work_batch FROM images")}


def test_parse_ids():
    assert wb.parse_ids("8327-8330, 8400 8329") == [8327, 8328, 8329, 8330, 8400]
    assert wb.parse_ids("5-3") == [3, 4, 5]
    for bad in ("abc", "3-", "-3", "3-x", "1.5"):
        with pytest.raises(ValueError):
            wb.parse_ids(bad)
    with pytest.raises(ValueError):
        wb.parse_ids("1-999999999")


def test_a_look_writes_nothing(run, store, data):
    before = snapshot(store)
    code, out = run("--source", "aihub_pm", "--name", "AI Hub 묶음")
    assert code == 0 and "묶을 것 6장" in out and "보기만 했습니다" in out
    assert "상태: 검수 대기 3 · 승인 2 · 건너뜀 1" in out
    assert "카메라 6대" in out
    assert store.list_work_batches() == [] and snapshot(store) == before
    code, out = run()                                   # 조건 없이 — 배치 목록
    assert code == 0 and "작업 배치 0개" in out and "배치 없음" in out and "8장" in out


def test_apply_makes_one_batch_and_changes_nothing_else(run, store, data):
    before = snapshot(store)
    code, out = run("--source", "aihub_pm", "--name", "  AI Hub   묶음 ", "--apply")
    assert code == 0, out
    (b,) = store.list_work_batches()
    assert (b["kind"], b["name"], b["hold"], b["total"]) == ("manual", "AI Hub 묶음", False, 6)
    assert b["counts"]["pending"] == 3 and b["counts"]["approved"] == 2 and b["counts"]["skipped"] == 1
    assert f"작업 {b['id']} 로 6장을 묶었습니다" in out
    got = batch_of(store)
    assert all(got[data[k]] == b["id"] for k in data if k.startswith("ah"))
    assert got[data["own0"]] is None and got[data["own1"]] is None
    assert snapshot(store) == before                    # 상태·분할·수정 시각·박스 수는 그대로
    code, out = run()
    assert f"작업 {b['id']:>3}" in out and "AI Hub 묶음" in out and "2장" in out


def test_images_already_in_a_batch_are_left_alone_unless_moved(run, store, data):
    first = store.create_work_batch("scan")
    store.set_work_batch([data["ah0"], data["ah1"]], first)
    code, out = run("--camera-prefix", "ah-", "--apply")
    assert code == 0 and "이미 다른 배치에 든 것 2장" in out and "건드리지 않습니다" in out
    got = batch_of(store)
    new = got[data["ah2"]]
    assert new not in (None, first) and got[data["ah0"]] == first and got[data["ah5"]] == new
    code, out = run("--camera-prefix", "ah-", "--status", "approved", "--move", "--hold", "--apply")
    assert code == 0, out
    got = batch_of(store)
    moved = got[data["ah0"]]
    assert moved not in (None, first, new) and got[data["ah1"]] == moved and got[data["ah2"]] == new
    assert store.get_work_batch(moved)["hold"] is True and store.get_work_batch(first)["total"] == 0


def test_picking_by_ids_camera_and_status(run, store, data):
    code, out = run("--ids", f"{data['ah2']}-{data['ah4']}", "--status", "pending", "--apply")
    assert code == 0 and "3장을 묶었습니다" in out
    code, out = run("--camera", "cam-01", "--status", "pending", "--status", "approved")
    assert code == 0 and "묶을 것 2장" in out
    code, out = run("--camera-prefix", "ah_")           # '_' 를 아무 글자로 읽지 않는다(LIKE 가 아니다)
    assert code == "조건에 맞는 이미지가 없습니다."
    code, out = run("--ids", f"{data['ah2']}")
    assert code == "묶을 이미지가 없습니다." and "이미 다른 배치에 든 것 1장" in out
    assert "id 로 읽을 수 없는" in str(run("--ids", "abc")[0])
    assert "60자까지" in str(run("--source", "aihub_pm", "--name", "가" * 61)[0])


def test_dissolve_removes_the_batch_and_keeps_the_images(run, store, data):
    run("--source", "aihub_pm", "--apply")
    (b,) = store.list_work_batches()
    before = snapshot(store)
    code, out = run("--dissolve", str(b["id"]))
    assert code == 0 and "이미지 6장은 그대로" in out and len(store.list_work_batches()) == 1      # 보기만
    code, out = run("--dissolve", str(b["id"]), "--apply")
    assert code == 0 and store.list_work_batches() == [] and set(batch_of(store).values()) == {None}
    assert snapshot(store) == before
    assert run("--dissolve", "999")[0] == "작업 배치 999 가 없습니다."
    assert "함께 쓰지 않습니다" in str(run("--dissolve", "1", "--source", "aihub_pm")[0])


def test_a_look_at_an_old_db_does_not_migrate_it(run, cfg):
    """옛 DB(작업 배치 열이 없다) — 보기만 하면 열을 더하지 않고 사본도 뜨지 않는다. --apply 때 더하고, 직전에 뜬다."""
    old_db(cfg.db, n=3)
    code, out = run()
    assert code == 0 and "아직 작업 배치 열이 없습니다" in out and "이미지 3장" in out
    code, out = run("--camera", "cam-01")
    assert code == 0 and "묶을 것 3장" in out
    con = sqlite3.connect(cfg.db)
    assert "work_batch" not in {r[1] for r in con.execute("PRAGMA table_info(images)")}
    con.close()
    assert not (cfg.db.parent / "backups").exists()

    code, out = run("--camera", "cam-01", "--name", "옛 묶음", "--apply")
    assert code == 0 and "더하기 직전의 DB 사본" in out and "3장을 묶었습니다" in out
    assert len(list((cfg.db.parent / "backups").glob("labeling_before_workbatch_*.db"))) == 1
    (b,) = Store(cfg.db).list_work_batches()
    assert (b["name"], b["total"]) == ("옛 묶음", 3)


def test_a_missing_db_stops_without_making_one(run, cfg):
    code, _ = run()
    assert "DB 가 없습니다" in str(code) and not cfg.db.exists()
