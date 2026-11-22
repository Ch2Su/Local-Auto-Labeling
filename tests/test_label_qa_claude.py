"""tools/label_qa_claude.py — 후보 크롭을 Claude API 로 보내는 도구. 여기서는 **밖으로 보내면 안 되는 그림이 나가지 않는가** 만 본다 (2026-11-23).

AI Hub 에서 받은 그림(source `aihub_pm`, 카메라 `ah-NNNN`)은 약관이 승인 없는 제3자 제공을 막아서, 사용자가 'Claude 는 AI Hub
그림을 열지 않는다' 고 정했다(docs/provenance.md '공개 데이터셋'). 이 도구는 그림을 PC 밖으로 보내는 유일한 코드라, 그 그림의
후보·대조군이 요청에 실리지 않는지를 끝까지 돌려 본다.

API 는 부르지 않는다 — `anthropic` 모듈을 가짜로 갈아 끼워 요청에 실린 그림을 받아 적고, 키도 읽지 않는다(`load_api_key` 를
간다). 그림은 한 가지 색으로 칠한 작은 JPEG 이다: 보내도 되는 그림은 회색, 보내면 안 되는 그림은 자홍색 — 요청에 실린 그림의
색으로 어느 쪽에서 왔는지 안다.
"""

from __future__ import annotations

import base64
import csv
import io
import json
import sqlite3
import sys
from collections import Counter
from types import SimpleNamespace

import pytest
import yaml
from PIL import Image

from helpers import CLASSES, ROOT, box

sys.path.insert(0, str(ROOT / "tools"))

import label_qa_claude as lqc  # noqa: E402

CAR = CLASSES.index("car")
SIZE = (640, 360)
GRAY, MAGENTA = (128, 128, 128), (255, 0, 255)
PRED = "400.0 20.0 480.0 90.0"          # 후보 박스(픽셀) — 80×70, 라벨(80×72)과 짧은 변이 비슷해 양성 대조가 만들어진다
FIELDS = ["kind", "image_id", "split", "pred_cls", "conf", "side_px", "pred_box"]


def labels():
    return [box(CAR, 0.3, 0.5, 0.125, 0.2), box(CAR, 0.7, 0.5, 0.125, 0.2)]


def is_magenta(c) -> bool:
    return c[0] > 180 and c[1] < 80 and c[2] > 180


def is_gray(c) -> bool:
    return all(90 < v < 170 for v in c)


@pytest.fixture
def data(store, tmp_path):
    """승인 이미지 16장. 보내도 되는 7장(후보 2 + 대조군 감 5), 보내면 안 되는 9장(후보 3 + 대조군으로 뽑힐 수 있는 6)."""

    def put(name, source, camera, color) -> int:
        path = tmp_path / "img" / camera / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", SIZE, color).save(path, "JPEG", quality=95)
        store.ingest_batch([{"path": str(path), "source": source, "width": SIZE[0], "height": SIZE[1],
                             "status": "approved", "boxes": labels(), "camera": camera}])
        with store.connect() as con:
            return int(con.execute("SELECT id FROM images WHERE path=?", (str(path),)).fetchone()[0])

    d = SimpleNamespace()
    d.n1 = put("n1.jpg", "collected", "cam-01", GRAY)
    d.n2 = put("n2.jpg", "collected", "cam-01", GRAY)
    d.pool = [put(f"p{i}.jpg", "collected", "cam-02", GRAY) for i in range(5)]
    d.h_both = put("h1.jpg", "aihub_pm", "ah-0001", MAGENTA)        # 넣은 그대로
    d.h_moved = put("h2.jpg", "aihub_pm", "cam-09", MAGENTA)        # 몇 장만 다른 카메라로 옮긴 꼴 — 출처만 남는다
    d.h_key = put("h3.jpg", "collected", "ah-0042", MAGENTA)        # 출처가 달라도 키가 그 카메라다
    d.h_pool = [put(f"hp{i}.jpg", "aihub_pm", f"ah-01{i:02d}", MAGENTA) for i in range(6)]
    d.held = {d.h_both, d.h_moved, d.h_key, *d.h_pool}
    d.open = {d.n1, d.n2, *d.pool}
    return d


def write_candidates(folder, rows) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "candidates.csv", "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        for kind, image_id in rows:
            wr.writerow({"kind": kind, "image_id": image_id, "split": "미배정", "pred_cls": "car", "conf": 0.9,
                         "side_px": 70.0, "pred_box": PRED})


@pytest.fixture
def qa(data, tmp_path):
    """label_qa 결과 폴더. missing 줄은 차례로 n1 · h_both · h_moved · n2 · h_key (사이의 unseen 줄은 이 도구가 읽지 않는다)."""
    folder = tmp_path / "qa"
    write_candidates(folder, [("missing", data.n1), ("unseen", data.n1), ("missing", data.h_both),
                              ("missing", data.h_moved), ("missing", data.n2), ("missing", data.h_key)])
    return folder


@pytest.fixture
def api(monkeypatch):
    """가짜 `anthropic`. sent 는 요청마다 실린 그림들의 평균 색, made 는 만든 클라이언트의 인자."""
    sent, made = [], []

    class Messages:
        def create(self, **kw):
            colors = []
            for part in kw["messages"][0]["content"]:
                if part["type"] == "image":
                    img = Image.open(io.BytesIO(base64.standard_b64decode(part["source"]["data"]))).convert("RGB")
                    colors.append(img.resize((1, 1), Image.BOX).getpixel((0, 0)))
            sent.append(colors)
            answer = {"verdict": "not_object", "object_class": None, "note": "test"}
            return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(answer))],
                                   usage=SimpleNamespace(input_tokens=10, output_tokens=5))

    class Client:
        def __init__(self, **kw):
            made.append(kw)
            self.messages = Messages()

    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=Client))
    monkeypatch.setattr(lqc, "load_api_key", lambda: "test-key")
    return SimpleNamespace(sent=sent, made=made)


def run(cfg, monkeypatch, qa, *extra):
    cfg.path.write_text(yaml.safe_dump(cfg.raw, allow_unicode=True), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["label_qa_claude.py", "--config", str(cfg.path), "--qa-dir", str(qa),
                                      "--controls", "2", "--workers", "1", *extra])
    lqc.main()
    return qa / "claude_claude-sonnet-5"


def read_items(out) -> list[dict]:
    return json.loads((out / "items.json").read_text(encoding="utf-8"))


def test_the_rule_is_source_or_camera_key():
    assert lqc.local_only("aihub_pm", "ah-0001")
    assert lqc.local_only("aihub_pm", "cam-09")             # 다른 카메라로 옮긴 장
    assert lqc.local_only("aihub_pm", None)
    assert lqc.local_only("collected", "ah-0042")
    assert not lqc.local_only("collected", "hw-04")
    assert not lqc.local_only("collected", "ah-12")         # 그 도구의 키 꼴(네 자리 이상)이 아니다
    assert not lqc.local_only("legacy", None)
    assert not lqc.local_only(None, None)


def test_the_rule_names_what_the_import_tool_writes():
    """넣는 도구가 출처 이름이나 키 꼴을 바꾸면 여기서 어긋난다 — 규칙을 두 곳에 적어 둔 값이다."""
    import import_aihub_pm as ah

    assert ah.SOURCE in lqc.LOCAL_ONLY_SOURCES
    for key in ah.next_keys(set(), 3) + ah.next_keys({"ah-9999"}, 2):       # ah-0001 … 과 다섯 자리로 넘어간 뒤
        assert ah.KEY_RE.match(key) and lqc.LOCAL_ONLY_CAMERA.match(key), key
    for key in ("ah-001", "ah-0001x", "xah-0001", "hw-01", "vid-07", ""):
        assert not ah.KEY_RE.match(key) and not lqc.LOCAL_ONLY_CAMERA.match(key), key


def test_build_items_leaves_local_only_images_out(cfg, data, qa, capsys):
    items, images, gt, held = lqc.build_items(cfg, qa, 2, 0)
    assert held == data.held
    assert set(images) == data.open                         # 그릴 수 있는 그림에 아예 없다
    assert set(gt) <= data.open
    assert {it["image_id"] for it in items} <= data.open    # 후보도 대조군도
    assert Counter(it["group"] for it in items) == {"candidate": 2, "control_pos": 2, "control_neg": 2}
    # 번호는 candidates.csv 의 missing 줄 순서 그대로다 — 빼는 그림 때문에 다른 후보의 키가 밀리지 않는다
    assert sorted(it["key"] for it in items if it["group"] == "candidate") == ["cand0000", "cand0003"]
    assert "3장의 후보 3건" in capsys.readouterr().out


def test_nothing_from_a_local_only_image_is_sent(cfg, monkeypatch, data, qa, api, capsys):
    out = run(cfg, monkeypatch, qa)
    # 요청에 실린 그림부터 본다 — 이 도구가 지켜야 하는 것 그 자체다
    assert not any(is_magenta(c) for colors in api.sent for c in colors), api.sent
    assert all(is_gray(c) for colors in api.sent for c in colors), api.sent
    assert all(len(colors) == 3 for colors in api.sent)                     # 요청마다 그림 세 장
    items = read_items(out)
    assert {it["image_id"] for it in items} <= data.open
    assert len(api.made) == 1 and len(api.sent) == len(items) == 6
    results = [json.loads(line) for line in open(out / "results.jsonl", encoding="utf-8")]
    assert {r["key"] for r in results} == {it["key"] for it in items}
    assert {p.name for p in (out / "crops").iterdir()} == {f"{it['key']}.jpg" for it in items}
    assert "밖으로 보내지 않는 그림" in capsys.readouterr().out


def test_dry_run_draws_no_crop_of_a_local_only_image(cfg, monkeypatch, data, qa, api, capsys):
    out = run(cfg, monkeypatch, qa, "--dry-run")
    items = read_items(out)
    assert {it["image_id"] for it in items} <= data.open
    assert {p.name for p in (out / "crops").iterdir()} == {f"{it['key']}.jpg" for it in items}
    for p in (out / "crops").iterdir():                                     # 그려 둔 크롭도 전부 회색 그림에서 왔다
        assert is_gray(Image.open(p).convert("RGB").resize((1, 1), Image.BOX).getpixel((0, 0))), p.name
    assert api.made == [] and api.sent == []
    assert "드라이런" in capsys.readouterr().out


def test_items_saved_before_the_rule_are_dropped_too(cfg, monkeypatch, data, qa, api, capsys):
    """이 규칙 전에 만든 `items.json` 에는 보내면 안 되는 그림의 후보·대조군이 들어 있을 수 있다 — 다시 돌려도 보내지 않는다."""
    out = qa / "claude_claude-sonnet-5"
    out.mkdir()
    item = {"box": [400.0, 20.0, 480.0, 90.0], "cls": "car", "conf": 0.9, "hide_gt": None, "split": "미배정", "side_px": 70.0}
    old = [{"key": "cand0000", "group": "candidate", "image_id": data.n1, **item},
           {"key": "cand0001", "group": "candidate", "image_id": data.h_both, **item},
           {"key": "neg000", "group": "control_neg", "image_id": data.h_pool[0], **item}]
    (out / "items.json").write_text(json.dumps(old, ensure_ascii=False), encoding="utf-8")

    run(cfg, monkeypatch, qa)
    assert len(api.sent) == 1 and all(is_gray(c) for c in api.sent[0])
    assert [p.name for p in (out / "crops").iterdir()] == ["cand0000.jpg"]
    said = capsys.readouterr().out
    assert "항목 2건을 뺍니다: {'candidate': 1, 'control_neg': 1}" in said
    assert "승인 이미지(또는 그 파일)가 없는" not in said                      # '지운 이미지' 로 잘못 알리지 않는다


def test_only_local_only_candidates_means_nothing_to_send(cfg, monkeypatch, data, tmp_path, api, capsys):
    folder = tmp_path / "qa-held"
    write_candidates(folder, [("missing", data.h_both), ("missing", data.h_pool[1])])
    with pytest.raises(SystemExit) as stop:
        run(cfg, monkeypatch, folder)
    assert "밖으로 보내지 않는 그림의 후보뿐" in str(stop.value)
    assert api.made == [] and api.sent == []
    assert not (folder / "claude_claude-sonnet-5" / "items.json").exists()
    assert list((folder / "claude_claude-sonnet-5" / "crops").iterdir()) == []
    assert "2장의 후보 2건" in capsys.readouterr().out


def test_a_database_from_before_the_camera_column_still_loads(cfg, tmp_path):
    """카메라 열이 생기기(2026-11-03) 전의 DB — 읽기 전용이라 열을 더하지 못한다. 출처로만 가른다."""
    con = sqlite3.connect(cfg.db)
    con.executescript(
        "CREATE TABLE classes(idx INTEGER, name TEXT);"
        "CREATE TABLE images(id INTEGER PRIMARY KEY, path TEXT, width INT, height INT, split TEXT, status TEXT, source TEXT);"
        "CREATE TABLE boxes(id INTEGER PRIMARY KEY, image_id INT, cls INT, cx REAL, cy REAL, w REAL, h REAL);"
        "INSERT INTO classes VALUES (0, 'car');"
        "INSERT INTO images VALUES (1, 'a.jpg', 640, 360, NULL, 'approved', 'legacy');"
        "INSERT INTO images VALUES (2, 'b.jpg', 640, 360, NULL, 'approved', 'aihub_pm');"
        "INSERT INTO boxes VALUES (1, 1, 0, 0.5, 0.5, 0.2, 0.2);"
        "INSERT INTO boxes VALUES (2, 2, 0, 0.5, 0.5, 0.2, 0.2);")
    con.commit()
    con.close()
    folder = tmp_path / "qa-old"
    write_candidates(folder, [("missing", 1), ("missing", 2)])
    items, images, gt, held = lqc.build_items(cfg, folder, 0, 0)
    assert (held, set(images), set(gt)) == ({2}, {1}, {1})
    assert [(it["key"], it["image_id"]) for it in items] == [("cand0000", 1)]
