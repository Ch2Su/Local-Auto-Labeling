"""tools/label_qa.py — 라벨과 예측이 어긋나는 곳 찾기. 여기서는 **가릴 영역이 덮은 자리를 견주지 않는가** 를 본다 (2026-11-16).

가릴 영역(검수 화면 `M`)이 넓이의 반 이상을 덮은 라벨은 학습 라벨에서 빠지고, 그 자리의 물체는 사람이 일부러 안 그린 것이다.
도구가 영역을 모르던 때는 덮어 둔 자리에 남은 자동 라벨 박스가 `unseen` 으로, 모델이 거기서 찾은 물체가 `missing` 으로 나왔다.
판정은 학습 세트와 같은 함수(`masks.drops`·`masks.applies_to`)라 test 분할은 그대로 견준다.

모델은 돌리지 않는다 — 예측 파일(`preds.jsonl`)을 손으로 써 두고 `--reuse` 로 분석만 한다.
"""

from __future__ import annotations

import csv
import json
import sqlite3
import sys

import pytest
import yaml

from helpers import CLASSES, ROOT, box

sys.path.insert(0, str(ROOT / "tools"))

import label_qa  # noqa: E402

PERSON, CAR, TRUCK = CLASSES.index("person"), CLASSES.index("car"), CLASSES.index("truck")
MASK = (0.0, 0.0, 0.5, 1.0)             # 그림의 왼쪽 반을 덮는다


@pytest.fixture
def data(store, add_image):
    """세 장 모두 같은 그림: 왼쪽(덮인 자리)에 남은 자동 라벨 차 하나, 오른쪽에 사람이 그린 트럭 하나.
    train·미배정 장은 왼쪽 반을 덮었고, test 장도 영역이 있지만(분할 전에 덮었다) test 라 적용하지 않는다."""
    boxes = [box(CAR, 0.2, 0.5, 0.2, 0.2), box(TRUCK, 0.8, 0.5, 0.2, 0.2)]
    ids = {"train": add_image("tr.jpg", camera="cam-01", boxes=boxes, split="train"),
           "none": add_image("no.jpg", camera="cam-02", boxes=boxes),
           "test": add_image("te.jpg", camera="cam-03", boxes=boxes, split="test"),
           "plain": add_image("pl.jpg", camera="cam-01", boxes=boxes, split="train")}
    for key in ("train", "none", "test"):
        store.set_masks(ids[key], [MASK])
    return ids


def preds_for(image_id: int) -> dict:
    """모델이 본 것: 오른쪽 트럭은 맞게, 왼쪽(덮인 자리)에서는 라벨 없는 사람을 하나 더 찾았고 차는 못 봤다."""
    return {"id": image_id, "w": 1280, "h": 720, "preds": [
        [TRUCK, 0.9, 0.7, 0.4, 0.9, 0.6],
        [PERSON, 0.8, 0.30, 0.10, 0.36, 0.30],
    ]}


def run(cfg, tmp_path, monkeypatch, data, *extra) -> list[dict]:
    cfg.path.write_text(yaml.safe_dump(cfg.raw, allow_unicode=True), encoding="utf-8")
    out = tmp_path / "qa"
    out.mkdir(exist_ok=True)
    (out / "preds.jsonl").write_text("".join(json.dumps(preds_for(i)) + "\n" for i in data.values()), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["label_qa.py", "--config", str(cfg.path), "--reuse", "--out", str(out),
                                      "--sheet", "0", *extra])
    label_qa.main()
    with open(out / "candidates.csv", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def kinds(rows, data) -> dict:
    name = {v: k for k, v in data.items()}
    return {key: sorted(r["kind"] for r in rows if name[int(r["image_id"])] == key) for key in data}


def test_masked_places_are_not_compared(cfg, tmp_path, monkeypatch, data, capsys):
    rows = run(cfg, tmp_path, monkeypatch, data)
    assert kinds(rows, data) == {
        "train": [], "none": [],                    # 덮인 자리의 차(unseen)·사람(missing)은 후보가 아니다
        "test": ["missing", "unseen"],              # test 는 영역을 적용하지 않는다 — 평가가 보는 대로 견준다
        "plain": ["missing", "unseen"],             # 영역이 없는 장은 전과 같다
    }
    said = capsys.readouterr().out
    assert "가릴 영역이 있는 2장에서 덮인 라벨 2개·예측 2개" in said and "--keep-masked" in said


def test_keep_masked_compares_everything(cfg, tmp_path, monkeypatch, data, capsys):
    rows = run(cfg, tmp_path, monkeypatch, data, "--keep-masked")
    assert all(v == ["missing", "unseen"] for v in kinds(rows, data).values())
    assert "가릴 영역이 덮은 라벨·예측도 견줬습니다" in capsys.readouterr().out


def test_half_covered_is_the_line_as_in_the_training_set():
    """반 이상 덮이면 뺀다(정확히 반 포함) — `masks.split_boxes` 가 학습 라벨에서 빼는 것과 같은 박스다."""
    from autolabel.masks import split_boxes

    rects = [MASK]
    labels = [(CAR, 0.5, 0.5, 0.2, 0.2), (CAR, 0.52, 0.5, 0.2, 0.2), (CAR, 0.2, 0.5, 0.1, 0.1)]    # 반 · 40% · 통째로
    assert label_qa.unmasked_labels(labels, rects) == [labels[1]]
    kept, _ = split_boxes([box(*r) for r in labels], rects)
    assert [(b.cx, b.cy) for b in kept] == [(0.52, 0.5)]
    preds = [[CAR, 0.9, 0.4, 0.4, 0.6, 0.6], [CAR, 0.9, 0.42, 0.4, 0.62, 0.6], [CAR, 0.9, 0.1, 0.1, 0.2, 0.2]]
    assert label_qa.unmasked_preds(preds, rects) == [preds[1]]


def test_a_database_from_before_masks_still_loads(tmp_path):
    """가릴 영역 표가 없는 옛 DB(보관 폴더)도 읽는다 — 영역 없음으로."""
    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.executescript(
        "CREATE TABLE classes(idx INTEGER, name TEXT);"
        "CREATE TABLE images(id INTEGER PRIMARY KEY, path TEXT, name TEXT, width INT, height INT, split TEXT, status TEXT);"
        "CREATE TABLE boxes(id INTEGER PRIMARY KEY, image_id INT, cls INT, cx REAL, cy REAL, w REAL, h REAL);"
        "INSERT INTO classes VALUES (0, 'car');"
        "INSERT INTO images VALUES (1, 'a.jpg', 'a.jpg', 100, 100, NULL, 'approved');"
        "INSERT INTO boxes VALUES (1, 1, 0, 0.5, 0.5, 0.2, 0.2);")
    con.commit()
    con.close()
    classes, images, gt, masks = label_qa.load_db(db)
    assert (classes, len(images), len(gt[1]), dict(masks)) == (["car"], 1, 1, {})
    assert label_qa.applied_rects(images, masks) == {}


def test_polygon_masks_are_read_and_judged_by_their_shape(store, add_image, cfg):
    """다각형 영역(2026-11-19) — DB 의 poly 열을 읽어 모양으로 판정한다(바깥 사각형으로 재면 덮이지 않은 박스까지 뺀다).
    poly 열이 생기기 전의 DB(읽기 전용이라 열을 더하지 못한다)는 사각형으로 읽는다."""
    tri = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))                                  # 왼쪽 위 절반(x+y≤1)
    i = add_image("t.jpg", camera="cam-01", boxes=[box(CAR, 0.2, 0.2, 0.2, 0.2), box(CAR, 0.8, 0.8, 0.2, 0.2)],
                  split="train")
    store.set_masks(i, [tri, MASK])
    _, images, gt, masks = label_qa.load_db(cfg.db)
    assert masks[i] == [tri, MASK]
    rects = label_qa.applied_rects(images, masks)
    assert [r[1:3] for r in label_qa.unmasked_labels(gt[i], rects[i])] == [(0.8, 0.8)]     # 오른쪽 아래 차는 모양 밖
    with store.connect() as con:
        con.execute("ALTER TABLE masks DROP COLUMN poly")                       # 옛 DB 꼴
    _, images, gt, old = label_qa.load_db(cfg.db)
    assert old[i] == [(0.0, 0.0, 1.0, 1.0), MASK]                               # 다각형 행은 바깥 사각형으로 읽힌다
    assert label_qa.unmasked_labels(gt[i], label_qa.applied_rects(images, old)[i]) == []   # 그렇게 재면 둘 다 빠진다


# ---------------------------------------------------- 모델이 모르는 클래스 (2026-11-16)

RESPONDER, CONE = CLASSES.index("responder"), CLASSES.index("cone")


def found(gt_rows, preds, known=None) -> list[tuple]:
    rec = {"w": 1280, "h": 720, "preds": preds}
    return sorted((kind, CLASSES[g] if g >= 0 else "", CLASSES[p] if p >= 0 else "")
                  for kind, g, p, *_ in label_qa.analyze_image(gt_rows, rec, 0.5, 24, known))


def test_labels_of_classes_the_model_does_not_know_are_not_compared():
    """8클래스 운영 모델은 responder·cone 을 모른다 — 그 라벨을 '클래스가 틀렸다'·'모델이 못 본다' 로 띄우지 않는다.

    고치기 전에는 responder 라벨마다 'responder → car' 가, cone 라벨마다 unseen 이 나왔다(2026-11-16 의 실제 DB: 후보 81건
    가운데 28건). 검수 화면의 예측 비교(app.js computeFindings)는 처음부터 이 규칙이었다 — 둘이 같은 규칙이어야 한다.
    """
    labels = [(RESPONDER, 0.3, 0.5, 0.2, 0.2),      # 견인차 — 모델은 여기를 car 로 본다
              (CONE, 0.8, 0.8, 0.1, 0.1),           # 라바콘 — 모델은 아무것도 못 본다
              (CAR, 0.6, 0.2, 0.2, 0.2)]            # 모델이 truck 이라고 확신하는 차 — 이것은 진짜 후보
    preds = [[CAR, 0.9, 0.2, 0.4, 0.4, 0.6], [TRUCK, 0.8, 0.5, 0.1, 0.7, 0.3]]
    eight = set(range(len(CLASSES))) - {RESPONDER, CONE}

    assert found(labels, preds) == [("class", "car", "truck"), ("class", "responder", "car"), ("unseen", "cone", "")]
    assert found(labels, preds, eight) == [("class", "car", "truck")]

    # 모르는 클래스의 라벨 자리에서 모델이 찾은 것은 '빠진 박스' 도 아니다 — 그 라벨과 짝이 된다
    assert found([(RESPONDER, 0.3, 0.5, 0.2, 0.2)], [[TRUCK, 0.9, 0.2, 0.4, 0.4, 0.6]], eight) == []
    # 라벨이 없는 자리의 확신 예측은 그대로 missing 이고, 아는 클래스의 못 본 라벨도 그대로 unseen 이다
    assert found([(RESPONDER, 0.3, 0.5, 0.2, 0.2), (CAR, 0.8, 0.8, 0.1, 0.1)], [[CAR, 0.9, 0.5, 0.1, 0.7, 0.3]], eight) == [
        ("missing", "", "car"), ("unseen", "car", "")]


def test_reuse_reads_the_model_classes_left_at_prediction_time(cfg, tmp_path, monkeypatch, data, capsys):
    """예측할 때 `meta.json` 에 남긴 `model_classes` 를 `--reuse` 가 읽는다. 없는 옛 예측 폴더는 전부 아는 것으로 본다."""
    out = tmp_path / "qa"
    out.mkdir()
    (out / "meta.json").write_text(json.dumps({"weights": "x.pt", "model_classes": [PERSON, TRUCK]}), encoding="utf-8")
    rows = run(cfg, tmp_path, monkeypatch, data)
    # 모델이 car 를 모른다 — 덮이지 않은 두 장(test·plain)의 차 라벨은 unseen 이 아니다. 사람(missing)은 그대로
    assert kinds(rows, data) == {"train": [], "none": [], "test": ["missing"], "plain": ["missing"]}
    assert "모델이 모르는 클래스" in capsys.readouterr().out

    (out / "meta.json").unlink()
    rows = run(cfg, tmp_path, monkeypatch, data)
    assert kinds(rows, data)["plain"] == ["missing", "unseen"]
    assert "모델이 모르는 클래스" not in capsys.readouterr().out
