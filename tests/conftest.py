"""회귀 테스트 공통 준비 (2026-11-12).

**실제 데이터는 건드리지 않는다.** DB·설정·카메라 등록 파일은 전부 pytest 의 임시 폴더(`tmp_path`)에 만든다 — `cfg` 가
`data.*` 경로를 빠짐없이 절대경로로 적는 것은 그래서다(하나라도 빼면 `config.resolve` 가 저장소의 `data/` 를 가리킨다).
GPU·모델·학습은 쓰지 않는다. 돌리는 법은 README '회귀 테스트'.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autolabel.config import Config  # noqa: E402
from autolabel.store import Store  # noqa: E402

from helpers import CLASSES  # noqa: E402


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """임시 폴더만 가리키는 설정. 분할 비율은 config.yaml 과 같은 0.2 / 0.15."""
    d = tmp_path / "data"
    d.mkdir()
    raw = {
        "data": {
            "seed_images": str(d / "seed" / "images"),
            "seed_labels": str(d / "seed" / "labels"),
            "unlabeled": str(d / "unlabeled"),
            "labeled": str(d / "labeled"),
            "dataset": str(d / "dataset"),
            "db": str(d / "labeling.db"),
            "cameras": str(d / "cameras.yaml"),
            "classes_txt": str(d / "classes.txt"),
        },
        "model": {"weights": str(tmp_path / "models" / "best.pt")},
        "train": {"val_ratio": 0.2, "test_ratio": 0.15},
        "backup": {"keep": 0},          # 시험에서는 하루 사본을 뜨지 않는다
        "classes": list(CLASSES),
    }
    return Config(raw, tmp_path / "config.yaml")


@pytest.fixture
def store(cfg: Config) -> Store:
    s = Store(cfg.db)
    s.set_classes(CLASSES)
    return s


@pytest.fixture
def add_image(store: Store, tmp_path: Path):
    """이미지 행을 하나 넣고 id 를 돌려준다. 파일은 만들지 않는다(행만 — dHash 가 필요하면 값으로 준다)."""

    def add(name: str, *, status: str = "approved", boxes=(), camera: str | None = None,
            dhash: str | None = None, split: str | None = None, size: tuple[int, int] = (1280, 720)) -> int:
        path = str(tmp_path / "img" / (camera or "_") / name)
        store.ingest_batch([{"path": path, "source": "collected" if camera else "legacy",
                             "width": size[0], "height": size[1], "status": status,
                             "boxes": list(boxes), "camera": camera}])
        with store.connect() as con:
            image_id = int(con.execute("SELECT id FROM images WHERE path=?", (path,)).fetchone()[0])
        if dhash:
            store.set_dhashes([(image_id, dhash)])
        if split:
            store.set_splits([(image_id, split)])
        return image_id

    return add
