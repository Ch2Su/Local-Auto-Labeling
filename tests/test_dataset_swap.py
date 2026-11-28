"""학습 세트 만들기 — 검사 → 옆 폴더에 만들기 → 바꿔 끼우기 (trainer.build_dataset, 2026-11-28 포터블 앱 4단계).

그 전에는 옛 세트를 먼저 지우고 검사했다 — 클래스·승인 라벨이 없거나 그림을 못 읽어 멈추면 세트가 없는 채로 남았다. 이제는
도중에 멈추면 옛 세트(그림·라벨·data.yaml)가 그대로고, 옆 작업 폴더(`dataset.building`·`dataset.old`)도 남지 않는다. 옛 세트의
학습 캐시(.npy·.cache)만 만들기 전에 지운다(디스크를 두 벌만큼 쓰지 않게).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from PIL import Image

from autolabel.trainer import build_dataset
from helpers import CLASSES, box, far_hashes

CAR = CLASSES.index("car")


@pytest.fixture
def photos(store, add_image):
    """카메라 셋(train·val·test 고정 분할)에 한 장씩 — 진짜 작은 JPEG."""
    hashes = far_hashes(3)
    ids = {}
    for k, sp in enumerate(("train", "val", "test")):
        ids[sp] = add_image(f"{sp}.jpg", camera=f"cam-{k}", boxes=[box(CAR)], dhash=hashes[k], split=sp)
        path = Path(store.get_image(ids[sp])["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 36), (40 * k, 90, 150)).save(path, quality=90)
    return ids


def files(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def leftovers(root: Path) -> list[str]:
    return sorted(p.name for p in root.parent.iterdir() if p.name.startswith(root.name + "."))


def test_a_rebuild_replaces_the_set_and_leaves_nothing_beside_it(cfg, store, photos):
    first = build_dataset(cfg, store)
    root = cfg.dataset
    assert first.data_yaml == root / "data.yaml" and first.data_yaml.is_file()
    assert yaml.safe_load(first.data_yaml.read_text(encoding="utf-8"))["path"] == str(root)   # 최종 자리 — 작업 폴더가 아니다
    (root / "images" / "train" / "stale.jpg").write_bytes(b"x")                          # 옛 세트에만 있던 것
    build_dataset(cfg, store)
    assert "images/train/stale.jpg" not in files(root)
    assert leftovers(root) == []


def test_a_failed_build_keeps_the_old_set(cfg, store, photos):
    build_dataset(cfg, store)
    root = cfg.dataset
    before = files(root)
    # 가릴 영역이 있는 그림을 못 읽게 — 칠하다 멈춘다(RuntimeError, id 를 말한다)
    store.set_masks(photos["train"], [(0.0, 0.0, 0.5, 0.5)])
    Path(store.get_image(photos["train"])["path"]).write_bytes(b"not an image")
    with pytest.raises(RuntimeError, match=rf"\(id {photos['train']}\)"):
        build_dataset(cfg, store)
    assert files(root) == before and leftovers(root) == []


def test_nothing_is_touched_when_there_is_nothing_to_build(cfg, store, photos):
    build_dataset(cfg, store)
    root = cfg.dataset
    before = files(root)
    for i in photos.values():
        store.set_status(i, "pending")
    with pytest.raises(RuntimeError, match="승인된 라벨이 하나도 없습니다"):
        build_dataset(cfg, store)
    assert files(root) == before and leftovers(root) == []


def test_caches_of_the_old_set_go_before_building(cfg, store, photos):
    build_dataset(cfg, store)
    root = cfg.dataset
    (root / "images" / "train" / "a.npy").write_bytes(b"0" * 10)
    (root / "labels" / "train.cache").write_bytes(b"0")
    store.set_masks(photos["train"], [(0.0, 0.0, 0.5, 0.5)])
    Path(store.get_image(photos["train"])["path"]).write_bytes(b"broken")
    with pytest.raises(RuntimeError):
        build_dataset(cfg, store)
    kept = files(root)
    assert "images/train/a.npy" not in kept and "labels/train.cache" not in kept    # 캐시만 갔다
    assert "data.yaml" in kept and any(f.startswith("labels/train/") for f in kept)


def test_leftovers_of_a_crash_are_cleared(cfg, store, photos):
    root = cfg.dataset
    for tag in ("building", "old"):
        stale = root.with_name(f"{root.name}.{tag}") / "images" / "x.jpg"
        stale.parent.mkdir(parents=True)
        stale.write_bytes(b"x")
    build_dataset(cfg, store)
    assert leftovers(root) == [] and (root / "data.yaml").is_file()


def test_the_old_set_comes_back_when_the_swap_fails(cfg, store, photos, monkeypatch):
    """바꿔 끼우기의 두 번째 이름 바꾸기가 실패하면(다른 프로그램이 새 폴더를 잡음 등) 옛 세트를 제자리로 돌린다."""
    build_dataset(cfg, store)
    root = cfg.dataset
    before = files(root)
    real = Path.rename

    def rename(self, target):
        if self.name.endswith(".building"):
            raise PermissionError("잡혀 있음")
        return real(self, target)

    monkeypatch.setattr(Path, "rename", rename)
    with pytest.raises(PermissionError):
        build_dataset(cfg, store)
    monkeypatch.setattr(Path, "rename", real)
    assert files(root) == before and leftovers(root) == []
