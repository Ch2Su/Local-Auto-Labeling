"""config.yaml 로딩과 경로 해석."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# src/autolabel/config.py -> 프로젝트 루트
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config.yaml"


def resolve(p: str | Path) -> Path:
    """설정에 적힌 경로를 프로젝트 루트 기준 절대경로로."""
    path = Path(p)
    return path if path.is_absolute() else (ROOT / path)


def pretrained_path(spec: str) -> str:
    """사전학습 가중치(yolo26s.pt, sam2.1_t.pt ...) 를 실제로 쓸 경로로 바꾼다.

    Ultralytics 는 `yolo26s.pt` 처럼 이름만 주면 **실행한 폴더**에서 찾고, 없으면 거기에
    받아 둔다. 그래서 실행 위치에 따라 프로젝트 루트에 가중치가 흩어졌고, WSL 학습은
    `/mnt/e/.../yolo26s.pt` 를 우연히 찾아 쓰고 있었다. 여기 한 곳에서 정한다.

      1) 프로젝트 기준 경로로 있으면 그것   (config 에 `weights/yolo26s.pt` 로 적은 경우)
      2) 이름만 적었으면 `weights/<이름>`   (옛 설정도 그대로 동작하게)
      3) 둘 다 없으면 받은 문자열 그대로    (Ultralytics 가 그 경로로 받아 온다)
    """
    p = resolve(spec)
    if p.exists():
        return str(p)
    w = ROOT / "weights" / Path(spec).name
    if w.exists():
        return str(w)
    return spec


@dataclass
class Config:
    raw: dict[str, Any]
    path: Path

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        cfg_path = Path(path) if path else DEFAULT_CONFIG
        if not cfg_path.exists():
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {cfg_path}")
        with open(cfg_path, "r", encoding="utf-8") as f:
            return cls(yaml.safe_load(f) or {}, cfg_path)

    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return default if cur is None else cur

    # --- 경로 ---
    @property
    def seed_images(self) -> Path:
        return resolve(self.get("data.seed_images", "data/seed/images"))

    @property
    def seed_labels(self) -> Path:
        return resolve(self.get("data.seed_labels", "data/seed/labels"))

    @property
    def seed_coco(self) -> Path | None:
        """COCO JSON 경로(폴더 또는 파일). 지정되면 .txt 대신 이쪽을 읽는다."""
        v = self.get("data.seed_coco")
        return resolve(v) if v else None

    @property
    def unlabeled(self) -> Path:
        return resolve(self.get("data.unlabeled", "data/unlabeled"))

    @property
    def labeled(self) -> Path:
        """검수를 마친 이미지를 옮겨 모아두는 곳."""
        return resolve(self.get("data.labeled", "data/labeled"))

    @property
    def dataset(self) -> Path:
        return resolve(self.get("data.dataset", "data/dataset"))

    @property
    def db(self) -> Path:
        return resolve(self.get("data.db", "data/labeling.db"))

    @property
    def classes_txt(self) -> Path:
        """사용자가 직접 편집할 수 있는 클래스 목록 파일. 기본값은 **DB 옆**이다.

        예전에는 `data/classes.txt` 로 박혀 있었다. 프로젝트를 탭으로 나눈 뒤로는
        가시광·열화상·화재가 그 한 파일을 서로 덮어쓰게 된다 — '클래스 추가' 가 여기에
        전체 목록을 적기 때문이다 (2026-09-21 에 DB 옆으로 옮김).
        """
        return resolve(self.get("data.classes_txt", self.db.parent / "classes.txt"))

    @property
    def cameras(self) -> Path:
        """카메라(출처 단위) 등록 파일. 기본값은 DB 옆 — 실제 설치 장소를 담아 git 에 넣지 않는다
        (provenance.py 머리말)."""
        return resolve(self.get("data.cameras", self.db.parent / "cameras.yaml"))

    @property
    def base_weights(self) -> str:
        return pretrained_path(str(self.get("model.base", "yolo26s.pt")))

    @property
    def weights(self) -> Path:
        return resolve(self.get("model.weights", "models/best.pt"))

    def active_weights(self) -> str:
        """자동 라벨링에 쓸 가중치: 학습된 게 있으면 그것, 없으면 사전학습."""
        return str(self.weights) if self.weights.exists() else self.base_weights

    @property
    def device(self) -> Any:
        return self.get("model.device", 0)

    @property
    def imgsz(self) -> int:
        return int(self.get("model.imgsz", 640))
