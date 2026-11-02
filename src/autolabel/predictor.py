"""학습된(또는 사전학습) YOLO 모델로 미라벨 이미지에 자동 라벨을 붙인다."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from .config import Config
from .scoring import image_uncertainty
from .store import Store
from .yolo_io import Box

ProgressFn = Callable[[int, int, str], None]


@dataclass
class PredictReport:
    processed: int = 0
    boxes: int = 0
    empty_images: int = 0
    dropped_unknown_class: int = 0
    weights: str = ""
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"가중치       : {self.weights}",
            f"처리한 이미지 : {self.processed}장",
            f"생성된 박스   : {self.boxes}개 (검출 0인 이미지 {self.empty_images}장)",
        ]
        if self.dropped_unknown_class:
            lines.append(
                f"  버린 검출 {self.dropped_unknown_class}개: "
                "모델 클래스명이 이 프로젝트의 클래스 목록에 없음"
            )
        if self.errors:
            lines.append(f"  오류 {len(self.errors)}건: {self.errors[:3]}")
        return "\n".join(lines)


def build_class_map(model_names: dict[int, str], our_classes: list[str]) -> dict[int, int]:
    """모델의 클래스 인덱스를 이 프로젝트의 클래스 인덱스로 이름 기준 매핑.

    학습을 마친 모델이면 이름이 그대로 일치한다. COCO 사전학습 가중치로
    콜드스타트하는 경우에는 이름이 겹치는 클래스만 살아남는다.
    """
    lookup = {name.strip().lower(): i for i, name in enumerate(our_classes)}
    return {
        int(midx): lookup[str(mname).strip().lower()]
        for midx, mname in model_names.items()
        if str(mname).strip().lower() in lookup
    }


class AutoLabeler:
    def __init__(self, cfg: Config, store: Store):
        self.cfg = cfg
        self.store = store
        self.weights = cfg.active_weights()
        self._model = None

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO  # 임포트가 느려서 실제 사용 시점에 로드

            self._model = YOLO(self.weights)
        return self._model

    def run(
        self,
        image_ids: Iterable[int] | None = None,
        *,
        progress: ProgressFn | None = None,
    ) -> PredictReport:
        """미라벨(또는 지정한) 이미지에 추론을 돌려 pending 상태로 만든다."""
        report = PredictReport(weights=self.weights)

        if image_ids is None:
            targets = [r["id"] for r in self.store.list_images(status="unlabeled", limit=10**9)]
        else:
            targets = list(image_ids)
        if not targets:
            return report

        our_classes = self.store.get_classes()
        if not our_classes:
            raise RuntimeError(
                "클래스 목록이 비어 있습니다. config 의 classes 또는 "
                f"{self.cfg.classes_txt} 를 채우고 'scan'을 먼저 실행하세요."
            )

        model = self._load()
        cmap = build_class_map(model.names, our_classes)
        if not cmap:
            raise RuntimeError(
                f"모델 클래스({list(model.names.values())[:10]})와 프로젝트 클래스"
                f"({our_classes[:10]})가 하나도 겹치지 않습니다. "
                "시드 데이터로 먼저 'train'을 실행하세요."
            )

        low = float(self.cfg.get("review.low_band", 0.30))
        high = float(self.cfg.get("review.high_band", 0.80))
        empty_score = float(self.cfg.get("review.empty_score", 0.60))

        rows = {r["id"]: r for r in self.store.list_images(status="all", limit=10**9)}
        total = len(targets)

        for n, image_id in enumerate(targets, 1):
            row = rows.get(image_id) or self.store.get_image(image_id)
            if row is None:
                continue
            path = Path(row["path"])
            if progress:
                progress(n, total, path.name)
            if not path.exists():
                report.errors.append(f"파일 없음: {path}")
                continue
            try:
                results = model.predict(
                    source=str(path),
                    conf=float(self.cfg.get("predict.conf", 0.25)),
                    iou=float(self.cfg.get("predict.iou", 0.6)),
                    max_det=int(self.cfg.get("predict.max_det", 100)),
                    # 추론 해상도는 학습 해상도와 달라도 된다. 최적값은 모델마다 달라서
                    # yolo26n 은 960 이(0.705 -> 0.719), yolo26s 는 640 이 가장 나았다
                    # (config.yaml 의 predict.imgsz 주석). 지정하지 않으면 학습과 같은 값.
                    imgsz=int(self.cfg.get("predict.imgsz") or self.cfg.imgsz),
                    device=self.cfg.device,
                    verbose=False,
                )
            except Exception as exc:
                report.errors.append(f"{path.name}: {exc}")
                continue

            boxes: list[Box] = []
            for res in results:
                if res.boxes is None:
                    continue
                xywhn = res.boxes.xywhn.tolist()
                clss = res.boxes.cls.tolist()
                confs = res.boxes.conf.tolist()
                for (cx, cy, w, h), c, cf in zip(xywhn, clss, confs):
                    mapped = cmap.get(int(c))
                    if mapped is None:
                        report.dropped_unknown_class += 1
                        continue
                    box = Box(mapped, cx, cy, w, h, float(cf), "auto").clamped()
                    if box.is_valid():
                        boxes.append(box)

            score = image_uncertainty(
                [b.conf for b in boxes], low=low, high=high, empty=empty_score
            )
            self.store.set_boxes(
                image_id, boxes, uncertainty=score, model_version=Path(self.weights).name,
                source="predict",
            )
            self.store.set_status(image_id, "pending")

            report.processed += 1
            report.boxes += len(boxes)
            if not boxes:
                report.empty_images += 1

        return report


# 프로세스에 올라와 있는 '예측 비교' 모델 하나. 프로젝트 탭마다 LiveDetector 가 있어도
# 메모리에는 하나만 둔다 (segmenter._active 와 같은 이유).
_live_active: "LiveDetector | None" = None


class LiveDetector:
    """검수 UI 의 '예측 비교'(P) — 지금 보는 이미지 한 장에 운영 가중치를 돌린다 (2026-09-23).

    자동 라벨링과 같은 가중치·해상도·NMS 를 쓰지만 **DB 에는 쓰지 않는다.** 라벨과 어긋나는
    곳을 고르는 비교는 브라우저가 한다(app.js `computeFindings`) — 사람이 박스를 고칠 때마다
    모델을 다시 돌리지 않아도 되게.

    - 다른 프로젝트에서 학습·자동 라벨링이 GPU 를 쓰는 중이면 **CPU 로** 돈다(`jobs.gpu_busy`).
      Ultralytics 는 첫 추론 때 장치를 정하고 바꾸지 않으므로, 장치가 바뀌면 다시 올린다.
    - 가중치 파일이 바뀌면(학습이 끝나 best.pt 가 교체되면) 다시 올린다.
    - 학습을 시작할 때 `release()` 로 내린다 (SAM 과 같다).
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._model = None
        self._key: tuple | None = None
        self._lock = threading.Lock()

    def _load(self, device):
        global _live_active
        w = self.cfg.active_weights()
        p = Path(w)
        key = (w, p.stat().st_mtime if p.exists() else 0.0, str(device))
        if self._model is None or self._key != key:
            if _live_active is not None and _live_active is not self:
                _live_active.release()
            import numpy as np
            from ultralytics import YOLO

            self._model = None
            model = YOLO(w)
            # 빈 이미지로 한 번 예열한다. 첫 추론은 CUDA 초기화로 10초 넘게 걸리는데, 그 안에서
            # NMS 가 시간 제한(약 2초)을 넘기면 Ultralytics 가 **검출을 도중에 자르고** 경고만
            # 낸다(2026-09-23 시험에서 'NMS time limit exceeded' 를 봤다). 실제 이미지는 예열 뒤에.
            model.predict(np.zeros((64, 64, 3), dtype=np.uint8), device=device, verbose=False)
            self._model = model
            self._key = key
        _live_active = self
        return self._model

    def release(self) -> None:
        global _live_active
        if _live_active is self:
            _live_active = None
        with self._lock:
            self._model = None
            self._key = None
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    def predict(self, path: Path, classes: list[str]) -> dict:
        """이미지 한 장의 예측. 좌표는 라벨과 같은 정규화 (cx, cy, w, h) — EXIF 회전을 적용한 화면 기준."""
        from .jobs import gpu_busy

        device = "cpu" if gpu_busy() else self.cfg.device
        with self._lock:
            model = self._load(device)
            t0 = time.perf_counter()
            res = model.predict(
                source=str(path),
                conf=float(self.cfg.get("predict.conf", 0.25)),
                iou=float(self.cfg.get("predict.iou", 0.6)),
                max_det=int(self.cfg.get("predict.max_det", 100)),
                imgsz=int(self.cfg.get("predict.imgsz") or self.cfg.imgsz),
                device=device,
                verbose=False,
            )[0]
            ms = (time.perf_counter() - t0) * 1000
            names = dict(model.names)
            weights = Path(self._key[0]).name

        cmap = build_class_map(names, classes)
        preds = []
        if res.boxes is not None:
            for (cx, cy, w, h), c, cf in zip(res.boxes.xywhn.tolist(), res.boxes.cls.tolist(),
                                              res.boxes.conf.tolist()):
                mapped = cmap.get(int(c))
                if mapped is None:
                    continue
                b = Box(mapped, cx, cy, w, h, float(cf), "auto").clamped()
                if b.is_valid():
                    preds.append({"cls": b.cls, "cx": b.cx, "cy": b.cy, "w": b.w, "h": b.h,
                                  "conf": round(b.conf, 4)})
        preds.sort(key=lambda p: -p["conf"])
        known = sorted(set(cmap.values()))
        return {
            "weights": weights,
            "device": str(device),
            "ms": round(ms, 1),
            # 이 모델이 아는 프로젝트 클래스. 모르는 클래스의 라벨은 비교하지 않는다
            # (예: visible-3 은 pickup 을 모른다 — 그 라벨을 '모델이 못 봄' 으로 띄우면 잡음이다).
            "model_classes": known,
            "unknown_classes": [c for i, c in enumerate(classes) if i not in known],
            "preds": preds,
        }
