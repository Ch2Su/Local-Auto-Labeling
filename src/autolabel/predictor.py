"""학습된(또는 사전학습) YOLO 모델로 미라벨 이미지에 자동 라벨을 붙인다."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from . import tiling
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
    skipped_changed: int = 0     # 도는 동안 사람이 바꿔서 그대로 둔 이미지 (store.apply_prediction)
    weights: str = ""
    tiles: dict | None = None    # 타일 추론으로 돌렸으면 그 설정 (tiling.settings)
    tiled: int = 0               # 그중 실제로 타일을 더한 이미지(긴 변이 min_side 를 넘는 것)
    class_iou: float = 0.0       # 통째 모드의 차량 겹침 정리 기준 (tiling.plain_class_iou — 0 이면 끔)
    merged: int = 0              # 그 정리로 뺀 박스 (통째 모드만 센다 — 타일 모드는 tiling.detect 안에서 정리한다)
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"가중치       : {self.weights}",
            f"처리한 이미지 : {self.processed}장",
            f"생성된 박스   : {self.boxes}개 (검출 0인 이미지 {self.empty_images}장)",
        ]
        if self.tiles:
            t = self.tiles
            lines.append(f"타일 추론     : 통째 + {t['grid']}×{t['grid']} 타일(겹침 {t['overlap']}) — 긴 변이 {t['min_side']}px"
                         f" 넘는 {self.tiled}장에 더했고 나머지는 통째만")
        if self.merged:
            lines.append(
                f"  겹친 차량 박스 {self.merged}개를 뺐습니다: 같은 자리(IoU {self.class_iou} 이상)에 클래스만 다른 차량 박스 — "
                "신뢰도 높은 쪽만 남김(predict.class_iou)"
            )
        if self.skipped_changed:
            lines.append(
                f"  건너뛴 이미지 {self.skipped_changed}장: 자동 라벨링을 시작한 뒤 사람이 승인·휴지통·수정해서 "
                "그대로 두었습니다"
            )
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
        tiles: bool = False,
    ) -> PredictReport:
        """미라벨(또는 지정한) 이미지에 추론을 돌려 pending 상태로 만든다.

        **시작할 때 본 뒤로 사람이 손댄 이미지는 덮지 않는다** (2026-11-05). 시작 때의 (상태, 수정 시각)을 떠 두고,
        쓸 때 `store.apply_prediction` 이 같은 트랜잭션에서 다시 견준다 — 예전에는 도는 동안 승인·휴지통·직접 수정한
        이미지가 예측 박스와 '검수 대기' 로 되돌아갔다. 건너뛴 수는 진행 로그와 report.skipped_changed 에 남는다.

        tiles=True 면 타일 추론(`tiling.detect` — 통째 + 긴 변이 큰 이미지에 K×K 타일, 2026-11-10 선택지)으로 돈다.
        작은 객체를 더 잡지만 오검출도 늘고 GPU 시간이 약 2배라 기본은 통째다(tiling 머리말·docs/experiments.md '타일 추론').

        통째에서도 차량 클래스끼리는 클래스가 달라도 `predict.class_iou`(0.6) 이상 겹치면 신뢰도 높은 박스만 남긴다
        (2026-11-20 사용자 결정 — `tiling.cross_class_keep`, 뺀 수는 report.merged). 타일 모드는 detect 안에서
        `predict.tiles.class_iou` 로 같은 일을 한다.
        """
        report = PredictReport(weights=self.weights)
        tile_cfg = tiling.settings(self.cfg) if tiles else None
        report.tiles = tile_cfg
        # 설정이 틀렸으면 대상을 고르거나 모델을 올리기 전에 멈춘다(tiling.settings 와 같이)
        class_iou = 0.0 if tile_cfg else tiling.plain_class_iou(self.cfg)
        report.class_iou = class_iou

        # 시작 때의 행. 모델을 올리는 몇 초 사이에 사람이 바꾼 것도 놓치지 않게 대상을 고르는 같은 순간에 뜬다.
        rows = {r["id"]: r for r in self.store.list_images(status="all", limit=10**9)}
        if image_ids is None:
            targets = [i for i, r in rows.items() if r["status"] == "unlabeled"]
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
        conf = float(self.cfg.get("predict.conf", 0.25))
        iou = float(self.cfg.get("predict.iou", 0.6))
        max_det = int(self.cfg.get("predict.max_det", 100))
        # 추론 해상도는 학습 해상도와 달라도 된다. 최적값은 모델마다 달라서
        # yolo26n 은 960 이(0.705 -> 0.719), yolo26s 는 640 이 가장 나았다
        # (config.yaml 의 predict.imgsz 주석). 지정하지 않으면 학습과 같은 값.
        imgsz = int(self.cfg.get("predict.imgsz") or self.cfg.imgsz)
        version = Path(self.weights).name

        imread = None
        if tile_cfg:
            import numpy as np
            from ultralytics.utils.patches import imread   # Ultralytics 가 경로로 읽을 때와 같은 함수(같은 방향)

            t = tile_cfg
            print(f"타일 추론: 통째 + {t['grid']}×{t['grid']} 타일(겹침 {t['overlap']}, 타일↔통째 NMS {t['cross_iou']})"
                  f" — 긴 변이 {t['min_side']}px 넘는 이미지에만 타일을 더한다"
                  + (f" · 차량끼리는 클래스가 달라도 IoU {t['class_iou']} 이상이면 하나만" if t["class_iou"] > 0 else ""),
                  flush=True)
            # 빈 이미지로 예열한다 — 성분마다 낮은 기준(RAW_*)으로 원검출을 받아 첫 추론이 CUDA 초기화로 길어지면
            # NMS 시간 제한에 걸려 검출이 잘릴 수 있다(LiveDetector·test_tiles 와 같은 이유)
            model.predict(np.zeros((64, 64, 3), np.uint8), imgsz=imgsz, device=self.cfg.device, verbose=False)
            version += f"+tiles{t['grid']}"
        # 통째의 차량 겹침 정리에 쓸 묶음 — 모델의 클래스 이름 가운데 tiling.VEHICLES 에 든 번호. 프로젝트 클래스로 옮겨지는
        # 것만 넣는다: 아래에서 버려질(dropped_unknown_class) 박스가 남을 박스를 지우지 않게
        group = frozenset(i for i in tiling.vehicle_ids(model.names) if i in cmap) if class_iou > 0 else frozenset()
        # 타일 모드의 차량 겹침 정리(tiling.detect 안)에도 같은 규칙을 준다 — 프로젝트 클래스로 옮겨지는 이름만 넘긴다
        # (2026-11-25, Gemini 리뷰 묶음 6: 전에는 타일 쪽만 모델의 차량 클래스를 전부 묶어, 저장 단계에서 버려질 박스가
        # 같은 자리의 남을 박스를 먼저 지울 수 있었다)
        names = model.names
        mapped_names = {int(i): n for i, n in (names.items() if hasattr(names, "items") else enumerate(names or []))
                        if int(i) in cmap}
        if group:
            print(f"차량 겹침 정리: 차량 클래스끼리는 클래스가 달라도 IoU {class_iou} 이상 겹치면 신뢰도 높은 하나만 남긴다"
                  " (predict.class_iou)", flush=True)

        total = len(targets)

        def changed(path: Path) -> None:
            report.skipped_changed += 1
            print(f"\n[건너뜀] {path.name}: 자동 라벨링을 시작한 뒤 사람이 바꿔서(승인·휴지통·수정) 그대로 둡니다.",
                  flush=True)

        for n, image_id in enumerate(targets, 1):
            row = rows.get(image_id) or self.store.get_image(image_id)
            if row is None:
                continue
            expect = (row["status"], row["updated_at"])
            path = Path(row["path"])
            if progress:
                progress(n, total, path.name)
            # 추론 전에 한 번 본다 — 이미 바뀐 이미지에 GPU 를 쓰지 않고, 정리(tidy)로 옮겨진 승인 이미지를
            # '파일 없음' 오류로 세지 않게. 덮을지 말지는 아래 apply_prediction 이 쓰기와 함께 다시 정한다.
            now = self.store.get_image(image_id)
            if now is None or (now["status"], now["updated_at"]) != expect:
                changed(path)            # 지워졌거나(휴지통 비우기) 바뀌었다
                continue
            if not path.exists():
                report.errors.append(f"파일 없음: {path}")
                continue
            # 검출: (cx, cy, w, h 정규화, 모델 클래스, 신뢰도) 목록
            try:
                if tile_cfg:
                    img = imread(str(path))
                    if img is None:
                        report.errors.append(f"{path.name}: 그림을 읽지 못함")
                        continue
                    det, used = tiling.detect(model, img, imgsz=imgsz, device=self.cfg.device, conf=conf, iou=iou,
                                              max_det=max_det, names=mapped_names, **tile_cfg)
                    ih, iw = img.shape[:2]
                    found = [((x1 + x2) / 2 / iw, (y1 + y2) / 2 / ih, (x2 - x1) / iw, (y2 - y1) / ih, c, cf)
                             for x1, y1, x2, y2, cf, c, *_ in det.tolist()]
                    report.tiled += int(used)
                else:
                    results = model.predict(source=str(path), conf=conf, iou=iou, max_det=max_det, imgsz=imgsz,
                                            device=self.cfg.device, verbose=False)
                    found = []
                    for res in results:
                        if res.boxes is None:
                            continue
                        cls_l, conf_l = res.boxes.cls.tolist(), res.boxes.conf.tolist()
                        keep = tiling.cross_class_keep(res.boxes.xyxy.tolist(), conf_l, cls_l, group, class_iou)
                        report.merged += int(len(keep) - keep.sum())
                        found += [(cx, cy, w, h, c, cf)
                                  for (cx, cy, w, h), c, cf, k in zip(res.boxes.xywhn.tolist(), cls_l, conf_l, keep) if k]
            except Exception as exc:
                report.errors.append(f"{path.name}: {exc}")
                continue

            boxes: list[Box] = []
            for cx, cy, w, h, c, cf in found:
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
            if not self.store.apply_prediction(
                image_id, boxes, expect=expect, uncertainty=score,
                model_version=version,
            ):
                changed(path)            # 추론하는 사이에 사람이 바꿨다
                continue

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
    모델을 다시 돌리지 않아도 되게. 자동 라벨링의 차량 겹침 정리(`predict.class_iou`, 2026-11-20)는
    걸지 않는다 — 모델이 낸 그대로 견준다(같은 차에 car·responder 가 겹쳐 나오면 둘 다 보인다).

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

        with self._lock:
            # 장치는 잠금을 쥔 뒤에 정한다 (2026-11-05). 밖에서 정하면 앞 요청을 기다리는 사이 학습이 시작돼
            # free_gpu() 가 모델을 내려도, 깨어난 요청이 예전 판단대로 GPU 에 다시 올렸다.
            device = "cpu" if gpu_busy() else self.cfg.device
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
