"""SAM 으로 클릭·거친 박스를 '사람이 그린 수준'의 박스로 바꿔 준다.

DB 는 마스크를 저장하지 않고 박스만 들고 있으므로(`store.py` 의 boxes 테이블),
여기서도 마스크는 중간 산물로만 쓰고 **외접 박스만** 돌려준다. 덕분에 스키마를
건드릴 필요가 없고, 나온 박스는 사람이 그린 것과 완전히 같은 취급을 받는다.

## 왜 sam2.1_t 인가 (실측, val 12장 / 객체 61개)

정답 박스에 ±15% 지터를 줘서 '대충 그린 박스'를 흉내내고, 그걸 프롬프트로 넣어
돌아온 마스크의 외접 박스가 정답에 얼마나 가까운지 쟀다.

| | mobile_sam | sam2.1_t |
|---|---|---|
| VRAM | 0.31 GiB | 0.55 GiB |
| 인코더 (이미지당 1회) | 18 ms | 13 ms |
| 박스 프롬프트 | 16 ms | 27 ms |
| 클릭 프롬프트 | 16 ms | 15 ms |
| 거친 박스 -> 다듬은 박스 IoU | 0.758 -> 0.825 | 0.758 -> **0.888** |
| 중심 1클릭만으로 | 0.475 | **0.755** |
| IoU < 0.5 (실패) | 5% | **0%** |

클릭 한 번(0.755)이 ±15% 로 손으로 그린 박스(0.758)와 같은 품질이다. mobile_sam 은
클릭이 쓸 만하지 않아 탈락했다.

## 한계 (알고 쓸 것)

- 위 수치는 **합성 지터**와 **정답 박스의 정중앙 클릭**으로 얻은 것이다. 실제
  사용자는 객체 위 아무 곳이나 찍고, 가늘거나 가려진 객체는 중심이 객체 밖일 수도
  있다. 실사용 품질은 이보다 낮다.
- SAM 은 클릭한 **보이는 조각**을 잡는다. 이 데이터셋의 COCO 박스는 가려져 조각난
  객체를 하나로 묶은 것이라(함정 4), 가려짐이 심하면 SAM 박스가 더 작게 나온다.
- Ultralytics 의 SAM Predictor 는 한 번에 이미지 하나만 물고 있다(`set_image` 가
  단일 이미지를 단정한다). 그래서 캐시 깊이가 1 이고, 전환은 락으로 직렬화한다.
  검수는 한 사람이 한 이미지를 보며 여러 번 찍는 패턴이라 이 깊이로 충분하다.
"""

from __future__ import annotations

import threading
from pathlib import Path

from .config import Config, pretrained_path
from .yolo_io import Box

DEFAULT_MODEL = "sam2.1_t.pt"

# 프로세스에 올라와 있는 SAM 하나. 프로젝트 탭마다 Segmenter 가 있어도 VRAM 에는
# 하나만 둔다 (2026-09-21). `_load` 가 새로 올리기 전에 이전 것을 내린다.
_active: "Segmenter | None" = None


class SegmenterUnavailable(RuntimeError):
    """SAM 을 쓸 수 없는 상태. 호출한 쪽이 503 으로 바꿔 내보낸다."""


class Segmenter:
    """클릭/박스 프롬프트 -> 박스. 모델은 처음 쓸 때 로드한다."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._model = None
        self._pred = None
        self._device: str | None = None               # 지금 올라가 있는 장치
        self._cur: tuple[str, float] | None = None   # (경로, mtime) — 인코딩해 둔 이미지
        self._lock = threading.Lock()
        self._failed: str | None = None              # 로드 실패 사유를 기억해 재시도 폭주를 막는다

    # ------------------------------------------------------------------ 설정

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.get("sam.enabled", True))

    @property
    def model_name(self) -> str:
        # weights/ 에 있으면 거기서 연다. 이름만 주면 Ultralytics 가 실행 폴더에 받아 둔다.
        return pretrained_path(str(self.cfg.get("sam.model") or DEFAULT_MODEL))

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "model": self.model_name,
            "loaded": self._pred is not None,
            "device": self._device,
            "error": self._failed,
        }

    # ------------------------------------------------------------------ 로드

    def _load(self, device):
        """`self._lock` 을 쥔 채로 부른다. device 가 바뀌면 내리고 그 장치에 다시 올린다.

        학습·자동 라벨링이 도는 동안에는 CPU 로 온다(`segment`). 작업을 시작할 때 `free_gpu()`
        가 SAM 을 내려도, 그 뒤 검수 UI 에서 C·R 을 누르면 예전에는 GPU 에 다시 올라가 학습과
        8GB 를 나눠 썼다 — 넘치면 에러 없이 느려지기만 한다(CLAUDE.md 함정 9, 2026-10-25).
        """
        global _active
        device = str(device)
        if self._pred is not None and self._device == device:
            _active = self
            return self._pred
        if self._pred is not None:
            self._unload()
        if not self.enabled:
            raise SegmenterUnavailable(
                "SAM 이 꺼져 있습니다. config.yaml 의 sam.enabled 를 true 로 두세요."
            )
        if self._failed:
            raise SegmenterUnavailable(self._failed)
        # 프로젝트 탭마다 Segmenter 가 따로 있다. 8GB 카드에서 SAM 을 여러 개 물고 있을
        # 이유가 없으므로(하나에 0.55GB) 새로 올리기 전에 다른 프로젝트 것을 내린다.
        if _active is not None and _active is not self:
            _active.release()
        try:
            from ultralytics import SAM  # 임포트가 느려서 실제 사용 시점에 로드

            model = SAM(self.model_name)
            # predictor 는 첫 호출 때 만들어진다. 더미 프롬프트로 준비시킨다.
            # (set_image 만으로는 predictor 가 없어서 AttributeError 가 난다.)
            import numpy as np

            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            # 장치는 predictor 가 처음 만들어질 때 정해지고 바뀌지 않는다 (LiveDetector 와 같다).
            model(source=dummy, bboxes=[[8, 8, 40, 40]], device=device, verbose=False, save=False)
            self._model, self._pred, self._device = model, model.predictor, device
            _active = self
        except Exception as exc:
            self._failed = f"{self.model_name} 를 불러오지 못했습니다: {exc}"
            raise SegmenterUnavailable(self._failed) from exc
        return self._pred

    def _ensure_image(self, path: Path) -> None:
        """이미 인코딩해 둔 이미지면 재사용한다 (인코딩 13ms 를 아낀다)."""
        key = (str(path), path.stat().st_mtime)
        if self._cur == key:
            return
        self._pred.reset_image()
        self._pred.set_image(str(path))
        self._cur = key

    def release(self) -> None:
        """VRAM 을 돌려준다. 학습을 시작하기 전에 부르면 좋다."""
        global _active
        if _active is self:
            _active = None
        with self._lock:
            self._unload()

    def _unload(self) -> None:
        """`self._lock` 을 쥔 채로 부른다 (Lock 은 재진입이 안 된다 — `_load` 가 장치를 바꿀 때)."""
        if self._pred is not None:
            try:
                self._pred.reset_image()
            except Exception:
                pass
        self._model = self._pred = None
        self._device = None
        self._cur = None
        try:
            import torch

            # 내린 뒤에도 memory_allocated 에 32MiB 가 남는데, 파이썬에서 닿는 CUDA 텐서는 0개였다
            # (2026-10-25) — 한 번이라도 GPU 연산을 한 프로세스가 갖는 cuBLAS 작업 공간으로 보인다.
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ------------------------------------------------------------------ 추론

    def segment(
        self,
        path: Path,
        *,
        point: tuple[float, float] | None = None,
        box: tuple[float, float, float, float] | None = None,
        cls: int = 0,
    ) -> tuple[Box, float]:
        """픽셀 좌표 프롬프트 -> 정규화 Box. 점수는 SAM 이 준 마스크 신뢰도.

        point 와 box 를 함께 주면 둘을 같이 프롬프트로 넣는다. 박스가 객체를
        가두고 점이 어느 것인지 가리키므로, 겹친 객체에서 가장 정확하다.
        """
        if point is None and box is None:
            raise ValueError("point 또는 box 중 하나는 있어야 합니다.")
        if not path.exists():
            raise FileNotFoundError(path)

        from .jobs import gpu_busy

        with self._lock:
            # 장치는 잠금을 쥔 뒤에 정한다 (2026-11-05, predictor.LiveDetector 와 같다) — 앞 요청을 기다리는 사이
            # 학습이 시작되면 예전 판단대로 GPU 에 다시 올렸다.
            device = "cpu" if gpu_busy() else self.cfg.device
            self._load(device)
            self._ensure_image(path)
            kw: dict = {}
            if box is not None:
                kw["bboxes"] = [list(box)]
            if point is not None:
                kw["points"] = [list(point)]
                kw["labels"] = [1]
            results = self._pred(**kw)

        if not results or results[0].masks is None or not len(results[0].masks.data):
            raise SegmenterUnavailable("SAM 이 마스크를 만들지 못했습니다.")

        res = results[0]
        mask = res.masks.data[0].cpu().numpy()
        ys, xs = mask.nonzero()
        if not len(xs):
            raise SegmenterUnavailable("마스크가 비어 있습니다.")

        h, w = mask.shape[-2:]
        x1, x2 = float(xs.min()), float(xs.max() + 1)
        y1, y2 = float(ys.min()), float(ys.max() + 1)
        score = 0.0
        try:
            score = float(res.boxes.conf[0]) if res.boxes is not None else 0.0
        except Exception:
            pass

        return (
            Box(
                cls,
                (x1 + x2) / 2 / w,
                (y1 + y2) / 2 / h,
                (x2 - x1) / w,
                (y2 - y1) / h,
                None,          # conf 는 비워 둔다 — 검출 신뢰도가 아니라 마스크 품질이다
                "sam",
            ).clamped(),
            score,
        )
