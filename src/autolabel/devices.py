"""학습·자동 라벨링·예측 비교·SAM 이 쓸 장치 (2026-11-28 — 포터블 앱 3단계, docs/portable/m01-survey.md 7절).

설정 `model.device` 는 그 전처럼 Ultralytics 가 받는 꼴을 그대로 넘긴다 — `0`·`"0,1"`·`"cuda:1"`·`"cpu"`·`"mps"`. 여기에
**`auto`** 를 더했다: 쓸 수 있는 것을 CUDA → Apple MPS → CPU 순으로 고른다. **설정에 키가 없으면 `auto`** 다(그 전에는 `0` —
CUDA 가 없는 PC 에서는 Ultralytics 가 오류를 냈다). 이 저장소의 `config.yaml` 은 `0` 을 적어 두어 이 PC 의 동작은 그대로다.

지키는 것(인수인계 문서 6.2절):
- **사람이 적은 장치는 바꾸지 않는다** — `0` 이 적혀 있는데 CUDA 가 없으면 Ultralytics 의 오류가 그대로 난다(몰래 CPU 로
  돌리지 않는다). `auto` 만 고른다.
- 학습 중 메모리가 모자라도 CPU 로 옮기지 않는다 — 여기는 시작할 때 한 번 고를 뿐이다.
- GPU 작업이 도는 동안 예측 비교·SAM 이 CPU 로 가는 것(`jobs.gpu_busy`)은 그 전 그대로 부르는 쪽이 정한다.

`torch` 는 고를 때(`auto` 일 때)만 불러 온다 — 설정을 읽는 것만으로 무거운 모듈이 올라오지 않게(`tests/test_devices.py`).
MPS 에서 SAM·타일 추론이 다 되는지는 확인하지 않았다(macOS 실기기에서 — 계획의 8단계).
"""

from __future__ import annotations

from typing import Any

AUTO = "auto"


def is_auto(spec: Any) -> bool:
    return isinstance(spec, str) and spec.strip().lower() == AUTO


def resolve_device(spec: Any) -> Any:
    """설정의 장치 → Ultralytics 에 넘길 값. `auto` 만 고르고 나머지는 적힌 그대로."""
    return auto_device() if is_auto(spec) else spec


def auto_device() -> Any:
    """CUDA(첫 GPU — 그 전 기본값 `0` 과 같은 꼴) → MPS → CPU."""
    import torch

    if torch.cuda.is_available():
        return 0
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"
