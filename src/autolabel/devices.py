"""학습·자동 라벨링·예측 비교·SAM 이 쓸 장치 (2026-11-28 — 포터블 앱 3단계, docs/portable/m01-survey.md 7절).

설정 `model.device` 는 그 전처럼 Ultralytics 가 받는 꼴을 그대로 넘긴다 — `0`·`"0,1"`·`"cuda:1"`·`"cpu"`·`"mps"`. 여기에
**`auto`** 를 더했다: CUDA 가 있으면 첫 GPU, 없으면 CPU. **설정에 키가 없으면 `auto`** 다(그 전에는 `0` — CUDA 가 없는 PC 에서는
Ultralytics 가 오류를 냈다). 이 저장소의 `config.yaml` 은 `0` 을 적어 두어 이 PC 의 동작은 그대로다.

**Apple GPU(MPS)는 `auto` 가 고르지 않는다 (2026-11-28 사용자 결정 — 'CPU 로 바꾸기').** 처음에는 CUDA → MPS → CPU 였는데, MPS 로
추론·학습·SAM 이 되는지 볼 실제 Mac 이 없고 GitHub 의 macOS 러너도 GPU 가 가상 장치라 MPS 를 못 쓴다(8단계 CI). 확인하지 않은
길을 남의 PC 에서 기본으로 쓰게 하지 않는다 — Mac 은 CPU 로 돌고(느리다), 쓰려는 사람은 `mps` 를 적는다. 그때 MPS 에 없는 연산이
멈추지 않고 CPU 로 넘어가게 macOS 에서는 `PYTORCH_ENABLE_MPS_FALLBACK=1` 을 기본으로 둔다(`autolabel/__init__.py`). Mac 에서
`auto` 가 CPU 를 고르면 그 사실과 `mps` 를 쓰는 법을 한 번 알린다(`_mac_note`). MPS 를 확인할 Mac 이 생기면 이 결정을 다시 본다.

지키는 것(인수인계 문서 6.2절):
- **사람이 적은 장치는 바꾸지 않는다** — `0` 이 적혀 있는데 CUDA 가 없으면 Ultralytics 의 오류가 그대로 난다(몰래 CPU 로
  돌리지 않는다). `auto` 만 고른다.
- 학습 중 메모리가 모자라도 CPU 로 옮기지 않는다 — 여기는 시작할 때 한 번 고를 뿐이다.
- GPU 작업이 도는 동안 예측 비교·SAM 이 CPU 로 가는 것(`jobs.gpu_busy`)은 그 전 그대로 부르는 쪽이 정한다.

`torch` 는 고를 때(`auto` 일 때)만 불러 온다 — 설정을 읽는 것만으로 무거운 모듈이 올라오지 않게(`tests/test_devices.py`).
`auto` 는 MPS 를 묻지도 않는다(`torch.backends.mps.is_available()` 을 부르지 않는다 — 가상 GPU 의 Mac 에서는 그것만으로 경고가 났다).
"""

from __future__ import annotations

import sys
from typing import Any

AUTO = "auto"
MAC_NOTE = ("[장치] auto → CPU. Apple GPU(MPS)는 확인하지 못해 auto 가 고르지 않습니다 — 쓰려면 config.yaml 의 "
            "model.device 를 mps 로 적으세요(MPS 에 없는 연산은 CPU 로 넘깁니다).")
_noted = False


def is_auto(spec: Any) -> bool:
    return isinstance(spec, str) and spec.strip().lower() == AUTO


def resolve_device(spec: Any) -> Any:
    """설정의 장치 → Ultralytics 에 넘길 값. `auto` 만 고르고 나머지는 적힌 그대로."""
    return auto_device() if is_auto(spec) else spec


def auto_device() -> Any:
    """CUDA(첫 GPU — 그 전 기본값 `0` 과 같은 꼴) → CPU. MPS 는 고르지 않는다(머리말)."""
    import torch

    if torch.cuda.is_available():
        return 0
    _mac_note()
    return "cpu"


def _mac_note() -> None:
    """Mac 에서 auto 가 CPU 를 골랐으면 한 번만 알린다 — Mac 쓰는 사람이 `mps` 를 쓸 수 있다는 것을 알게."""
    global _noted
    if sys.platform == "darwin" and not _noted:
        _noted = True
        print(MAC_NOTE)
