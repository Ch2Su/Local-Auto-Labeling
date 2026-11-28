"""로컬 환경 자동 이미지 라벨링 파이프라인."""

import os as _os
import sys as _sys

__version__ = "0.1.0"

# Ultralytics 가 없는 패키지를 **지금 쓰는 파이썬 환경에 스스로 설치하지 않게** 한다 (2026-11-25, Gemini 리뷰 묶음 6).
# 그 라이브러리는 `check_requirements` 로 모자란 패키지를 만나면 `pip install` 을 돌린다 — 예: 갈아 끼운 `PIL.Image.open` 이
# 못 알아보는 그림에서 `pi-heif` (yolo_io.plain_image_open 머리말). 이 `.venv` 는 다른 프로젝트도 쓴다.
# Ultralytics 가 import 될 때 읽는 값이라 이 패키지가 처음 import 될 때(그 전에) 둔다. 이미 정해 둔 값은 건드리지 않는다 —
# 자동 설치를 쓰려면 환경 변수 `YOLO_AUTOINSTALL=true` 로 켠다. 이 프로젝트가 쓰는 길(학습·추론·SAM)에 필요한 패키지는
# requirements.txt 에 다 있다. WSL 학습 스크립트(wsl/train_wsl.py)는 이 패키지를 import 하지 않아 해당 없다.
#
# macOS 에서는 Apple GPU(MPS)에 없는 연산을 CPU 로 넘긴다 (2026-11-28 — devices.py 머리말). MPS 는 `auto` 가 고르지 않고
# config 에 `mps` 를 적은 사람만 쓴다(실제 Mac 으로 확인하지 못했다) — 그때 연산 하나가 없다고 멈추지 않게. PyTorch 가 MPS 대체
# 커널을 torch 를 불러 올 때 정하므로 그 전에(여기서) 둔다 — 이 패키지보다 torch 를 먼저 import 한 프로세스에는 안 먹는다.


def _set_env_defaults(environ=None, platform: str | None = None) -> None:
    env = _os.environ if environ is None else environ
    env.setdefault("YOLO_AUTOINSTALL", "false")
    if (_sys.platform if platform is None else platform) == "darwin":
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


_set_env_defaults()
