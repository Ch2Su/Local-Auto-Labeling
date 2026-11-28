# 잠금 파일 (locks/)

2026-11-28, 포터블 앱 7단계([docs/portable/m01-survey.md](../docs/portable/m01-survey.md)). `pyproject.toml` 은 '이 범위면 된다' 는
범위이고, 여기의 잠금 파일은 OS·GPU 마다 **실제로 깔 판**을 하나씩 정한다(간접 의존성까지 58개 남짓). 손으로 고치지 말고
`python locks/compile.py` 로 다시 만든다.

| 파일 | 대상 | Python | 확인한 것 |
|---|---|---|---|
| `win-cu134-py315.txt` | Windows x64 · NVIDIA (CUDA 13.4 빌드) | 3.15 | **이 데스크탑 `.venv` 의 판 그대로**(그 venv 의 `pip freeze` 를 제약으로 골랐다) — 회귀 테스트 809개가 이 판에서 돈다 |
| `win-cpu-py314.txt` | Windows x64 · CPU | 3.14 | 새 가상 환경(3.14.10)에 이것만 깔고 회귀 테스트 801개 통과(2026-11-28, 건너뜀 0) |
| `linux-cpu-py314.txt` | Linux x86_64 (glibc 2.28+) · CPU | 3.14 | GitHub Actions `ubuntu-latest` 에서 807개 통과·건너뜀 0(2026-11-28, 8단계). 그 전에 WSL 의 새 venv 로도 |
| `macos-arm64-py314.txt` | macOS 14+ · Apple Silicon (CPU — MPS 는 적은 사람만) | 3.14 | GitHub Actions `macos-latest` 에서 807개 통과·건너뜀 0(2026-11-28). 그 러너는 MPS 를 못 써(가상 GPU) CPU 로만 — MPS 는 실제 Mac 에서 볼 것 |
| `../requirements.txt` | Windows x64 · NVIDIA (CUDA 13.0 빌드) | 3.14 | 노트북의 개발 환경(직접 쓰는 패키지만 고정, 2026-10-31). 그대로 둔다 |

배포용 잠금(3.14)은 개발 잠금의 판을 제약으로 골라 **같은 판**이다 — 다른 것은 플랫폼에만 있는 패키지(Linux·macOS 의 `uvloop`,
Windows 의 `colorama`)와 PyTorch 빌드(`+cpu` / `+cu134` / macOS 는 PyPI 그대로)뿐이다. 배포 대상 Python 을 3.14 로 둔 것은 잠정이다
(노트북이 3.14 이고 휠이 넓게 있다 — 사용자가 정할 것, m01-survey 8절 2번).

## 깔기

새 가상 환경에서(이 저장소의 `.venv` 는 다른 프로젝트도 빌려 쓰니 거기에 깔지 말 것):

```bash
python -m venv .venv-new            # 그 OS 의 Python 3.14 (데스크탑 잠금은 3.15)
.venv-new/Scripts/python -m pip install -r locks/win-cpu-py314.txt     # macOS·Linux 는 .venv-new/bin/python
.venv-new/Scripts/python -m pip install -e . --no-deps
.venv-new/Scripts/python -m pytest  # 회귀 테스트 (화면 JS 테스트는 node 가 있어야 돈다)
```

- PyTorch 를 받는 곳은 잠금 파일 머리의 `--extra-index-url` 줄이 정한다(`+cpu`·`+cu134`). macOS 는 PyPI 의 torch 다.
- **Mac 은 CPU 로 돈다** — Apple GPU(MPS)는 실제 Mac 으로 확인하지 못해 장치 `auto` 가 고르지 않는다(2026-11-28 사용자 결정, `devices.py`).
  쓰려면 `config.yaml` 의 `model.device` 를 `mps` 로 적는다(확인 안 된 길 — macOS 에서는 MPS 에 없는 연산을 CPU 로 넘기는
  `PYTORCH_ENABLE_MPS_FALLBACK=1` 이 기본이다). 이 저장소의 config.yaml 은 `0`(첫 CUDA GPU)이라, 다른 PC 에서는 `auto` 로 바꿔 쓴다.
- 편집 설치는 `src/local_auto_labeling.egg-info/` 를 만든다(git 은 무시한다 — `.gitignore`).
- **설치는 편집 모드(`-e`)만 된다** — `pip install .` 로 site-packages 에 넣으면 `paths.APP` 이 소스 폴더를 못 찾는다(pyproject.toml 머리말).
- 데스크탑 잠금의 PyYAML 6.0.3 은 Python 3.15 휠이 없어 소스에서 만든다 — MSVC 가 없으면 `PYYAML_FORCE_LIBYAML=0` 을 두고 깐다.
- WSL·Runpod 학습은 이 잠금이 아니라 conda 환경 `yolo_final` 이다(CLAUDE.md 'WSL 학습', docs/runpod.md).
- 해시(`--require-hashes`)는 아직 넣지 않았다.

## 다시 만들기

```bash
.venv/Scripts/python locks/compile.py                    # 배포용 셋 (uv 가 PATH 에 있어야 한다 — 메타데이터만 받고 깔지 않는다)
.venv/Scripts/python locks/compile.py --dev-venv .venv   # + 데스크탑 개발 잠금을 그 venv 에 깔린 판으로 (venv 는 읽기만)
.venv/Scripts/python -m pytest tests/test_deps.py        # 범위·잠금·import 가 서로 맞는지
```

`pyproject.toml` 의 범위를 바꾸거나 `.venv` 의 패키지를 올렸으면 둘 다 돌린다. **Ultralytics 는 범위의 위쪽이 확인한 판(8.4.212)이다** —
올리려면 CLAUDE.md 함정 8 의 순서를 소스로 다시 보고 위쪽을 고칠 것.
