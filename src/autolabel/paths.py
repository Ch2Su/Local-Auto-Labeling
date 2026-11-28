"""프로그램·리소스·사용자 데이터의 자리를 한 곳에서 정한다 (2026-11-28 — 포터블 앱 1단계, docs/portable/m01-survey.md 7절).

그 전에는 `config.py` 가 `ROOT = 저장소 루트` 하나로 셋을 함께 가리켰다 — 설정 파일(`config.yaml`), 설정의 상대경로(`data/…`·
`runs/`·`models/`·`export/`), 사전학습 가중치(`weights/`)는 '사용자 데이터' 이고, 화면 파일(`web/`)은 '프로그램에 딸린 읽기 전용
리소스' 이고, '저장소 안이면 .gitignore 를 둔다' 같은 것은 '소스 저장소' 다. 압축을 풀어 쓰는 배포판에서는 이 셋의 자리가 다르다
(실행 파일 옆 · 번들 내부 · 사용자 데이터 폴더). 여기서 그 셋을 나눠 둔다.

**개발 모드(지금 이 저장소에서 `labeler.py`·`python -m` 으로 돌릴 때)는 셋이 전과 같은 값이다** — 사용자 데이터 = 저장소 루트,
리소스 = `src/autolabel`, 소스 저장소 = 저장소 루트. 그래서 이 단계는 동작을 바꾸지 않는다(`tests/test_paths.py`).

사용자 데이터 자리를 고르는 순서:
1. 환경 변수 `LAL_USER_DATA` 가 있으면 그 폴더 (개발 모드에서도 — 다른 데이터 폴더로 띄워 볼 때).
2. 배포판(PyInstaller 등으로 얼린 실행 — `sys.frozen`)이고 실행 파일 폴더의 한 단계 위에 `portable.flag` 가 있으면 그 옆의
   `UserData/` (인수인계 문서 3.2절의 `LocalAutoLabeling/{App,UserData,portable.flag}`).
3. 배포판인데 표식이 없으면 OS 의 기본 자리(`default_user_data_dir`).
4. 개발 모드면 저장소 루트.

**배포판 쪽(2·3번과 `resource_dir`)은 아직 패키징을 해 보지 않은 설계다** — 번들 안의 리소스 자리(`_MEIPASS/autolabel`)는
패키징 단계(인수인계 문서 Milestone 0.3)에서 실제 빌드로 확인할 것. 여기서는 고르는 규칙만 테스트한다.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

ENV_USER_DATA = "LAL_USER_DATA"
PORTABLE_FLAG = "portable.flag"
PORTABLE_DATA = "UserData"
APP_DIR_NAME = "LocalAutoLabeling"          # Windows·macOS 의 사용자 데이터 폴더 이름
XDG_DIR_NAME = "local-auto-labeling"        # Linux(XDG) 의 폴더 이름 — 소문자·하이픈이 관례

_PACKAGE_DIR = Path(__file__).resolve().parent          # src/autolabel
_SOURCE_DIR = _PACKAGE_DIR.parents[1]                   # 저장소 루트 (개발 모드)


@dataclass(frozen=True)
class AppPaths:
    install_dir: Path            # 실행하는 프로그램이 있는 곳 — 개발 모드 = 저장소 루트, 배포판 = 실행 파일 폴더
    resource_dir: Path           # 프로그램에 딸린 읽기 전용 리소스(`web/`) — 개발 모드 = src/autolabel
    user_data_dir: Path          # 설정·데이터·학습 결과·가중치 캐시의 기준 — 설정의 상대경로는 여기서 푼다
    source_dir: Path | None      # 소스 저장소(git) — 배포판에는 없다
    portable: bool = False       # 포터블 폴더의 UserData/ 를 쓰는가
    frozen: bool = False         # 배포판(얼린 실행)인가

    @classmethod
    def discover(cls, *, environ: Mapping[str, str] | None = None, frozen: bool | None = None,
                 executable: str | Path | None = None, bundle_dir: str | Path | None = None,
                 platform: str | None = None, home: str | Path | None = None) -> AppPaths:
        """지금 실행에 맞는 자리. 인자는 테스트가 흉내 내려는 것이다 — 비우면 지금 프로세스의 값을 쓴다."""
        env = os.environ if environ is None else environ
        if frozen is None:
            frozen = bool(getattr(sys, "frozen", False))
        override = env.get(ENV_USER_DATA, "").strip()

        if not frozen:
            user = _absolute(override, home) if override else _SOURCE_DIR
            return cls(install_dir=_SOURCE_DIR, resource_dir=_PACKAGE_DIR, user_data_dir=user,
                       source_dir=_SOURCE_DIR)

        exe_dir = Path(executable or sys.executable).resolve().parent
        bundle = Path(bundle_dir or getattr(sys, "_MEIPASS", exe_dir))
        portable_root = exe_dir.parent
        if override:
            user, portable = _absolute(override, home), False
        elif (portable_root / PORTABLE_FLAG).is_file():
            user, portable = portable_root / PORTABLE_DATA, True
        else:
            user, portable = default_user_data_dir(platform=platform, environ=env, home=home), False
        return cls(install_dir=exe_dir, resource_dir=bundle / "autolabel", user_data_dir=user, source_dir=None,
                   portable=portable, frozen=True)


def default_user_data_dir(*, platform: str | None = None, environ: Mapping[str, str] | None = None,
                          home: str | Path | None = None) -> Path:
    """포터블 표식이 없는 배포판의 사용자 데이터 자리 — OS 의 관례를 따른다(인수인계 문서 5.2절).

    Windows `%LOCALAPPDATA%\\LocalAutoLabeling`(없으면 `~\\AppData\\Local\\…`), macOS `~/Library/Application Support/…`,
    그 밖(Linux) `$XDG_DATA_HOME/local-auto-labeling`(없으면 `~/.local/share/…`)."""
    plat = sys.platform if platform is None else platform
    env = os.environ if environ is None else environ
    base_home = Path(home) if home is not None else Path.home()
    if plat.startswith("win"):
        local = env.get("LOCALAPPDATA", "").strip()
        return (Path(local) if local else base_home / "AppData" / "Local") / APP_DIR_NAME
    if plat == "darwin":
        return base_home / "Library" / "Application Support" / APP_DIR_NAME
    xdg = env.get("XDG_DATA_HOME", "").strip()
    return (Path(xdg) if xdg else base_home / ".local" / "share") / XDG_DIR_NAME


def _absolute(p: str, home: str | Path | None) -> Path:
    """환경 변수로 받은 자리를 절대경로로 — `~` 는 집 폴더로 푼다(테스트가 준 home 이 있으면 그것)."""
    path = Path(p)
    if p.startswith("~"):
        rest = p[1:].lstrip("/\\")
        path = (Path(home) if home is not None else Path.home()) / rest
    return path.resolve()


APP = AppPaths.discover()
