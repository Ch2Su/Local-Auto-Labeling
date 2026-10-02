"""검수 UI 상단 탭에 뜨는 프로젝트(데이터셋) 목록.

프로젝트 하나 = config 파일 하나다. 데이터 폴더·DB·가중치·학습 결과 폴더가 전부
config 로 갈리므로 여기서는 '어떤 config 를 어떤 이름으로
띄울지'만 정한다. 서버는 프로젝트마다 `create_app(cfg)` 를 만들어 `/p/<name>` 에 붙인다
— 앱이 통째로 따로라 Store·작업큐·SAM 이 섞이지 않는다.

**서버 하나에 '현재 프로젝트' 를 두고 전환하는 방식은 쓰지 않았다.** 검수 UI 는 박스를
0.6초 뒤 자동 저장하므로, 브라우저 탭 두 개를 열어 두면 엉뚱한 프로젝트에 저장될 수 있다.

`projects.yaml` 이 없으면 `config.yaml` 하나짜리 목록으로 돈다(예전과 같은 동작).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import Config, resolve

PROJECTS_YAML = "projects.yaml"
_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")     # 주소에 들어가므로 좁게 받는다


@dataclass
class Project:
    name: str            # 주소에 쓰는 이름 (/p/<name>/)
    title: str           # 탭에 보이는 이름
    config_path: Path
    cfg: Config


def load_projects(path: str | Path | None = None) -> list[Project]:
    """projects.yaml 을 읽어 프로젝트 목록을 만든다. 없으면 config.yaml 하나만.

    config 파일이 없는 항목은 **건너뛰고 경고만 낸다** — 프로젝트 하나를 아직 안 만들었다고
    검수 UI 전체가 안 뜨면 곤란하다.
    """
    p = resolve(path or PROJECTS_YAML)
    if not p.exists():
        cfg_path = resolve("config.yaml")
        return [Project("visible", "가시광", cfg_path, Config.load(cfg_path))]

    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    out: list[Project] = []
    seen: set[str] = set()
    for item in raw.get("projects") or []:
        name = str(item.get("name", "")).strip()
        cfg_rel = str(item.get("config", "")).strip()
        if not _NAME.match(name):
            print(f"[{PROJECTS_YAML}] 이름 '{name}' 은 영소문자·숫자·_- 만 됩니다. 건너뜁니다.")
            continue
        if name in seen:
            print(f"[{PROJECTS_YAML}] 이름 '{name}' 이 중복입니다. 뒤의 것을 건너뜁니다.")
            continue
        cfg_path = resolve(cfg_rel)
        if not cfg_rel or not cfg_path.exists():
            print(f"[{PROJECTS_YAML}] '{name}' 의 설정 파일 {cfg_rel or '(없음)'} 을 찾지 못해 건너뜁니다.")
            continue
        seen.add(name)
        out.append(Project(name, str(item.get("title") or name), cfg_path, Config.load(cfg_path)))

    if not out:
        raise RuntimeError(f"{p} 에서 쓸 수 있는 프로젝트를 하나도 찾지 못했습니다.")
    return out
