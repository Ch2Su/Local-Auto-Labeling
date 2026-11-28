"""프로그램·리소스·사용자 데이터의 자리 (paths.py — 2026-11-28, 포터블 앱 1단계 docs/portable/m01-survey.md).

이 단계는 동작을 바꾸지 않는다 — 개발 모드(이 저장소에서 돌림)에서는 사용자 데이터·소스 저장소가 저장소 루트, 리소스가
`src/autolabel` 로 그 전의 `ROOT` 와 같아야 한다. 배포판(얼린 실행) 쪽은 아직 패키징을 해 보지 않은 설계라 고르는 규칙만 본다 —
실행 파일·번들 자리·OS·환경 변수를 인자로 흉내 낸다(파일은 포터블 표식 하나만 실제로 만든다).
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

from autolabel import config, export
from autolabel.paths import APP, ENV_USER_DATA, PORTABLE_FLAG, AppPaths, default_user_data_dir
from helpers import ROOT


# ------------------------------------------------------------------ 개발 모드 — 그 전과 같은 값

def test_dev_mode_keeps_every_place_where_it_was():
    p = AppPaths.discover(environ={})
    assert (p.install_dir, p.user_data_dir, p.source_dir) == (ROOT, ROOT, ROOT)
    assert p.resource_dir == ROOT / "src" / "autolabel" and not p.frozen and not p.portable


def test_config_and_server_read_their_places_from_here():
    """설정 파일·상대경로의 기준·사전학습 가중치·화면 파일이 이 한 곳을 거친다 — 값은 그 전과 같다(개발 모드)."""
    from autolabel import server

    assert config.ROOT == APP.user_data_dir and config.DEFAULT_CONFIG == APP.user_data_dir / "config.yaml"
    assert config.resolve("data/x.db") == APP.user_data_dir / "data" / "x.db"
    assert config.resolve(APP.user_data_dir / "abs") == APP.user_data_dir / "abs"            # 절대경로는 그대로
    assert server.WEB_DIR == APP.resource_dir / "web" and (server.WEB_DIR / "index.html").is_file()
    if APP.user_data_dir == ROOT:           # LAL_USER_DATA 를 두고 돌린 것이 아니면 — 그 전의 ROOT 그대로
        assert Path(config.pretrained_path("yolo26n.pt")) == ROOT / "weights" / "yolo26n.pt"


def test_only_paths_py_works_out_where_things_are_from_its_own_file():
    """패키지에서 `__file__` 로 자리를 구하는 곳은 paths.py 하나다 — 다른 모듈이 저장소 루트를 따로 구하면 배포판에서 어긋난다.
    `wsl_train.py` 는 WSL 스크립트(저장소의 wsl/)를 찾는다 — 2단계(WSL 을 선택 기능으로)에서 옮길 곳이라 지금은 둔다."""
    allowed = {"paths.py", "wsl_train.py"}
    found = sorted(p.name for p in (ROOT / "src" / "autolabel").glob("*.py")
                   if p.name not in allowed and re.search(r"\b__file__\b", p.read_text(encoding="utf-8")))
    assert found == []


# ------------------------------------------------------------------ 사용자 데이터를 다른 곳으로

def test_an_environment_variable_moves_only_the_user_data(tmp_path):
    p = AppPaths.discover(environ={ENV_USER_DATA: str(tmp_path / "data root")})
    assert p.user_data_dir == (tmp_path / "data root").resolve()
    assert (p.install_dir, p.resource_dir, p.source_dir) == (ROOT, ROOT / "src" / "autolabel", ROOT)
    home = tmp_path / "home"
    assert AppPaths.discover(environ={ENV_USER_DATA: "~/lal"}, home=home).user_data_dir == (home / "lal").resolve()
    assert AppPaths.discover(environ={ENV_USER_DATA: "   "}).user_data_dir == ROOT        # 빈 값은 없는 것


# ------------------------------------------------------------------ 배포판 (얼린 실행) — 고르는 규칙

@pytest.fixture
def bundle(tmp_path):
    """LocalAutoLabeling/App/LocalAutoLabeling.exe 와 번들 폴더 — 인수인계 문서 3.2절의 꼴."""
    root = tmp_path / "LocalAutoLabeling"
    app = root / "App"
    app.mkdir(parents=True)
    exe = app / "LocalAutoLabeling.exe"
    exe.write_bytes(b"")
    return root, exe, app / "_internal"


def test_a_portable_folder_keeps_its_data_beside_the_app(bundle):
    root, exe, internal = bundle
    (root / PORTABLE_FLAG).write_text("", encoding="utf-8")
    p = AppPaths.discover(environ={}, frozen=True, executable=exe, bundle_dir=internal, platform="win32")
    assert p.user_data_dir == root / "UserData" and p.portable and p.frozen
    assert p.install_dir == exe.parent.resolve() and p.resource_dir == internal / "autolabel"
    assert p.source_dir is None                                   # 배포판에는 소스 저장소가 없다


def test_without_the_flag_the_data_goes_to_the_os_place(bundle, tmp_path):
    _, exe, internal = bundle
    home = tmp_path / "home"
    win = AppPaths.discover(environ={"LOCALAPPDATA": str(tmp_path / "Local")}, frozen=True, executable=exe,
                            bundle_dir=internal, platform="win32", home=home)
    assert win.user_data_dir == tmp_path / "Local" / "LocalAutoLabeling" and not win.portable
    mac = AppPaths.discover(environ={}, frozen=True, executable=exe, bundle_dir=internal, platform="darwin", home=home)
    assert mac.user_data_dir == home / "Library" / "Application Support" / "LocalAutoLabeling"


@pytest.mark.parametrize("plat, env, expect", [
    ("win32", {"LOCALAPPDATA": "{t}/L"}, "{t}/L/LocalAutoLabeling"),
    ("win32", {}, "{h}/AppData/Local/LocalAutoLabeling"),
    ("darwin", {}, "{h}/Library/Application Support/LocalAutoLabeling"),
    ("linux", {"XDG_DATA_HOME": "{t}/xdg"}, "{t}/xdg/local-auto-labeling"),
    ("linux", {}, "{h}/.local/share/local-auto-labeling"),
    ("linux", {"XDG_DATA_HOME": "  "}, "{h}/.local/share/local-auto-labeling"),
])
def test_os_default_places(tmp_path, plat, env, expect):
    home = tmp_path / "home"
    fill = lambda s: s.format(t=tmp_path.as_posix(), h=home.as_posix())
    got = default_user_data_dir(platform=plat, environ={k: fill(v) for k, v in env.items()}, home=home)
    assert got == Path(fill(expect))


def test_the_environment_variable_wins_over_the_portable_flag(bundle, tmp_path):
    root, exe, internal = bundle
    (root / PORTABLE_FLAG).write_text("", encoding="utf-8")
    p = AppPaths.discover(environ={ENV_USER_DATA: str(tmp_path / "elsewhere")}, frozen=True, executable=exe,
                          bundle_dir=internal)
    assert p.user_data_dir == (tmp_path / "elsewhere").resolve() and not p.portable


# ------------------------------------------------------------------ 소스 저장소에 매인 동작

def test_export_guards_git_only_inside_the_source_repository(cfg, tmp_path, monkeypatch):
    """내보낸 폴더에 `.gitignore` 를 두는 것은 소스 저장소 안일 때만이다 — 배포판(소스 저장소 없음)에서는 두지 않는다."""
    inside = tmp_path / "repo" / "export" / "a"
    monkeypatch.setattr(export, "APP", dataclasses.replace(APP, source_dir=tmp_path / "repo"))
    export._prepare(inside, cfg)
    assert (inside / ".gitignore").is_file()
    outside = tmp_path / "out" / "b"
    export._prepare(outside, cfg)
    assert not (outside / ".gitignore").exists()
    monkeypatch.setattr(export, "APP", dataclasses.replace(APP, source_dir=None))
    frozen = tmp_path / "repo" / "export" / "c"
    export._prepare(frozen, cfg)
    assert not (frozen / ".gitignore").exists()
