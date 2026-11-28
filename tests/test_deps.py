"""의존성 범위·잠금 파일·import 가 서로 맞는지 (pyproject.toml·locks/·requirements.txt — 2026-11-28 포터블 앱 7단계).

네트워크도 설치도 쓰지 않는다 — 파일을 읽어 견주기만 한다(세 OS 의 CI 에서도 같게 돈다). 범위를 바꾸면 `python locks/compile.py`
로 잠금을 다시 만들고 이것을 돌린다(locks/README.md).
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path

import pytest

from helpers import ROOT

specifiers = pytest.importorskip("packaging.specifiers")
requirements = pytest.importorskip("packaging.requirements")

PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
LOCKS = sorted((ROOT / "locks").glob("*.txt"))
# import 이름 → 배포 이름 (같으면 적지 않는다). starlette 는 fastapi 가 데려온다(fastapi 의 일부로 쓴다).
DIST = {"PIL": "pillow", "cv2": "opencv-python", "yaml": "pyyaml", "starlette": "fastapi", "imageio_ffmpeg": "imageio-ffmpeg"}
# 일부러 범위에 넣지 않은 도구 의존성 — 그 도구가 다른 가상 환경에서 돈다(tools/ovd_propose.py 머리말)
ELSEWHERE = {"transformers"}


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def reqs(items) -> dict:
    out = {}
    for s in items:
        r = requirements.Requirement(s)
        out[norm(r.name)] = r
    return out


DEPS = reqs(PYPROJECT["project"]["dependencies"])
EXTRAS = {k: reqs(v) for k, v in PYPROJECT["project"]["optional-dependencies"].items()}


def pins(path: Path) -> dict[str, str]:
    """잠금(요구 사항) 파일의 `이름==판` — 주석·옵션 줄은 뺀다."""
    out = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)(\[[^\]]*\])?==([^\s;#]+)", ln.strip())
        if m:
            out[norm(m.group(1))] = m.group(3)
    return out


def third_party(files) -> dict[str, set[str]]:
    """파일들이 import 하는 바깥 패키지(맨 위 이름) → 그 파일들."""
    std = set(sys.stdlib_module_names)
    local = {p.stem for p in (ROOT / "tools").glob("*.py")} | {p.stem for p in (ROOT / "wsl").glob("*.py")} \
        | {p.stem for p in (ROOT / "tests").glob("*.py")} | {"autolabel"}
    out: dict[str, set[str]] = {}
    for f in files:
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module] if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module else [])
            for m in names:
                top = m.split(".")[0]
                if top not in std and top not in local:
                    out.setdefault(top, set()).add(f.name)
    return out


def test_what_the_package_imports_is_declared():
    """패키지(src/autolabel)가 import 하는 바깥 패키지는 전부 pyproject 의 dependencies 에 있다."""
    used = third_party(sorted((ROOT / "src" / "autolabel").glob("*.py")))
    assert {"torch", "ultralytics", "cv2", "fastapi", "PIL", "yaml"} <= set(used)      # 검사가 무언가를 보고 있는지
    missing = {m: sorted(f) for m, f in used.items() if norm(DIST.get(m, m)) not in DEPS}
    assert not missing, f"pyproject.toml 의 dependencies 에 없다: {missing}"


def test_what_tests_and_tools_import_is_declared_somewhere():
    """테스트는 dependencies + test 묶음, 도구는 거기에 더해 다른 묶음 중 하나(일부러 뺀 것만 빼고)."""
    declared = set(DEPS) | {n for extra in EXTRAS.values() for n in extra}
    for folder, allowed in (("tests", set(DEPS) | set(EXTRAS["test"])), ("tools", declared), ("wsl", set(DEPS))):
        used = third_party(sorted((ROOT / folder).glob("*.py")))
        missing = {m: sorted(f)[:3] for m, f in used.items()
                   if norm(DIST.get(m, m)) not in allowed and m not in ELSEWHERE}
        assert not missing, f"{folder}/ 가 import 하는데 pyproject.toml 에 없다: {missing}"


@pytest.mark.parametrize("lock", LOCKS, ids=lambda p: p.name)
def test_each_lock_pins_every_dependency_within_its_range(lock):
    got = pins(lock)
    for name, req in {**DEPS, **EXTRAS["test"]}.items():
        assert name in got, f"{lock.name} 에 {name} 이 없다 — python locks/compile.py 로 다시 만들 것"
        assert req.specifier.contains(got[name], prereleases=True), \
            f"{lock.name} 의 {name}=={got[name]} 이 pyproject 범위 '{req.specifier}' 밖이다"


@pytest.mark.parametrize("lock", LOCKS, ids=lambda p: p.name)
def test_each_lock_gets_torch_from_the_right_place(lock):
    """`+cpu`·`+cu134` 빌드는 PyTorch 인덱스에만 있다 — 잠금 머리에 그 인덱스 줄이 있어야 pip 가 찾는다. macOS 는 PyPI 그대로."""
    text = lock.read_text(encoding="utf-8")
    torch = pins(lock)["torch"]
    local = torch.partition("+")[2]
    if local:
        assert f"--extra-index-url https://download.pytorch.org/whl/{local}" in text
        assert pins(lock)["torchvision"].endswith("+" + local)
        assert local in lock.name                                      # 이름이 빌드를 말한다
    else:
        assert "macos" in lock.name and "download.pytorch.org" not in text


@pytest.mark.parametrize("lock", LOCKS, ids=lambda p: p.name)
def test_locks_carry_no_local_paths(lock):
    """저장소는 공개다 — 잠금을 만든 PC 의 자리(임시 폴더의 제약 파일 등)가 적히면 안 된다."""
    text = lock.read_text(encoding="utf-8")
    # 드라이브 글자 자리(C:\ · C:/) — https:// 의 's:/' 는 앞이 글자라 빠진다
    assert not re.search(r"(?<![A-Za-z])[A-Za-z]:[\\/]|/Users/|/home/|AppData|\s-c\s", text), lock.name


def test_the_release_locks_share_the_dev_versions():
    """배포용 잠금(3.14)은 개발 잠금(이 PC 에서 테스트가 도는 판)과 같은 판이다 — 빌드 꼬리(+cpu 등)와 플랫폼에만 있는 패키지 밖."""
    dev = {k: v.partition("+")[0] for k, v in pins(ROOT / "locks" / "win-cu134-py315.txt").items()}
    for lock in LOCKS:
        if lock.name.endswith("py314.txt"):
            diff = {k: (dev[k], v) for k, v in pins(lock).items() if k in dev and v.partition("+")[0] != dev[k]}
            assert not diff, f"{lock.name} 이 개발 잠금과 판이 다르다: {diff}"


def test_the_laptop_requirements_fit_the_ranges():
    """노트북의 requirements.txt(그대로 둔다)도 pyproject 범위 안이다 — 범위의 아래쪽이 그 판이다."""
    got = pins(ROOT / "requirements.txt")
    for name, req in DEPS.items():
        assert name in got and req.specifier.contains(got[name], prereleases=True), (name, got.get(name))


def test_the_web_files_and_the_entry_point_are_declared():
    """화면 파일이 package-data 에 들고(나중에 휠을 만들 때), `labeler` 명령이 가리키는 함수가 있다."""
    globs = PYPROJECT["tool"]["setuptools"]["package-data"]["autolabel"]
    web = ROOT / "src" / "autolabel"
    for f in (web / "web").rglob("*"):
        if f.is_file():
            assert any(f.relative_to(web).match(g) for g in globs), f
    mod, _, func = PYPROJECT["project"]["scripts"]["labeler"].partition(":")
    tree = ast.parse((ROOT / "src" / Path(*mod.split("."))).with_suffix(".py").read_text(encoding="utf-8"))
    assert any(isinstance(n, ast.FunctionDef) and n.name == func for n in tree.body)


def test_ultralytics_is_capped_at_a_checked_version():
    """Ultralytics 는 범위 위쪽이 있어야 한다 — 정지·스냅샷이 내부 순서에 기대서 올리기 전에 소스로 봐야 한다(함정 8)."""
    spec = DEPS["ultralytics"].specifier
    assert any(s.operator in ("<=", "<", "==") for s in spec), spec
