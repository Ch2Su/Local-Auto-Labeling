"""잠금 파일 다시 만들기 (2026-11-28, 포터블 앱 7단계 — locks/README.md).

    python locks/compile.py                      # 배포용 잠금(Python 3.14 — Windows CPU·Linux CPU·macOS arm64)만
    python locks/compile.py --dev-venv .venv     # + 이 PC(데스크탑)의 개발 잠금을 그 venv 에 깔린 판으로

`uv`(https://docs.astral.sh/uv/)가 PATH 에 있어야 한다 — 패키지 목록(메타데이터)을 받아 고르기만 하고 아무것도 설치하지 않는다.
배포용 잠금은 개발 잠금(`win-cu134-py315.txt`)의 판을 제약으로 쓴다 — 그 판이 이 PC 에서 테스트를 통과한 판이라서. 그 플랫폼에
같은 판이 없을 때만 다른 판이 들어간다. 개발 잠금은 그 venv 의 `pip freeze` 를 제약으로 써서 깔린 판 그대로 적는다(venv 는
읽기만 한다 — 다른 프로젝트도 쓰는 `.venv` 다). 범위(pyproject.toml)를 바꾸면 이것을 돌리고 `tests/test_deps.py` 를 돌린다.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEV = "win-cu134-py315.txt"
TORCH = "https://download.pytorch.org/whl/{}"

# 파일 이름 → (설명, Python, uv 의 --python-platform, PyTorch 인덱스(없으면 PyPI 만))
TARGETS = {
    "win-cpu-py314.txt": ("Windows x64 · CPU", "3.14", "x86_64-pc-windows-msvc", "cpu"),
    "linux-cpu-py314.txt": ("Linux x86_64 (glibc 2.28+) · CPU", "3.14", "x86_64-manylinux_2_28", "cpu"),
    "macos-arm64-py314.txt": ("macOS 14+ Apple Silicon · CPU (MPS 는 확인 전 — 적은 사람만)", "3.14", "aarch64-apple-darwin", None),
}
# torch 2.15.0 의 macOS 휠은 macOS 14 부터다(uv 는 macOS 13 을 기본으로 골라 '휠이 없다' 로 멈췄다 — 2026-11-28)
MACOS_MIN = "14.0"
DEV_TARGET = ("Windows x64 · NVIDIA CUDA 13.4 — 데스크탑 개발 환경(.venv) 그대로", "3.15", "x86_64-pc-windows-msvc", "cu134")

NOTE = {
    DEV: ("# PyYAML 6.0.3 은 cp315 휠이 없어 소스에서 만든다 — MSVC 가 없으면 PYYAML_FORCE_LIBYAML=0 을 두고 깔면 순수 Python 판이 된다.\n"
          "# torch 를 cu134 인덱스에서 받는다(아래 --extra-index-url). 드라이버는 CUDA 13.4 이상(이 PC 는 621.24 — 13.5).\n"),
    "macos-arm64-py314.txt": ("# Apple GPU(MPS)는 실제 Mac 으로 확인하지 못해 장치 auto 가 고르지 않는다(CPU 로 돈다 — 2026-11-28 사용자 결정).\n"
                              "# 쓰려면 config.yaml 의 model.device 를 mps 로(MPS 에 없는 연산은 CPU 로 넘긴다 — devices.py 머리말).\n"),
}


def uv() -> str:
    exe = shutil.which("uv")
    if not exe:
        raise SystemExit("uv 를 찾지 못했다 — https://docs.astral.sh/uv/ 로 깔거나 PATH 에 둘 것")
    return exe


def strip_local(lock: Path, out: Path) -> None:
    """개발 잠금을 제약 파일로 — `+cu134` 같은 로컬 판·`--index` 줄·주석을 뺀다(==2.15.0 은 2.15.0+cpu 도 받는다)."""
    lines = []
    for ln in lock.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)(\[[^\]]*\])?==([^\s;+]+)", ln)
        if m:
            lines.append(f"{m.group(1)}=={m.group(3)}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compile_one(name: str, target: tuple, constraint: Path) -> None:
    title, py, platform, index = target
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / name
        cmd = [uv(), "pip", "compile", str(ROOT / "pyproject.toml"), "--extra", "test", "--python-version", py,
               "--python-platform", platform, "-c", str(constraint), "--emit-index-url", "--no-header", "--quiet",
               "-o", str(raw)]
        if index:
            cmd += ["--extra-index-url", TORCH.format(index), "--index-strategy", "unsafe-best-match"]
        env = {**os.environ, "MACOSX_DEPLOYMENT_TARGET": MACOS_MIN} if "apple" in platform else None
        subprocess.run(cmd, check=True, cwd=ROOT, env=env)
        body = "\n".join(ln for ln in raw.read_text(encoding="utf-8").splitlines()
                         if not re.match(r"^\s*#\s+-c\s", ln))   # 제약 파일 자리(임시 폴더)는 적지 않는다
    head = (f"# {title} · Python {py}\n"
            "# pyproject.toml 의 의존성 + test 묶음을 uv 로 고른 잠금이다 — 손으로 고치지 말고 `python locks/compile.py` 로\n"
            "# 다시 만든다(locks/README.md). 깔기(새 가상 환경에서):  python -m pip install -r locks/" + name + "\n"
            "#                                그다음 소스 폴더에서:  python -m pip install -e . --no-deps\n"
            + NOTE.get(name, ""))
    (HERE / name).write_text(head + "\n" + body.strip() + "\n", encoding="utf-8", newline="\n")
    print(f"locks/{name}: {sum(1 for ln in body.splitlines() if '==' in ln)}개")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dev-venv", type=Path, help="개발 잠금을 만들 venv(그 안에 깔린 판 그대로 적는다 — 읽기만 한다)")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        if args.dev_venv:
            py = args.dev_venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
            freeze = subprocess.run([str(py), "-m", "pip", "freeze", "--exclude-editable"], check=True,
                                    capture_output=True, text=True).stdout
            frozen = Path(tmp) / "freeze.txt"
            frozen.write_text("\n".join(ln for ln in freeze.splitlines() if "==" in ln) + "\n", encoding="utf-8")
            compile_one(DEV, DEV_TARGET, frozen)
        constraint = Path(tmp) / "verified.txt"
        strip_local(HERE / DEV, constraint)
        for name, target in TARGETS.items():
            compile_one(name, target, constraint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
