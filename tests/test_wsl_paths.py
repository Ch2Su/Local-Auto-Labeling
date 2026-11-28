"""WSL 스크립트가 저장소 자리를 스스로 구하는가 (2026-11-28 — Grok 리뷰를 대조하다 찾음).

노트북에서는 저장소가 E: 라 WSL 에서 `/mnt/e/local_auto_labeling` 이었고, 스크립트 몇 곳이 그 경로를 그대로 적어 두었다:
`wsl/sync_dataset.sh` 의 기본 원본(인자 없이 돌리면 데스크탑에서 '데이터셋이 없습니다'), `wsl/train_wsl.py` 의 안내 두 줄(동기화
명령·학습 뒤 가중치 복사 명령 — 11-27 의 n·m 1280 학습 로그에도 틀린 경로가 찍혔다). 웹 UI 의 WSL 버튼은 경로를 넘겨서
(`wsl_train.to_linux`) 영향이 없었고, 손으로 돌릴 때만 틀렸다. 이제 스크립트가 제 자리에서 저장소를 구한다.
"""

from __future__ import annotations

import ast
import re
import sys

from helpers import ROOT

sys.path.insert(0, str(ROOT / "wsl"))
import train_wsl as T  # noqa: E402

WSL = ROOT / "wsl"


def test_train_wsl_points_at_its_own_repository_when_the_dataset_is_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["train_wsl.py", "--project", str(tmp_path / "runs"),
                                      "--data", str(tmp_path / "none" / "data.yaml")])
    assert T.main() == 1
    out = capsys.readouterr().out
    assert str(T.REPO / "wsl" / "sync_dataset.sh") in out and "/mnt/e" not in out
    assert T.REPO == ROOT


def test_no_wsl_script_runs_with_a_fixed_drive_path():
    """주석·머리말의 예시(노트북 /mnt/e/…)는 괜찮다 — 실행되는 글자(파이썬 문자열·셸 명령 줄)에 드라이브가 박혀 있으면 안 된다."""
    found = []
    for p in sorted(WSL.glob("*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        docs = {id(n.body[0].value) for n in ast.walk(tree)
                if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body
                and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs \
                    and re.search(r"/mnt/[a-z]/", node.value):
                found.append(f"{p.name}:{node.lineno}")
    for p in sorted(WSL.glob("*.sh")):
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if not line.lstrip().startswith("#") and re.search(r"/mnt/[a-z]/", line):
                found.append(f"{p.name}:{n}")
    assert found == []


def test_sync_dataset_defaults_to_the_dataset_next_to_itself():
    text = (WSL / "sync_dataset.sh").read_text(encoding="utf-8")
    assert re.search(r'^REPO="\$\(cd "\$\(dirname "\$\{BASH_SOURCE\[0\]\}"\)/\.\." && pwd\)"', text, re.M)
    assert re.search(r'^SRC="\$\{1:-\$REPO/data/visible/dataset\}"', text, re.M)
