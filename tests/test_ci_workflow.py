"""GitHub Actions 워크플로(.github/workflows/tests.yml)가 잠금 파일·규칙과 맞는지 (2026-11-28 포터블 앱 8단계).

CI 는 push 해야 돌아서, 고친 워크플로가 틀렸는지는 올린 뒤에야 안다 — 그 전에 여기서 파일만으로 본다(네트워크 없음).
"""

from __future__ import annotations

import re

import yaml

from helpers import ROOT

WF = ROOT / ".github" / "workflows" / "tests.yml"
DOC = yaml.safe_load(WF.read_text(encoding="utf-8"))
JOB = DOC["jobs"]["pytest"]
STEPS = JOB["steps"]
MATRIX = JOB["strategy"]["matrix"]["include"]


def test_every_release_lock_runs_on_its_own_os():
    """배포용 잠금(locks/*-py314.txt)마다 그 OS 의 러너가 하나씩 — 잠금을 더하면 여기에도 더할 것. 개발 잠금은 GPU 라 CI 밖."""
    want = {p.as_posix().split("/")[-1] for p in (ROOT / "locks").glob("*-py314.txt")}
    got = {m["lock"].split("/")[-1]: m["os"] for m in MATRIX}
    assert set(got) == want
    for lock, os_ in got.items():
        assert os_.split("-")[0] == {"win": "windows", "linux": "ubuntu", "macos": "macos"}[lock.split("-")[0]], (lock, os_)
        assert (ROOT / "locks" / lock).is_file()


def test_python_matches_the_locks():
    py = next(s for s in STEPS if str(s.get("uses", "")).startswith("actions/setup-python"))["with"]["python-version"]
    assert all(f"py{py.replace('.', '')}" in m["lock"] for m in MATRIX), py


def test_actions_are_pinned_to_a_commit():
    """태그(@v7)는 옮겨질 수 있다 — 바깥 액션은 40자리 커밋 해시로."""
    uses = [s["uses"] for s in STEPS if "uses" in s]
    assert uses and all(re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", u) for u in uses), uses


def test_the_workflow_cannot_write_and_keeps_bytes():
    assert DOC["permissions"] == {"contents": "read"}
    assert DOC["defaults"]["run"]["shell"] == "bash"            # pwsh 는 가운데 명령의 실패를 놓친다
    checkout = next(i for i, s in enumerate(STEPS) if str(s.get("uses", "")).startswith("actions/checkout"))
    assert any("core.autocrlf false" in s.get("run", "") for s in STEPS[:checkout])     # 체크아웃 전에
    assert STEPS[checkout]["with"]["persist-credentials"] is False
    assert "secrets." not in WF.read_text(encoding="utf-8")


def test_skipped_tests_fail_the_run():
    """CI 에서 건너뛴 테스트(node 없음 등)는 그 OS 에서 확인하지 않은 것 — 실패로 친다."""
    run = next(s["run"] for s in STEPS if s.get("name") == "Regression tests")
    assert "pytest" in run and "skipped" in run and "exit 1" in run


def test_it_runs_on_push_and_pull_requests():
    on = DOC.get("on", DOC.get(True))                             # PyYAML(YAML 1.1)은 키 on 을 True 로 읽는다
    assert "push" in on and "pull_request" in on
