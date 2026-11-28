"""WSL 학습은 Windows 의 선택 기능이다 (2026-11-28, 포터블 앱 2단계 — docs/portable/m01-survey.md).

- 서버를 불러 와도 WSL 모듈(`wsl_train`)은 올라오지 않는다 — WSL 요청이 올 때만 불러 온다.
- `wsl_train.supported()` 는 서브프로세스 없이 '여기서 있을 수 있는 기능인가'(Windows·WSL 스크립트)를 본다. `check()` 는 그것부터
  본다 — Windows 가 아니면 wsl.exe 를 찾거나 부르지 않는다.
- Windows 가 아니면 `/api/wsl` 이 `supported: false` 를 주고 화면은 WSL 버튼을 감춘다. 작업 요청은 GPU 를 잡기 전에 400.

다른 OS 는 `sys.platform` 을 갈아 흉내 낸다(실제 macOS·Linux 에서는 돌려 보지 않았다 — 계획의 8단계).
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from autolabel import jobs, wsl_train
from helpers import ROOT, WEB, run_node
from test_js_parity import find_node, js_function

ORIGIN = "http://127.0.0.1:7860"


def test_loading_the_server_does_not_load_the_wsl_module():
    code = "import sys; import autolabel.server; print('autolabel.wsl_train' in sys.modules)"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT,
                          env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")}, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "False"


def test_supported_only_on_windows_with_the_scripts(monkeypatch, tmp_path):
    assert wsl_train.supported("win32") == (True, "")
    assert wsl_train.supported("linux") == (False, "WSL 학습은 Windows 에서만 됩니다.")
    assert wsl_train.supported("darwin")[0] is False
    monkeypatch.setattr(wsl_train, "SCRIPT_TRAIN", tmp_path / "none" / "train_wsl.py")
    ok, why = wsl_train.supported("win32")
    assert not ok and "스크립트가 없습니다" in why
    assert wsl_train.SCRIPTS == ROOT / "wsl"                      # 개발 모드는 저장소의 wsl/


def test_off_windows_the_check_never_touches_wsl(cfg, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("WSL 을 부르면 안 된다")

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(wsl_train.subprocess, "run", boom)
    wsl_train._check_cache.clear()
    assert wsl_train.check(cfg) == (False, "WSL 학습은 Windows 에서만 됩니다.")


@pytest.fixture
def client(cfg, store):
    from fastapi.testclient import TestClient

    from autolabel.server import create_app

    app = create_app(cfg, label="시험", bind_host="127.0.0.1")
    with TestClient(app, base_url=ORIGIN) as c:
        yield c


def test_off_windows_the_api_says_unsupported_and_refuses_jobs(client, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(wsl_train.subprocess, "run", lambda *a, **k: pytest.fail("WSL 을 부르면 안 된다"))
    wsl_train._check_cache.clear()
    got = client.get("/api/wsl").json()
    assert (got["supported"], got["available"], got["resume"]) == (False, False, None)
    for kind in ("train_wsl", "resume_wsl"):
        r = client.post(f"/api/jobs/{kind}", json={})
        assert r.status_code == 400 and "Windows" in r.json()["detail"]
    assert client.get("/api/state").json()["job"] is None              # 작업도, GPU 잡기도 없었다
    assert jobs._gpu_owner is None


def test_the_screen_hides_the_wsl_button_when_the_pc_cannot_have_it(tmp_path):
    node = find_node()
    if node is None:
        pytest.skip("node 가 없어 화면(JS) 쪽을 돌릴 수 없습니다")
    app = (WEB / "app.js").read_text(encoding="utf-8")
    script = "\n".join([
        "const els = {};",
        "const mk = () => ({ disabled: false, title: '', textContent: '', _h: new Set(),",
        "  classList: { toggle(n, v) { v ? this._h.add(n) : this._h.delete(n); }, contains(n) { return this._h.has(n); } } });",
        "for (const k of ['#btn-train-wsl', '#btn-resume-wsl', '#btn-resume', '#model-dot']) {",
        "  els[k] = mk(); els[k].classList._h = els[k]._h; }",
        "els['#btn-resume-wsl'].classList.toggle('hidden', true); els['#btn-resume'].classList.toggle('hidden', true);",
        "const $ = (s) => els[s];",
        "const S = {};",
        "let reply = null;",
        "const api = async () => { if (reply instanceof Error) throw reply; return reply; };",
        js_function(app, "syncModelDot"),
        js_function(app, "refreshWslButtons"),
        "const hid = (s) => els[s].classList.contains('hidden');",
        "(async () => {",
        "  const out = {};",
        "  reply = { supported: false, available: false, reason: 'WSL 학습은 Windows 에서만 됩니다.', resume: null };",
        "  await refreshWslButtons(); out.off = [hid('#btn-train-wsl'), hid('#btn-resume-wsl')];",
        "  reply = { supported: true, available: true, reason: '', resume: { epoch: 3, total: 10 } };",
        "  await refreshWslButtons(); out.on = [hid('#btn-train-wsl'), hid('#btn-resume-wsl'), els['#btn-train-wsl'].disabled];",
        "  reply = { available: false, reason: '옛 서버' };",                      # 옛 서버는 supported 를 안 준다 — 보이되 꺼짐
        "  await refreshWslButtons(); out.old = [hid('#btn-train-wsl'), els['#btn-train-wsl'].disabled];",
        "  reply = new Error('down');",
        "  await refreshWslButtons(); out.down = [hid('#btn-train-wsl'), els['#btn-train-wsl'].disabled];",
        "  console.log(JSON.stringify(out));",
        "})().catch((e) => { console.error(e.stack || e); process.exit(1); });",
    ])
    got = run_node(node, script, tmp_path)
    assert got["off"] == [True, True]
    assert got["on"] == [False, False, False]
    assert got["old"] == [False, True] and got["down"] == [False, True]
