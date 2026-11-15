"""학습이 끝났을 때 운영 가중치(models/best.pt)를 바꾸는가 — **사람이 멈춘 학습은 바꾸지 않는다** (2026-11-16 사용자 결정).

전에는 멈출 때마다 그 시점의 best.pt 가 운영 가중치를 덮었다(epoch 3 짜리가 0.72 짜리를 덮은 적이 있고, 두 번 끊으면
best.prev.pt 까지 밀려났다). Windows 학습(`trainer._finish`)과 WSL 학습(`wsl_train.train`)이 같은 규칙이어야 한다.
학습은 돌리지 않는다 — 결과 폴더와 WSL 쪽 출력만 흉내 낸다.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from autolabel import trainer, wsl_train


@pytest.fixture
def run(cfg, tmp_path) -> Path:
    """운영 가중치가 이미 있고, 방금 끝난 학습 폴더에 best.pt·resume.pt 가 있다."""
    cfg.weights.parent.mkdir(parents=True, exist_ok=True)
    cfg.weights.write_bytes(b"operating")
    d = tmp_path / "runs" / "visible"
    (d / "weights").mkdir(parents=True)
    (d / "weights" / "best.pt").write_bytes(b"this run")
    (d / "weights" / "resume.pt").write_bytes(b"snapshot")
    return d


def finish(cfg, run, stopped: bool) -> dict:
    results = SimpleNamespace(save_dir=str(run), results_dict={"metrics/mAP50-95(B)": 0.5})
    return trainer._finish(cfg, results, {"stopped": stopped, "snapshot": str(run / "weights" / "resume.pt")}, {})


def test_a_stopped_training_leaves_the_operating_weights_alone(cfg, run, capsys):
    out = finish(cfg, run, stopped=True)
    assert cfg.weights.read_bytes() == b"operating"
    assert not cfg.weights.with_name("best.prev.pt").exists()            # 밀려난 것도 없다
    assert (out["promoted"], out["previous_weights"], out["stopped_early"]) == (False, None, True)
    assert out["run_best"] == str(run / "weights" / "best.pt") and (run / "weights" / "best.pt").exists()
    assert (run / "weights" / "resume.pt").exists()                      # 이어하기는 그대로 된다
    said = capsys.readouterr().out
    assert "운영 가중치" in said and "바꾸지 않았습니다" in said and str(run / "weights" / "best.pt") in said


def test_a_finished_training_is_promoted_and_keeps_the_previous_one(cfg, run):
    out = finish(cfg, run, stopped=False)
    prev = cfg.weights.with_name("best.prev.pt")
    assert cfg.weights.read_bytes() == b"this run" and prev.read_bytes() == b"operating"
    assert (out["promoted"], out["previous_weights"], out["stopped_early"]) == (True, str(prev), False)
    assert not (run / "weights" / "resume.pt").exists()                  # 끝난 학습의 스냅샷은 치운다


def test_stopping_twice_never_loses_the_operating_weights(cfg, run):
    """고치기 전의 사고 — 한 학습을 두 번 끊으면 best.pt·best.prev.pt 가 둘 다 중간 결과가 됐다."""
    finish(cfg, run, stopped=True)
    (run / "weights" / "best.pt").write_bytes(b"this run, later")
    finish(cfg, run, stopped=True)
    assert cfg.weights.read_bytes() == b"operating" and not cfg.weights.with_name("best.prev.pt").exists()
    finish(cfg, run, stopped=False)                                      # 이어서 끝까지 가면 그때 올린다
    assert cfg.weights.read_bytes() == b"this run, later"
    assert cfg.weights.with_name("best.prev.pt").read_bytes() == b"operating"


def wsl(cfg, run, monkeypatch, stopped: bool) -> dict:
    settings = SimpleNamespace(distro="T", python="py", project="/runs", dataset="/data", name="visible", workers=1,
                               cache="ram", stop_file=run.parent / "STOP", cmd=lambda *a: list(a),
                               unc=lambda p: run if p == "/runs/visible" else Path(p))
    monkeypatch.setattr(wsl_train, "check", lambda cfg: (True, ""))
    monkeypatch.setattr(wsl_train.WslSettings, "from_cfg", classmethod(lambda cls, cfg: settings))
    monkeypatch.setattr(wsl_train, "train_cmd", lambda cfg, s: ["train"])
    monkeypatch.setattr(wsl_train, "build_dataset", lambda cfg, store: SimpleNamespace(summary=lambda: "요약"))
    monkeypatch.setattr(wsl_train, "_read_metrics", lambda run_dir: {})
    tail = f"RESULT_DIR=/runs/visible\nSTOPPED={int(stopped)}\n"
    # 복사 단계(on_cancel 없이 부른다)는 그냥 끝나고, 학습 단계에서만 중단됐다
    monkeypatch.setattr(wsl_train, "_stream", lambda cmd, job, on_cancel=None, keep_cancel=None:
                        (0, tail, stopped and on_cancel is not None))
    return wsl_train.train(cfg, None, SimpleNamespace(cancel_requested=False))


def test_wsl_training_follows_the_same_rule(cfg, run, monkeypatch, capsys):
    out = wsl(cfg, run, monkeypatch, stopped=True)
    assert cfg.weights.read_bytes() == b"operating" and not cfg.weights.with_name("best.prev.pt").exists()
    assert (out["promoted"], out["previous_weights"], out["stopped_early"]) == (False, None, True)
    assert out["resume_snapshot"] == "/runs/visible/weights/resume.pt"
    assert "바꾸지 않았습니다" in capsys.readouterr().out

    out = wsl(cfg, run, monkeypatch, stopped=False)
    assert cfg.weights.read_bytes() == b"this run"
    assert (out["promoted"], out["stopped_early"]) == (True, False)
    assert cfg.weights.with_name("best.prev.pt").read_bytes() == b"operating"
