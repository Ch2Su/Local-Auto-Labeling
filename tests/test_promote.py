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


# ---------------------------------------------------------------- 2026-11-17 리뷰: 마지막 epoch 중단·운영 가중치 없음

class FakeModel:
    """`add_callback` 만 받는 대역 — 붙은 콜백을 Ultralytics 가 부르는 순서로 일으킨다."""

    def __init__(self) -> None:
        self.cbs: dict[str, list] = {}

    def add_callback(self, event: str, fn) -> None:
        self.cbs.setdefault(event, []).append(fn)

    def fire(self, event: str, trainer) -> None:
        for fn in self.cbs.get(event, []):
            fn(trainer)


def train_wsl():
    import importlib.util

    from helpers import ROOT
    spec = importlib.util.spec_from_file_location("train_wsl", ROOT / "wsl" / "train_wsl.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stop_during(kind: str, run: Path, epoch: int, epochs: int) -> dict:
    """epoch(0부터) 의 배치 도중에 중단을 요청하고 그 epoch 의 검증·저장 뒤(on_fit_epoch_end)까지 간 상태."""
    model = FakeModel()
    (run / "weights" / "last.pt").write_bytes(b"last")
    (run / "weights" / "resume.pt").unlink()
    if kind == "windows":
        state = trainer.attach_callbacks(model, lambda: True)
    else:
        stop_file = run.parent / "STOP"
        stop_file.write_text("stop", encoding="utf-8")
        state = {"stopped": False, "snapped": False, "snapshot": None}
        train_wsl().attach(model, stop_file, state)
    fake = SimpleNamespace(stop=False, epoch=epoch, epochs=epochs, last=str(run / "weights" / "last.pt"))
    model.fire("on_train_batch_end", fake)
    assert fake.stop and state["stopped"]                                # 요청은 받았다
    model.fire("on_fit_epoch_end", fake)
    model.fire("on_fit_epoch_end", fake)                                 # final_eval 이 한 번 더 부른다
    return state


@pytest.mark.parametrize("kind", ["windows", "wsl"])
def test_a_stop_in_the_last_epoch_is_a_finished_training(cfg, run, kind):
    """마지막 epoch 에서 멈추면 이어 붙일 것이 없어 스냅샷을 안 남긴다 — '멈춘 학습' 으로 두면 승격도 이어하기도 안 됐다."""
    state = stop_during(kind, run, epoch=99, epochs=100)
    assert state["stopped"] is False and state["snapshot"] is None
    assert not (run / "weights" / "resume.pt").exists()
    (run / "weights" / "resume.pt").write_bytes(b"from an earlier stop")
    out = finish(cfg, run, stopped=state["stopped"])
    assert cfg.weights.read_bytes() == b"this run" and (out["promoted"], out["stopped_early"]) == (True, False)
    assert not (run / "weights" / "resume.pt").exists()                  # 예전 중단의 스냅샷으로 되감지 않는다


@pytest.mark.parametrize("kind", ["windows", "wsl"])
def test_a_stop_before_the_last_epoch_is_still_a_stopped_training(cfg, run, kind):
    state = stop_during(kind, run, epoch=49, epochs=100)
    assert state["stopped"] is True and (run / "weights" / "resume.pt").read_bytes() == b"last"
    out = finish(cfg, run, stopped=state["stopped"])
    assert cfg.weights.read_bytes() == b"operating" and out["promoted"] is False


def test_a_stopped_first_training_becomes_the_operating_weights(cfg, run, monkeypatch, capsys):
    """운영 가중치가 아직 없으면 지킬 것이 없다 — 안 올리면 자동 라벨링이 사전학습(COCO) 가중치로 돈다."""
    cfg.weights.unlink()
    out = finish(cfg, run, stopped=True)
    assert cfg.weights.read_bytes() == b"this run" and not cfg.weights.with_name("best.prev.pt").exists()
    assert (out["promoted"], out["previous_weights"], out["stopped_early"]) == (True, None, True)
    assert (run / "weights" / "resume.pt").exists()                      # 멈춘 학습인 것은 그대로 — 이어할 수 있다
    said = capsys.readouterr().out
    assert "아직 없어" in said and "올렸습니다" in said and "바꾸지 않았습니다" not in said

    cfg.weights.unlink()
    out = wsl(cfg, run, monkeypatch, stopped=True)                       # WSL 도 같은 한 곳(promote_if_finished)을 지난다
    assert cfg.weights.read_bytes() == b"this run" and out["promoted"] is True
    out = wsl(cfg, run, monkeypatch, stopped=True)                       # 이제 운영 가중치가 있으니 다시 멈추면 그대로
    assert out["promoted"] is False


def test_the_cli_summary_does_not_show_the_operating_weights_as_a_stopped_runs_result(cfg, run):
    from autolabel.cli import _weights_line

    stopped = _weights_line(finish(cfg, run, stopped=True))
    assert str(run / "weights" / "best.pt") in stopped and "그대로" in stopped
    assert stopped.index(str(run / "weights" / "best.pt")) < stopped.index(str(cfg.weights))
    done = _weights_line(finish(cfg, run, stopped=False))
    assert done.strip() == f"가중치    : {cfg.weights}"
    assert _weights_line({"weights": "w.pt"}).strip() == "가중치    : w.pt"      # 옛 꼴의 결과(키 없음)는 전처럼
