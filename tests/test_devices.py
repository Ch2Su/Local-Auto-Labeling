"""장치 고르기 (devices.py — 2026-11-28, 포터블 앱 3단계).

`auto` 만 CUDA → MPS → CPU 로 고르고, 사람이 적은 장치(`0`·`cpu`·`cuda:1`·`mps`·`0,1`)는 그대로 넘긴다 — CUDA 가 없는데 `0` 이
적혀 있어도 몰래 CPU 로 바꾸지 않는다(Ultralytics 의 오류가 그대로 난다). 설정에 키가 없으면 `auto` 다(그 전에는 `0`).
`torch` 의 CUDA·MPS 확인은 갈아 끼워 흉내 낸다 — 실제 macOS(MPS)·GPU 없는 PC 에서 돌려 본 것은 아니다.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch

from autolabel.config import Config
from autolabel.devices import auto_device, is_auto, resolve_device
from helpers import ROOT


@pytest.fixture
def machine(monkeypatch):
    """이 PC 를 CUDA·MPS 가 있고 없는 PC 로 흉내 낸다."""
    def set_(cuda: bool, mps: bool):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    return set_


@pytest.mark.parametrize("cuda, mps, expect", [
    (True, False, 0), (True, True, 0), (False, True, "mps"), (False, False, "cpu"),
])
def test_auto_prefers_cuda_then_mps_then_cpu(machine, cuda, mps, expect):
    machine(cuda, mps)
    assert auto_device() == expect
    assert resolve_device("auto") == expect and resolve_device(" AUTO ") == expect


@pytest.mark.parametrize("spec", [0, 1, "0", "0,1", "cuda:1", "cpu", "mps", "-1"])
def test_what_a_person_wrote_is_passed_through(machine, spec):
    machine(False, False)                   # CUDA 가 없어도 0 을 CPU 로 바꾸지 않는다
    assert resolve_device(spec) == spec and not is_auto(spec)


def test_config_device_defaults_to_auto_and_keeps_what_is_written(machine, tmp_path):
    machine(False, False)
    path = tmp_path / "config.yaml"
    assert Config({}, path).device == "cpu"                                       # 키가 없으면 auto
    assert Config({"model": {"device": None}}, path).device == "cpu"              # 빈 값도
    assert Config({"model": {"device": "auto"}}, path).device == "cpu"
    assert Config({"model": {"device": 0}}, path).device == 0                     # 이 저장소의 config.yaml 처럼
    machine(False, True)
    assert Config({}, path).device == "mps"


def test_reading_the_config_does_not_load_torch():
    """`auto` 는 장치를 쓸 때(`cfg.device`)에만 torch 를 불러 온다 — 설정·서버를 읽는 것만으로 무거운 모듈이 올라오지 않게."""
    code = "import sys; import autolabel.config, autolabel.server; print('torch' in sys.modules)"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT,
                          env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "False"
