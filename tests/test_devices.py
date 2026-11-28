"""장치 고르기 (devices.py — 2026-11-28, 포터블 앱 3단계).

`auto` 만 CUDA → CPU 로 고르고, 사람이 적은 장치(`0`·`cpu`·`cuda:1`·`mps`·`0,1`)는 그대로 넘긴다 — CUDA 가 없는데 `0` 이
적혀 있어도 몰래 CPU 로 바꾸지 않는다(Ultralytics 의 오류가 그대로 난다). 설정에 키가 없으면 `auto` 다(그 전에는 `0`).
**Apple GPU(MPS)는 `auto` 가 고르지 않는다**(2026-11-28 사용자 결정 — 실제 Mac 으로 확인하지 못했다, devices.py 머리말): `mps` 를
적은 사람만 쓰고, macOS 에서는 MPS 에 없는 연산을 CPU 로 넘기는 환경 변수를 기본으로 둔다(`autolabel._set_env_defaults`).
`torch` 의 CUDA·MPS 확인은 갈아 끼워 흉내 낸다 — 실제 macOS(MPS)·GPU 없는 PC 에서 돌려 본 것은 아니다.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch

import autolabel
import autolabel.devices as devices
from autolabel.config import Config
from autolabel.devices import MAC_NOTE, auto_device, is_auto, resolve_device
from helpers import ROOT


@pytest.fixture
def machine(monkeypatch):
    """이 PC 를 CUDA·MPS 가 있고 없는 PC 로 흉내 낸다. auto 는 MPS 를 묻지도 않아야 한다 — 물으면 센다."""
    asked = []

    def set_(cuda: bool, mps: bool):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: (asked.append(1), mps)[1])
    set_.asked = asked
    monkeypatch.setattr(devices, "_noted", False)
    return set_


@pytest.mark.parametrize("cuda, mps, expect", [
    (True, False, 0), (True, True, 0), (False, True, "cpu"), (False, False, "cpu"),
])
def test_auto_picks_cuda_or_cpu_and_never_mps(machine, cuda, mps, expect):
    """MPS 가 있어도 auto 는 CPU 다(확인하지 못한 길을 기본으로 쓰지 않는다) — MPS 를 묻지도 않는다(가상 GPU 의 Mac 에서는 그것만으로 경고)."""
    machine(cuda, mps)
    assert auto_device() == expect
    assert resolve_device("auto") == expect and resolve_device(" AUTO ") == expect
    assert machine.asked == []


def test_a_mac_is_told_once_how_to_use_mps(machine, monkeypatch, capsys):
    machine(False, True)
    monkeypatch.setattr(sys, "platform", "darwin")
    assert auto_device() == "cpu" and auto_device() == "cpu"
    assert capsys.readouterr().out.count(MAC_NOTE) == 1 and "mps" in MAC_NOTE
    monkeypatch.setattr(devices, "_noted", False)
    monkeypatch.setattr(sys, "platform", "linux")                    # Mac 이 아니면 알리지 않는다
    auto_device()
    assert capsys.readouterr().out == ""


def test_the_mps_fallback_is_a_mac_default_that_respects_what_is_set():
    """macOS 에서만 PYTORCH_ENABLE_MPS_FALLBACK=1 을 기본으로 — 이미 정해 둔 값(0 포함)은 그대로. YOLO_AUTOINSTALL 기본도 같은 함수."""
    mac: dict = {}
    autolabel._set_env_defaults(mac, "darwin")
    assert mac == {"YOLO_AUTOINSTALL": "false", "PYTORCH_ENABLE_MPS_FALLBACK": "1"}
    for plat in ("win32", "linux"):
        other: dict = {}
        autolabel._set_env_defaults(other, plat)
        assert other == {"YOLO_AUTOINSTALL": "false"}
    kept = {"PYTORCH_ENABLE_MPS_FALLBACK": "0", "YOLO_AUTOINSTALL": "true"}
    autolabel._set_env_defaults(kept, "darwin")
    assert kept == {"PYTORCH_ENABLE_MPS_FALLBACK": "0", "YOLO_AUTOINSTALL": "true"}


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
    assert Config({}, path).device == "cpu"                                       # MPS 가 있어도 auto 는 CPU
    assert Config({"model": {"device": "mps"}}, path).device == "mps"             # 적은 사람만 MPS


def test_reading_the_config_does_not_load_torch():
    """`auto` 는 장치를 쓸 때(`cfg.device`)에만 torch 를 불러 온다 — 설정·서버를 읽는 것만으로 무거운 모듈이 올라오지 않게."""
    code = "import sys; import autolabel.config, autolabel.server; print('torch' in sys.modules)"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT,
                          env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "False"
