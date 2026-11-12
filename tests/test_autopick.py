"""autopick 단위 시험 — 손으로 만든 검출로 고르기 규칙을 확인한다 (2026-11-12).

`tools/auto_capture.py` 를 만든 세션이 저장소 밖에서 쓴 검증용을 그대로 옮겼다(영상·모델·서버 없이 `autopick.select` 만 본다).
끝의 둘(`test_default_gap_…`·`test_cli_defaults_…`)은 같은 날 간격 기본값을 3 → 20 초로 바꾸며 더했다.
pytest 로 다른 테스트와 함께 돌고, pytest 없이도 돈다:
    .venv\\Scripts\\python.exe tests\\test_autopick.py
"""
import sys
from pathlib import Path

import numpy as np

for _up in [Path.cwd(), *Path(__file__).resolve().parents]:
    if (_up / "src" / "autolabel").is_dir():
        sys.path.insert(0, str(_up / "src"))
        break
from autolabel import autopick  # noqa: E402
from autolabel.autopick import Cand, Det, Params  # noqa: E402

SIG = autopick.SIG
CAR, PERSON, CONE = 2, 4, 9           # config.yaml 순서: bicycle bus car motorcycle person scooter truck pickup responder cone


def sig(color):
    return np.full((SIG, SIG, 3), color, np.uint8)


def det(cls, conf, box, color):
    return Det(cls, conf, box, sig(color))


def cand(i, t, dets, sharp=100.0, mean=120.0, flat=0.3, same=None):
    return Cand(i, t, dets, sharp, mean, flat, same)


PARKED = (0.10, 0.60, 0.25, 0.80)
LANE = (0.50, 0.50, 0.65, 0.70)
W1 = [1.0] * 10


def test_parked_car_counted_once():
    # 60초, 1초 간격. 서 있는 차는 내내 있고, 10·30·50초에 다른 차가 같은 차선 자리를 지난다(색이 다름)
    cands = []
    for t in range(60):
        d = [det(CAR, 0.9, PARKED, (40, 40, 200))]
        if t in (10, 30, 50):
            d.append(det(CAR, 0.9, LANE, (20 + t * 3, 200, 20)))
        cands.append(cand(t, float(t), d))
    p = Params(per_min=10, gap=3, background=0, explore=0)
    r = autopick.select(cands, p, W1, every=1.0)
    main = [pk for pk in r.picks if pk.kind == "main"]
    ts = sorted(pk.t for pk in main)
    # 첫 장은 두 대가 다 새것(10·30·50 중 하나), 그다음은 지나가는 차만 새것 — 서 있는 차만 있는 프레임은 고르지 않는다
    assert set(ts) == {10.0, 30.0, 50.0}, ts
    assert r.stop == "gain", r.stop
    first = min(main, key=lambda x: -x.gain)
    assert first.gain == 2.0 and sorted(pk.gain for pk in main) == [1.0, 1.0, 2.0]


def test_same_spot_same_look_is_covered():
    # 같은 자리·같은 외형(같은 차가 서 있다) → 두 번째는 새것이 아니다
    cands = [cand(0, 0.0, [det(CAR, 0.9, LANE, (100, 100, 100))]),
             cand(1, 10.0, [det(CAR, 0.9, LANE, (104, 98, 101))])]
    r = autopick.select(cands, Params(per_min=60, gap=1, background=0, explore=0), W1, every=1.0)
    assert [pk.t for pk in r.picks] == [0.0], r.picks


def test_class_agnostic_identity():
    # 같은 차가 car ↔ pickup 으로 오가도 같은 차
    cands = [cand(0, 0.0, [det(CAR, 0.9, LANE, (100, 100, 100))]),
             cand(1, 10.0, [det(7, 0.9, LANE, (100, 100, 100))])]
    r = autopick.select(cands, Params(per_min=60, gap=1, background=0, explore=0), W1, every=1.0)
    assert len(r.picks) == 1


def test_gap_and_budget():
    # 매 초 새 차 — 예산 per_min 6 × 1분 = 6장, gap 5초
    cands = [cand(t, float(t), [det(CAR, 0.9, (0.1 * (t % 8), 0.1, 0.1 * (t % 8) + 0.1, 0.2), (t * 4 % 255, 0, 0))])
             for t in range(60)]
    p = Params(per_min=6, gap=5, background=0, explore=0)
    r = autopick.select(cands, p, W1, every=1.0)
    ts = sorted(pk.t for pk in r.picks)
    assert len(ts) == 6, ts
    assert all(b - a >= 5 for a, b in zip(ts, ts[1:])), ts
    assert r.stop == "budget"


def test_rare_class_first():
    w = autopick.class_weights([0, 0, 1000, 0, 5000, 0, 0, 0, 0, 3])
    assert abs(w[PERSON] - 1.0) < 1e-9 and w[CONE] > w[CAR] > 1.0 and max(w) <= 8.0
    cands = [cand(0, 0.0, [det(PERSON, 0.9, (0.1, 0.1, 0.2, 0.3), (10, 10, 10)),
                           det(PERSON, 0.9, (0.3, 0.1, 0.4, 0.3), (20, 20, 20))]),
             cand(1, 20.0, [det(CONE, 0.9, (0.5, 0.8, 0.52, 0.85), (0, 120, 250))])]
    r = autopick.select(cands, Params(per_min=1, min_total=1, gap=1, background=0, explore=0), w, every=1.0)
    assert [pk.t for pk in r.picks] == [20.0], r.picks       # 예산 1장 — 라바콘 하나가 사람 둘보다 값지다


def test_uncertain_bonus():
    # 같은 수의 새 객체면 애매한 쪽(놓칠 뻔한 검출)이 먼저
    sure = cand(0, 0.0, [det(CAR, 0.95, LANE, (1, 1, 1))])
    vague = cand(1, 30.0, [det(CAR, 0.2, PARKED, (200, 200, 200))])
    r = autopick.select([sure, vague], Params(per_min=1, gap=1, background=0, explore=0), W1, every=1.0)
    assert [pk.t for pk in r.picks] == [30.0]
    assert abs(r.picks[0].gain - 2.0) < 1e-9


def test_quality_drops():
    cands = [cand(0, 0.0, [], sharp=100),
             cand(1, 1.0, [], same=0),                       # 멈춘 화면
             cand(2, 2.0, [], flat=0.99),                     # 신호 없음
             cand(3, 3.0, [], mean=5.0),                     # 어두움
             cand(4, 4.0, [], mean=250.0),                   # 하얌
             cand(5, 5.0, [], sharp=10.0)]                   # 흐림 (중앙값 100 의 0.35 배 미만)
    cands += [cand(i, float(i), [], sharp=100.0 + i) for i in range(6, 12)]
    q = autopick.quality(cands, Params())
    assert q == {1: "frozen", 2: "blank", 3: "dark", 4: "bright", 5: "blur"}, q


def test_background_far_from_picks_and_explore_seeded():
    cands = []
    for t in range(120):
        d = [det(CAR, 0.9, (0.1, 0.1, 0.3, 0.3), (t, 0, 0))] if 50 <= t < 60 else []
        cands.append(cand(t, float(t), d))
    p = Params(per_min=10, gap=3, background=0.1, explore=0.1, min_gain=1.0, seed=7)
    r1 = autopick.select(cands, p, W1, every=1.0)
    r2 = autopick.select(cands, p, W1, every=1.0)
    assert [(x.i, x.kind) for x in r1.picks] == [(x.i, x.kind) for x in r2.picks]     # 같은 씨앗이면 같다
    bg = [x for x in r1.picks if x.kind == "background"]
    main = [x for x in r1.picks if x.kind == "main"]
    assert bg and main
    # 첫 빈 화면은 본 고르기(50~59초)에서 가장 먼 끝(119초 또는 0초)
    assert bg[0].t in (0.0, 119.0) or min(b.t for b in bg) == 0.0 or max(b.t for b in bg) == 119.0, bg
    ts = sorted(x.t for x in r1.picks)
    assert all(b - a >= 3 for a, b in zip(ts, ts[1:])), ts
    assert r1.counts()["explore"] == r1.budgets["explore"]


def test_distinct_gate():
    # 빈 화면·탐색은 고른 것과 화면이 거의 같으면 빼고, 다르면 넣는다
    base = np.zeros((180, 320), np.uint8)
    other = base.copy()
    other[:, :20] = 200          # 6% 바뀜

    def gray(i):
        return other if i == 5 else base

    cands = [cand(0, 0.0, [det(CAR, 0.9, LANE, (1, 1, 1))])] + [cand(i, float(i * 10), []) for i in range(1, 8)]
    p = Params(per_min=60, gap=1, background=0.5, explore=0.0, explore_change=0.3)
    r = autopick.select(cands, p, W1, every=1.0, gray=gray)
    bg = [x.i for x in r.picks if x.kind == "background"]
    assert bg == [5], bg


def test_empty_and_none():
    r = autopick.select([], Params(), W1, every=1.0)
    assert r.picks == [] and r.budget == 0
    cands = [cand(i, float(i), []) for i in range(30)]
    r = autopick.select(cands, Params(per_min=2), W1, every=1.0)
    assert r.stop == "none"


def test_speed_large():
    # 5000 후보 × 20 검출 — 고르기만 몇 초 안에
    import time
    rng = np.random.default_rng(0)
    cands = []
    for i in range(5000):
        ds = []
        for k in range(20):
            x, y = rng.random(2) * 0.9
            ds.append(Det(int(rng.integers(0, 10)), float(rng.random()), (x, y, x + 0.05, y + 0.08),
                          rng.integers(0, 255, (SIG, SIG, 3), dtype=np.uint8)))
        cands.append(cand(i, float(i), ds))
    t0 = time.perf_counter()
    r = autopick.select(cands, Params(per_min=4, max_total=300, gap=3), W1, every=1.0)
    dt = time.perf_counter() - t0
    print(f"  5000 후보 × 20 검출: {len(r.picks)}장 고름, {dt:.2f}초")
    assert dt < 30


def test_default_gap_spreads_over_dense_traffic():
    # 2026-11-12 에 이 PC 의 고속도로 영상에서 본 것 — 지나가는 차는 후보마다 자리가 달라 늘 새 객체라서, 간격이 3초(처음 기본값)일
    # 때는 차가 가장 많은 구간에 고르기가 몰리고 나머지 구간은 비었다. 기본값(20초)이면 영상에 퍼진다.
    # 270초, 매 초 새 차 4대(100~124초에는 12대). 박스 자리는 90초마다 겹치지만 색이 달라 같은 차가 아니다.
    def frame(t):
        n = 12 if 100 <= t < 125 else 4
        y = (t % 90) / 100.0
        return [det(CAR, 0.9, (0.05 * k, y, 0.05 * k + 0.04, y + 0.008), (t % 256, t * 5 % 256, 128)) for k in range(n)]

    cands = [cand(t, float(t), frame(t)) for t in range(270)]
    r = autopick.select(cands, Params(background=0, explore=0), W1, every=1.0)
    ts = sorted(pk.t for pk in r.picks)
    assert len(ts) == r.budget == 9, ts
    assert all(b - a >= 20 for a, b in zip(ts, ts[1:])), ts
    assert sum(100 <= t < 125 for t in ts) <= 2, ts                  # 붐비는 25초에서는 두 장까지만
    assert min(ts) < 100 and max(ts) >= 125, ts                      # 그 앞뒤 구간에서도 고른다
    old = autopick.select(cands, Params(gap=3, background=0, explore=0), W1, every=1.0)
    assert all(100 <= pk.t < 125 for pk in old.picks) and len(old.picks) == 9    # 처음 기본값이면 아홉 장이 다 그 구간


def test_cli_defaults_match_params():
    # tools/auto_capture.py 의 고르기 옵션 기본값은 Params 의 기본값과 같아야 한다 — 두 곳에 적혀 있어 한쪽만 고치기 쉽다
    # (2026-11-12 에 gap 을 3 → 20 초로 바꾸며 더했다). 도구는 main() 안에서 옵션을 만들어서 글자로 읽는다 — add_argument 줄의
    # 모양('"--이름", type=…, default=값')을 바꾸면 여기도 맞출 것.
    import re

    src = (Path(__file__).resolve().parents[1] / "tools" / "auto_capture.py").read_text(encoding="utf-8")
    pairs = {"--per-min": "per_min", "--max": "max_total", "--min": "min_total", "--gap": "gap", "--match-iou": "match_iou",
             "--match-sig": "match_sig", "--uncertain-bonus": "uncertain_bonus", "--min-gain": "min_gain",
             "--background": "background", "--explore": "explore", "--explore-change": "explore_change",
             "--blur-ratio": "blur_ratio"}
    d = Params()
    for flag, name in pairs.items():
        m = re.search(rf'add_argument\("{re.escape(flag)}", type=(?:float|int), default=([0-9.]+)', src)
        assert m, f"tools/auto_capture.py 에서 {flag} 의 기본값을 찾지 못했습니다"
        assert float(m.group(1)) == float(getattr(d, name)), (flag, m.group(1), getattr(d, name))
    assert d.gap == 20.0


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"ok   {name}")
            except AssertionError as exc:
                fails += 1
                print(f"FAIL {name}: {exc}")
    sys.exit(1 if fails else 0)
