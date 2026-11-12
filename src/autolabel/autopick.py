"""영상 후보 프레임 자동 고르기 (2026-11-12, 사용자 요청 — '영상 캡처' 에서 사람이 하던 고르기를 대신한다).

`tools/auto_capture.py` 가 영상 캡처 세션의 후보(몇 초마다 한 장)마다 운영 가중치로 검출하고 화질을 잰 뒤 여기서 고른다.
이 모듈은 **고르기만** 한다 — 영상·모델·서버를 모른다(검출 결과를 손으로 만들어 시험할 수 있게, 나중에 화면에서도 부를 수 있게).

순서:
1. **버린다** — 앞 프레임과 픽셀까지 같은 것(멈춘 화면 — 세션의 `s`), 거의 한 색인 것(신호 없음·암전 — 픽셀의 `blank_flat`
   이상이 중앙값 ±4 안. 표준편차로 가르면 안개·흐린 주차장처럼 대비가 낮은 정상 화면(공개 영상에서 8.4)과 흐린 화면(5.4)이 섞이고,
   글자 오버레이가 있는 검은 화면은 표준편차가 커서 못 거른다), 너무 어둡거나 하얀 것(평균 밝기), 흐린 것(라플라시안 분산이
   **이 영상** 중앙값의 `blur_ratio` 배 미만 — 카메라마다 선명도가 달라 절대값으로는 못 가른다).
2. **본 고르기(main)** — '아직 고른 프레임에 없는 객체' 가 많은 프레임부터 욕심쟁이로 고른다. 프레임 값 = 그런 검출마다
   `클래스 가중치 × (1 + uncertain_bonus × 애매함)` 의 합. 애매함은 검수 우선순위와 같은 식(`scoring.box_ambiguity`,
   config `review.low_band`·`high_band`)이라, 자동 라벨링이 버릴 낮은 신뢰도 검출(놓칠 뻔한 객체)일수록 값이 크다.
   **같은 객체 = 박스 IoU ≥ `match_iou` 이고 외형(박스를 12×12 컬러로 줄인 것)의 평균 절대 차이 ≤ `match_sig`.** 클래스는 보지
   않는다(같은 차가 car·pickup 으로 오가도 같은 차다). 그래서 서 있는 차·간판 오검출은 한 번 고르면 다시 값을 주지 않고, 같은
   자리를 지나는 **다른** 차는 외형이 달라 새 객체가 된다. 고른 프레임끼리는 `gap` 초 넘게 떨어져야 한다. 새 객체 값이
   `min_gain` 아래로 떨어지면(볼 것이 더 없는 장면) 예산이 남아도 멈춘다.
3. **빈 화면 몫(background)** — 검출이 하나도 없는 프레임을, 고른 프레임들에서 시간으로 가장 먼 것부터(배경·오검출 억제용).
4. **탐색 몫(explore)** — 모델이 못 보는 객체(모르는 클래스, 놓친 작은 객체)도 들어오게 남은 프레임에서 무작위로. 씨앗은 영상
   지문으로 정해(`seed`) 다시 돌려도 같다.
3·4 는 고른 어느 프레임과도 화면이 `explore_change` % 넘게 달라야 한다(`capture.changed_pct` — 영상 캡처 칸의 숫자와 같은 값).

**트래커를 쓰지 않은 이유.** 후보가 1초 간격이라 빠른 차는 한 칸 사이에 크게 움직여 트래커(ByteTrack)의 ID 가 끊기기 쉽고
(재 보지는 않았다), 그러면 같은 차가 '새 객체' 로 여러 번 잡힌다. 또 Ultralytics 트래커는 `lap` 패키지가 없으면 실행 중에
설치하려 든다(8.4.142 `trackers/utils/matching.py`). 위치+외형으로 같은 객체를 가리는 쪽은 간격과 상관없고 의존성도 없다. 대신
**움직이는 같은 차는 자리가 바뀌면 새 객체로 센다** — 크기·배경이 달라진 같은 차라 학습에 손해는 아니고, 장수는 예산
(`per_min`·`max_total`)과 `gap` 이 막는다.

기본값은 2026-11-12 에 공개 영상 두 편(주차장을 내려다본 768×432)과 yolo26n(COCO)으로만 돌려 본 첫 값이다 — 이 데이터·운영 가중치로 다시
맞출 것. 맞출 근거는 도구가 남기는 프레임별 기록(`autocapture/frames/*.jsonl`)에 있다.
"""

from __future__ import annotations

import bisect
import heapq
import math
import random
from dataclasses import dataclass, field
from statistics import median
from typing import Callable, Sequence

import numpy as np

from .scoring import box_ambiguity

SIG = 12                    # 외형 서명 한 변(px) — SIG×SIG 컬러
DROP_NAMES = {"frozen": "멈춘 화면", "blank": "신호 없음", "dark": "어두움", "bright": "하얌", "blur": "흐림"}
KIND_NAMES = {"main": "새 객체", "background": "빈 화면", "explore": "탐색"}


@dataclass
class Det:
    """검출 하나. box 는 정규화 (x1, y1, x2, y2), sig 는 (SIG, SIG, 3) uint8 — `make_sig`."""

    cls: int
    conf: float
    box: tuple[float, float, float, float]
    sig: np.ndarray


@dataclass
class Cand:
    """후보 프레임 하나 (세션 프레임 번호 i, 영상 안 시각 t초)."""

    i: int
    t: float
    dets: list[Det] = field(default_factory=list)
    sharp: float | None = None   # 축소 흑백의 라플라시안 분산 (`frame_stats`)
    mean: float | None = None    # 축소 흑백의 평균 밝기
    flat: float | None = None    # 축소 흑백에서 중앙값 ±4 안인 픽셀 비율 (1 이면 한 색)
    same: int | None = None      # 픽셀까지 같은 앞 프레임 번호 (세션의 s)


@dataclass
class Params:
    per_min: float = 2.0          # 영상 1분당 최대 장수
    max_total: int = 300          # 영상 하나에서 최대 장수
    min_total: int = 1            # 짧은 영상도 이만큼은 (볼 것이 있으면)
    gap: float = 3.0              # 고른 프레임 사이 최소 간격(초)
    match_iou: float = 0.5        # 같은 객체로 볼 박스 IoU
    match_sig: float = 12.0       # 같은 객체로 볼 외형 평균 절대 차이(0~255)
    uncertain_bonus: float = 1.0  # 애매한 검출의 덧값 (애매함 1 이면 값이 2배)
    low_band: float = 0.30        # 애매함 계산 — config review.low_band
    high_band: float = 0.80       # config review.high_band
    min_gain: float = 1.0         # 본 고르기를 멈추는 새 객체 값 (확실한 새 객체 하나 = 가중치 1)
    background: float = 0.05      # 예산 가운데 빈 화면 몫
    explore: float = 0.10         # 예산 가운데 탐색 몫
    explore_change: float = 0.3   # 빈 화면·탐색 프레임이 고른 프레임과 달라야 할 바뀐 픽셀 %
    blank_flat: float = 0.95
    dark: float = 12.0
    bright: float = 243.0
    blur_ratio: float = 0.35
    seed: int = 0


@dataclass
class Pick:
    i: int
    t: float
    kind: str                     # main | background | explore
    gain: float = 0.0             # 고를 때의 새 객체 값 (main 만)
    novel: dict[int, int] = field(default_factory=dict)   # 고를 때 새 객체의 클래스별 수 (main 만)
    fresh: list[int] = field(default_factory=list)        # 고를 때 새 객체였던 검출 번호 (main 만 — 대조 시트가 칠한다)


@dataclass
class Result:
    picks: list[Pick]
    dropped: dict[int, str]       # 프레임 번호 → 버린 이유 (DROP_NAMES 의 키)
    budget: int                   # 이 영상의 최대 장수
    budgets: dict[str, int]       # 몫별 최대 장수
    duration: float               # 후보가 덮는 영상 길이(초)
    stop: str = ""                # 본 고르기가 멈춘 이유 (budget | gain | none)

    def counts(self) -> dict[str, int]:
        out = {k: 0 for k in KIND_NAMES}
        for p in self.picks:
            out[p.kind] += 1
        return out


# ───────────────────────────── 측정 (도구가 프레임마다 부른다) ─────────────────────────────

def make_sig(frame: np.ndarray, box: Sequence[float]) -> np.ndarray:
    """박스 영역을 SIG×SIG 컬러로 줄인 외형 서명. frame 은 BGR 원본, box 는 정규화 xyxy."""
    import cv2

    h, w = frame.shape[:2]
    x1 = min(max(int(box[0] * w), 0), w - 1)
    y1 = min(max(int(box[1] * h), 0), h - 1)
    x2 = min(max(int(math.ceil(box[2] * w)), x1 + 1), w)
    y2 = min(max(int(math.ceil(box[3] * h)), y1 + 1), h)
    return cv2.resize(frame[y1:y2, x1:x2], (SIG, SIG), interpolation=cv2.INTER_AREA)


def frame_stats(gray: np.ndarray) -> tuple[float, float, float]:
    """(선명도, 평균 밝기, 한 색 비율) — gray 는 `capture.small_gray` (가로 320px 흑백)."""
    import cv2

    flat = float((np.abs(gray.astype(np.int16) - int(np.median(gray))) <= 4).mean())
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()), float(gray.mean()), flat


def class_weights(counts: Sequence[int], *, cap: float = 8.0) -> list[float]:
    """클래스별 가중치 — 박스가 적은 클래스일수록 크게: sqrt((가장 많은 수 + 1) / (이 클래스 수 + 1)), cap 에서 자른다.
    박스가 하나도 없으면(새 DB) 모두 1."""
    if not counts or max(counts) <= 0:
        return [1.0] * len(counts)
    top = max(counts)
    return [min(cap, math.sqrt((top + 1) / (max(0, n) + 1))) for n in counts]


# ───────────────────────────── 고르기 ─────────────────────────────

def quality(cands: Sequence[Cand], p: Params) -> dict[int, str]:
    """버릴 프레임과 이유. 흐림은 나머지(버리지 않은 것)의 선명도 중앙값과 견준다."""
    out: dict[int, str] = {}
    for c in cands:
        if c.same is not None:
            out[c.i] = "frozen"
        elif c.flat is not None and c.flat >= p.blank_flat:
            out[c.i] = "blank"
        elif c.mean is not None and c.mean < p.dark:
            out[c.i] = "dark"
        elif c.mean is not None and c.mean > p.bright:
            out[c.i] = "bright"
    sharp = [c.sharp for c in cands if c.i not in out and c.sharp is not None]
    if p.blur_ratio > 0 and len(sharp) >= 5:
        limit = p.blur_ratio * median(sharp)
        for c in cands:
            if c.i not in out and c.sharp is not None and c.sharp < limit:
                out[c.i] = "blur"
    return out


def plan_budget(cands: Sequence[Cand], p: Params, every: float) -> tuple[int, float]:
    """(이 영상의 최대 장수, 후보가 덮는 길이 초)."""
    if not cands:
        return 0, 0.0
    ts = [c.t for c in cands]
    duration = max(ts) - min(ts) + max(0.0, every)
    n = math.ceil(duration / 60.0 * p.per_min - 1e-9)
    return max(0, min(p.max_total, max(p.min_total, n))), duration


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(K,4) × (M,4) 정규화 xyxy → (K,M) IoU."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-12)


class _Pool:
    """고른 프레임들의 검출 — '이미 본 객체' 목록. 후보마다 어디까지 대조했는지 기억해 새로 더해진 것만 견준다
    (고른 것은 늘기만 하니 한 번 덮인 검출은 계속 덮인다)."""

    def __init__(self, p: Params):
        self.p = p
        self.boxes = np.zeros((0, 4), np.float32)
        self.sigs = np.zeros((0, SIG * SIG * 3), np.int16)

    def add(self, dets: Sequence[Det]) -> None:
        if not dets:
            return
        self.boxes = np.vstack([self.boxes, np.array([d.box for d in dets], np.float32)])
        self.sigs = np.vstack([self.sigs, np.stack([d.sig.reshape(-1) for d in dets]).astype(np.int16)])

    def cover(self, boxes: np.ndarray, sigs: np.ndarray, covered: np.ndarray, start: int) -> int:
        """covered 를 제자리에서 갱신하고, 다음에 대조를 시작할 위치를 돌려준다."""
        end = len(self.boxes)
        todo = np.flatnonzero(~covered)
        if end <= start or todo.size == 0:
            return end
        iou = _iou(boxes[todo], self.boxes[start:end])
        k, m = np.nonzero(iou >= self.p.match_iou)
        if k.size:
            dist = np.abs(sigs[todo[k]].astype(np.int16) - self.sigs[start + m]).mean(axis=1)
            covered[np.unique(todo[k[dist <= self.p.match_sig]])] = True
        return end


def select(cands: Sequence[Cand], p: Params, weights: Sequence[float], *, every: float,
           gray: Callable[[int], np.ndarray | None] | None = None) -> Result:
    """후보에서 넣을 프레임을 고른다. weights 는 프로젝트 클래스 번호별 가중치(`class_weights`), gray(i) 는 프레임 i 의
    `capture.small_gray` (빈 화면·탐색 몫의 화면 비교에만 쓴다 — None 이면 비교하지 않는다)."""
    by_i = {c.i: c for c in cands}
    dropped = quality(cands, p)
    budget, duration = plan_budget(cands, p, every)
    n_bg = int(budget * p.background + 0.5)
    n_ex = int(budget * p.explore + 0.5)
    n_main = budget - n_bg - n_ex
    while budget and n_main < 1:          # 몫 비율이 커도 본 고르기에 한 장은 남긴다 (탐색 → 빈 화면 순으로 줄인다)
        if n_ex:
            n_ex -= 1
        else:
            n_bg -= 1
        n_main = budget - n_bg - n_ex
    ok = [c for c in cands if c.i not in dropped]

    picks: list[Pick] = []
    times: list[float] = []          # 고른 프레임 시각(정렬)

    def far_enough(t: float) -> bool:
        k = bisect.bisect_left(times, t)
        if k < len(times) and times[k] - t < p.gap - 1e-9:
            return False
        return not (k > 0 and t - times[k - 1] < p.gap - 1e-9)

    def take(c: Cand, kind: str, gain: float = 0.0, novel: dict[int, int] | None = None,
             fresh: list[int] | None = None) -> None:
        picks.append(Pick(c.i, c.t, kind, round(gain, 4), novel or {}, fresh or []))
        bisect.insort(times, c.t)

    stop = _main(ok, by_i, p, weights, n_main, far_enough, take)
    chosen = {pk.i for pk in picks}
    distinct = _distinct_fn(picks, p, gray)

    # --- 3. 빈 화면: 고른 프레임들에서 시간으로 가장 먼 것부터 (아무것도 안 골랐으면 가운데부터)
    bg = [c for c in ok if not c.dets and c.i not in chosen]
    if bg and n_bg:
        ts = np.array([c.t for c in bg], np.float64)
        alive = np.ones(len(bg), bool)
        mid = (min(c.t for c in cands) + max(c.t for c in cands)) / 2
        far: np.ndarray | None = None
        taken = misses = 0
        while taken < n_bg and misses < _MAX_MISSES and alive.any():
            if far is None:           # 고른 것이 바뀌었을 때만 다시 잰다
                far = (np.min(np.abs(ts[:, None] - np.array(times)[None, :]), axis=1) if times
                       else -np.abs(ts - mid))
                far[~alive] = -np.inf
            k = int(np.argmax(far))
            alive[k] = False
            far[k] = -np.inf
            c = bg[k]
            if far_enough(c.t) and distinct(c):
                take(c, "background")
                chosen.add(c.i)
                far = None
                taken += 1
            else:
                misses += 1

    # --- 4. 탐색: 남은 프레임에서 무작위로
    rng = random.Random(p.seed)
    pool = [c for c in ok if c.i not in chosen]
    rng.shuffle(pool)
    taken = misses = 0
    for c in pool:
        if taken >= n_ex or misses >= _MAX_MISSES:
            break
        if far_enough(c.t) and distinct(c):
            take(c, "explore")
            taken += 1
        else:
            misses += 1

    picks.sort(key=lambda x: x.t)
    return Result(picks, dropped, budget, {"main": n_main, "background": n_bg, "explore": n_ex}, duration, stop)


_MAX_MISSES = 300   # 빈 화면·탐색에서 이만큼 연달아 못 쓰면 그만 찾는다 (멈춘 듯한 장면에서 끝없이 견주지 않게)


def _main(ok: Sequence[Cand], by_i: dict[int, Cand], p: Params, weights: Sequence[float], n_main: int,
          far_enough: Callable[[float], bool], take: Callable[..., None]) -> str:
    """본 고르기 — 게으른 욕심쟁이. 값은 고를수록 줄기만 하므로(덮인 검출은 계속 덮인다), 다시 잰 값이 다음 후보의 옛 값
    이상이면 그것이 지금 최대다. 멈춘 이유를 돌려준다: budget(예산을 채움) | gain(새 객체 값이 min_gain 미만) |
    exhausted(남은 후보가 다 gap 안) | none(검출 있는 후보가 없음)."""

    def value(d: Det) -> float:
        w = weights[d.cls] if 0 <= d.cls < len(weights) else 1.0
        return w * (1.0 + p.uncertain_bonus * box_ambiguity(d.conf, p.low_band, p.high_band))

    pool = _Pool(p)
    state: dict[int, list] = {}
    heap: list[tuple[float, int, float, int]] = []
    for c in ok:
        if not c.dets:
            continue
        vals = np.array([value(d) for d in c.dets], np.float64)
        boxes = np.array([d.box for d in c.dets], np.float32)
        sigs = np.stack([d.sig.reshape(-1) for d in c.dets])          # uint8 — 견줄 때만 int16 으로
        state[c.i] = [vals, boxes, sigs, np.zeros(len(c.dets), bool), 0]
        heapq.heappush(heap, (-float(vals.sum()), -len(c.dets), c.t, c.i))
    if not heap:
        return "none"
    taken = 0
    while heap:
        if taken >= n_main:
            return "budget"
        _, nd, t, i = heapq.heappop(heap)
        if not far_enough(t):
            continue                  # 고른 것은 늘기만 하니 다시 멀어질 일이 없다
        st = state[i]
        st[4] = pool.cover(st[1], st[2], st[3], st[4])
        g = float(st[0][~st[3]].sum())
        if heap and g < -heap[0][0] - 1e-9:
            heapq.heappush(heap, (-g, nd, t, i))
            continue
        if g < p.min_gain:
            return "gain"
        c = by_i[i]
        novel: dict[int, int] = {}
        fresh = [k for k, cov in enumerate(st[3]) if not cov]
        for k in fresh:
            novel[c.dets[k].cls] = novel.get(c.dets[k].cls, 0) + 1
        take(c, "main", g, novel, fresh)
        pool.add(c.dets)
        taken += 1
    return "budget" if taken >= n_main else "exhausted"


def _distinct_fn(picks: list[Pick], p: Params, gray: Callable[[int], np.ndarray | None] | None
                 ) -> Callable[[Cand], bool]:
    """후보가 고른 어느 프레임과도 화면이 explore_change % 넘게 다른가 (`capture.changed_pct`). 흑백은 한 번만 읽는다."""
    from .capture import changed_pct

    if gray is None or p.explore_change <= 0:
        return lambda c: True
    cache: dict[int, np.ndarray | None] = {}

    def g_of(i: int) -> np.ndarray | None:
        if i not in cache:
            cache[i] = gray(i)
        return cache[i]

    def distinct(c: Cand) -> bool:
        a = g_of(c.i)
        if a is None:
            return True
        for pk in picks:
            b = g_of(pk.i)
            if b is not None and b.shape == a.shape and changed_pct(a, b) < p.explore_change:
                return False
        return True

    return distinct
