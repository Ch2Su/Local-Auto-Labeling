"""검수 우선순위(불확실도) 계산.

사람의 검수 시간이 병목이므로, 모델이 헷갈려 하는 이미지를 먼저 보여준다.
- 신뢰도가 high_band 이상인 박스는 거의 맞으니 볼 가치가 적다.
- low_band 이하인 박스는 오검출일 가능성이 커서 볼 가치가 크다.
- 검출이 하나도 없는 이미지는 '놓친 객체'가 있을 수 있어 중간값을 준다.
"""

from __future__ import annotations


def box_ambiguity(conf: float, low: float, high: float) -> float:
    """박스 하나의 애매함을 0(확실)~1(매우 애매)로 환산."""
    if high <= low:
        return 0.0 if conf >= high else 1.0
    if conf >= high:
        return 0.0
    if conf <= low:
        return 1.0
    return (high - conf) / (high - low)


def image_uncertainty(
    confs: list[float | None],
    *,
    low: float = 0.30,
    high: float = 0.80,
    empty: float = 0.60,
) -> float:
    """이미지 한 장의 검수 우선순위 점수. 클수록 먼저 봐야 한다."""
    values = [c for c in confs if c is not None]
    if not values:
        # 박스가 있는데 conf가 전부 없다면 사람이 손본 라벨이므로 볼 필요 없음.
        return 0.0 if confs else empty
    per_box = [box_ambiguity(c, low, high) for c in values]
    worst = max(per_box)
    mean = sum(per_box) / len(per_box)
    # 최악의 박스 하나만으로도 검수할 이유가 되지만, 전반적으로 애매한 쪽을 더 올린다.
    return round(0.6 * worst + 0.4 * mean, 6)
