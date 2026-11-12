"""테스트들이 함께 쓰는 값·작은 함수. (픽스처는 conftest.py)"""

from __future__ import annotations

import random
from pathlib import Path

from autolabel.yolo_io import Box

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "src" / "autolabel" / "web"
CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck", "scooter", "pickup", "responder", "cone"]


def box(cls: int = 2, cx: float = 0.5, cy: float = 0.5, w: float = 0.25, h: float = 0.25,
        conf: float | None = 0.875, origin: str = "auto") -> Box:
    return Box(cls, cx, cy, w, h, conf, origin)


def keys(boxes) -> list[tuple]:
    """박스 목록을 견주기 쉬운 값으로 — Box 든 이력의 dict 든."""
    get = lambda b, k: b[k] if isinstance(b, dict) else getattr(b, k)
    return [(get(b, "cls"), *(round(get(b, k), 7) for k in ("cx", "cy", "w", "h")), get(b, "conf"), get(b, "origin"))
            for b in boxes]


def far_hashes(n: int, seed: int = 1) -> list[str]:
    """서로 멀리 떨어진(해밍 거리 12 초과) dHash n 개 — 우연히 한 묶음이 되지 않게."""
    rng = random.Random(seed)
    out: list[str] = []
    while len(out) < n:
        h = rng.getrandbits(64)
        if bin(h).count("1") > 12 and all(bin(h ^ int(o, 16)).count("1") > 12 for o in out):
            out.append(f"{h:016x}")
    return out


def plain_name(i: int) -> str:
    """이름 규칙(연번·스크린샷)에 걸리지 않는 이름 — 숫자가 없다."""
    return f"p{chr(97 + i // 26 % 26)}{chr(97 + i % 26)}{'x' * (i // 676)}.jpg"
