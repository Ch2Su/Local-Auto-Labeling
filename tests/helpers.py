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


def pixel_rects(width: int, height: int, n: int, seed: int = 11) -> list[tuple[float, float, float, float]]:
    """픽셀에 딱 맞춘 영역(정규화 xyxy) n 개 — `tools/mask_images.py --rect` 가 만드는 꼴."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        x1, x2 = sorted(rng.sample(range(width + 1), 2))
        y1, y2 = sorted(rng.sample(range(height + 1), 2))
        out.append((x1 / width, y1 / height, x2 / width, y2 / height))
    return out


def screen_round_trip(r) -> tuple[float, float, float, float]:
    """화면이 영역을 {cx, cy, w, h} 로 들었다가 저장할 때 xyxy 로 되돌리는 계산 — app.js 의 maskFromXyxy·maskToXyxy 와 같은
    식이다(같은지는 test_js_parity.py 가 node 로 확인한다). 끝자리가 달라질 수 있다."""
    x1, y1, x2, y2 = r
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def plain_name(i: int) -> str:
    """이름 규칙(연번·스크린샷)에 걸리지 않는 이름 — 숫자가 없다."""
    return f"p{chr(97 + i // 26 % 26)}{chr(97 + i % 26)}{'x' * (i // 676)}.jpg"
