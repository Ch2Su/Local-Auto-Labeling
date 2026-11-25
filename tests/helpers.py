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


def random_poly(rng: random.Random, *, k: int | None = None, grid: tuple[int, int] | None = None,
                simple: bool = True, cx: float | None = None, cy: float | None = None,
                r: float | None = None) -> tuple[tuple[float, float], ...]:
    """다각형 하나(정규화 꼭짓점, 그린 순서) — 가릴 영역 다각형(2026-11-19)의 시험 입력.

    simple 이면 한 점 둘레로 각도 순서대로 찍어 변이 엇갈리지 않는 모양(오목할 수 있다), 아니면 아무 점이나 이어 꼬일 수
    있는 모양. grid=(W, H) 면 꼭짓점이 픽셀에 딱 맞는다(화면에서 픽셀을 눌러 찍은 꼴). 점은 0~1 안이다."""
    k = k or rng.randint(3, 9)
    cx = rng.random() if cx is None else cx
    cy = rng.random() if cy is None else cy
    r = 0.05 + 0.4 * rng.random() if r is None else r
    if simple:
        import math
        angles = sorted(rng.uniform(0, 2 * math.pi) for _ in range(k))
        raw = [(cx + r * rng.uniform(0.3, 1.0) * math.cos(a), cy + r * rng.uniform(0.3, 1.0) * math.sin(a)) for a in angles]
    else:
        raw = [(cx + r * rng.uniform(-1, 1), cy + r * rng.uniform(-1, 1)) for _ in range(k)]
    pts = [(min(1.0, max(0.0, x)), min(1.0, max(0.0, y))) for x, y in raw]
    if grid:
        w, h = grid
        pts = [(round(x * w) / w, round(y * h) / h) for x, y in pts]
    return tuple(pts)


def screen_round_trip(r) -> tuple[float, float, float, float]:
    """화면이 영역을 {cx, cy, w, h} 로 들었다가 저장할 때 xyxy 로 되돌리는 계산 — app.js 의 maskFromXyxy·maskToXyxy 와 같은
    식이다(같은지는 test_js_parity.py 가 node 로 확인한다). 끝자리가 달라질 수 있다."""
    x1, y1, x2, y2 = r
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def run_node(node: str, script: str, folder: Path):
    """script 를 node 로 돌려 표준 출력의 JSON 을 돌려준다 — 화면 코드(JS)의 함수를 글자로 떼어 돌리는 테스트용."""
    import json
    import subprocess

    path = folder / "run.js"
    path.write_text(script, encoding="utf-8")
    done = subprocess.run([node, str(path)], capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert done.returncode == 0, f"node 실패:\n{done.stderr}"
    return json.loads(done.stdout)


def plain_name(i: int) -> str:
    """이름 규칙(연번·스크린샷)에 걸리지 않는 이름 — 숫자가 없다."""
    return f"p{chr(97 + i // 26 % 26)}{chr(97 + i % 26)}{'x' * (i // 676)}.jpg"


class StreamBody:
    """요청 본문 대역 — `stream()` 이 조각을 차례로 주다가 fail 이 있으면 그것을 던진다. 몇 조각을 가져갔는지 센다
    (`taken`). 본문을 그대로 받는 처리 함수(영상·이미지 올리기)를 `post_endpoint` 로 꺼내 직접 부를 때 쓴다."""

    def __init__(self, chunks, fail: BaseException | None = None):
        self.chunks, self.fail, self.taken = list(chunks), fail, 0

    async def stream(self):
        for c in self.chunks:
            self.taken += 1
            yield c
        if self.fail is not None:
            raise self.fail


def post_endpoint(app, path: str):
    """서버 앱에서 그 주소의 POST 처리 함수 — 본문을 조각으로 흘려 넣으려고 직접 부른다(TestClient 는 본문을 한 번에 준다)."""
    return next(r.endpoint for r in app.router.routes
                if getattr(r, "path", None) == path and "POST" in getattr(r, "methods", ()))
