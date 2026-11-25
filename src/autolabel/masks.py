"""가릴 영역 (2026-11-09) — 사람을 포함해 라벨하기 애매한 물체(가를 수 없는 무리·조각만 보이는 사람 등, 판정은
docs/label-rules.md)를 **학습 세트를 만들 때만** 회색(114)으로 칠한다.

**왜.** YOLO(Ultralytics) 에는 '이 자리는 보지 마' 하는 무시 영역이 없다. 애매한 물체(가를 수 없는 사람 무리 등)를 그리지 않고 두면 모델은
그것을 '그 물체 아님(배경)'으로 배우고, 장을 통째로 빼면 나머지 라벨까지 잃는다. 그 자리만 학습 때 레터박스·모자이크
바탕색(114)으로 칠하면 거기서는 아무것도 배우지 않는다. 처음(11-09 오후)에는 그림 파일에 직접 칠했다(그때의
`tools/mask_images.py`). 원본을 바꾸지 않으려고 영역을 DB `masks` 표에 두고 학습 세트를 만들 때만 칠하게 바꿨다 —
검수 화면에서는 M(가릴 영역) 모드로 그린다.

**모양 (2026-11-19 다각형 추가, 사용자 요청 — 대각선 모양을 덮으려면 작은 사각형을 여러 개 그려야 했다).**
- 사각형 `(x1, y1, x2, y2)` 와 다각형 `((x, y), (x, y), …)`(꼭짓점 3개 이상, 그린 순서) 두 가지다. 좌표는 박스와 같은
  화면(EXIF 회전을 적용한) 기준 정규화 값. 둘을 가르는 것은 `is_poly`(첫 값이 점인가) 하나다.
- JSON(변경 이력·템플릿·API)도 같은 꼴이다 — 사각형은 숫자 넷 `[x1, y1, x2, y2]`, 다각형은 점 목록 `[[x, y], …]`.
  DB `masks` 표는 x1~y2 에 바깥 사각형을, `poly` 열에 꼭짓점 JSON 을 둔다(사각형은 NULL — `row_values`·`from_row`).
- 꼬인 다각형(변끼리 엇갈림)은 홀짝 규칙이다 — 덮는 몫·칠하기·화면 그림이 모두 같은 규칙이다.

**규칙.**
- 학습 세트(`trainer.build_dataset`)와 이미지를 함께 내보내기(`export.export_yolo`)가 칠한 JPEG 을 쓴다. 영역들의
  합집합이 박스 넓이의 반(DROP) 이상을 덮으면 그 박스는 **그 라벨 파일에서만** 뺀다 — DB 박스는 그대로다.
- **test 의 영역은 '채점에서 빼는 자리' 다 (2026-11-24 사용자 결정)** — test 그림은 **칠하지 않는다**(평가는 실제 화면
  그대로 — `paints`). 대신 반 넘게 덮인 라벨은 test 라벨 파일에서도 빼고(`drops` — 모든 분할에 같다), 영역이 칠할 픽셀을
  `ignore/<분할>/<이름>.png`(라벨 파일과 같은 자리에 `labels` 대신 `ignore`)로 남겨, 평가 도구가 그 자리에 반 넘게 든 예측을
  헛 박스로 세지 않는다(`wsl/eval_test.py` 의 `IgnoreValidator` — `tools/test_baseline.py`·`test_tiles.py` 도 그것을 쓴다).
  COCO 의 iscrowd·VOC 의 difficult 와 같은 생각이다. 전에는 서버가 test 이미지에 영역을 받지 않았고(11-09) test 에서는 애매한
  물체도 박스로 그리게 했는데(11-10), 라벨링이 분할보다 먼저라 학습 규칙대로 덮은 그림이 나중에 test 로 가면 영역은 무시되고
  그 물체가 라벨 없이 채점됐다 — 정말 못 다는 물체는 짐작한 박스나 빈칸이 점수를 흔들었다.
- **'어느 박스가 라벨에서 빠지는가' 는 `drops`(덮인 몫이 DROP 이상), '그림을 칠하는가' 는 `paints`(test 빼고)** — 학습 세트·
  내보내기·데이터 점검(health.py)·평가 세트가 함께 쓴다(2026-11-15) — 그쪽에 판정을 따로 적지 말 것. 결과가 같은지는
  tests/test_health_dataset.py 가 견준다.
- 라벨된 박스와 겹친 몫은 그 물체도 칠해진 채로 학습된다 — 조금은 괜찮고(가려진 물체와 같다), 크게 덮어야 하면 그림
  가장자리면 잘라 낸다(`tools/crop_images.py`). 검수 화면은 반 미만으로 덮이는 박스를 '⚠ n% 가림' 으로 알린다.
- 원본을 직접 읽는 분석 도구(domain_review 등)는 영역을 모른다(읽기 전용이라 그대로 둔다). `tools/label_qa.py` 는 11-16 부터
  본다(`drops` — 11-24 부터 test 도).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageOps

IGNORE_DIR = "ignore"       # 평가에서 뺄 자리의 그림(`write_ignore`) — 라벨 폴더 `labels` 와 같은 자리의 이 이름 폴더

GRAY = 114                  # Ultralytics 레터박스·모자이크 바탕과 같은 회색
DROP = 0.5                  # 박스 넓이의 이만큼 이상이 가려지면 학습 라벨에서 뺀다
MAX_MASKS = 500             # 한 장에 둘 수 있는 영역 수(다각형도 하나로 센다) — 실수로 수천 개가 들어오지 않게만 막는다
                            # (11-11 에 50 → 500, 사용자 요청: 애매한 물체를 하나씩 촘촘히 덮으려고. app.js 의 MASK_MAX 도 같이)
MAX_POINTS = 100            # 다각형 하나의 꼭짓점 한도 (2026-11-19) — 덮는 몫이 변 수의 제곱으로 느는 것을 묶는다.
                            # 서버가 받을 때(400)와 화면(app.js 의 MASK_MAX_PTS)이 막는다. clean 은 이것으로 버리지 않는다
MIN_SIDE = 1e-4             # 이보다 좁은 영역은 버린다(박스의 yolo_io.Box.is_valid 와 같은 하한)

Rect = tuple[float, float, float, float]
Poly = tuple[tuple[float, float], ...]
Mask = Rect | Poly


def is_poly(m) -> bool:
    """다각형인가 — 첫 값이 점((x, y))이면 다각형, 숫자면 사각형(x1, y1, x2, y2)."""
    return len(m) > 0 and isinstance(m[0], (tuple, list))


def as_mask(m) -> Mask:
    """JSON·DB 에서 읽은 값(리스트)을 이 모듈의 꼴(튜플, float)로. 고르거나 자르지는 않는다(그것은 clean)."""
    if is_poly(m):
        return tuple((float(p[0]), float(p[1])) for p in m)
    return tuple(float(v) for v in m)


def bbox(m) -> Rect:
    """영역의 바깥 사각형(정규화 xyxy). 사각형은 그대로."""
    if not is_poly(m):
        return (m[0], m[1], m[2], m[3])
    xs = [p[0] for p in m]
    ys = [p[1] for p in m]
    return (min(xs), min(ys), max(xs), max(ys))


def to_wire(m) -> list:
    """JSON 에 쓰는 꼴 — 사각형 [x1, y1, x2, y2], 다각형 [[x, y], …] (API·템플릿·이력)."""
    if is_poly(m):
        return [[float(p[0]), float(p[1])] for p in m]
    return [float(v) for v in m]


def row_values(m) -> tuple[float, float, float, float, str | None]:
    """DB masks 행의 값 (x1, y1, x2, y2, poly) — 다각형은 바깥 사각형 + 꼭짓점 JSON(반올림하지 않는다), 사각형은 poly NULL.
    x1~y2 에 바깥 사각형을 두는 것은 poly 열을 모르는 옛 코드도 그 자리 둘레를 가리게 하려는 것이다."""
    if not is_poly(m):
        return (float(m[0]), float(m[1]), float(m[2]), float(m[3]), None)
    x1, y1, x2, y2 = bbox(m)
    return (float(x1), float(y1), float(x2), float(y2), json.dumps(to_wire(m), separators=(",", ":")))


def from_row(x1, y1, x2, y2, poly=None) -> Mask:
    """DB masks 행 → 영역. poly 가 있으면 다각형(x1~y2 는 그 바깥 사각형이라 버린다)."""
    if poly:
        return as_mask(json.loads(poly))
    return (float(x1), float(y1), float(x2), float(y2))


def _clean_rect(r) -> Rect | None:
    try:
        x1, y1, x2, y2 = (float(v) for v in r)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return None
    x1, x2 = sorted((min(1.0, max(0.0, x1)), min(1.0, max(0.0, x2))))
    y1, y2 = sorted((min(1.0, max(0.0, y1)), min(1.0, max(0.0, y2))))
    if x2 - x1 > MIN_SIDE and y2 - y1 > MIN_SIDE:
        return (x1, y1, x2, y2)
    return None


def poly_area(pts) -> float:
    """다각형의 넓이(신발끈 공식, 부호 없음). 꼬인 다각형은 실제 덮는 넓이와 다르다 — 크기 하한을 보는 데만 쓴다."""
    s = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2


def _clean_poly(m) -> Poly | None:
    pts: list[tuple[float, float]] = []
    for p in m:
        try:
            x, y = (float(v) for v in p)
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        q = (min(1.0, max(0.0, x)), min(1.0, max(0.0, y)))
        if not pts or pts[-1] != q:                 # 같은 점을 두 번 찍은 것(더블클릭)은 하나로
            pts.append(q)
    while len(pts) > 1 and pts[-1] == pts[0]:       # 닫는 점(첫 점과 같은 끝점)은 뺀다 — 변은 늘 끝에서 처음으로 닫힌다
        pts.pop()
    if len(pts) < 3:
        return None
    x1, y1, x2, y2 = bbox(pts)
    if x2 - x1 <= MIN_SIDE or y2 - y1 <= MIN_SIDE or poly_area(pts) <= MIN_SIDE * MIN_SIDE:
        return None                                 # 한 줄로 늘어선 점·점 하나 크기
    return tuple(pts)


def clean(rects: Iterable) -> list[Mask]:
    """영역 목록을 0~1 로 자르고, 좁거나 숫자가 아닌 것은 버린다. 순서는 그대로 둔다.

    사각형은 좌표를 0~1 로 자르고 x1<x2·y1<y2 로 맞춘다. 다각형은 꼭짓점마다 0~1 로 자르고, 잇달아 같은 점과 닫는 점을
    빼고, 3개 미만·한 줄로 늘어선 것은 버린다(꼭짓점 순서는 그대로 — '같은 자리' 판정이 순서대로 견준다).
    """
    out: list[Mask] = []
    for r in rects:
        try:
            poly = is_poly(r)
        except TypeError:
            continue
        c = _clean_poly(r) if poly else _clean_rect(r)
        if c is not None:
            out.append(c)
    return out


# '같은 자리' (2026-11-12) — 두 영역의 네 좌표가 모두 SAME 안이면 같은 영역으로 본다. 화면(app.js 의 sameRect·MASK_SAME)도
# 같은 식이다 — 고치면 둘 다(tests/test_js_parity.py 가 대조한다). 다각형(2026-11-19)은 꼭짓점 수가 같고 **순서대로** 모든
# 좌표가 SAME 안일 때다(같은 모양을 다른 점에서 시작해 그린 것은 다른 영역으로 본다 — 템플릿·이력은 받은 순서를 그대로 둔다).
#
# 그 전에는 소수 7자리로 반올림한 값(mask_key, 화면은 xyxyKey)을 견줬다. 반올림에는 경계가 있어서, 경계 위의 값은 끝자리
# 잡음만으로 옆 칸으로 넘어간다 — 그리고 픽셀에 딱 맞춘 좌표는 2560·1280 폭에서 그 경계(소수 8번째 자리가 정확히 5,
# 예: 1326/2560 = 0.51796875)에 자주 걸린다. 화면은 영역을 {cx, cy, w, h} 로 들었다 xyxy 로 되돌려 끝자리가 달라지므로,
# 같은 템플릿을 두 번 더하면 겹쳐 들어가거나 빼기가 그 영역을 남길 수 있었다(무작위 2만 개 실측: 2560×1440 에서 화면을 한 번
# 오가면 키가 바뀌는 영역 8.4%, JS 의 Math.round 와 Python 의 round 가 다른 영역 22.8%. 손으로 그린 영역은 0).
#
# SAME 은 '7자리로 반올림하며 생기는 차(5e-8 까지 — 변경 이력의 영역은 그렇게 반올림돼 있다, store._masks_json)' 보다 크고
# '7자리 한 칸(1e-7)' 보다 작게 잡았다: 이력에서 되읽은 영역은 원래 영역과 같은 자리이고, 한 칸 떨어진 영역은 다른 자리다.
# 2560px 그림에서 0.0002px 이다.
SAME = 7.5e-8


def same_rect(a: Sequence, b: Sequence) -> bool:
    """두 영역이 같은 자리인가 — 사각형끼리는 네 좌표, 다각형끼리는 꼭짓점 수가 같고 순서대로 모든 좌표의 차가 SAME
    이하. 사각형과 다각형은 늘 다르다(이름은 사각형만 있던 때의 것이다)."""
    pa, pb = is_poly(a), is_poly(b)
    if pa != pb:
        return False
    if pa:
        return len(a) == len(b) and all(abs(p[0] - q[0]) <= SAME and abs(p[1] - q[1]) <= SAME
                                        for p, q in zip(a, b))
    return (abs(a[0] - b[0]) <= SAME and abs(a[1] - b[1]) <= SAME
            and abs(a[2] - b[2]) <= SAME and abs(a[3] - b[3]) <= SAME)


def has_rect(rects: Iterable[Sequence], r: Sequence) -> bool:
    """rects 안에 r 과 같은 자리의 영역이 있나."""
    return any(same_rect(c, r) for c in rects)


def same_rects(a: Sequence[Sequence], b: Sequence[Sequence]) -> bool:
    """영역 목록 둘이 같은가 — 개수가 같고 순서대로 같은 자리(영역이 '바뀌었나' 를 가른다)."""
    return len(a) == len(b) and all(same_rect(x, y) for x, y in zip(a, b))


# 가릴 영역 템플릿 (2026-11-10, 사용자 요청) — 이름을 붙여 둔 영역 묶음을 이미지에 더하고 뺀다. 적용하면 이미지의
# 영역으로 **복사**되므로, 빼기는 템플릿과 같은 자리(same_rect)의 영역만 뺀다 — 손으로 옮긴 영역은 그 이미지의 것이다.
# 검수 화면의 한 장 적용(app.js applyMaskTemplate)도 같은 규칙이다. 다각형도 같다.

def merge(cur: Sequence[Sequence], add: Sequence[Sequence]) -> list[Mask]:
    """cur 뒤에 add 가운데 cur 에 없는(같은 자리가 아닌) 영역만 붙인다. 순서는 그대로."""
    out = [as_mask(r) for r in cur]
    for r in add:
        r = as_mask(r)
        if not has_rect(out, r):
            out.append(r)
    return out


def subtract(cur: Sequence[Sequence], rem: Sequence[Sequence]) -> list[Mask]:
    """cur 에서 rem 과 같은 자리의 영역을 뺀다."""
    gone = [as_mask(r) for r in rem]
    return [c for c in (as_mask(r) for r in cur) if not has_rect(gone, c)]


def covered(box, rects: Sequence) -> float:
    """박스 넓이 가운데 영역들의 **합집합**이 덮는 몫(0~1). 겹친 영역을 두 번 세지 않는다.

    **사각형만 걸린 박스**는 좌표 압축으로 정확히 잰다 — 박스에 걸친 영역들의 경계로 칸을 나눠, 영역마다 제 칸에 표시한
    뒤 표시된 칸의 넓이를 줄(i)·칸(j) 순서로 더한다. 11-11 전에는 칸마다 모든 영역을 견줘 한 박스에 걸친 영역 p 개에 p³
    이었다(한도를 500 으로 올리며 바꿈 — 실측: 영역 200개가 걸친 박스 하나 0.95초 → 7ms, 500개 13초 → 57ms. 값은 옛 식과
    같다 — 무작위 6천 건에서 차이 1e-15 이하). **다각형이 하나라도 걸리면**(2026-11-19) 그 박스는 세로 띠로 나눠 잰다
    (`_slab_union` — 역시 근사 없이 정확하다). 사각형만 걸린 박스의 값은 다각형을 더하기 전과 끝자리까지 같다.
    합은 sum() 이 아니라 차례로 더한다 — 3.12+ 의 sum() 은 보정 합이라 화면(JS)과 끝자리가 달라진다. 정규화 좌표로 재도
    비율은 같다(가로·세로를 각각 늘려도 넓이 비는 그대로다). 웹 화면(app.js 의 maskCover·maskSlabs)도 같은 식이다 —
    고치면 둘 다 고칠 것(tests/test_js_parity.py 가 끝자리까지 견준다).
    """
    bx1, by1 = box.cx - box.w / 2, box.cy - box.h / 2
    bx2, by2 = box.cx + box.w / 2, box.cy + box.h / 2
    area = (bx2 - bx1) * (by2 - by1)
    if area <= 0:
        return 0.0
    parts = []
    polys = []
    for m in rects:
        if is_poly(m):
            x1, y1, x2, y2 = bbox(m)
            if x2 > bx1 and x1 < bx2 and y2 > by1 and y1 < by2:
                polys.append(m)
            continue
        x1, y1, x2, y2 = m
        a, t, c, d = max(x1, bx1), max(y1, by1), min(x2, bx2), min(y2, by2)
        if c > a and d > t:
            parts.append((a, t, c, d))
    if polys:
        return min(1.0, _slab_union(bx1, by1, bx2, by2, parts, polys) / area)
    if not parts:
        return 0.0
    xs = sorted({v for p in parts for v in (p[0], p[2])})
    ys = sorted({v for p in parts for v in (p[1], p[3])})
    xi = {v: i for i, v in enumerate(xs)}
    yi = {v: j for j, v in enumerate(ys)}
    ny = len(ys) - 1
    cell = bytearray((len(xs) - 1) * ny)        # 칸 (i, j) 가 덮였나 — 줄 i 마다 ny 칸
    for a, t, c, d in parts:
        j1, j2 = yi[t], yi[d]
        for i in range(xi[a], xi[c]):
            cell[i * ny + j1:i * ny + j2] = b"\x01" * (j2 - j1)
    inside = 0.0
    for i in range(len(xs) - 1):
        dx = xs[i + 1] - xs[i]
        row = i * ny
        for j in range(ny):
            if cell[row + j]:
                inside += dx * (ys[j + 1] - ys[j])
    return min(1.0, inside / area)


def _slab_union(bx1: float, by1: float, bx2: float, by2: float, parts: list, polys: list) -> float:
    """박스 [bx1, bx2]×[by1, by2] 안에서 사각형 조각(parts — 이미 박스로 자른 (a, t, c, d))과 다각형들의 합집합 넓이.

    세로 띠로 나눈다. 띠의 경계는 박스·사각형의 x, 다각형 꼭짓점의 x, 그리고 변끼리·변과 가로선(박스와 사각형의 위아래)이
    엇갈리는 x 다 — 그 사이에서는 아무것도 서로 엇갈리지 않아 세로 단면에 걸린 길이(합집합)가 x 에 대해 직선이고, 직선은
    '가운데 단면 × 띠 폭' 이 넓이와 정확히 같다. 다각형의 안은 홀짝 규칙(가운데 세로선이 변과 몇 번 만나는가)으로 가른다.
    박스 밖의 엇갈림은 단면을 박스로 자르면 꺾이지 않으니 박스에 닿는 변끼리만 본다. 화면 app.js 의 maskSlabs 와 연산 순서까지
    같은 식이다 — 고치면 둘 다.
    """
    per = []
    for pts in polys:
        n = len(pts)
        per.append([(pts[i][0], pts[i][1], pts[(i + 1) % n][0], pts[(i + 1) % n][1]) for i in range(n)])
    xs = {bx1, bx2}
    lines = [by1, by2]
    for a, t, c, d in parts:
        xs.add(a)
        xs.add(c)
        lines.append(t)
        lines.append(d)
    near = []
    for es in per:
        for e in es:
            xs.add(e[0])
            if (max(e[0], e[2]) >= bx1 and min(e[0], e[2]) <= bx2
                    and max(e[1], e[3]) >= by1 and min(e[1], e[3]) <= by2):
                near.append(e)
    for i in range(len(near)):
        xa, ya, xb, yb = near[i]
        for h in lines:
            if (ya - h) * (yb - h) < 0:
                xs.add(xa + (h - ya) * (xb - xa) / (yb - ya))
        for j in range(i + 1, len(near)):
            xc, yc, xd, yd = near[j]
            rx, ry, sx, sy = xb - xa, yb - ya, xd - xc, yd - yc
            den = rx * sy - ry * sx
            if den == 0:
                continue
            qx, qy = xc - xa, yc - ya
            t = (qx * sy - qy * sx) / den
            u = (qx * ry - qy * rx) / den
            if 0 < t < 1 and 0 < u < 1:
                xs.add(xa + t * rx)
    cuts = sorted(v for v in xs if bx1 <= v <= bx2)
    inside = 0.0
    for k in range(len(cuts) - 1):
        x0, x1 = cuts[k], cuts[k + 1]
        xm = (x0 + x1) / 2
        spans = []
        for a, t, c, d in parts:
            if a < xm < c:
                spans.append((t, d))
        for es in per:
            ys = []
            for xa, ya, xb, yb in es:
                if (xa <= xm < xb) or (xb <= xm < xa):
                    ys.append(ya + (xm - xa) * (yb - ya) / (xb - xa))
            ys.sort()
            for j in range(0, len(ys) - 1, 2):
                lo, hi = max(ys[j], by1), min(ys[j + 1], by2)
                if hi > lo:
                    spans.append((lo, hi))
        if not spans:
            continue
        spans.sort()
        lo, hi = spans[0]
        run = 0.0
        for s, e in spans[1:]:
            if s > hi:
                run += hi - lo
                lo, hi = s, e
            elif e > hi:
                hi = e
        run += hi - lo
        inside += (x1 - x0) * run
    return inside


def paints(split: str | None) -> bool:
    """이 분할의 그림에 가릴 영역을 **칠하는가** — test 만 칠하지 않는다(평가는 실제 화면 그대로).

    반 넘게 가려진 박스를 라벨에서 빼는 것(`drops`)은 분할과 상관없다 — test 에서는 그 자리를 채점에서 뺀다(머리말, 2026-11-24
    사용자 결정. 그 전 `applies_to` 는 test 에 칠하기·빼기를 둘 다 안 했다). **이 예외는 여기 한 곳에 둔다** — 학습 세트
    (trainer.build_dataset)·내보내기(export_yolo)가 함께 쓴다.
    """
    return split != "test"


def drops(box, rects: Sequence) -> bool:
    """영역들이 이 박스를 학습 라벨에서 빼는가 — 합집합이 박스 넓이의 DROP 이상을 덮는다(split_boxes·health 가 함께 쓴다)."""
    return bool(rects) and covered(box, rects) >= DROP


def split_boxes(boxes: Sequence, rects: Sequence) -> tuple[list, list]:
    """(학습 라벨에 남길 박스, 가려져 뺄 박스). 영역이 없으면 전부 남긴다."""
    if not rects:
        return list(boxes), []
    keep, drop = [], []
    for b in boxes:
        (drop if drops(b, rects) else keep).append(b)
    return keep, drop


def pixel_rect(r: Rect, width: int, height: int) -> tuple[int, int, int, int]:
    """정규화 xyxy → 칠할 픽셀 [x1, x2) × [y1, y2). 반올림이라 픽셀로 그은 영역은 그대로 돌아온다."""
    x1, y1, x2, y2 = r
    return (max(0, min(width, round(x1 * width))), max(0, min(height, round(y1 * height))),
            max(0, min(width, round(x2 * width))), max(0, min(height, round(y2 * height))))


def poly_pixels(pts, width: int, height: int) -> tuple[int, int, np.ndarray] | None:
    """다각형이 칠하는 픽셀 — (첫 줄, 첫 칸, bool[줄, 칸]) 또는 None(한 픽셀도 없음).

    **픽셀 중심이 다각형 안(홀짝 규칙)이면 칠한다** — 덮는 몫(covered)이 재는 넓이와 같은 뜻이다. 줄마다 중심 높이의 가로선이
    변과 만나는 x 에서 칠하기를 켜고 끈다(그 x 이상의 중심부터 — 칸 ceil(x − 0.5)). 변은 아래쪽 끝을 넣고 위쪽 끝은 뺀다
    (꼭짓점을 두 번 세지 않게).
    """
    P = [(float(x) * width, float(y) * height) for x, y in pts]
    xs = [p[0] for p in P]
    ys = [p[1] for p in P]
    r0, r1 = max(0, math.ceil(min(ys) - 0.5)), min(height, math.ceil(max(ys) - 0.5))
    c0, c1 = max(0, math.ceil(min(xs) - 0.5)), min(width, math.ceil(max(xs) - 0.5))
    if r1 <= r0 or c1 <= c0:
        return None
    rows = np.arange(r0, r1, dtype=np.float64) + 0.5
    flip = np.zeros((r1 - r0, c1 - c0 + 1), np.uint8)        # 켜고 끄는 횟수 — 짝홀만 보니 넘쳐도(256) 된다
    n = len(P)
    for i in range(n):
        xa, ya = P[i]
        xb, yb = P[(i + 1) % n]
        if ya == yb:
            continue
        lo, hi = (ya, yb) if ya < yb else (yb, ya)
        hit = np.nonzero((rows >= lo) & (rows < hi))[0]
        if not len(hit):
            continue
        x = xa + (rows[hit] - ya) * (xb - xa) / (yb - ya)
        col = np.clip(np.ceil(x - 0.5), c0, c1).astype(np.int64) - c0
        np.add.at(flip, (hit, col), 1)
    inside = (np.cumsum(flip, axis=1, dtype=np.uint8) & 1).astype(bool)[:, :c1 - c0]
    return r0, c0, inside


def paint_array(arr: np.ndarray, rects: Sequence) -> np.ndarray:
    """그림 배열(줄, 칸[, 색])의 영역을 그 자리에서 회색으로 칠하고 돌려준다. 회색은 모든 채널이 같아 RGB·BGR 이 같다."""
    h, w = arr.shape[:2]
    for r in rects:
        if is_poly(r):
            hit = poly_pixels(r, w, h)
            if hit is not None:
                r0, c0, inside = hit
                arr[r0:r0 + inside.shape[0], c0:c0 + inside.shape[1]][inside] = GRAY
            continue
        x1, y1, x2, y2 = pixel_rect(r, w, h)
        arr[y1:y2, x1:x2] = GRAY
    return arr


def write_ignore(src: Path, dst: Path, rects: Sequence) -> None:
    """평가에서 뺄 자리의 그림 — 원본(EXIF 회전을 적용한 화면 크기)에서 영역이 **칠할 바로 그 픽셀**(`paint_array`)을 흰색으로
    한 1비트 PNG 를 dst 에 쓴다 (2026-11-24, 머리말 'test 의 영역'). 평가 도구는 프로젝트 코드를 import 하지 않으니(pod·WSL)
    영역 좌표 대신 이 그림을 읽는다(`wsl/eval_test.py` 의 `IgnoreValidator` — 예측 박스 안 픽셀의 반 이상이 흰색이면 뺀다).
    반쯤 쓴 파일이 남지 않게 임시 이름으로 쓴 뒤 바꾼다."""
    from .yolo_io import display_size

    w, h = display_size(src)
    arr = paint_array(np.zeros((h, w), np.uint8), rects)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    Image.fromarray(arr > 0).save(tmp, "PNG")
    tmp.replace(dst)


def paint(img: Image.Image, rects: Sequence) -> Image.Image:
    """회전을 적용한 RGB 그림의 영역을 회색으로 칠한 새 그림."""
    return Image.fromarray(paint_array(np.array(img.convert("RGB")), rects))


def write_masked(src: Path, dst: Path, rects: Sequence, *, quality: int = 95) -> None:
    """원본을 EXIF 회전을 적용해 열고 영역을 칠해 JPEG 으로 쓴다(원본은 그대로).

    회전을 적용해 쓰므로 새 파일에는 EXIF 가 없다 — 라벨이 회전을 적용한 화면 기준이라 그대로 맞는다
    (CLAUDE.md 'EXIF 회전 이미지'). 반쯤 쓴 파일이 남지 않게 임시 이름으로 쓴 뒤 바꾼다.
    """
    from .yolo_io import plain_image_open       # Ultralytics 가 갈아 끼운 Image.open 을 거치지 않는다(그 머리말)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with plain_image_open()(src) as im:
        out = paint(ImageOps.exif_transpose(im), rects)
    tmp = dst.with_name(dst.name + ".tmp")
    out.save(tmp, "JPEG", quality=quality)
    tmp.replace(dst)


def clip_poly(pts, rect: Sequence[float]) -> list[tuple[float, float]]:
    """다각형을 사각형 [x1, x2]×[y1, y2] 안으로 자른다(Sutherland–Hodgman — 네 변으로 차례로 자름). 좌표계는 받은 그대로라
    픽셀로 넣으면 픽셀로 나온다(tools/crop_images.py 가 그림을 자를 때). 다 밖이면 빈 목록, 3개 미만이면 쓸 수 없다."""
    x1, y1, x2, y2 = (float(v) for v in rect)

    def cut_x(a, b, x):
        return (x, a[1] + (x - a[0]) * (b[1] - a[1]) / (b[0] - a[0]))

    def cut_y(a, b, y):
        return (a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1]), y)

    out = [(float(x), float(y)) for x, y in pts]
    for keep, cut in ((lambda p: p[0] >= x1, lambda a, b: cut_x(a, b, x1)),
                      (lambda p: p[0] <= x2, lambda a, b: cut_x(a, b, x2)),
                      (lambda p: p[1] >= y1, lambda a, b: cut_y(a, b, y1)),
                      (lambda p: p[1] <= y2, lambda a, b: cut_y(a, b, y2))):
        src, out = out, []
        for i, cur in enumerate(src):
            prev = src[i - 1]
            if keep(cur):
                if not keep(prev):
                    out.append(cut(prev, cur))
                out.append(cur)
            elif keep(prev):
                out.append(cut(prev, cur))
        if not out:
            break
    return out
