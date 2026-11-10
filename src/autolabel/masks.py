"""가릴 영역 (2026-11-09) — 사람을 포함해 라벨하기 애매한 물체(가를 수 없는 무리·조각만 보이는 사람 등, 판정은
docs/label-rules.md)를 **학습 세트를 만들 때만** 회색(114)으로 칠한다.

**왜.** YOLO(Ultralytics) 에는 '이 자리는 보지 마' 하는 무시 영역이 없다. 애매한 물체(가를 수 없는 사람 무리 등)를 그리지 않고 두면 모델은
그것을 '그 물체 아님(배경)'으로 배우고, 장을 통째로 빼면 나머지 라벨까지 잃는다. 그 자리만 학습 때 레터박스·모자이크
바탕색(114)으로 칠하면 거기서는 아무것도 배우지 않는다. 처음(11-09 오후)에는 그림 파일에 직접 칠했다(그때의
`tools/mask_images.py`). 원본을 바꾸지 않으려고 영역을 DB `masks` 표에 두고 학습 세트를 만들 때만 칠하게 바꿨다 —
검수 화면에서는 M(가릴 영역) 모드로 그린다.

**규칙.**
- 좌표는 박스와 같은 화면(EXIF 회전을 적용한) 기준 정규화 xyxy `(x1, y1, x2, y2)`.
- 학습 세트(`trainer.build_dataset`)와 이미지를 함께 내보내기(`export.export_yolo`)가 칠한 JPEG 을 쓴다. 영역들의
  합집합이 박스 넓이의 반(DROP) 이상을 덮으면 그 박스는 **그 라벨 파일에서만** 뺀다 — DB 박스는 그대로다.
- **test 는 칠하지 않는다** — 평가는 실제 화면 그대로여야 한다. 서버는 test 이미지(분할이 test 이거나 cameras.yaml 에서
  test 로 고정한 카메라)에 영역을 받지 않고, 학습 세트도 test 분할은 원본·박스 그대로 쓴다. `tools/test_baseline.py` 는
  원래부터 원본 그림과 DB 박스를 읽는다.
- 라벨된 박스와 겹친 몫은 그 물체도 칠해진 채로 학습된다 — 조금은 괜찮고(가려진 물체와 같다), 크게 덮어야 하면 그림
  가장자리면 잘라 낸다(`tools/crop_images.py`). 검수 화면은 반 미만으로 덮이는 박스를 '⚠ n% 가림' 으로 알린다.
- 원본을 직접 읽는 분석 도구(label_qa·test_tiles·domain_review 등)는 영역을 모른다(읽기 전용이라 그대로 둔다) —
  label_qa 는 칠한 자리의 물체를 '빠짐' 후보로 낼 수 있다.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageOps

GRAY = 114                  # Ultralytics 레터박스·모자이크 바탕과 같은 회색
DROP = 0.5                  # 박스 넓이의 이만큼 이상이 가려지면 학습 라벨에서 뺀다
MAX_MASKS = 50              # 한 장에 둘 수 있는 영역 수 (실수로 수천 개가 들어오지 않게)
MIN_SIDE = 1e-4             # 이보다 좁은 영역은 버린다(박스의 yolo_io.Box.is_valid 와 같은 하한)

Rect = tuple[float, float, float, float]


def clean(rects: Iterable[Sequence[float]]) -> list[Rect]:
    """정규화 xyxy 목록을 0~1 로 자르고, 좁거나 숫자가 아닌 것은 버린다. 순서는 그대로 둔다."""
    out: list[Rect] = []
    for r in rects:
        try:
            x1, y1, x2, y2 = (float(v) for v in r)
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            continue
        x1, x2 = sorted((min(1.0, max(0.0, x1)), min(1.0, max(0.0, x2))))
        y1, y2 = sorted((min(1.0, max(0.0, y1)), min(1.0, max(0.0, y2))))
        if x2 - x1 > MIN_SIDE and y2 - y1 > MIN_SIDE:
            out.append((x1, y1, x2, y2))
    return out


def mask_key(rects: Iterable[Sequence[float]]) -> list[tuple]:
    """영역 목록이 '바뀌었나' 를 가르는 값 (store._box_key 처럼 부동소수 잡음은 같은 것으로 본다)."""
    return [tuple(round(float(v), 7) for v in r) for r in rects]


def covered(box, rects: Sequence[Rect]) -> float:
    """박스 넓이 가운데 영역들의 **합집합**이 덮는 몫(0~1). 겹친 영역을 두 번 세지 않는다.

    좌표 압축으로 정확히 잰다 — 영역이 많아야 몇 개라 칸 수가 작다. 정규화 좌표로 재도 비율은 같다(가로·세로를 각각
    늘려도 넓이 비는 그대로다). 웹 화면(app.js 의 maskCover)도 같은 식이다 — 고치면 둘 다 고칠 것.
    """
    bx1, by1 = box.cx - box.w / 2, box.cy - box.h / 2
    bx2, by2 = box.cx + box.w / 2, box.cy + box.h / 2
    area = (bx2 - bx1) * (by2 - by1)
    if area <= 0:
        return 0.0
    parts = []
    for x1, y1, x2, y2 in rects:
        a, t, c, d = max(x1, bx1), max(y1, by1), min(x2, bx2), min(y2, by2)
        if c > a and d > t:
            parts.append((a, t, c, d))
    if not parts:
        return 0.0
    xs = sorted({v for p in parts for v in (p[0], p[2])})
    ys = sorted({v for p in parts for v in (p[1], p[3])})
    inside = 0.0
    for i in range(len(xs) - 1):
        mx = (xs[i] + xs[i + 1]) / 2
        for j in range(len(ys) - 1):
            my = (ys[j] + ys[j + 1]) / 2
            if any(p[0] <= mx < p[2] and p[1] <= my < p[3] for p in parts):
                inside += (xs[i + 1] - xs[i]) * (ys[j + 1] - ys[j])
    return min(1.0, inside / area)


def split_boxes(boxes: Sequence, rects: Sequence[Rect]) -> tuple[list, list]:
    """(학습 라벨에 남길 박스, 가려져 뺄 박스). 영역이 없으면 전부 남긴다."""
    if not rects:
        return list(boxes), []
    keep, drop = [], []
    for b in boxes:
        (drop if covered(b, rects) >= DROP else keep).append(b)
    return keep, drop


def pixel_rect(r: Rect, width: int, height: int) -> tuple[int, int, int, int]:
    """정규화 xyxy → 칠할 픽셀 [x1, x2) × [y1, y2). 반올림이라 픽셀로 그은 영역은 그대로 돌아온다."""
    x1, y1, x2, y2 = r
    return (max(0, min(width, round(x1 * width))), max(0, min(height, round(y1 * height))),
            max(0, min(width, round(x2 * width))), max(0, min(height, round(y2 * height))))


def paint(img: Image.Image, rects: Sequence[Rect]) -> Image.Image:
    """회전을 적용한 RGB 그림의 영역을 회색으로 칠한 새 그림."""
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]
    for r in rects:
        x1, y1, x2, y2 = pixel_rect(r, w, h)
        arr[y1:y2, x1:x2] = GRAY
    return Image.fromarray(arr)


def write_masked(src: Path, dst: Path, rects: Sequence[Rect], *, quality: int = 95) -> None:
    """원본을 EXIF 회전을 적용해 열고 영역을 칠해 JPEG 으로 쓴다(원본은 그대로).

    회전을 적용해 쓰므로 새 파일에는 EXIF 가 없다 — 라벨이 회전을 적용한 화면 기준이라 그대로 맞는다
    (CLAUDE.md 'EXIF 회전 이미지'). 반쯤 쓴 파일이 남지 않게 임시 이름으로 쓴 뒤 바꾼다.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        out = paint(ImageOps.exif_transpose(im), rects)
    tmp = dst.with_name(dst.name + ".tmp")
    out.save(tmp, "JPEG", quality=quality)
    tmp.replace(dst)
