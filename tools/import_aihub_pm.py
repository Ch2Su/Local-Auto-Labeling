"""AI Hub '개인형 이동장치 안전 데이터'(PM) 프레임을 **검수 대기(pending)** 로 넣는다 — 운영 모델 예측 + AI Hub 의 PM
박스를 단서로 보탠 후보.

**사용자 결정(2026-11-22).** 시범으로 100장: 연출 CCTV 의 킥보드 프레임 60장 + 실증 CCTV 중 덜 붐비는 40장(오토바이·자전거,
야간 10장). 블랙박스는 뺀다(차 안 눈높이라 CCTV 가 아니다). 조사 기록은 docs/provenance.md '공개 데이터셋' 의 같은 이름 항목,
그 수치를 낸 스크립트는 runs/aihub_pm/.

**왜 검수 대기인가.** 이 데이터의 라벨은 사람이 탄 이륜 탈것(PM)뿐이고 **박스 하나가 탄 사람 + 탈것**이다. 우리 라벨은 탈것만
그리고 탄 사람은 person 으로 따로 그리며, 차·보행자·세워 둔 탈것도 전부 그린다 — 그래서 AI Hub 박스를 그대로 넣지 않는다.
운영 모델 예측을 먼저 넣고(자동 라벨링과 같은 설정 — predict.conf·iou·max_det·imgsz 와 차량 겹침 정리 predict.class_iou),
PM 박스는 '여기에 이 종류의 탈것과 탄 사람이 있다' 는 단서로만 쓴다:

- PM 박스 안에서 모델이 **탈것을 찾았으면**(모델의 bicycle·motorcycle·scooter 박스가 넓이의 0.7 이상 PM 박스 안에 있고 PM
  박스의 0.10 이상을 차지 — IoU 큰 것부터 하나씩 짝) 그 박스를 쓴다. 클래스가 AI Hub 종류와 다르면 **AI Hub 종류로 바꾼다**
  (출처 'aihub', 신뢰도 없음 — 검수 화면에 '수동' 처럼 보인다). 11-22 에 어긋난 36곳을 그림으로 보니 31곳은 AI Hub 가 맞았다
  (모델이 킥보드를 bicycle 로 잡은 22곳 중 21곳). 코드 15 는 종류가 섞여 있어(그 코드의 크롭 절반이 자전거) 모델 클래스를 둔다.
- **탈것을 못 찾았으면** PM 박스에서 어림한 자리에 후보 박스를 넣는다(출처 'aihub'). 자리는 모델이 탈것과 탄 사람을 둘 다
  찾은 PM 박스 616곳에서 잰 중앙값이다(`VEH_IN_PM` — 킥보드는 PM 박스의 아래 64%). 짝이 안 된 같은 종류의 모델 박스가 그
  후보와 IoU 0.1 이상 겹치거나 중심이 PM 박스 안이면 '모양만 다른 같은 물체' 로 보고 넣지 않는다.
- **탄 사람을 못 찾았으면**(모델의 person 박스가 넓이의 0.6 이상 PM 박스 안에 있고 PM 박스의 0.15 이상) person 후보를 넣는다
  (`PERSON_IN_PM`). 그 후보와 IoU 0.3 이상인 person 박스가 있으면 넣지 않는다.
- 후보는 **자리만 어림한 것**이다 — 앞·뒤에서 본 킥보드처럼 사람에 가린 탈것은 검수에서 박스를 고쳐야 한다. 모델 예측(차·
  보행자·세워 둔 탈것)도 평소처럼 사람이 봐야 한다.

**PM 코드 → 종류.** 13–19 오토바이 · 20–27 자전거 · 28–36 킥보드(34 는 없다). AI Hub 페이지에는 코드표가 없어 코드별 크롭과
운영 모델 예측으로 확인한 대응이다. 좌표가 숫자가 아니거나(검증 라벨에 NaN 1개) 크기가 0 이하인 박스, 화면 안에 넓이의 반도
안 남는 박스, 같은 물체에 또 붙은 박스(IoU 0.8 이상 — 학습 라벨의 0.7%)는 버린다.

**고르는 법.** 라벨 색인(학습 라벨 58만 개의 PM 박스를 한 파일로 — 처음 한 번 7분쯤, `<DB 폴더>/aihub_pm/index.jsonl`)에서
묶음(`BUCKETS`)마다 그 종류의 PM 박스(높이 `--min-height` 80px 이상)가 있는 프레임을 찾고, **영상마다 한 장**만 쓴다. 영상
순서와 프레임은 `--seed` 로 섞는다(같은 입력·같은 DB 면 같은 프레임을 고른다). 후보를 하나씩 읽어
1. 운영 모델을 돌려 예측 박스가 `--max-boxes`(16)개를 넘으면 건너뛴다 — 검수할 박스가 많은 붐비는 화면(실증 주간 300장의
   78% 가 16개를 넘었다, 연출 킥보드 200장은 7%).
2. 이미 고른 프레임이나 DB 의 AI Hub 이미지와 **같은 카메라 자리**면 건너뛴다(`same_camera`) — 둘 중 하나면 같은 자리다:
   32×18 흑백 축소본의 상관이 `SAME_SCENE`(0.70) 이상(11-22 에 영상 쌍을 그림으로 보니 0.70–0.75 는 8쌍 중 7쌍이 같은 자리,
   0.60–0.65 는 8쌍 중 1쌍), 또는 ORB 특징점으로 맞춘 호모그래피의 인라이어가 12개 이상이고 그 변환으로 두 화면이 반 이상
   겹침. **밝기 상관만으로는 같은 자리의 주간·야간, 조금 돌아간 화면을 놓친다** — 처음에 상관만으로 고른 100장에서 특징점
   정합이 같은 자리 5쌍을 더 찾았다(그림으로 확인 — 주간·야간 한 쌍의 상관은 0.35 였다). 같은 자리를 여러 번 찍은 영상이
   많다: 영상 300개씩을 이 판정으로 묶으면 실증 137자리(가장 큰 묶음 35개)·연출 168자리(가장 큰 묶음 86개)다.
   **같은 교차로를 다른 자리·다른 줌으로 찍은 것은 여전히 다른 카메라로 들어간다**(프로젝트 기준으로도 다른 단위지만 배경이
   겹친다 — 분할을 정할 때 볼 것). 견주는 시간은 이미 있는 화면 수에 비례한다(한 쌍 10ms 쯤 — 수천 장을 넣게 되면 줄일 것).
묶음의 몫이 차거나, 후보가 떨어지거나, 몫의 30배(적어도 100장)를 읽을 때까지 돈다. 이 도구가 전에 넣은 영상(DB 의 이름으로
안다 — 휴지통 포함)은 다시 고르지 않는다. `--scale 0.5` 처럼 몫을 줄이거나 늘릴 수 있고, 다시 돌리면 새 영상에서 그만큼 더 넣는다.

**카메라.** 이 데이터에는 카메라 번호가 없다. 고른 프레임마다(화면이 다른 것만 골랐다) 카메라 `ah-NNNN` 을 `cameras.yaml` 에
등록한다(`provenance.add_camera` — kind dataset, place `AI Hub PM 영상 <영상 id>`, 쓰기 전 파일은 `backups/cameras_<시각>.yaml`
에 한 번). 분할은 카메라째라, 같은 자리의 다른 영상을 나중에 넣게 되면 같은 키로 넣어야 한다(지금은 같은 화면이면 건너뛴다).

**쓰는 것.** 그림은 AI Hub 의 JPEG 바이트 그대로 `<unlabeled>/ah-NNNN/<AI Hub 이름>.jpg` 에 쓰고(풀린 폴더에 없으면 zip 에서
그 항목만 꺼낸다), DB 에 source='aihub_pm'·camera·status='pending' 으로 넣은 뒤 domain 을 'cctv' 로 정한다. 촬영 시각은
모른다. 이 도구가 넣은 경로는 scan 이 이미 아는 경로라 건너뛴다. model_version 은 `<가중치 파일>+aihub`.

**순서.** 기본은 보기만 한다(파일·DB·등록 파일을 건드리지 않는다 — 고르고, 읽고, 추론까지 해서 셈과 대조 시트
`<DB 폴더>/aihub_pm/preview_<시각>_NN.jpg` 를 보인다). `--apply` 면 DB 를 `labeling.db.bak_aihubpm_<시각>` 으로 떠 두고
카메라 등록 → 그림 쓰기 → DB 순으로 쓴다. 도중에 실패하면 이번에 쓴 그림과 등록을 되돌린다. 목록은
`runs/aihub_pm/import_<시각>.tsv`. 추론은 기본 CPU 다(`--device 0` 으로 GPU — 학습이 돌고 있으면 쓰지 말 것).

**주의.**
- `labeler.py label --all` 은 검수 대기 이미지를 모델 예측**만**으로 덮어써 AI Hub 후보를 지운다 — 쓰지 말 것. 웹 UI 의 자동
  라벨링은 검수 대기를 건드리지 않는다.
- **약관(AI Hub 데이터 이용정책):** 인공지능 학습용으로만, 승인 없는 제3자 제공 금지, 국외 반출은 수행기관·NIA 합의 필요. 이
  그림(카메라 `ah-*`)을 git 에 넣지 말고, Runpod·Roboflow 처럼 밖으로 올리기 전에 사용자에게 먼저 물을 것(`data/` 는 git 밖).
- 간판·표지판에 실제 상호·지명이 보인다 — 카메라 키·문서·커밋에 옮기지 말 것.

    python tools/import_aihub_pm.py                      # 무엇을 넣을지 보기만 (대조 시트)
    python tools/import_aihub_pm.py --device 0 --apply   # 시범 100장
    python tools/import_aihub_pm.py --scale 0.5          # 묶음마다 몫을 절반으로
    python tools/import_aihub_pm.py --reindex            # 라벨 색인을 다시 만든다
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import math
import os
import random
import re
import sqlite3
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # tools/ -> 저장소 루트

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from autolabel import tiling  # noqa: E402
from autolabel.config import Config, resolve  # noqa: E402
from autolabel.grouping import base_name  # noqa: E402
from autolabel.predictor import build_class_map  # noqa: E402
from autolabel.provenance import add_camera, load_cameras, remove_cameras  # noqa: E402
from autolabel.scoring import image_uncertainty  # noqa: E402
from autolabel.store import Store  # noqa: E402
from autolabel.yolo_io import Box  # noqa: E402

AIHUB_ROOT = Path(r"E:\datasets\120.개인형 이동장치 안전 데이터\01.데이터")
SOURCE = "aihub_pm"          # images.source — 공개 데이터셋은 데이터셋 이름을 쓴다(docs/provenance.md 'DB 변경')
ORIGIN = "aihub"             # boxes.origin — AI Hub 단서로 넣거나 클래스를 바꾼 박스
RIGHTS = "AI Hub(NIA) 개인형 이동장치 안전 데이터 — AI 학습용으로만, 제3자 제공·국외 반출 금지"
KEY_RE = re.compile(r"^ah-(\d{4,})$")
# <영상 id>_<클립>_<프레임>_<C|B 기기>_<D|N 주야>_<F|R 날씨>_<0|1 연출>
NAME_RE = re.compile(r"^([A-Z]\d{6})_(\d{3})_(\d{4})_([CB])_([DN])_([FR])_([01])$")

BIKES = ("bicycle", "motorcycle", "scooter")
MIXED_CODES = frozenset({15})   # 종류가 섞인 코드 — 모델 클래스를 AI Hub 종류로 바꾸지 않는다

DUP_IOU = 0.8            # 같은 물체에 또 붙은 PM 박스
CLIP_KEEP = 0.5          # 화면 안에 원래 넓이의 이만큼도 안 남는 PM 박스는 버린다
VEH_INSIDE = 0.7         # 모델 탈것 박스 넓이 중 PM 박스 안에 든 몫
VEH_MIN_SHARE = 0.10     # 그 겹침이 PM 박스에서 차지하는 몫
RIDER_INSIDE = 0.6       # 모델 person 박스 넓이 중 PM 박스 안에 든 몫
RIDER_MIN_SHARE = 0.15
NEAR_IOU = 0.1           # 탈것 후보와 짝 없는 같은 종류 모델 박스가 이만큼 겹치면 같은 물체로 본다
PERSON_NEAR_IOU = 0.3    # 사람 후보와 모델 person 박스
SAME_SCENE = 0.70        # 밝기 축소본 상관이 이 이상이면 같은 화면(카메라 자리)
THUMB = (32, 18)
SAME_EDGES = 0.80        # 윤곽 방향 무늬(edge_signature)의 내적이 이 이상이면 같은 자리로 본다
EDGE_SIZE, EDGE_CELL, EDGE_BINS = (160, 96), 16, 8
ORB_WIDTH = 960          # 특징점은 이 폭으로 줄인 그림에서 찾는다
ORB_FEATURES = 2000
ORB_INLIERS = 12         # 호모그래피 인라이어가 이 이상이고
ORB_OVERLAP = 0.5        # 그 변환으로 두 화면이 이만큼 겹치면 같은 카메라 자리
TRIES, TRIES_MIN = 30, 100   # 묶음마다 몫의 TRIES 배(적어도 TRIES_MIN 장)까지 읽어 보고 못 채우면 그만둔다

# PM 박스(탄 사람 + 탈것) 안에서 우리 기준의 탈것·사람 박스가 놓이는 자리 (x1, y1, x2, y2 — PM 박스 폭·높이에 대한 몫).
# 2026-11-22 에 운영 가중치가 탈것(코드와 같은 종류)과 탄 사람을 둘 다 찾은 PM 박스에서 잰 중앙값이다
# (오토바이 275 · 자전거 169 · 킥보드 172곳 — runs/aihub_pm/ 의 표본 500장).
VEH_IN_PM = {"motorcycle": (0.03, 0.20, 0.98, 0.99), "bicycle": (0.05, 0.40, 0.97, 0.99),
             "scooter": (0.13, 0.36, 0.91, 1.00)}
PERSON_IN_PM = {"motorcycle": (0.25, 0.02, 0.82, 0.81), "bicycle": (0.18, 0.01, 0.90, 0.82),
                "scooter": (0.07, 0.01, 0.95, 0.89)}

Rect = tuple[float, float, float, float]
Pred = tuple            # (클래스 이름, 신뢰도, x1, y1, x2, y2) — 픽셀


@dataclass(frozen=True)
class Bucket:
    """고를 묶음 하나 — 이 폴더들에서 이 종류의 PM 박스가 있는 프레임을 count 장."""
    name: str
    folders: tuple[str, ...]
    kind: str
    count: int


# 시범 100장(2026-11-22 사용자 결정): 연출 킥보드 60 + 실증 오토바이·자전거 40(야간 10)
BUCKETS = (
    Bucket("연출 주간 맑음 킥보드", ("CCTV/주간/맑음/연출",), "scooter", 46),
    Bucket("연출 주간 우천 킥보드", ("CCTV/주간/우천/연출",), "scooter", 6),
    Bucket("연출 야간 킥보드", ("CCTV/야간/맑음/연출", "CCTV/야간/우천/연출"), "scooter", 8),
    Bucket("실증 주간 오토바이", ("CCTV/주간/맑음/실증", "CCTV/주간/우천/실증"), "motorcycle", 15),
    Bucket("실증 주간 자전거", ("CCTV/주간/맑음/실증", "CCTV/주간/우천/실증"), "bicycle", 15),
    Bucket("실증 야간 오토바이", ("CCTV/야간/맑음/실증", "CCTV/야간/우천/실증"), "motorcycle", 5),
    Bucket("실증 야간 자전거", ("CCTV/야간/맑음/실증", "CCTV/야간/우천/실증"), "bicycle", 5),
)


# ───────────────────────────── PM 라벨 ─────────────────────────────

def pm_type(code) -> str | None:
    """PM_code → 탈것 종류(우리 클래스 이름). 모르는 코드는 None."""
    try:
        c = int(code)
    except (TypeError, ValueError):
        return None
    if 13 <= c <= 19:
        return "motorcycle"
    if 20 <= c <= 27:
        return "bicycle"
    if 28 <= c <= 36:
        return "scooter"
    return None


@dataclass(frozen=True)
class PM:
    code: int
    kind: str
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def box(self) -> Rect:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def height(self) -> float:
        return self.y2 - self.y1


def area(a: Rect) -> float:
    return max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])


def inter(a: Rect, b: Rect) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def iou(a: Rect, b: Rect) -> float:
    i = inter(a, b)
    u = area(a) + area(b) - i
    return i / u if u > 0 else 0.0


def inside(inner: Rect, outer: Rect) -> float:
    """inner 의 넓이 가운데 outer 와 겹친 몫."""
    a = area(inner)
    return inter(inner, outer) / a if a > 0 else 0.0


def clean_pm(raw, w: int, h: int) -> tuple[list[PM], collections.Counter]:
    """라벨의 PM 줄 `[code, x, y, 폭, 높이]` 들 → (쓸 PM 박스, 버린 이유별 수). 좌표는 화면 안으로 자른다."""
    out: list[PM] = []
    why: collections.Counter = collections.Counter()
    for row in raw or []:
        if not isinstance(row, (list, tuple)) or len(row) != 5:
            why["꼴이 틀림"] += 1
            continue
        code, *pts = row
        kind = pm_type(code)
        if kind is None:
            why["모르는 코드"] += 1
            continue
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in pts):
            why["좌표가 숫자가 아님"] += 1
            continue
        x, y, bw, bh = (float(v) for v in pts)
        if bw <= 0 or bh <= 0:
            why["크기가 0 이하"] += 1
            continue
        x1, y1, x2, y2 = max(0.0, x), max(0.0, y), min(float(w), x + bw), min(float(h), y + bh)
        if x2 <= x1 or y2 <= y1 or (x2 - x1) * (y2 - y1) < CLIP_KEEP * bw * bh:
            why["화면 밖"] += 1
            continue
        pm = PM(int(code), kind, x1, y1, x2, y2)
        if any(iou(pm.box, q.box) >= DUP_IOU for q in out):
            why["같은 물체에 또 하나"] += 1
            continue
        out.append(pm)
    return out, why


def place(p: Rect, frac: Rect) -> Rect:
    """PM 박스 p 안의 몫 frac 자리."""
    w, h = p[2] - p[0], p[3] - p[1]
    return (p[0] + frac[0] * w, p[1] + frac[1] * h, p[0] + frac[2] * w, p[1] + frac[3] * h)


@dataclass
class Hints:
    recls: dict[int, str] = field(default_factory=dict)       # 모델 박스 번호 → AI Hub 종류로 바꿀 클래스
    added: list[tuple] = field(default_factory=list)          # 후보 (클래스 이름, x1, y1, x2, y2)
    notes: collections.Counter = field(default_factory=collections.Counter)


def plan_hints(pm: list[PM], preds: list[Pred]) -> Hints:
    """PM 박스를 단서로 모델 예측에 보탤 것을 정한다(머리말 '왜 검수 대기인가'). 모델 박스는 하나도 지우지 않는다."""
    out = Hints()
    pairs = []
    for i, p in enumerate(pm):
        for j, d in enumerate(preds):
            if d[0] in BIKES and inside(d[2:], p.box) >= VEH_INSIDE and inter(d[2:], p.box) >= VEH_MIN_SHARE * area(p.box):
                pairs.append((iou(d[2:], p.box), -i, -j))     # IoU 큰 것부터, 같으면 앞 번호부터
    veh: dict[int, int] = {}
    used: set[int] = set()
    for _, ni, nj in sorted(pairs, reverse=True):
        i, j = -ni, -nj
        if i in veh or j in used:
            continue
        veh[i] = j
        used.add(j)

    for i, p in enumerate(pm):
        j = veh.get(i)
        if j is not None:
            if preds[j][0] == p.kind:
                out.notes["탈것 찾음"] += 1
            elif p.code in MIXED_CODES:
                out.notes["종류가 섞인 코드 — 모델 클래스 그대로"] += 1
            else:
                out.recls[j] = p.kind
                out.notes[f"클래스를 AI Hub 종류로 ({preds[j][0]}→{p.kind})"] += 1
        else:
            cand = place(p.box, VEH_IN_PM[p.kind])
            near = False
            for k, d in enumerate(preds):
                if k in used or d[0] != p.kind:
                    continue
                cx, cy = (d[2] + d[4]) / 2, (d[3] + d[5]) / 2
                if iou(cand, d[2:]) >= NEAR_IOU or (p.x1 <= cx <= p.x2 and p.y1 <= cy <= p.y2):
                    near = True
                    used.add(k)               # 이 모델 박스가 그 탈것이다 — 다른 PM 박스의 것으로 다시 세지 않는다
                    break
            if near:
                out.notes["탈것 모양만 다름"] += 1
            else:
                out.added.append((p.kind, *cand))
                out.notes[f"탈것 후보 ({p.kind})"] += 1
        rider = any(d[0] == "person" and inside(d[2:], p.box) >= RIDER_INSIDE
                    and inter(d[2:], p.box) >= RIDER_MIN_SHARE * area(p.box) for d in preds)
        if rider:
            out.notes["탄 사람 찾음"] += 1
            continue
        cand = place(p.box, PERSON_IN_PM[p.kind])
        if any(d[0] == "person" and iou(cand, d[2:]) >= PERSON_NEAR_IOU for d in preds):
            out.notes["탄 사람 모양만 다름"] += 1
        else:
            out.added.append(("person", *cand))
            out.notes["탄 사람 후보"] += 1
    return out


# ───────────────────────────── 모델 ─────────────────────────────

def to_box(cls: int, x1: float, y1: float, x2: float, y2: float, w: int, h: int,
           conf: float | None, origin: str) -> Box:
    return Box(cls, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h, conf, origin).clamped()


class Labeler:
    """운영 모델 예측(자동 라벨링과 같은 설정) + PM 단서로 한 장의 박스를 만든다."""

    def __init__(self, cfg: Config, classes: list[str], weights: str | None, device, model=None):
        self.classes = classes
        self.idx = {n: i for i, n in enumerate(classes)}
        missing = sorted(set(BIKES + ("person",)) - set(classes))
        if missing:
            raise SystemExit(f"DB 에 없는 클래스: {missing} (지금 {classes})")
        self.weights = str(resolve(weights)) if weights else cfg.active_weights()   # 자동 라벨링과 같은 가중치
        if model is None:
            from ultralytics import YOLO
            model = YOLO(self.weights)
        self.model = model
        self.cmap = build_class_map(model.names, classes)       # 모델 클래스 번호 → 프로젝트 클래스 번호
        if not self.cmap:
            raise SystemExit(f"모델 클래스({list(model.names.values())[:10]})와 프로젝트 클래스가 하나도 겹치지 않습니다")
        self.kw = dict(conf=float(cfg.get("predict.conf", 0.25)), iou=float(cfg.get("predict.iou", 0.6)),
                       max_det=int(cfg.get("predict.max_det", 100)),
                       imgsz=int(cfg.get("predict.imgsz") or cfg.imgsz), device=device, verbose=False)
        # 통째 자동 라벨링과 같은 차량 겹침 정리(predictor.AutoLabeler.run)
        self.class_iou = tiling.plain_class_iou(cfg)
        self.group = (frozenset(i for i in tiling.vehicle_ids(model.names) if i in self.cmap)
                      if self.class_iou > 0 else frozenset())
        self.bands = (float(cfg.get("review.low_band", 0.30)), float(cfg.get("review.high_band", 0.80)),
                      float(cfg.get("review.empty_score", 0.60)))
        self.version = f"{Path(self.weights).name}+aihub"

    def detect(self, img) -> list[Pred]:
        """그림 배열(BGR) → 프로젝트 클래스로 옮긴 예측 [(이름, 신뢰도, x1, y1, x2, y2)]."""
        r = self.model.predict(source=img, **self.kw)[0]
        if r.boxes is None:
            return []
        cls_l, conf_l, xyxy = r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()
        keep = tiling.cross_class_keep(xyxy, conf_l, cls_l, self.group, self.class_iou)
        return [(self.classes[self.cmap[int(c)]], float(cf), *(float(v) for v in b))
                for c, cf, b, k in zip(cls_l, conf_l, xyxy, keep) if k and int(c) in self.cmap]

    def boxes(self, preds: list[Pred], pm: list[PM], w: int, h: int) -> tuple[list[Box], float, Hints]:
        hints = plan_hints(pm, preds)
        out = []
        for j, d in enumerate(preds):
            if j in hints.recls:
                out.append(to_box(self.idx[hints.recls[j]], *d[2:], w, h, None, ORIGIN))
            else:
                out.append(to_box(self.idx[d[0]], *d[2:], w, h, round(d[1], 4), "auto"))
        out += [to_box(self.idx[name], *b, w, h, None, ORIGIN) for name, *b in hints.added]
        out = [b for b in out if b.is_valid()]
        score = image_uncertainty([d[1] for d in preds], low=self.bands[0], high=self.bands[1], empty=self.bands[2])
        return out, score, hints


# ───────────────────────────── 색인·그림 읽기 ─────────────────────────────

def index_path(cfg: Config) -> Path:
    return cfg.db.parent / "aihub_pm" / "index.jsonl"


def build_index(label_root: Path, out: Path) -> int:
    """라벨 json 을 전부 읽어 한 줄에 `[폴더, 이름, 폭, 높이, [[code, x, y, 폭, 높이], …], 영역 수]` 로 쓴다."""
    files = sorted((p.parent.relative_to(label_root).as_posix(), p.name) for p in label_root.rglob("*.json"))
    if not files:
        raise SystemExit(f"{label_root} 에 라벨 json 이 없습니다")
    print(f"라벨 색인을 만든다: {len(files):,}개 ({label_root})", flush=True)

    def load(item):
        rel, name = item
        try:
            d = json.loads((label_root / rel / name).read_bytes())
            ann = d.get("annotations") or {}
            pm = [[p.get("PM_code"), *(p.get("points") or [])] for p in ann.get("PM") or []]
            ds = d.get("description") or {}
            return [rel, name[:-5], ds.get("imageWidth"), ds.get("imageHeight"), pm, len(ann.get("environment") or [])]
        except Exception as exc:                   # 깨진 json 한 줄 때문에 전부를 버리지 않는다
            return [rel, name[:-5], None, None, f"ERR {type(exc).__name__}", 0]

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    t = time.time()
    with ThreadPoolExecutor(16) as ex, tmp.open("w", encoding="utf-8") as fo:
        for k, row in enumerate(ex.map(load, files, chunksize=256), 1):
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")
            if k % 100000 == 0:
                print(f"  {k:,}/{len(files):,} ({time.time() - t:.0f}초)", flush=True)
    tmp.replace(out)
    out.with_name("index.meta.json").write_text(
        json.dumps({"root": str(label_root), "files": len(files), "built": datetime.now().isoformat(timespec="seconds")},
                   ensure_ascii=False), encoding="utf-8")
    return len(files)


def read_index(path: Path):
    """색인의 줄 → (폴더, 이름, 폭, 높이, PM 줄들). 읽지 못한 라벨(PM 자리가 글자)은 건너뛴다."""
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            rel, stem, w, h, pm, *_ = json.loads(line)
            if isinstance(pm, list):
                yield rel, stem, w, h, pm


class Frames:
    """원천 그림을 읽는다 — 풀린 폴더에 있으면 거기서, 없으면 옆의 zip 에서 그 항목만."""

    def __init__(self, image_root: Path):
        self.root = image_root
        self.zips = sorted(image_root.glob("*.zip"))
        self._names: dict[str, tuple[int, zipfile.ZipInfo]] | None = None
        self._open: dict[int, zipfile.ZipFile] = {}

    def _zip_names(self) -> dict[str, tuple[int, zipfile.ZipInfo]]:
        if self._names is None:
            self._names = {}
            for k, zp in enumerate(self.zips):
                z = zipfile.ZipFile(zp)
                self._open[k] = z
                for info in z.infolist():
                    if not info.is_dir():
                        self._names.setdefault(info.filename, (k, info))
        return self._names

    def read(self, rel: str, stem: str) -> bytes | None:
        p = self.root / rel / f"{stem}.jpg"
        if p.is_file():
            return p.read_bytes()
        hit = self._zip_names().get(f"{rel}/{stem}.jpg")
        if hit is None:
            return None
        return self._open[hit[0]].read(hit[1])     # zipfile 이 CRC 를 확인한다

    def close(self) -> None:
        for z in self._open.values():
            z.close()
        self._open.clear()
        self._names = None


def signature(data: bytes) -> np.ndarray:
    """그림 바이트 → 32×18 흑백 축소본(평균 0·표준편차 1). 두 장의 밝기 무늬가 같은지는 `scene_corr` 로 본다."""
    im = Image.open(io.BytesIO(data))
    im.draft("L", (THUMB[0] * 8, THUMB[1] * 8))
    a = np.asarray(im.convert("L").resize(THUMB, Image.BILINEAR), dtype=np.float32).ravel()
    return (a - a.mean()) / (a.std() + 1e-6)


def scene_corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b) / a.size


@dataclass
class Scene:
    """한 프레임의 화면 지문 — 같은 카메라 자리인지 `same_camera` 로 견준다."""
    sig: np.ndarray                 # 밝기 축소본(signature)
    edges: np.ndarray               # 칸마다의 윤곽 방향 히스토그램(edge_signature)
    size: tuple[int, int]           # 특징점을 찾은 그림의 (폭, 높이) — 폭 ORB_WIDTH 로 줄인 것
    pts: np.ndarray                 # ORB 특징점 자리 (N×2)
    desc: np.ndarray | None         # ORB 기술자 (N×32)


def edge_signature(gray: np.ndarray) -> np.ndarray:
    """흑백 그림 → 160×96 으로 줄여 16px 칸마다 윤곽 방향(8칸, 방향의 부호는 안 봄)의 세기 히스토그램을 이은 단위 벡터.
    밝기가 달라져도(주간·야간) 차선·건물 윤곽의 방향은 남는다 — 두 장의 내적이 `SAME_EDGES` 이상이면 같은 자리로 본다."""
    import cv2

    g = cv2.GaussianBlur(cv2.resize(gray, EDGE_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32), (0, 0), 1.0)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.minimum((np.arctan2(gy, gx) % np.pi) / np.pi * EDGE_BINS, EDGE_BINS - 1).astype(int)
    out = []
    for y in range(0, EDGE_SIZE[1], EDGE_CELL):
        for x in range(0, EDGE_SIZE[0], EDGE_CELL):
            hist = np.bincount(ang[y:y + EDGE_CELL, x:x + EDGE_CELL].ravel(),
                               weights=mag[y:y + EDGE_CELL, x:x + EDGE_CELL].ravel(), minlength=EDGE_BINS)
            out.append(hist / (np.linalg.norm(hist) + 1e-6))
    v = np.concatenate(out).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-6)


def scene_of(data: bytes) -> Scene:
    import cv2

    g = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise ValueError("그림을 읽지 못했습니다")
    h, w = g.shape
    size = (ORB_WIDTH, max(1, round(h * ORB_WIDTH / w)))
    small = cv2.createCLAHE(2.0, (8, 8)).apply(cv2.resize(g, size))      # 밤 화면도 특징점이 잡히게 대비를 편다
    kps, desc = cv2.ORB_create(ORB_FEATURES).detectAndCompute(small, None)
    pts = np.float32([k.pt for k in kps]).reshape(-1, 2)
    return Scene(signature(data), edge_signature(g), size, pts, desc)


def orb_match(a: Scene, b: Scene) -> tuple[int, float]:
    """두 화면의 ORB 특징점을 짝지어 호모그래피를 맞춘다 → (인라이어 수, a 의 화면이 b 의 화면과 겹치는 몫).

    반복 무늬(횡단보도 줄)는 다른 장소끼리도 열 개 남짓 짝이 맞지만, 그때의 호모그래피는 화면을 뒤집거나 찌그러뜨려 겹치는
    몫이 0 이다 — 2026-11-22 에 밝기 상관만으로 고른 100장의 4,950쌍에서 같은 자리 5쌍은 인라이어 14–139·겹침 0.79–1.00,
    나머지는 인라이어 13 이하이고 그중 겹침이 반 이상인 쌍은 인라이어 7 이하였다(`runs/aihub_pm/scene_probe.py`)."""
    import cv2

    if a.desc is None or b.desc is None or len(a.pts) < 8 or len(b.pts) < 8:
        return 0, 0.0
    good = [m[0] for m in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(a.desc, b.desc, k=2)
            if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
    if len(good) < 8:
        return 0, 0.0
    src = np.float32([a.pts[m.queryIdx] for m in good]).reshape(-1, 1, 2)
    dst = np.float32([b.pts[m.trainIdx] for m in good]).reshape(-1, 1, 2)
    cv2.setRNGSeed(20261122)                     # RANSAC 표본이 실행마다 같게 — 같은 입력이면 같은 프레임을 고른다
    hm, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if hm is None:
        return 0, 0.0
    frame_a = np.float32([[0, 0], [a.size[0], 0], [a.size[0], a.size[1]], [0, a.size[1]]])
    try:
        warped = cv2.perspectiveTransform(frame_a.reshape(-1, 1, 2), hm).reshape(-1, 2)
    except cv2.error:
        return int(mask.sum()), 0.0
    if not np.isfinite(warped).all() or not cv2.isContourConvex(warped):
        return int(mask.sum()), 0.0
    quad = [(float(x), float(y)) for x, y in warped]
    if _area(quad) <= 0:                         # 뒤집힌 변환 — 같은 카메라의 화면은 거울상이 되지 않는다
        return int(mask.sum()), 0.0
    return int(mask.sum()), _area(_clip(quad, b.size[0], b.size[1])) / float(b.size[0] * b.size[1])


def _area(poly: list[tuple[float, float]]) -> float:
    """다각형의 부호 있는 넓이(화면 좌표에서 시계 방향이 양수 — 프레임 네 귀를 적은 순서)."""
    return sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1])) / 2.0


def _clip(poly: list[tuple[float, float]], w: float, h: float) -> list[tuple[float, float]]:
    """볼록 다각형을 화면 [0, w]×[0, h] 로 자른다(변마다 차례로 — 서덜랜드·호지먼). `cv2.intersectConvexConvex` 는 변이 정확히
    겹치는 두 다각형(같은 화면끼리)의 넓이를 틀리게 낸다(같은 그림 두 장에 0.63)."""
    for axis, bound, keep_less in ((0, 0.0, False), (0, float(w), True), (1, 0.0, False), (1, float(h), True)):
        out: list[tuple[float, float]] = []
        for p, q in zip(poly, poly[1:] + poly[:1]):
            pin = p[axis] <= bound if keep_less else p[axis] >= bound
            qin = q[axis] <= bound if keep_less else q[axis] >= bound
            if pin:
                out.append(p)
            if pin != qin:
                t = (bound - p[axis]) / (q[axis] - p[axis])
                out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
        poly = out
        if not poly:
            break
    return poly


def same_camera(a: Scene, b: Scene) -> bool:
    """같은 카메라 자리로 볼 것인가 — 셋 중 하나면 그렇다: 밝기 무늬가 같다(SAME_SCENE) · 윤곽 방향 무늬가 같다(SAME_EDGES) ·
    특징점이 한 화면으로 맞는다(ORB). **틀려서 다른 자리를 같다고 하면 그 후보를 건너뛸 뿐이지만, 같은 자리를 다르다고 하면 같은
    배경이 다른 카메라 키로 들어가 분할을 가로지를 수 있다** — 그래서 넉넉하게 잡는다."""
    if scene_corr(a.sig, b.sig) >= SAME_SCENE or float(a.edges @ b.edges) >= SAME_EDGES:
        return True
    inliers, overlap = orb_match(a, b)
    return inliers >= ORB_INLIERS and overlap >= ORB_OVERLAP


def video_of(name: str) -> str | None:
    """파일 이름(정리 꼬리표는 뗀다) → AI Hub 영상 id. 이 데이터의 이름이 아니면 None."""
    m = NAME_RE.match(base_name(name))
    return m.group(1) if m else None


# ───────────────────────────── 고르기 ─────────────────────────────

@dataclass
class Pick:
    bucket: str
    rel: str
    stem: str
    video: str
    data: bytes
    w: int
    h: int
    pm: list[PM]
    preds: list[Pred]
    boxes: list[Box]
    score: float
    hints: Hints
    scene: Scene
    camera: str = ""


def candidates(index, buckets, min_height: float) -> dict[str, dict[str, list]]:
    """색인 → {묶음 이름: {영상 id: [(폴더, 이름, PM 줄들), …]}} — 그 묶음 종류의 PM 박스(높이 min_height 이상)가 있는 프레임."""
    by_folder: dict[str, list[Bucket]] = collections.defaultdict(list)
    for b in buckets:
        for f in b.folders:
            by_folder[f].append(b)
    out: dict[str, dict[str, list]] = {b.name: collections.defaultdict(list) for b in buckets}
    for rel, stem, w, h, raw in index:
        hit = by_folder.get(rel)
        if not hit or not w or not h:
            continue
        video = video_of(stem)
        if video is None:
            continue
        pm, _ = clean_pm(raw, w, h)
        tall = {p.kind for p in pm if p.height >= min_height}
        for b in hit:
            if b.kind in tall:
                out[b.name][video].append((rel, stem, raw))
    return out


def decode(data: bytes):
    """JPEG 바이트 → (BGR 배열, 폭, 높이). 못 읽으면 None."""
    import cv2

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return img, img.shape[1], img.shape[0]


def choose(buckets, cands, frames: Frames, lab: Labeler, rng: random.Random, *, taken: set[str],
           scenes: list[Scene], max_boxes: int, log=print) -> tuple[list[Pick], dict[str, collections.Counter]]:
    """묶음마다 몫만큼 고른다. taken(이미 쓴 영상)·scenes(이미 있는 화면)는 고르는 대로 늘어난다.

    붐비는지(모델 한 번, 30ms 쯤)를 먼저 보고 같은 화면인지(이미 있는 화면마다 특징점 정합 10ms 쯤)를 나중에 본다 — 실증분은
    대부분 붐벼서 걸러지는데, 순서가 거꾸로면 그 프레임마다 화면 백 개와 견준다. 묶음마다 `TRIES`(몫의 배수)만큼 읽어도 몫을 못
    채우면 그만둔다(같은 자리 영상만 남은 폴더에서 끝없이 돌지 않게)."""
    import cv2

    # Ultralytics 는 import 때 OpenCV 스레드를 하나로 줄인다(DataLoader 와 부딪히지 않게). 여기는 워커가 없으니 고르는 동안만
    # 되돌린다 — 그대로 두면 특징점 정합이 3배쯤 느리다(2026-11-22 실측: 한 쌍 35ms → 9ms).
    threads = cv2.getNumThreads()
    cv2.setNumThreads(max(1, min(16, os.cpu_count() or 1)))
    picks: list[Pick] = []
    report: dict[str, collections.Counter] = {}
    try:
        for b in buckets:
            why: collections.Counter = collections.Counter()
            report[b.name] = why
            videos = sorted(cands.get(b.name, {}))
            rng.shuffle(videos)
            why["후보 영상"] = len(videos)
            got = tries = 0
            limit = max(TRIES_MIN, TRIES * b.count)
            for video in videos:
                if got >= b.count:
                    break
                if video in taken:
                    why["이미 쓴 영상"] += 1
                    continue
                if tries >= limit:
                    why["그만 봄(읽은 수 한도)"] = tries
                    break
                tries += 1
                rel, stem, raw = rng.choice(cands[b.name][video])
                data = frames.read(rel, stem)
                if data is None:
                    why["그림 없음"] += 1
                    continue
                try:
                    dec = decode(data)
                except Exception:
                    dec = None
                if dec is None:
                    why["그림을 못 읽음"] += 1
                    continue
                img, w, h = dec
                preds = lab.detect(img)
                if len(preds) > max_boxes:
                    why["붐빔"] += 1
                    continue
                scene = scene_of(data)
                if any(same_camera(scene, s) for s in scenes):
                    why["같은 화면"] += 1
                    continue
                pm, _ = clean_pm(raw, w, h)
                if not any(p.kind == b.kind for p in pm):
                    why["화면 크기가 라벨과 다름"] += 1       # 라벨의 크기로는 있던 박스가 실제 크기로는 안 남는다
                    continue
                boxes, score, hints = lab.boxes(preds, pm, w, h)
                picks.append(Pick(b.name, rel, stem, video, data, w, h, pm, preds, boxes, score, hints, scene))
                taken.add(video)
                scenes.append(scene)
                got += 1
                if got % 10 == 0:
                    log(f"  {b.name}: {got}/{b.count}")
            why["고름"] = got
    finally:
        cv2.setNumThreads(threads)
    return picks, report


def next_keys(taken: set[str], n: int) -> list[str]:
    """`ah-0001` … 의 다음 번호 n 개(가장 큰 번호 + 1 부터, 쓰인 키는 건너뛴다)."""
    k = max((int(m.group(1)) for key in taken if (m := KEY_RE.match(key))), default=0) + 1
    out: list[str] = []
    while len(out) < n:
        key = f"ah-{k:04d}"
        if key not in taken:
            out.append(key)
        k += 1
    return out


def used_keys(cfg: Config, store: Store) -> set[str]:
    """등록 파일·DB(휴지통 포함)·카메라 폴더가 쓰는 키."""
    keys = set(load_cameras(cfg.cameras)) | {k for k in store.camera_counts(with_trashed=True) if k}
    for root in (cfg.unlabeled, cfg.labeled / "images", cfg.labeled / "labels"):
        if root.is_dir():
            keys |= {p.name for p in root.iterdir() if p.is_dir()}
    return keys


def existing(store: Store) -> tuple[set[str], list[Scene]]:
    """이 도구가 전에 넣은 이미지(휴지통 포함) → (영상 id 들, 화면 지문들 — 파일이 남은 것만)."""
    videos: set[str] = set()
    scenes: list[Scene] = []
    with store.connect() as con:
        rows = con.execute("SELECT name, path FROM images WHERE source=?", (SOURCE,)).fetchall()
    for r in rows:
        v = video_of(r["name"])
        if v:
            videos.add(v)
        p = Path(r["path"])
        if p.is_file():
            try:
                scenes.append(scene_of(p.read_bytes()))
            except Exception:
                pass
    return videos, scenes


def view_text(rel: str) -> str:
    """'CCTV/주간/맑음/연출' → '연출 · 주간 · 맑음' (등록 파일의 view)."""
    parts = rel.split("/")
    return " · ".join(reversed(parts[1:])) if len(parts) == 4 else rel


# ───────────────────────────── 대조 시트 ─────────────────────────────

_COLORS = {"person": (0, 255, 255), "car": (0, 200, 0), "truck": (255, 128, 0), "bus": (255, 0, 255),
           "pickup": (255, 150, 150), "bicycle": (80, 160, 255), "motorcycle": (255, 60, 60), "scooter": (160, 255, 0)}


def _font(size: int = 12):
    """대조 시트의 글꼴 — 한글이 보이게 맑은 고딕을 찾고, 없으면 기본 글꼴(한글은 네모로 나온다)."""
    from PIL import ImageFont

    for name in ("malgun.ttf", "NanumGothic.ttf", "AppleGothic.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def write_sheets(picks: list[Pick], classes: list[str], out_dir: Path, stamp: str, per: int = 12) -> list[Path]:
    """고른 프레임을 박스와 함께 그려 본다 — 모델 박스는 클래스 색으로 가늘게, AI Hub 단서 박스는 노랑 굵게(이름표), AI Hub 의
    PM 박스(탄 사람 + 탈것)는 회색 가는 선."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cw, ch, cols = 640, 360, 3
    font = _font()
    paths: list[Path] = []
    for page in range(0, len(picks), per):
        part = picks[page:page + per]
        rows = (len(part) + cols - 1) // cols
        sheet = Image.new("RGB", (cw * cols, ch * rows), (24, 24, 24))
        for k, p in enumerate(part):
            im = Image.open(io.BytesIO(p.data)).convert("RGB").resize((cw, ch), Image.LANCZOS)
            dr = ImageDraw.Draw(im)
            sx, sy = cw / p.w, ch / p.h
            for q in p.pm:
                dr.rectangle((q.x1 * sx, q.y1 * sy, q.x2 * sx, q.y2 * sy), outline=(170, 170, 170), width=1)
            for b in p.boxes:
                x1, y1 = (b.cx - b.w / 2) * cw, (b.cy - b.h / 2) * ch
                x2, y2 = (b.cx + b.w / 2) * cw, (b.cy + b.h / 2) * ch
                name = classes[b.cls]
                if b.origin == ORIGIN:
                    dr.rectangle((x1, y1, x2, y2), outline=(255, 230, 0), width=2)
                    dr.text((x1 + 2, y2 + 1), name, fill=(255, 230, 0), font=font)
                else:
                    dr.rectangle((x1, y1, x2, y2), outline=_COLORS.get(name, (200, 200, 200)), width=1)
            label = f"{p.camera or '?'}  {p.stem}  [{p.bucket}]  박스 {len(p.boxes)}"
            dr.rectangle((0, 0, dr.textlength(label, font=font) + 8, 16), fill=(0, 0, 0))
            dr.text((4, 1), label, fill=(255, 255, 0), font=font)
            sheet.paste(im, ((k % cols) * cw, (k // cols) * ch))
        path = out_dir / f"preview_{stamp}_{page // per + 1:02d}.jpg"
        sheet.save(path, quality=85)
        paths.append(path)
    return paths


# ───────────────────────────── 넣기 ─────────────────────────────

def backup(cfg: Config, stamp: str) -> Path:
    bak = cfg.db.with_name(f"{cfg.db.name}.bak_aihubpm_{stamp}")
    with sqlite3.connect(cfg.db) as src, sqlite3.connect(bak) as dst:
        src.backup(dst)
    return bak


def apply(cfg: Config, store: Store, picks: list[Pick], version: str, stamp: str) -> tuple[int, Path]:
    """카메라 등록 → 그림 쓰기 → DB. 도중에 실패하면 이번에 쓴 그림과 등록을 되돌린다. → (넣은 장수, 목록 파일)."""
    backup_dir = cfg.db.parent / "backups"
    registered: list[str] = []
    written: list[Path] = []
    try:
        for n, p in enumerate(picks):
            add_camera(cfg.cameras, p.camera,
                       {"kind": "dataset", "place": f"AI Hub PM 영상 {p.video}", "mount": "unknown",
                        "view": view_text(p.rel), "resolution": f"{p.w}x{p.h}", "rights": RIGHTS},
                       backup_dir=backup_dir if n == 0 else None,        # 쓰기 전 파일은 한 번만 남긴다
                       comment=f"tools/import_aihub_pm.py {stamp}" if n == 0 else None)
            registered.append(p.camera)
        items = []
        for p in picks:
            dst = cfg.unlabeled / p.camera / f"{p.stem}.jpg"
            if dst.exists():
                raise RuntimeError(f"이미 있는 파일: {dst}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(p.data)
            written.append(dst)
            items.append({"path": str(dst.resolve()), "source": SOURCE, "width": p.w, "height": p.h,
                          "status": "pending", "boxes": p.boxes, "uncertainty": p.score, "model_version": version,
                          "camera": p.camera, "captured_at": None})
        store.ingest_batch(items)
    except BaseException:
        for f in written:
            f.unlink(missing_ok=True)
        for d in {f.parent for f in written}:
            try:
                d.rmdir()
            except OSError:
                pass
        if registered:
            try:
                remove_cameras(cfg.cameras, registered)
            except Exception as exc:                 # 되돌리지 못한 등록은 알린다 — 빈 카메라라 scan 은 막지 않는다
                print(f"[경고] 등록을 되돌리지 못했습니다({exc}) — {cfg.cameras} 에서 {registered[0]}…{registered[-1]} 를 볼 것")
        raise
    paths = [it["path"] for it in items]
    with store.connect() as con:
        ids = [r[0] for k in range(0, len(paths), 500)
               for r in con.execute(f"SELECT id FROM images WHERE path IN ({','.join('?' * len(paths[k:k + 500]))})",
                                    paths[k:k + 500])]
    store.set_domain(ids, "cctv")
    manifest = write_list(picks, resolve("runs") / "aihub_pm" / f"import_{stamp}.tsv")
    return len(ids), manifest


def write_list(picks: list[Pick], path: Path) -> Path:
    """고른 프레임의 목록(묶음·카메라·영상·AI Hub 이름·PM 코드·박스 수)을 TSV 로."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["bucket\tcamera\tvideo\tfolder\tname\tpm\tauto_boxes\taihub_boxes"]
    for p in picks:
        lines.append(f"{p.bucket}\t{p.camera}\t{p.video}\t{p.rel}\t{p.stem}\t"
                     f"{' '.join(f'{q.code}:{q.kind}' for q in p.pm)}\t"
                     f"{sum(1 for b in p.boxes if b.origin == 'auto')}\t{sum(1 for b in p.boxes if b.origin == ORIGIN)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def scaled(buckets, scale: float) -> tuple[Bucket, ...]:
    return tuple(Bucket(b.name, b.folders, b.kind, max(0, round(b.count * scale))) for b in buckets)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--root", type=Path, default=AIHUB_ROOT, help="AI Hub '01.데이터' 폴더")
    ap.add_argument("--scale", type=float, default=1.0, help="묶음마다의 몫에 곱한다 (1.0 = 시범 100장)")
    ap.add_argument("--seed", type=int, default=20261122, help="영상·프레임을 섞는 씨앗")
    ap.add_argument("--min-height", type=float, default=80.0, help="고를 종류의 PM 박스 최소 높이(px)")
    ap.add_argument("--max-boxes", type=int, default=16, help="모델 예측 박스가 이보다 많은 프레임은 건너뛴다")
    ap.add_argument("--weights", default=None, help="모델 (기본: 자동 라벨링과 같은 운영 가중치)")
    ap.add_argument("--device", default="cpu", help="추론 장치 (기본 cpu, GPU 는 0)")
    ap.add_argument("--reindex", action="store_true", help="라벨 색인을 다시 만든다")
    ap.add_argument("--no-sheet", action="store_true", help="대조 시트를 그리지 않는다")
    ap.add_argument("--apply", action="store_true", help="실제로 쓴다 (기본은 보기만)")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    store = Store(cfg.db)
    classes = store.get_classes()
    label_root = args.root / "1.Training" / "라벨링데이터"
    image_root = args.root / "1.Training" / "원천데이터"
    if not label_root.is_dir() or not image_root.is_dir():
        raise SystemExit(f"{args.root} 에 1.Training\\라벨링데이터·원천데이터 폴더가 없습니다 (--root)")
    idx = index_path(cfg)
    if args.reindex or not idx.is_file():
        build_index(label_root, idx)
    buckets = scaled(BUCKETS, args.scale)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("색인에서 후보를 찾는다…", flush=True)
    cands = candidates(read_index(idx), buckets, args.min_height)
    taken, scenes = existing(store)
    n_before = len(taken)
    lab = Labeler(cfg, classes, args.weights, args.device)
    frames = Frames(image_root)
    try:
        picks, report = choose(buckets, cands, frames, lab, random.Random(args.seed), taken=taken, scenes=scenes,
                               max_boxes=args.max_boxes)
    finally:
        frames.close()
    for p, key in zip(picks, next_keys(used_keys(cfg, store), len(picks))):
        p.camera = key

    print("\n묶음 | 몫 | 고름 | 후보 영상 | 건너뜀")
    for b in buckets:
        why = report[b.name]
        skipped = {k: v for k, v in why.items() if k not in ("고름", "후보 영상")}
        print(f"{b.name} | {b.count} | {why['고름']} | {why['후보 영상']:,} | {skipped or '-'}")
    n_auto = collections.Counter(classes[b.cls] for p in picks for b in p.boxes if b.origin == "auto")
    n_hint = collections.Counter(classes[b.cls] for p in picks for b in p.boxes if b.origin == ORIGIN)
    notes: collections.Counter = collections.Counter()
    for p in picks:
        notes.update(p.hints.notes)
    print(f"합계 {len(picks)}장 (이 도구가 전에 넣은 영상 {n_before}개는 빼고 골랐다)")
    print(f"  모델 박스 {sum(n_auto.values()):,}:", dict(sorted(n_auto.items())))
    print(f"  AI Hub 단서로 넣거나 클래스를 바꾼 박스 {sum(n_hint.values()):,}:", dict(sorted(n_hint.items())))
    print("  PM 박스마다:", dict(notes.most_common()))
    if picks and not args.no_sheet:
        sheets = write_sheets(picks, classes, idx.parent, stamp)
        print(f"  대조 시트 {len(sheets)}장: {sheets[0].parent}\\preview_{stamp}_NN.jpg")

    if not args.apply:
        if picks:
            print(f"  목록: {write_list(picks, idx.parent / f'preview_{stamp}.tsv')}")
        print(f"\n(보기만 했다. 넣으려면 --apply — {cfg.unlabeled}\\ah-NNNN\\ 에 쓰고 카메라를 등록하고 DB 에 검수 대기로 넣는다)")
        return 0
    if not picks:
        print("넣을 것이 없다.")
        return 0
    print(f"DB 를 떠 두었다: {backup(cfg, stamp)}")
    n, manifest = apply(cfg, store, picks, lab.version, stamp)
    print(f"\n검수 대기로 {len(picks)}장을 넣었다(도메인 CCTV {n}장, 카메라 {picks[0].camera}–{picks[-1].camera}). 목록: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
