"""VIRAT Ground Release 2.0 영상에서 프레임을 뽑아 **검수 대기(pending)** 로 넣는다 — 운영 모델 예측 + 모델이 그 자리에서
못 찾은 VIRAT 박스.

**사용자 결정(2026-11-07).** 장면(카메라)마다 50장. 박스는 처음에 'VIRAT 라벨 + 모델이 더 찾은 것' 으로 넣었다가, 같은
날 사용자가 지금 모델로 새로 라벨링하기를 원해 **모델 예측을 먼저 쓰고, 모델이 그 자리에서 아무것도 못 찾은 VIRAT 박스만
남기는** 쪽으로 바꿨다(`--relabel` 로 이미 넣은 550장도 바꿨다). 모델 예측만 쓰면 그 550장의 VIRAT 박스 8,360개 중
2,326개(사람 1,113·차 845·두 바퀴 368 — 대부분 먼 사람·가려진 차)가 사라졌다 — 검수 때 다 그리지 않으면 학습에서
배경으로 배운다. VIRAT 은 행동과 무관한 주차 차량·사람을 비워 두고(배포본 안내서 3.1, 0502 주차장은 표본 한 장에
VIRAT 차 2대·모델 20여 대), 운영 모델은 먼 사람을 놓친다(표본 66장에서 VIRAT 사람 박스의 36%) — 둘이 서로 메운다.

**장면 = 카메라** `virat-XXYY`(`data/visible/cameras.yaml` 에 먼저 등록 — kind dataset, docs/provenance.md). 프레임은
`<unlabeled>/virat-XXYY/<클립>_f<프레임 6자리>.jpg`(JPEG 95, 원본 해상도)로 쓰고 DB 에 source='virat'·camera·
status='pending' 로 넣은 뒤 domain 을 'cctv' 로 정한다. 촬영 시각은 모른다(클립 이름의 초는 원 영상 안의 위치다).
이 도구가 넣은 경로는 scan 이 이미 아는 경로라 건너뛴다(출처가 'collected' 로 바뀌지 않는다).

**고르는 법.** 객체 주석이 있는 클립만 본다(영상 14개는 주석 파일이 없어 뺀다). 사람·차·자전거 GT 가 하나라도 있는
프레임을 `--gap` 초(기본 5) 이상 떨어지게 추리고, 장면의 몫을 클립마다 후보 수에 비례해 나눈 뒤 클립 안에서는 고르게
뽑는다. 정해진 입력이면 늘 같은 프레임을 고른다. 이미 DB 에 있는 VIRAT 프레임(이름으로 안다 — 휴지통 행 포함)은
장면 몫에 세고, 그 클립에서는 그 프레임들에서 가장 먼 후보부터 고른다 — `--per-scene` 을 올려 다시 돌리면 모자란
만큼만, 기존 프레임과 떨어진 곳에서 더 넣는다.

**박스.**
- 모델(`--weights`, 기본 운영 가중치) 예측은 자동 라벨링과 같은 설정(predict.conf·iou·max_det·imgsz)으로 전부 넣는다
  (출처 'auto', 신뢰도). 운영 모델(visible-3)은 pickup 을 모른다 — 픽업은 검수에서 고친다.
- VIRAT 종류 1 person → person, 2 car·3 vehicle → car, 5 bike → bicycle(오토바이도 bike 다 — 검수에서 가른다).
  0(구역 상자)·4 object 는 버리고, 열 수·형이 틀린 줄은 건너뛴다.
- 같은 무리(사람 / 차: car·bus·truck·pickup / 두 바퀴: bicycle·motorcycle·scooter)끼리 IoU 0.5 이상을 큰 것부터 하나씩
  짝지어 짝이 생긴 VIRAT 박스는 버린다(모델 박스로 갈음). 짝이 없는 VIRAT 박스도, 짝 없는 같은 무리 모델 박스와 IoU 0.1
  이상 겹치거나 그 중심을 품으면 버린다(같은 물체를 모양만 다르게 그린 것). 남은 VIRAT 박스만 출처 'virat'·신뢰도
  없음(검수 화면에 '수동')으로 더한다. `--virat-only` 면 모델 없이 VIRAT 박스만 넣는다.

**다시 라벨링(`--relabel`).** 새 프레임을 넣지 않고, DB 의 VIRAT 이미지 가운데 검수 대기이고 사람이 손대지 않은 것
(변경 이력이 자동 라벨링(predict) 것뿐인 것)을 위 규칙으로 다시 라벨링한다 — 주석은 파일에서 다시 읽고 모델은 저장된
JPEG 에 돌린다. `store.apply_prediction` 으로 바꾸므로 이전 박스는 변경 이력에 남고, 도는 사이 사람이 바꾼 이미지는
덮지 않는다. 모델을 새로 학습한 뒤 남은 검수 대기를 다시 라벨링할 때도 쓴다.

**순서.** 기본은 보기만 한다(파일·DB 를 건드리지 않는다 — 고르고, 프레임을 읽고, 추론까지 해서 셈을 보인다).
`--apply` 면 DB 를 `labeling.db.bak_virat_<시각>`(다시 라벨링은 `bak_viratrelabel_`)으로 떠 두고 쓴다. 넣을 때는 넣은
목록을 `runs/virat_import/<시각>.tsv` 에 남긴다. 추론은 기본 CPU 다(`--device 0` 으로 GPU — 학습이 돌고 있으면 쓰지
말 것).

**주의.** `labeler.py label --all` 은 검수 대기 이미지를 모델 예측**만**으로 덮어써 VIRAT 박스를 지운다 — 쓰지 말 것
(다시 라벨링은 `--relabel`). 웹 UI 의 자동 라벨링은 검수 대기를 건드리지 않는다. VIRAT Release 2.0 은 재배포 금지로
본다 — 프레임을 밖에 올리지(로보플로 등) 말 것(`data/` 는 git 밖이다).

    python tools/import_virat.py                          # 무엇을 넣을지 보기만
    python tools/import_virat.py --apply                  # 장면당 50장
    python tools/import_virat.py --per-scene 100 --apply  # 장면당 100장이 되게 더 넣기
    python tools/import_virat.py --relabel --device 0 --apply   # 이미 넣은 검수 대기를 지금 모델로 다시
    python tools/import_virat.py --scenes 0000,0502       # 몇 장면만
"""

from __future__ import annotations

import argparse
import bisect
import collections
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # tools/ -> 저장소 루트

from autolabel.config import Config, resolve  # noqa: E402
from autolabel.grouping import base_name  # noqa: E402
from autolabel.provenance import load_cameras  # noqa: E402
from autolabel.scoring import image_uncertainty  # noqa: E402
from autolabel.store import Store  # noqa: E402
from autolabel.yolo_io import Box  # noqa: E402

VIRAT_ROOT = Path(r"E:\datasets\VIRAT_Ground_partial\VIRAT Ground Dataset")
GT_CLASS = {1: "person", 2: "car", 3: "car", 5: "bicycle"}       # 0 구역 상자·4 object 는 버린다
GROUP = {"person": "person", "car": "vehicle", "bus": "vehicle", "truck": "vehicle", "pickup": "vehicle",
         "bicycle": "bike", "motorcycle": "bike", "scooter": "bike"}
MATCH_IOU = 0.5          # VIRAT 박스와 모델 박스를 같은 물체로 짝짓는 IoU
NEAR_IOU = 0.1           # 짝 없는 VIRAT 박스를 '모양만 다른 같은 물체' 로 보는 IoU (또는 모델 박스 중심을 품음)
OBJ_SUFFIX = ".viratdata.objects.txt"
NAME_RE = re.compile(r"^(VIRAT_S_\d{6}(?:_\d{2}_\d{6}_\d{6})?)_f(\d{6})$")
RIGHTS = "VIRAT Release 2.0 — 개인별 동의 약관, 재배포 금지로 볼 것"
HUMAN_SOURCES = "('ui', 'bulk', 'restore', 'revert')"    # box_history.source 중 사람이 한 것


def scene_of(clip: str) -> str:
    return clip[8:12]                    # VIRAT_S_XXYYZZ… → XXYY


def camera_of(scene: str) -> str:
    return f"virat-{scene}"


def frame_file(clip: str, frame: int) -> str:
    return f"{clip}_f{frame:06d}.jpg"


@dataclass
class Pick:
    scene: str
    clip: str
    frame: int
    gt: list                             # [(클래스 이름, x1, y1, x2, y2)] 원본 픽셀


def load_objects(path: Path) -> tuple[dict[int, list], int]:
    """{프레임: [(클래스 이름, x1, y1, x2, y2)]} 와 건너뛴(틀린) 줄 수. 열: id 길이 프레임 x y w h 종류."""
    frames: dict[int, list] = collections.defaultdict(list)
    bad = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        p = line.split()
        if not p:
            continue
        try:
            if len(p) != 8:
                raise ValueError
            _oid, _dur, fr, x, y, w, h, typ = (int(v) for v in p)
        except ValueError:
            bad += 1
            continue
        name = GT_CLASS.get(typ)
        if name and w > 0 and h > 0 and fr >= 0:
            frames[fr].append((name, float(x), float(y), float(x + w), float(y + h)))
    return frames, bad


def video_meta(path: Path) -> tuple[float, int]:
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        return cap.get(cv2.CAP_PROP_FPS) or 30.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def thin(frames: list[int], gap: int, taken: list[int]) -> list[int]:
    """오름차순 프레임에서, 앞에서부터 서로 그리고 이미 넣은 프레임(taken)과 gap 이상 떨어진 것만."""
    taken = sorted(taken)
    out: list[int] = []
    for f in frames:
        if out and f - out[-1] < gap:
            continue
        i = bisect.bisect_left(taken, f)
        if (i < len(taken) and taken[i] - f < gap) or (i > 0 and f - taken[i - 1] < gap):
            continue
        out.append(f)
    return out


def allocate(quota: int, sizes: dict[str, int]) -> dict[str, int]:
    """quota 를 후보 수에 비례해 나눈다(최대 나머지 방식, 후보 수를 넘지 않게, 같으면 이름순)."""
    out = {k: 0 for k in sizes}
    total = sum(sizes.values())
    quota = min(quota, total)
    if quota <= 0:
        return out
    raw = {k: quota * n / total for k, n in sizes.items()}
    for k in sizes:
        out[k] = min(int(raw[k]), sizes[k])
    rest = quota - sum(out.values())
    for k in sorted(sizes, key=lambda k: (-(raw[k] - int(raw[k])), k)):
        if rest <= 0:
            break
        if out[k] < sizes[k]:
            out[k] += 1
            rest -= 1
    for k in sorted(sizes):              # 위에서 후보 수에 막혀 남은 몫
        while rest > 0 and out[k] < sizes[k]:
            out[k] += 1
            rest -= 1
    return out


def spread(cands: list[int], k: int, taken: list[int]) -> list[int]:
    """후보에서 k 개를 고르게. 이미 넣은 프레임(taken)이 없으면 구간 가운데들을, 있으면 기존·이번에 고른 프레임에서
    가장 먼 후보를 하나씩(같으면 앞의 것) — 늘 가운데를 고르면 더 넣을 때마다 기존 프레임의 `--gap` 바로 옆이 뽑혔다
    (2026-11-07 시험: 5초 떨어진 거의 같은 프레임)."""
    n = len(cands)
    if k >= n:
        return list(cands)
    if not taken:
        return [cands[int((i + 0.5) * n / k)] for i in range(k)]
    taken = sorted(taken)

    def gap_to(f: int) -> int:
        i = bisect.bisect_left(taken, f)
        return min(abs(f - taken[j]) for j in (i - 1, i) if 0 <= j < len(taken))

    dist = [gap_to(f) for f in cands]
    out: list[int] = []
    for _ in range(k):
        j = max(range(n), key=lambda i: (dist[i], -i))
        out.append(cands[j])
        dist = [min(d, abs(f - cands[j])) for d, f in zip(dist, cands)]
    return sorted(out)


def existing_frames(store: Store) -> dict[str, list[int]]:
    """이미 DB 에 있는 VIRAT 프레임 — {클립: [프레임]}. 이름으로 알아보고(정리 꼬리표는 뗀다) 휴지통 행도 센다.
    카메라로도 찾는다 — 쓰다 끊긴 프레임을 scan 이 'collected' 로 넣었어도 다시 고르지 않게."""
    out: dict[str, list[int]] = collections.defaultdict(list)
    with store.connect() as con:
        for (name,) in con.execute("SELECT name FROM images WHERE source='virat' OR camera LIKE 'virat-%'"):
            m = NAME_RE.match(base_name(name))
            if m:
                out[m.group(1)].append(int(m.group(2)))
    return out


def choose(root: Path, scenes: set[str] | None, per_scene: int, gap_sec: float,
           have: dict[str, list[int]]) -> tuple[list[Pick], dict[str, dict]]:
    adir, vdir = root / "annotations", root / "videos_original"
    by_scene: dict[str, list[str]] = collections.defaultdict(list)
    for p in sorted(adir.glob(f"*{OBJ_SUFFIX}")):
        clip = p.name[: -len(OBJ_SUFFIX)]
        if (vdir / f"{clip}.mp4").exists() and (not scenes or scene_of(clip) in scenes):
            by_scene[scene_of(clip)].append(clip)
    picks: list[Pick] = []
    info: dict[str, dict] = {}
    for sc, clips in sorted(by_scene.items()):
        cands: dict[str, list[int]] = {}
        bad = 0
        for c in clips:                  # 1차: 후보 프레임만 남긴다(주석 전체를 장면째 들고 있지 않게)
            frames, b = load_objects(adir / f"{c}{OBJ_SUFFIX}")
            bad += b
            fps, n = video_meta(vdir / f"{c}.mp4")
            cands[c] = thin(sorted(f for f in frames if f < n), max(1, round(gap_sec * fps)), have.get(c, []))
        n_have = sum(len(have.get(c, [])) for c in clips)
        alloc = allocate(max(0, per_scene - n_have), {c: len(v) for c, v in cands.items()})
        for c in clips:                  # 2차: 고른 프레임의 GT 만 다시 읽는다
            want = spread(cands[c], alloc[c], have.get(c, []))
            if not want:
                continue
            frames, _ = load_objects(adir / f"{c}{OBJ_SUFFIX}")
            picks += [Pick(sc, c, f, frames[f]) for f in want]
        info[sc] = {"clips": len(clips), "have": n_have, "new": sum(alloc.values()),
                    "cands": sum(len(v) for v in cands.values()), "bad": bad}
    return picks, info


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def merge(gt: list, preds: list) -> tuple[list[int], collections.Counter]:
    """gt [(이름, x1, y1, x2, y2)], preds [(이름, 신뢰도, x1, y1, x2, y2)] → (남길 VIRAT 박스 번호, 버린 이유별 수).

    모델 예측이 먼저다. 같은 무리끼리 IoU 0.5 이상을 큰 것부터 하나씩 짝지은 VIRAT 박스는 모델 박스로 갈음하고, 짝 없는
    VIRAT 박스 중 짝 없는 같은 무리 모델 박스와 IoU 0.1 이상이거나 그 중심을 품은 것도 버린다(모양만 다른 같은 물체 —
    짝 지어진 모델 박스는 보지 않는다: 겹쳐 선 두 사람 중 하나만 찾았으면 나머지 하나는 남긴다).
    """
    pairs = sorted(((iou(g[1:], p[2:]), gi, pj) for gi, g in enumerate(gt) for pj, p in enumerate(preds)
                    if GROUP[g[0]] == GROUP.get(p[0])), reverse=True)
    used_g: set[int] = set()
    used_p: set[int] = set()
    for v, gi, pj in pairs:
        if v < MATCH_IOU:
            break
        if gi in used_g or pj in used_p:
            continue
        used_g.add(gi)
        used_p.add(pj)
    why: collections.Counter = collections.Counter()
    keep: list[int] = []
    for gi, g in enumerate(gt):
        if gi in used_g:
            why["모델과 짝"] += 1
            continue
        near = False
        for pj, p in enumerate(preds):
            if pj in used_p or GROUP.get(p[0]) != GROUP[g[0]]:
                continue
            cx, cy = (p[2] + p[4]) / 2, (p[3] + p[5]) / 2
            if iou(g[1:], p[2:]) >= NEAR_IOU or (g[1] <= cx <= g[3] and g[2] <= cy <= g[4]):
                near = True
                break
        if near:
            why["모양만 다름"] += 1
        else:
            keep.append(gi)
    return keep, why


def to_box(cls: int, x1: float, y1: float, x2: float, y2: float, w: int, h: int,
           conf: float | None, origin: str) -> Box:
    return Box(cls, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h, conf, origin).clamped()


class Labeler:
    """모델 예측 + 모델이 못 찾은 VIRAT 박스로 한 장의 박스를 만든다(넣기·다시 라벨링이 같이 쓴다)."""

    def __init__(self, cfg: Config, classes: list[str], weights: str | None, device: str, use_model: bool):
        self.class_idx = {n: i for i, n in enumerate(classes)}
        self.classes = classes
        self.model = None
        self.weights = str(resolve(weights)) if weights else cfg.active_weights()   # 자동 라벨링과 같은 가중치
        if use_model:
            from ultralytics import YOLO
            self.model = YOLO(self.weights)
            unknown = sorted(set(self.model.names.values()) - set(GROUP))
            if unknown:
                raise SystemExit(f"모델 클래스 {unknown} 를 어느 무리로 볼지 모릅니다 (GROUP)")
        self.kw = dict(conf=float(cfg.get("predict.conf", 0.25)), iou=float(cfg.get("predict.iou", 0.6)),
                       max_det=int(cfg.get("predict.max_det", 100)),
                       imgsz=int(cfg.get("predict.imgsz") or cfg.imgsz), device=device, verbose=False)
        self.bands = (float(cfg.get("review.low_band", 0.30)), float(cfg.get("review.high_band", 0.80)),
                      float(cfg.get("review.empty_score", 0.60)))
        self.version = "virat" if self.model is None else f"{Path(self.weights).name}+virat"
        self.n_model: collections.Counter = collections.Counter()
        self.n_virat: collections.Counter = collections.Counter()
        self.dropped: collections.Counter = collections.Counter()

    def boxes(self, source, gt: list) -> tuple[list[Box], float, int, int]:
        """source 는 이미지 배열이나 경로. → (박스, 불확실도, 너비, 높이)."""
        preds, w, h = [], None, None
        if self.model is not None:
            r = self.model.predict(source, **self.kw)[0]
            h, w = r.orig_shape
            for k, c, xyxy in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
                preds.append((self.model.names[int(k)], float(c), *map(float, xyxy)))
        else:
            if isinstance(source, (str, Path)):
                from PIL import Image
                with Image.open(source) as im:
                    w, h = im.size
            else:
                h, w = source.shape[:2]
        keep, why = merge(gt, preds) if preds else (list(range(len(gt))), collections.Counter())
        out = [to_box(self.class_idx[p[0]], *p[2:], w, h, round(p[1], 4), "auto") for p in preds
               if p[0] in self.class_idx]
        out += [to_box(self.class_idx[gt[i][0]], *gt[i][1:], w, h, None, "virat") for i in keep]
        out = [b for b in out if b.is_valid()]
        self.n_model.update(self.classes[b.cls] for b in out if b.origin == "auto")
        self.n_virat.update(self.classes[b.cls] for b in out if b.origin == "virat")
        self.dropped.update(why)
        score = image_uncertainty([p[1] for p in preds], low=self.bands[0], high=self.bands[1], empty=self.bands[2])
        return out, score, w, h

    def report(self) -> None:
        print(f"  모델 박스 {sum(self.n_model.values()):,}:", dict(sorted(self.n_model.items())))
        print(f"  모델이 못 찾아 남긴 VIRAT 박스 {sum(self.n_virat.values()):,}:", dict(sorted(self.n_virat.items())))
        if self.dropped:
            print("  버린 VIRAT 박스:", dict(self.dropped.most_common()))


def backup(cfg: Config, tag: str, stamp: str) -> None:
    bak = cfg.db.with_name(f"{cfg.db.name}.bak_{tag}_{stamp}")
    with sqlite3.connect(cfg.db) as src, sqlite3.connect(bak) as dst:
        src.backup(dst)
    print(f"DB 를 떠 두었다: {bak}")


def relabel(args, cfg: Config, store: Store, lab: Labeler, scenes: set[str] | None, stamp: str) -> int:
    """DB 의 VIRAT 검수 대기(사람이 손대지 않은 것)를 지금 규칙·모델로 다시 라벨링한다."""
    adir = args.root / "annotations"
    with store.connect() as con:
        rows = con.execute("SELECT id, path, name, status, updated_at FROM images WHERE source='virat' ORDER BY id"
                           ).fetchall()
        human = {r[0] for r in con.execute(f"SELECT DISTINCT image_id FROM box_history WHERE source IN {HUMAN_SOURCES}")}
    targets, skipped = [], collections.Counter()
    for r in rows:
        m = NAME_RE.match(base_name(r["name"]))
        if not m:
            skipped["이름으로 클립·프레임을 모름"] += 1
        elif scenes and scene_of(m.group(1)) not in scenes:
            continue
        elif r["status"] != "pending":
            skipped[f"상태가 {r['status']}"] += 1
        elif r["id"] in human:
            skipped["사람이 고친 이력이 있음"] += 1
        elif not Path(r["path"]).exists():
            skipped["파일 없음"] += 1
        else:
            targets.append((m.group(1), int(m.group(2)), r))
    print(f"다시 라벨링할 것 {len(targets)}장" + (f" (건너뜀 {dict(skipped)})" if skipped else ""))
    if args.apply and targets:
        backup(cfg, "viratrelabel", stamp)
    done = changed = 0
    gt_cache: tuple[str, dict] | None = None
    for i, (clip, frame, r) in enumerate(sorted(targets, key=lambda t: (t[0], t[1])), 1):
        if gt_cache is None or gt_cache[0] != clip:
            obj = adir / f"{clip}{OBJ_SUFFIX}"
            gt_cache = (clip, load_objects(obj)[0] if obj.exists() else {})
        boxes, score, _w, _h = lab.boxes(r["path"], gt_cache[1].get(frame, []))
        if args.apply:
            if store.apply_prediction(r["id"], boxes, expect=(r["status"], r["updated_at"]), uncertainty=score,
                                      model_version=lab.version):
                done += 1
            else:
                changed += 1             # 도는 사이 사람이 승인·수정했다 — 덮지 않는다
        if i % 100 == 0:
            print(f"  {i}/{len(targets)}", flush=True)
    lab.report()
    if args.apply:
        print(f"\n다시 라벨링 {done}장" + (f", 도는 사이 바뀌어 그대로 둔 것 {changed}장" if changed else ""))
    else:
        print("\n(보기만 했다. 바꾸려면 --apply — DB 를 떠 두고 이전 박스는 변경 이력에 남긴다)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--root", type=Path, default=VIRAT_ROOT, help="VIRAT Ground Dataset 폴더")
    ap.add_argument("--scenes", default="", help="쉼표로 장면만 (예: 0000,0502). 비우면 전부")
    ap.add_argument("--per-scene", type=int, default=50, help="장면마다 이 장수가 되게(이미 넣은 것 포함)")
    ap.add_argument("--gap", type=float, default=5.0, help="같은 클립 안 프레임 사이 최소 초")
    ap.add_argument("--weights", default=None, help="모델 (기본: config 의 model.weights)")
    ap.add_argument("--device", default="cpu", help="추론 장치 (기본 cpu, GPU 는 0)")
    ap.add_argument("--virat-only", action="store_true", help="모델 없이 VIRAT 박스만")
    ap.add_argument("--relabel", action="store_true", help="새로 넣지 않고 이미 넣은 검수 대기를 다시 라벨링")
    ap.add_argument("--quality", type=int, default=95, help="JPEG 품질 (tools/extract_frames.py 와 같다)")
    ap.add_argument("--apply", action="store_true", help="실제로 쓴다 (기본은 보기만)")
    args = ap.parse_args()

    import cv2

    cfg = Config.load(args.config)
    store = Store(cfg.db)
    classes = store.get_classes()
    missing = sorted(set(GROUP) - set(classes))
    if missing:
        raise SystemExit(f"DB 에 없는 클래스: {missing} (지금 {classes})")
    scenes = {s.strip() for s in args.scenes.split(",") if s.strip()} or None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lab = Labeler(cfg, classes, args.weights, args.device, use_model=not args.virat_only)
    if args.relabel:
        return relabel(args, cfg, store, lab, scenes, stamp)

    have = existing_frames(store)
    picks, info = choose(args.root, scenes, args.per_scene, args.gap, have)
    if not info:
        raise SystemExit(f"{args.root} 에서 주석과 영상이 다 있는 클립을 찾지 못했습니다")

    cams = load_cameras(cfg.cameras)
    unregistered = [s for s in sorted(info) if camera_of(s) not in cams]
    if unregistered:
        msg = (f"{cfg.cameras} 에 등록 안 된 카메라: {', '.join(camera_of(s) for s in unregistered)} — "
               "아래를 cameras: 밑에 더할 것(docs/provenance.md):\ncameras:\n" + "".join(
                   f"  {camera_of(s)}:\n    kind: dataset\n    place: VIRAT scene {s}\n    mount: rooftop\n"
                   f"    rights: {RIGHTS}\n    split:\n" for s in unregistered))
        if args.apply:
            raise SystemExit(msg)
        print("[경고] " + msg)

    vdir = args.root / "videos_original"
    items: list[dict] = []
    rows: list[str] = []
    fails: list[str] = []
    per_scene_new: collections.Counter = collections.Counter()
    if args.apply and picks:
        backup(cfg, "virat", stamp)

    picks.sort(key=lambda p: (p.clip, p.frame))
    cap, cap_clip = None, None
    for i, p in enumerate(picks, 1):
        if p.clip != cap_clip:
            if cap is not None:
                cap.release()
            cap, cap_clip = cv2.VideoCapture(str(vdir / f"{p.clip}.mp4")), p.clip
        cap.set(cv2.CAP_PROP_POS_FRAMES, p.frame)
        ok, img = cap.read()
        if not ok:
            fails.append(f"{p.clip} f{p.frame}")
            continue
        boxes, score, w, h = lab.boxes(img, p.gt)
        dst = cfg.unlabeled / camera_of(p.scene) / frame_file(p.clip, p.frame)
        if args.apply:
            okj, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            if not okj:
                fails.append(f"{p.clip} f{p.frame} (JPEG)")
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(buf.tobytes())          # 앞서 넣다 끊긴 파일이 있으면 덮는다(DB 에 없는 것만 골랐다)
            items.append({"path": str(dst.resolve()), "source": "virat", "width": w, "height": h,
                          "status": "pending", "boxes": boxes, "uncertainty": score, "model_version": lab.version,
                          "camera": camera_of(p.scene), "captured_at": None})
        per_scene_new[p.scene] += 1
        rows.append(f"{p.scene}\t{p.clip}\t{p.frame}\t{dst}\t"
                    f"{sum(1 for b in boxes if b.origin == 'auto')}\t{sum(1 for b in boxes if b.origin == 'virat')}")
        if i % 50 == 0:
            print(f"  {i}/{len(picks)} 프레임", flush=True)
    if cap is not None:
        cap.release()

    print(f"\n장면 | 클립 | 이미 넣음 | 후보 | 새로 {'넣음' if args.apply else '넣을 것'} | 틀린 주석 줄")
    for sc, d in sorted(info.items()):
        print(f"{sc} | {d['clips']} | {d['have']} | {d['cands']:,} | {per_scene_new[sc]} | {d['bad']}")
    print(f"합계 새 프레임 {sum(per_scene_new.values())}장")
    lab.report()
    if fails:
        print(f"  [경고] 읽지 못한 프레임 {len(fails)}: {fails[:5]}")

    if not args.apply:
        print(f"\n(보기만 했다. 넣으려면 --apply — {cfg.unlabeled}\\virat-XXXX\\ 에 쓰고 DB 에 검수 대기로 넣는다)")
        return 0
    if not items:
        print("넣을 것이 없다.")
        return 0
    store.ingest_batch(items)
    paths = [it["path"] for it in items]
    with store.connect() as con:
        ids = [r[0] for k in range(0, len(paths), 500)
               for r in con.execute(f"SELECT id FROM images WHERE path IN ({','.join('?' * len(paths[k:k + 500]))})",
                                    paths[k:k + 500])]
    store.set_domain(ids, "cctv")
    log_dir = resolve("runs") / "virat_import"
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest = log_dir / f"{stamp}.tsv"
    manifest.write_text("scene\tclip\tframe\tpath\tauto_boxes\tvirat_boxes\n" + "\n".join(rows) + "\n",
                        encoding="utf-8")
    print(f"\n검수 대기로 {len(items)}장을 넣었다(도메인 CCTV {len(ids)}장). 목록: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
