"""VIRAT Ground Release 2.0 영상에서 프레임을 뽑아 **검수 대기(pending)** 로 넣는다 — VIRAT 라벨 + 모델 보충.

**사용자 결정(2026-11-07).** 장면(카메라)마다 50장, VIRAT 라벨에 운영 모델이 더 찾은 박스를 보탠다. 표본 66프레임
(장면마다 6장)에서 VIRAT 박스 1,096개 말고도 모델이 496개를 더 찾았다 — 0502 주차장은 VIRAT 차가 2대인 프레임에서
모델이 진짜 주차 차량 20여 대를 찾았다(VIRAT 은 행동과 무관한 주차 차량·사람을 비워 둔다, 배포본 안내서 3.1). 거꾸로
0100 광장의 먼 사람은 VIRAT 에만 있다(그 표본에서 모델이 VIRAT 사람 박스의 16% 만 찾았다). 그래서 둘을 합쳐 사람이
한 장씩 본다.

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
- VIRAT 종류 1 person → person, 2 car·3 vehicle → car, 5 bike → bicycle(오토바이도 bike 다 — 검수에서 가른다).
  0(구역 상자)·4 object 는 버리고, 열 수·형이 틀린 줄은 건너뛴다. 출처 'virat', 신뢰도 없음(검수 화면에 '수동').
- 모델(`--weights`, 기본 운영 가중치) 예측과 같은 무리(사람 / 차: car·bus·truck·pickup / 두 바퀴: bicycle·
  motorcycle·scooter)끼리 IoU 0.5 이상을 큰 것부터 하나씩 짝짓는다. 짝이 생긴 VIRAT 박스는 모델의 세부 클래스를
  따른다(VIRAT car 가 모델 truck 이면 truck) — 자리는 VIRAT 그대로.
- 짝이 없는 모델 박스는, 같은 무리의 VIRAT 박스와 IoU 0.3 이상이거나 70% 이상이 그 안에 들면 버린다(같은 물체를
  VIRAT 이 크게 그린 것 — 0100·0500 표본에서 봤다). 나머지는 출처 'auto'·모델 신뢰도로 더한다. `--no-fill` 이면
  VIRAT 박스만 넣는다. 운영 모델(visible-3)은 pickup 을 모른다 — 픽업은 검수에서 고친다.

**넣는 순서.** 기본은 보기만 한다(파일·DB 를 건드리지 않는다 — 고르고, 프레임을 읽고, 추론까지 해서 셈을 보인다).
`--apply` 면 DB 를 `labeling.db.bak_virat_<시각>` 으로 떠 두고, 프레임을 쓰고, 한 번에 넣은 뒤, 넣은 목록을
`runs/virat_import/<시각>.tsv` 에 남긴다. 추론은 기본 CPU 다(`--device 0` 으로 GPU — 학습이 돌고 있으면 쓰지 말 것).

**주의.** `labeler.py label --all` 은 검수 대기 이미지도 모델 예측으로 덮어쓴다 — 넣은 뒤에는 쓰지 말 것(웹 UI 의 자동
라벨링은 검수 대기를 건드리지 않는다). VIRAT Release 2.0 은 재배포 금지로 본다 — 프레임을 밖에 올리지(로보플로 등)
말 것(`data/` 는 git 밖이다).

    python tools/import_virat.py                          # 무엇을 넣을지 보기만
    python tools/import_virat.py --apply                  # 장면당 50장
    python tools/import_virat.py --per-scene 100 --apply  # 장면당 100장이 되게 더 넣기
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
from autolabel.store import Store  # noqa: E402
from autolabel.yolo_io import Box  # noqa: E402

VIRAT_ROOT = Path(r"E:\datasets\VIRAT_Ground_partial\VIRAT Ground Dataset")
GT_CLASS = {1: "person", 2: "car", 3: "car", 5: "bicycle"}       # 0 구역 상자·4 object 는 버린다
GROUP = {"person": "person", "car": "vehicle", "bus": "vehicle", "truck": "vehicle", "pickup": "vehicle",
         "bicycle": "bike", "motorcycle": "bike", "scooter": "bike"}
MATCH_IOU = 0.5          # VIRAT 박스와 모델 박스를 같은 물체로 짝짓는 IoU
SAME_IOU = 0.3           # 짝이 없는 모델 박스를 같은 물체로 보고 버리는 IoU
SAME_INSIDE = 0.7        # …또는 모델 박스가 VIRAT 박스 안에 든 비율
OBJ_SUFFIX = ".viratdata.objects.txt"
NAME_RE = re.compile(r"^(VIRAT_S_\d{6}(?:_\d{2}_\d{6}_\d{6})?)_f(\d{6})$")
RIGHTS = "VIRAT Release 2.0 — 개인별 동의 약관, 재배포 금지로 볼 것"


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
    """후보에서 k 개를 고르게. 이미 넣은 프레임(taken)이 없으면 구간 가운데들을, 있으면 이미 고른 것에서 가장 먼
    후보를 하나씩(같으면 앞의 것) — 늘 가운데를 고르면 더 넣을 때마다 기존 프레임의 `--gap` 바로 옆이 뽑혔다
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


def inside(p, g) -> float:
    """p 의 넓이 중 g 안에 든 비율."""
    ix = max(0.0, min(p[2], g[2]) - max(p[0], g[0]))
    iy = max(0.0, min(p[3], g[3]) - max(p[1], g[1]))
    area = (p[2] - p[0]) * (p[3] - p[1])
    return ix * iy / area if area > 0 else 0.0


def merge(gt: list, preds: list) -> tuple[list[str], list]:
    """gt [(이름, x1, y1, x2, y2)], preds [(이름, 신뢰도, x1, y1, x2, y2)] → (gt 마다의 클래스, 더할 모델 박스)."""
    pairs = sorted(((iou(g[1:], p[2:]), gi, pj) for gi, g in enumerate(gt) for pj, p in enumerate(preds)
                    if GROUP[g[0]] == GROUP.get(p[0])), reverse=True)
    names = [g[0] for g in gt]
    used_g: set[int] = set()
    used_p: set[int] = set()
    for v, gi, pj in pairs:
        if v < MATCH_IOU:
            break
        if gi in used_g or pj in used_p:
            continue
        used_g.add(gi)
        used_p.add(pj)
        names[gi] = preds[pj][0]         # 같은 무리 안의 세부 클래스(truck·bus·motorcycle…)
    extra = []
    for pj, p in enumerate(preds):
        if pj in used_p:
            continue
        same = [g[1:] for g in gt if GROUP[g[0]] == GROUP.get(p[0])]
        if any(iou(g, p[2:]) >= SAME_IOU or inside(p[2:], g) >= SAME_INSIDE for g in same):
            continue
        extra.append(p)
    return names, extra


def to_box(cls: int, x1: float, y1: float, x2: float, y2: float, w: int, h: int,
           conf: float | None, origin: str) -> Box:
    return Box(cls, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h, conf, origin).clamped()


def camera_yaml(scenes: list[str]) -> str:
    return "cameras:\n" + "".join(
        f"  {camera_of(s)}:\n    kind: dataset\n    place: VIRAT scene {s}\n    mount: rooftop\n"
        f"    rights: {RIGHTS}\n    split:\n" for s in scenes)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--root", type=Path, default=VIRAT_ROOT, help="VIRAT Ground Dataset 폴더")
    ap.add_argument("--scenes", default="", help="쉼표로 장면만 (예: 0000,0502). 비우면 전부")
    ap.add_argument("--per-scene", type=int, default=50, help="장면마다 이 장수가 되게(이미 넣은 것 포함)")
    ap.add_argument("--gap", type=float, default=5.0, help="같은 클립 안 프레임 사이 최소 초")
    ap.add_argument("--weights", default=None, help="보충할 모델 (기본: config 의 model.weights)")
    ap.add_argument("--device", default="cpu", help="추론 장치 (기본 cpu, GPU 는 0)")
    ap.add_argument("--no-fill", action="store_true", help="VIRAT 박스만 넣는다(모델 보충 없음)")
    ap.add_argument("--quality", type=int, default=95, help="JPEG 품질 (tools/extract_frames.py 와 같다)")
    ap.add_argument("--apply", action="store_true", help="실제로 프레임을 쓰고 DB 에 넣는다")
    args = ap.parse_args()

    import cv2

    cfg = Config.load(args.config)
    store = Store(cfg.db)
    classes = store.get_classes()
    class_idx = {n: i for i, n in enumerate(classes)}
    missing = sorted(set(GROUP) - set(class_idx))
    if missing:
        raise SystemExit(f"DB 에 없는 클래스: {missing} (지금 {classes})")
    scenes = {s.strip() for s in args.scenes.split(",") if s.strip()} or None

    have = existing_frames(store)
    picks, info = choose(args.root, scenes, args.per_scene, args.gap, have)
    if not info:
        raise SystemExit(f"{args.root} 에서 주석과 영상이 다 있는 클립을 찾지 못했습니다")

    cams = load_cameras(cfg.cameras)
    unregistered = [s for s in sorted(info) if camera_of(s) not in cams]
    if unregistered:
        msg = (f"{cfg.cameras} 에 등록 안 된 카메라: {', '.join(camera_of(s) for s in unregistered)} — "
               f"아래를 cameras: 밑에 더할 것(docs/provenance.md):\n{camera_yaml(unregistered)}")
        if args.apply:
            raise SystemExit(msg)
        print("[경고] " + msg)

    model = None
    weights = args.weights or cfg.get("model.weights")
    if not args.no_fill:
        from ultralytics import YOLO
        model = YOLO(str(resolve(weights)))
        unknown = sorted(set(model.names.values()) - set(GROUP))
        if unknown:
            raise SystemExit(f"모델 클래스 {unknown} 를 어느 무리로 볼지 모릅니다 (GROUP)")
    imgsz, conf, iou_nms = cfg.get("predict.imgsz", 640), cfg.get("predict.conf", 0.25), cfg.get("predict.iou", 0.7)
    version = "virat" if model is None else f"virat+{Path(weights).stem}"

    vdir = args.root / "videos_original"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    items: list[dict] = []
    rows: list[str] = []
    n_virat: collections.Counter = collections.Counter()
    n_auto: collections.Counter = collections.Counter()
    refined: collections.Counter = collections.Counter()
    fails: list[str] = []
    per_scene_new: collections.Counter = collections.Counter()
    if args.apply and picks:
        bak = cfg.db.with_name(f"{cfg.db.name}.bak_virat_{stamp}")
        with sqlite3.connect(cfg.db) as src, sqlite3.connect(bak) as dst:
            src.backup(dst)
        print(f"DB 를 떠 두었다: {bak}")

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
        h, w = img.shape[:2]
        preds = []
        if model is not None:
            r = model.predict(img, imgsz=imgsz, conf=conf, iou=iou_nms, device=args.device, verbose=False)[0]
            for k, c, xyxy in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
                preds.append((model.names[int(k)], float(c), *map(float, xyxy)))
        names, extra = merge(p.gt, preds)
        boxes = [to_box(class_idx[name], *g[1:], w, h, None, "virat") for g, name in zip(p.gt, names)]
        boxes += [to_box(class_idx[e[0]], *e[2:], w, h, round(e[1], 4), "auto") for e in extra]
        boxes = [b for b in boxes if b.is_valid()]
        dst = cfg.unlabeled / camera_of(p.scene) / frame_file(p.clip, p.frame)
        if args.apply:
            okj, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            if not okj:
                fails.append(f"{p.clip} f{p.frame} (JPEG)")
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(buf.tobytes())          # 앞서 넣다 끊긴 파일이 있으면 덮는다(DB 에 없는 것만 골랐다)
            items.append({"path": str(dst.resolve()), "source": "virat", "width": w, "height": h,
                          "status": "pending", "boxes": boxes, "uncertainty": 0.0, "model_version": version,
                          "camera": camera_of(p.scene), "captured_at": None})
        per_scene_new[p.scene] += 1
        n_virat.update(classes[b.cls] for b in boxes if b.origin == "virat")
        n_auto.update(classes[b.cls] for b in boxes if b.origin == "auto")
        refined.update(f"{g[0]}→{name}" for g, name in zip(p.gt, names) if name != g[0])
        rows.append(f"{p.scene}\t{p.clip}\t{p.frame}\t{dst}\t"
                    f"{sum(1 for b in boxes if b.origin == 'virat')}\t{sum(1 for b in boxes if b.origin == 'auto')}")
        if i % 50 == 0:
            print(f"  {i}/{len(picks)} 프레임", flush=True)
    if cap is not None:
        cap.release()

    print(f"\n장면 | 클립 | 이미 넣음 | 후보 | 새로 {'넣음' if args.apply else '넣을 것'} | 틀린 주석 줄")
    for sc, d in sorted(info.items()):
        print(f"{sc} | {d['clips']} | {d['have']} | {d['cands']:,} | {per_scene_new[sc]} | {d['bad']}")
    print(f"합계 새 프레임 {sum(per_scene_new.values())}장 · 박스 VIRAT {sum(n_virat.values()):,} + 모델 보충 "
          f"{sum(n_auto.values()):,}")
    print("  VIRAT 박스:", dict(sorted(n_virat.items())))
    print("  모델 보충:", dict(sorted(n_auto.items())))
    if refined:
        print("  VIRAT 클래스를 모델 세부 클래스로:", dict(refined.most_common()))
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
    manifest.write_text("scene\tclip\tframe\tpath\tvirat_boxes\tauto_boxes\n" + "\n".join(rows) + "\n",
                        encoding="utf-8")
    print(f"\n검수 대기로 {len(items)}장을 넣었다(도메인 CCTV {len(ids)}장). 목록: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
