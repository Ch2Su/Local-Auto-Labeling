"""타일 추론(SAHI 식) — 통째 추론에 K×K 타일 추론을 더해 작은 객체를 더 잡는다.

측정(`tools/test_tiles.py`, 2026-11-03 — docs/experiments.md '타일 추론')과 운영(자동 라벨링의 '타일' 선택지 —
`AutoLabeler.run(tiles=True)`, 웹 UI '자동 라벨링 (타일)', `labeler.py label --tiles`, 2026-11-10)이 **이 모듈을 함께 쓴다**
— 같은 코드라야 잰 것과 같은 결과가 나온다. 함수는 test_tiles.py 에 있던 것을 그대로 옮겼다.

**운영 조합(`detect`)은 측정에서 고른 것이다**(변형 `g2+full:big`, `--cross-iou 0.3`):
- 통째 1번 + 긴 변이 `min_side`(800px)를 넘는 이미지에만 2×2 타일(겹침 20%)을 더해 각각 imgsz 로 추론한다.
- 타일 안쪽 경계에 닿아 잘린 박스는 버린다 — 이웃 타일이나 통째 추론이 온전한 박스를 낸다(안 버리면 FP 5배·mAP −0.06).
- 합치기는 신뢰도 순 클래스별 NMS 인데, 같은 성분끼리(통째끼리·타일끼리)는 predict.iou(0.6), **타일↔통째 사이에만**
  cross_iou(0.3) — 통째 추론 안의 겹친 정상 객체는 그대로 두고 타일이 같은 객체를 다시 낸 중복만 지운다(`merge_cross`).
- **차량 클래스끼리는 클래스가 달라도 class_iou(0.6) 이상 겹치면 신뢰도 높은 쪽만 남긴다 (2026-11-17 사용자 요청).** 클래스별
  NMS 라서 통째가 car, 타일이 truck 이라고 낸 같은 차가 둘 다 남았다 — 그날 DB 에서 타일로 라벨링한 172장 가운데 64장에
  다른 클래스끼리 IoU 0.5 넘게 겹친 박스가 있었고 132쌍 중 127쌍이 차량끼리였다(통째로 라벨링한 46장은 4장). 사람과 탈것
  (person–motorcycle·bicycle)은 실제로 겹치는 정상 라벨이라 묶지 않는다 — `VEHICLES` 에 든 이름끼리만 본다. 남는 쪽의
  클래스가 맞다는 보장은 없다(겹침만 없앤다). 위 측정(11-03)은 이 규칙이 없던 때의 것이고, 측정 도구에서는
  `test_tiles.py --class-iou` 를 줄 때만 운영 지표(conf 0.25 쪽)에 건다 — mAP 쪽은 낮은 신뢰도의 다른 클래스 추측도 세므로 걸지 않는다.
- **통째 자동 라벨링도 같은 차량 규칙으로 정리한다 (2026-11-20 사용자 결정 — `cross_class_keep`, `predict.class_iou`).** 새 풀
  프레임 평가 175장에서 통째도 같은 차에 car·responder 처럼 클래스만 다른 박스를 겹쳐 냈다(운영 가중치 중복 42개 → 정리하면
  10개, 찾음·클래스까지 그대로 — docs/experiments.md '타일 모드로 중복이 얼마나 주는지'). 줄인 몫은 전부 이 클래스를 넘는
  병합이라(그 측정에서 같은 클래스 NMS 는 아무것도 지우지 않았다), 통째에는 차량 규칙만 걸고 모델이 낸 같은 클래스 박스는
  그대로 둔다. 예측 비교(P)·label_qa 는 걸지 않는다(모델이 낸 그대로 견준다).
- 그 측정(옛 CCTV test 793장, visible-3)에서 통째 대비 F1 같음(0.890), 재현율 +1.8pt, FP +22%, mAP50-95 +0.035, GPU 추론
  시간 약 2배(긴 변 800px 넘는 이미지만). 이득은 대부분 아주 작은(640 환산 16px 미만) 객체에서 나오고, 늘어난 FP 는 라벨
  누락보다 진짜 오검출(신호등·기둥·도로 글씨 등)이 많았다 — 그래서 기본 자동 라벨링은 통째 그대로이고 이것은 고르는 선택지다.
- 성분마다 원검출을 낮은 기준(RAW_* — 측정의 mAP 조건과 같다)으로 받아 합친 **뒤에** 운영 기준(conf·max_det)으로 거른다.
  보통 자동 라벨링(통째, `model.predict(conf=0.25)`)과 길이 달라, 타일을 안 쓰는 작은 이미지도 이론상 조금 다를 수 있다 —
  11-10 운영 m 으로 작은 이미지 6장을 견주니 박스·클래스는 같고 좌표·신뢰도만 1e-7 아래(부동소수 반올림)로 달랐다.
- 11-10 운영 m(8클래스) CPU 확인: 보관 DB 사본의 CCTV test 큰 이미지 40장(사람 박스 316)에서 통째 → 타일이 재현율 0.921 →
  0.943, 오검출 24 → 29, 640 환산 16px 미만 30/39 → 34/39, 시간 0.48 → 1.62초/장(CPU — GPU 로는 측정에서 약 2배).
"""

from __future__ import annotations

import time

import numpy as np

RAW_CONF, RAW_IOU, RAW_MAXDET = 0.001, 0.7, 300   # 성분마다 받는 원검출 (측정의 mAP 조건과 같다)
MIN_SIDE = 800       # 긴 변이 이 값 이하인 이미지에는 타일을 쓰지 않는다 (측정의 `big` 옵션)

# config.yaml 의 predict.tiles 가 없을 때의 값 — 측정에서 고른 조합
DEFAULTS = {"grid": 2, "overlap": 0.2, "edge_px": 3, "min_side": MIN_SIDE, "cross_iou": 0.3, "class_iou": 0.6}
# config.yaml 의 predict.class_iou 가 없을 때 — 통째 자동 라벨링의 차량 겹침 정리(2026-11-20, cross_class_keep)
PLAIN_CLASS_IOU = 0.6

# 클래스가 달라도 같은 물체일 수 있는 묶음 — 차량 판정(car·pickup·truck·bus·responder)은 모델이 자주 헷갈린다.
# 사람·자전거·오토바이·킥보드·라바콘은 넣지 않는다(탄 사람과 탈것은 실제로 겹친다).
VEHICLES = ("car", "truck", "bus", "pickup", "responder")


def vehicle_ids(names) -> frozenset[int]:
    """모델의 클래스 이름(dict 번호→이름 또는 list)에서 VEHICLES 에 든 번호."""
    items = names.items() if hasattr(names, "items") else enumerate(names or [])
    return frozenset(int(i) for i, n in items if n in VEHICLES)


def tile_grid(w: int, h: int, k: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """K×K 타일의 (x0, y0, x1, y1). 타일 크기는 모두 같고 마지막 타일은 이미지 끝에 붙인다."""
    tw = min(w, int(np.ceil(w / (k - (k - 1) * overlap))))
    th = min(h, int(np.ceil(h / (k - (k - 1) * overlap))))
    xs = [min(int(round(i * tw * (1 - overlap))), w - tw) for i in range(k)]
    ys = [min(int(round(i * th * (1 - overlap))), h - th) for i in range(k)]
    return [(x, y, x + tw, y + th) for y in ys for x in xs]


def run_component(model, img: np.ndarray, comp: str, *, overlap: float, edge_px: int, imgsz: int, device,
                  conf: float = RAW_CONF, iou: float = RAW_IOU, max_det: int = RAW_MAXDET) -> tuple[np.ndarray, float]:
    """한 성분('full' 또는 'gK')의 검출을 이미지 좌표로. 열 7개: x1 y1 x2 y2 conf cls touch(타일 안쪽 경계에 닿아 잘림).
    두 번째 값은 추론에 든 초(디코딩·합치기 제외)."""
    h, w = img.shape[:2]
    boxes = [(0, 0, w, h)] if comp == "full" else tile_grid(w, h, int(comp[1:]), overlap)
    crops = [np.ascontiguousarray(img[y0:y1, x0:x1]) for x0, y0, x1, y1 in boxes]
    t0 = time.perf_counter()
    res = model.predict(source=crops, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
                        device=device, verbose=False, batch=len(crops))
    dt = time.perf_counter() - t0
    rows = []
    e = edge_px
    for (x0, y0, x1, y1), r in zip(boxes, res):
        d = r.boxes.data.cpu().numpy().astype(np.float32)
        if not len(d):
            continue
        touch = np.zeros(len(d), bool)
        if comp != "full":
            tw, th = x1 - x0, y1 - y0
            if x0 > 0:
                touch |= d[:, 0] <= e
            if x1 < w:
                touch |= d[:, 2] >= tw - e
            if y0 > 0:
                touch |= d[:, 1] <= e
            if y1 < h:
                touch |= d[:, 3] >= th - e
        d[:, [0, 2]] += x0
        d[:, [1, 3]] += y0
        rows.append(np.concatenate([d, touch[:, None].astype(np.float32)], 1))
    return (np.concatenate(rows) if rows else np.zeros((0, 7), np.float32)), dt


def merge(d: np.ndarray, iou_thr: float, max_det: int, min_conf: float) -> np.ndarray:
    """클래스별 NMS 로 겹친 검출을 합친다 (신뢰도 내림차순으로 돌려준다)."""
    import torch
    from torchvision.ops import batched_nms

    d = d[d[:, 4] >= min_conf]
    if not len(d):
        return d
    keep = batched_nms(torch.from_numpy(d[:, :4]), torch.from_numpy(d[:, 4]),
                       torch.from_numpy(d[:, 5]).long(), iou_thr)[:max_det]
    return d[keep.numpy()]


def merge_cross(d: np.ndarray, iou_same: float, iou_cross: float, max_det: int, min_conf: float, *,
                group: frozenset[int] | None = None, iou_class: float = 0.0) -> np.ndarray:
    """신뢰도 순 클래스별 NMS 인데 기준이 둘이다 — 같은 성분(통째끼리·타일끼리)은 iou_same, 서로 다른 성분은 iou_cross.
    열 8(인덱스 7)이 성분 표식(0 통째 / 1 타일)이다.

    통째 추론 안의 겹친 정상 객체는 건드리지 않고(iou_same 은 모델의 NMS 와 같은 수준), 타일이 통째와 같은 객체를
    다시 낸 중복만 낮은 기준으로 지운다. 성분이 하나뿐이면 보통 NMS 와 같다.

    `group`(클래스 번호 묶음)과 `iou_class` 를 주면 그 묶음 안에서는 **클래스가 달라도** IoU 가 iou_class 이상인 것을 같은
    물체로 보고 신뢰도 높은 쪽만 남긴다(성분은 안 본다 — 모듈 머리말). 안 주면 전과 같다."""
    import torch
    from torchvision.ops import box_iou

    d = d[d[:, 4] >= min_conf]
    if not len(d):
        return d
    d = d[np.argsort(-d[:, 4], kind="stable")]
    n = len(d)
    iou = box_iou(torch.from_numpy(d[:, :4]), torch.from_numpy(d[:, :4])).numpy()
    same_cls = d[:, 5][:, None] == d[:, 5][None, :]
    thr = np.where(d[:, 7][:, None] == d[:, 7][None, :], iou_same, iou_cross)
    conflict = (iou >= thr) & same_cls
    if group and iou_class > 0:
        in_group = np.isin(d[:, 5].astype(int), list(group))
        conflict |= (iou >= iou_class) & ~same_cls & in_group[:, None] & in_group[None, :]
    removed = np.zeros(n, bool)
    keep = []
    for i in range(n):
        if removed[i]:
            continue
        keep.append(i)
        if len(keep) >= max_det:
            break
        removed[i + 1:] |= conflict[i, i + 1:]
    return d[keep]


def cross_class_keep(xyxy, conf, cls, group: frozenset[int], iou_class: float) -> np.ndarray:
    """통째 추론(성분 하나)의 검출에서 남길 것 — 원래 순서의 bool 배열 (2026-11-20, 통째 자동 라벨링).

    `group`(차량 클래스 번호 — vehicle_ids) 안에서 **클래스가 다른** 두 박스의 IoU 가 iou_class 이상이면 신뢰도 높은 쪽만
    남긴다. merge_cross(group=, iou_class=)의 차량 규칙과 같은 순서다: 신뢰도 내림차순(같으면 앞의 것이 먼저)으로 남기고,
    지워진 박스는 다른 박스를 지우지 못한다. 같은 클래스끼리와 묶음 밖 클래스(사람·탈것 등)는 건드리지 않는다 — 같은 클래스
    NMS 는 모델이 낸 그대로 둔다(merge_cross 와 다른 점). group 이 비었거나 iou_class 가 0 이면 전부 남긴다.
    입력은 numpy 로 바꿀 수 있는 것(좌표는 픽셀이든 정규화든 같은 단위면 된다)."""
    xyxy = np.asarray(xyxy, np.float32).reshape(-1, 4)
    n = len(xyxy)
    keep = np.ones(n, bool)
    if n < 2 or not group or iou_class <= 0:
        return keep
    cls = np.asarray(cls).reshape(-1).astype(int)
    in_group = np.isin(cls, list(group))
    if in_group.sum() < 2:
        return keep
    import torch
    from torchvision.ops import box_iou

    order = np.argsort(-np.asarray(conf, np.float32).reshape(-1), kind="stable")
    b, c, g = xyxy[order], cls[order], in_group[order]
    iou = box_iou(torch.from_numpy(np.ascontiguousarray(b)), torch.from_numpy(np.ascontiguousarray(b))).numpy()
    conflict = (iou >= iou_class) & (c[:, None] != c[None, :]) & g[:, None] & g[None, :]
    removed = np.zeros(n, bool)
    for i in range(n):
        if not removed[i]:
            removed[i + 1:] |= conflict[i, i + 1:]
    keep[order[removed]] = False
    return keep


def plain_class_iou(cfg) -> float:
    """config.yaml 의 predict.class_iou(없으면 PLAIN_CLASS_IOU) — 통째 자동 라벨링의 차량 겹침 정리. 0 이면 끔.
    타일 모드는 이것이 아니라 predict.tiles.class_iou(settings)를 쓴다."""
    raw = cfg.get("predict.class_iou")
    value = PLAIN_CLASS_IOU if raw is None else float(raw)
    if not 0 <= value <= 1:
        raise ValueError(f"predict.class_iou 는 0(끔)~1 이어야 합니다: {value}")
    return value


def settings(cfg) -> dict:
    """config.yaml 의 predict.tiles(없으면 DEFAULTS)."""
    raw = cfg.get("predict.tiles") or {}
    out = dict(DEFAULTS)
    out.update({k: raw[k] for k in DEFAULTS if raw.get(k) is not None})
    out["grid"], out["edge_px"], out["min_side"] = int(out["grid"]), int(out["edge_px"]), int(out["min_side"])
    out["overlap"], out["cross_iou"] = float(out["overlap"]), float(out["cross_iou"])
    out["class_iou"] = float(out["class_iou"])
    if not 0 <= out["class_iou"] <= 1:
        raise ValueError(f"predict.tiles.class_iou 는 0(끔)~1 이어야 합니다: {out['class_iou']}")
    if out["grid"] < 2:
        raise ValueError(f"predict.tiles.grid 는 2 이상이어야 합니다: {out['grid']}")
    return out


def detect(model, img: np.ndarray, *, imgsz: int, device, conf: float, iou: float, max_det: int,
           grid: int, overlap: float, edge_px: int, min_side: int, cross_iou: float,
           class_iou: float = 0.0, names=None) -> tuple[np.ndarray, bool]:
    """운영용 타일 추론 한 장 — (검출, 타일을 썼나). 검출의 열: x1 y1 x2 y2 conf cls touch comp (이미지 픽셀 좌표).

    측정 변형 'gK+full:big' + cross_iou 와 같은 순서·기준이다(test_tiles.compose → merge_cross): 타일 성분을 앞에,
    통째를 뒤에 이어 붙이고(같은 신뢰도에서는 앞의 것이 남는다), 잘린 타일 박스를 버린 뒤 합친다.
    `class_iou` 가 0 보다 크면 차량 클래스끼리의 겹침도 지운다(`names` 는 모델의 클래스 이름 — 기본은 model.names).
    """
    h, w = img.shape[:2]
    comps = [f"g{grid}", "full"] if max(w, h) > min_side else ["full"]
    parts = []
    for c in comps:
        d, _ = run_component(model, img, c, overlap=overlap, edge_px=edge_px, imgsz=imgsz, device=device)
        parts.append(np.concatenate([d, np.full((len(d), 1), c != "full", np.float32)], 1))
    d = np.concatenate(parts)
    d = d[d[:, 6] == 0]                     # 타일 안쪽 경계에 닿아 잘린 박스는 버린다
    group = vehicle_ids(names if names is not None else getattr(model, "names", None)) if class_iou > 0 else None
    return merge_cross(d, iou, cross_iou, max_det, conf, group=group, iou_class=class_iou), len(comps) > 1
