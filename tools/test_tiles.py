"""타일 추론(SAHI 식)이 통째 추론보다 나은지 test 로 잰다 (2026-11-03).

**측정용이다 — 운영 코드(`AutoLabeler`·`LiveDetector`)는 건드리지 않는다.** 결과가 좋으면 그때 운영 코드에
옮긴다. DB 는 `test_baseline.build_split` 처럼 읽기 전용으로만 열고, test 분할을 복사해 재고 지운다.

    python tools/test_tiles.py                                  # models/best.pt, 기본 변형 6개
    python tools/test_tiles.py --limit 30                       # 맛보기 (앞 30장)
    python tools/test_tiles.py --variants full,g2+full,g3+full  # 변형을 직접 고른다

**변형** 은 `성분+성분:옵션` 으로 적는다. 성분은 `full`(이미지 통째) 과 `gK`(K×K 타일, 겹침 `--overlap`).
옵션 `noedge` 는 타일 안쪽 경계에 닿아 잘린 박스를 버리지 **않는다**(기본은 버린다 — 이웃 타일이나
통째 추론이 온전한 박스를 낸다), `big` 은 긴 변이 800px 넘는 이미지에만 타일을 쓴다.
성분마다 한 번만 추론하고 변형은 그 결과를 합쳐 만든다.

**점수는 두 가지로 잰다.**
- mAP: conf 0.001·NMS 0.7·max_det 300 (Ultralytics `val` 과 같은 조건, 매칭 함수도 같다). 모델이 아는
  클래스(0–6)의 평균을 먼저 보인다 — 운영 가중치 visible-3 은 pickup 을 몰라 평균에 0 이 끼기 때문이다.
- 운영 조건: conf 0.25·NMS 0.6·max_det 100 (`predict:` 설정) 에서 IoU 0.5 로 맞춘 P·R·F1·FP 수와, 객체
  크기별(640 입력으로 줄였을 때의 변 길이) 재현율. 자동 라벨링이 실제로 내는 것은 이쪽이다.

`full` 을 `model.val` 과 견준다(`--no-check-val` 로 끈다). **model.val 은 NMS 를 `multi_label=True` 로 돌려**
한 박스가 여러 클래스 후보를 내므로(`ultralytics/models/yolo/detect/val.py`) 단일 라벨인 이 평가기보다
mAP 가 약 0.01 높게 나온다(11-03 visible-3 test: 0.5869 대 0.5731). 자동 라벨링은 단일 라벨이라 이쪽이 실제에 가깝다.
변형끼리 비교는 같은 평가기로 한 것이라 이 차이와 상관없다.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # tools/ -> 저장소 루트
sys.path.insert(0, str(Path(__file__).resolve().parent))               # test_baseline 을 같은 폴더에서

from autolabel.config import Config, resolve
from test_baseline import build_split, class_names, sha256, write_yaml

IOUV = np.linspace(0.5, 0.95, 10)
MAP_CONF, MAP_IOU, MAP_MAXDET = 0.001, 0.7, 300
SIZE_BINS = [(0, 16, "<16"), (16, 32, "16-32"), (32, 64, "32-64"), (64, 1e9, "64+")]
BIG_SIDE = 800          # `big` 옵션: 긴 변이 이 값 이하인 이미지는 타일을 쓰지 않는다

# 이름: (보여 줄 이름, 이미지를 받을지). test_baseline.DOMAINS 와 같은 기준에 CCTV 를 더했다.
DOMAINS = {
    "cctv": ("CCTV", lambda it: it["source"] not in ("unlabeled", "o365") and "stabilized_" not in it["path"].lower()),
    "noncctv": ("비CCTV", lambda it: it["source"] == "unlabeled"),
    "drone": ("드론", lambda it: "stabilized_" in it["path"].lower()),
    "all": ("전체", lambda it: it["source"] != "o365"),
}


def parse_args():
    p = argparse.ArgumentParser(description="타일 추론 대 통째 추론을 test 로 비교한다 (측정용, DB 읽기 전용).")
    p.add_argument("--config", default=None)
    p.add_argument("--weights", default=None, help="기본: config 의 운영 가중치(models/best.pt)")
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--imgsz", type=int, default=None, help="기본: model.imgsz — 타일도 이 크기로 추론한다")
    p.add_argument("--variants", default="full,g2+full,g3+full,g2,g2+full:noedge,g2+full:big")
    p.add_argument("--overlap", type=float, default=0.2, help="타일끼리 겹치는 비율 (타일 한 변 기준)")
    p.add_argument("--edge-px", type=int, default=3, help="타일 안쪽 경계에서 이 px 안에 닿은 박스를 '잘린 것' 으로 본다")
    p.add_argument("--op-conf", type=float, default=0.25,
                   help="운영 조건 P·R·FP 를 셀 신뢰도 (기본 0.25 = predict.conf). 올리면 FP 가 같은 수준에서 재현율을 견줄 수 있다")
    p.add_argument("--limit", type=int, default=0, help="앞 N장만 (맛보기)")
    p.add_argument("--out", default=None, help="만들 폴더 (기본: runs/test_tiles/<날짜>_<가중치>_<split>)")
    p.add_argument("--keep", action="store_true", help="복사한 이미지 폴더를 지우지 않는다")
    p.add_argument("--no-check-val", action="store_true", help="model.val 과의 교차 확인을 건너뛴다")
    return p.parse_args()


# ---------------------------------------------------------------- 변형
def parse_variant(spec: str) -> dict:
    head, *opts = spec.split(":")
    comps = head.split("+")
    for c in comps:
        if c != "full" and not (c.startswith("g") and c[1:].isdigit() and int(c[1:]) >= 2):
            raise SystemExit(f"알 수 없는 성분 '{c}' ({spec}) — full 이나 g2, g3 … 만 쓸 수 있습니다")
    bad = [o for o in opts if o not in ("noedge", "big")]
    if bad:
        raise SystemExit(f"알 수 없는 옵션 {bad} ({spec}) — noedge, big 만 쓸 수 있습니다")
    if "big" in opts and "full" not in comps:
        raise SystemExit(f"'big' 은 작은 이미지에서 통째 추론으로 돌아가므로 full 성분이 있어야 합니다 ({spec})")
    return {"name": spec, "comps": comps, "noedge": "noedge" in opts, "big": "big" in opts}


def tile_grid(w: int, h: int, k: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """K×K 타일의 (x0, y0, x1, y1). 타일 크기는 모두 같고 마지막 타일은 이미지 끝에 붙인다."""
    tw = min(w, int(np.ceil(w / (k - (k - 1) * overlap))))
    th = min(h, int(np.ceil(h / (k - (k - 1) * overlap))))
    xs = [min(int(round(i * tw * (1 - overlap))), w - tw) for i in range(k)]
    ys = [min(int(round(i * th * (1 - overlap))), h - th) for i in range(k)]
    return [(x, y, x + tw, y + th) for y in ys for x in xs]


def run_component(model, img: np.ndarray, comp: str, args, imgsz: int, device) -> tuple[np.ndarray, float]:
    """한 성분의 검출을 이미지 좌표로. 열 7개: x1 y1 x2 y2 conf cls touch(타일 안쪽 경계에 닿아 잘림)."""
    h, w = img.shape[:2]
    boxes = [(0, 0, w, h)] if comp == "full" else tile_grid(w, h, int(comp[1:]), args.overlap)
    crops = [np.ascontiguousarray(img[y0:y1, x0:x1]) for x0, y0, x1, y1 in boxes]
    t0 = time.perf_counter()
    res = model.predict(source=crops, imgsz=imgsz, conf=MAP_CONF, iou=MAP_IOU, max_det=MAP_MAXDET,
                        device=device, verbose=False, batch=len(crops))
    dt = time.perf_counter() - t0
    rows = []
    e = args.edge_px
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


def compose(var: dict, comps: dict[str, np.ndarray], w: int, h: int) -> np.ndarray:
    use = ["full"] if var["big"] and max(w, h) <= BIG_SIDE else var["comps"]
    d = np.concatenate([comps[c] for c in use])
    if not var["noedge"]:
        d = d[d[:, 6] == 0]
    return d


def merge(d: np.ndarray, iou_thr: float, max_det: int, min_conf: float) -> np.ndarray:
    """클래스별 NMS 로 겹친 검출을 합친다 (신뢰도 내림차순으로 돌려준다)."""
    from torchvision.ops import batched_nms

    d = d[d[:, 4] >= min_conf]
    if not len(d):
        return d
    keep = batched_nms(torch.from_numpy(d[:, :4]), torch.from_numpy(d[:, 4]),
                       torch.from_numpy(d[:, 5]).long(), iou_thr)[:max_det]
    return d[keep.numpy()]


# ---------------------------------------------------------------- 평가
def match(pcls: np.ndarray, tcls: np.ndarray, iou: np.ndarray, iouv: np.ndarray):
    """Ultralytics `match_predictions`(use_scipy=False) 와 같은 규칙. (N, len(iouv)) TP 표와 IoU 0.5 에서 맞은 GT."""
    correct = np.zeros((len(pcls), len(iouv)), bool)
    hit = np.zeros(len(tcls), bool)
    if not len(pcls) or not len(tcls):
        return correct, hit
    iou = iou * (tcls[:, None] == pcls)
    for i, t in enumerate(iouv):
        m = np.array(np.nonzero(iou >= t)).T
        if not m.shape[0]:
            continue
        if m.shape[0] > 1:
            m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
            m = m[np.unique(m[:, 1], return_index=True)[1]]
            m = m[np.unique(m[:, 0], return_index=True)[1]]
        correct[m[:, 1].astype(int), i] = True
        if i == 0:
            hit[m[:, 0].astype(int)] = True
    return correct, hit


def read_gt(label: Path, w: int, h: int) -> np.ndarray:
    """(M, 5): cls x1 y1 x2 y2 (픽셀). 정규화 좌표는 회전을 적용한 화면 기준이다 (CLAUDE.md 'EXIF 회전')."""
    rows = []
    if label.exists():
        for line in label.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) >= 5:
                c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:5])
                rows.append([c, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(rows, np.float32).reshape(-1, 5)


def eval_image(det_map: np.ndarray, det_op: np.ndarray, gt: np.ndarray, w: int, h: int,
               imgsz: int, known: int) -> dict:
    from ultralytics.utils.metrics import box_iou

    tbox = torch.from_numpy(gt[:, 1:5])
    out = {"tcls": gt[:, 0].astype(int)}
    iou = box_iou(tbox, torch.from_numpy(det_map[:, :4])).numpy() if len(det_map) and len(gt) else np.zeros((len(gt), len(det_map)))
    out["tp"], _ = match(det_map[:, 5].astype(int), out["tcls"], iou, IOUV)
    out["conf"], out["pcls"] = det_map[:, 4], det_map[:, 5].astype(int)
    # 운영 조건: IoU 0.5 한 단계만
    iou = box_iou(tbox, torch.from_numpy(det_op[:, :4])).numpy() if len(det_op) and len(gt) else np.zeros((len(gt), len(det_op)))
    ok, hit = match(det_op[:, 5].astype(int), out["tcls"], iou, IOUV[:1])
    out["op_tp"], out["op_fp"] = int(ok.sum()), int((~ok[:, 0]).sum()) if len(ok) else 0
    known_gt = out["tcls"] < known
    out["op_hit"] = hit[known_gt]
    side = np.sqrt(np.maximum(gt[:, 3] - gt[:, 1], 0) * np.maximum(gt[:, 4] - gt[:, 2], 0)) * imgsz / max(w, h)
    out["op_side"] = side[known_gt]
    return out


def aggregate(recs: list[dict], known: int) -> dict:
    from ultralytics.utils.metrics import ap_per_class

    r = {"images": len(recs)}
    tcls = np.concatenate([x["tcls"] for x in recs]) if recs else np.zeros(0, int)
    tp = np.concatenate([x["tp"] for x in recs]) if recs else np.zeros((0, 10), bool)
    conf = np.concatenate([x["conf"] for x in recs]) if recs else np.zeros(0)
    pcls = np.concatenate([x["pcls"] for x in recs]) if recs else np.zeros(0, int)
    r["boxes"] = int(len(tcls))
    per = {}
    if len(tp):
        _, _, _, _, _, ap, uc, *_ = ap_per_class(tp, conf, pcls, tcls)
        for k, c in enumerate(uc):
            per[int(c)] = (float(ap[k].mean()), float(ap[k, 0]))
    else:
        for c in np.unique(tcls):
            per[int(c)] = (0.0, 0.0)
    kn = [v for c, v in per.items() if c < known]
    r["map_known"] = float(np.mean([v[0] for v in kn])) if kn else 0.0
    r["map50_known"] = float(np.mean([v[1] for v in kn])) if kn else 0.0
    r["map_all"] = float(np.mean([v[0] for v in per.values()])) if per else 0.0
    r["per_class"] = {c: round(v[0], 4) for c, v in sorted(per.items())}
    tpn, fpn = sum(x["op_tp"] for x in recs), sum(x["op_fp"] for x in recs)
    hit = np.concatenate([x["op_hit"] for x in recs]) if recs else np.zeros(0, bool)
    side = np.concatenate([x["op_side"] for x in recs]) if recs else np.zeros(0)
    r["op"] = {"tp": tpn, "fp": fpn, "gt": int(len(hit)),
               "p": tpn / max(tpn + fpn, 1), "r": float(hit.mean()) if len(hit) else 0.0}
    r["op"]["f1"] = 2 * r["op"]["p"] * r["op"]["r"] / max(r["op"]["p"] + r["op"]["r"], 1e-9)
    r["recall_by_size"] = {
        lab: (int(((side >= lo) & (side < hi)).sum()),
              float(hit[(side >= lo) & (side < hi)].mean()) if ((side >= lo) & (side < hi)).any() else None)
        for lo, hi, lab in SIZE_BINS}
    return r


# ---------------------------------------------------------------- 출력
def print_domain(title: str, rows: list[tuple[str, dict, float]], names: list[str], known: int, op_conf: float) -> None:
    base = rows[0][1]
    print(f"\n[{title}] {base['images']}장 / 박스 {base['boxes']}개   (맨 위 {rows[0][0]} 대비)")
    print(f"  {'변형':<18}{'mAP50-95':>9}{'(차)':>8}{'mAP50':>8}{'mAP8':>8}   {'P':>6}{'R':>6}{'F1':>6}{'FP':>7}{'(차)':>7}  {'초/장':>6}")
    for name, a, sec in rows:
        o, bo = a["op"], base["op"]
        print(f"  {name:<18}{a['map_known']:>9.4f}{a['map_known'] - base['map_known']:>+8.4f}{a['map50_known']:>8.4f}"
              f"{a['map_all']:>8.4f}   {o['p']:>6.3f}{o['r']:>6.3f}{o['f1']:>6.3f}{o['fp']:>7d}{o['fp'] - bo['fp']:>+7d}  {sec:>6.3f}")
    print("  (mAP50-95·mAP50 은 모델이 아는 클래스 0–%d 의 평균, mAP8 은 pickup 까지 넣은 평균 — 운영 조건 P·R·F1·FP 는 conf %.2f·IoU 0.5)" % (known - 1, op_conf))
    print(f"\n  클래스별 mAP50-95 ({title})")
    print(f"  {'변형':<18}" + "".join(f"{names[c][:7]:>9}" for c in range(known)))
    for name, a, _ in rows:
        print(f"  {name:<18}" + "".join(f"{a['per_class'].get(c, float('nan')):>9.4f}" for c in range(known)))
    print(f"\n  객체 크기별 재현율 (운영 조건, 640 입력 환산 변 길이) ({title})")
    print(f"  {'변형':<18}" + "".join(f"{lab + ' (' + str(base['recall_by_size'][lab][0]) + ')':>14}" for _, _, lab in SIZE_BINS))
    for name, a, _ in rows:
        print(f"  {name:<18}" + "".join(
            f"{(a['recall_by_size'][lab][1] if a['recall_by_size'][lab][1] is not None else float('nan')):>14.3f}" for _, _, lab in SIZE_BINS))


def main():
    args = parse_args()
    cfg = Config.load(args.config)
    weights = str(resolve(args.weights)) if args.weights else cfg.active_weights()
    imgsz = args.imgsz or cfg.imgsz
    variants = [parse_variant(s) for s in args.variants.split(",") if s.strip()]
    out = resolve(args.out) if args.out else resolve(
        f"runs/test_tiles/{date.today():%Y%m%d}_{Path(weights).stem}_{args.split}")
    names = class_names(cfg)

    items = build_split(cfg, args.split, out)
    if args.limit:
        items = items[:args.limit]
    print(f"{args.split} {len(items)}장 -> {out}")

    from ultralytics import YOLO
    from ultralytics.utils.patches import imread

    model = YOLO(weights)
    known = len(model.names)
    print(f"가중치 {weights} (sha256 {sha256(Path(weights))[:12]}…) / imgsz {imgsz} / 모델 클래스 {known}개"
          f" / end2end {getattr(model.model, 'end2end', '?')} / 겹침 {args.overlap} / 잘림 판정 {args.edge_px}px", flush=True)

    # 첫 추론은 CUDA 초기화로 길어 시간을 왜곡한다 (LiveDetector 와 같은 이유로 빈 이미지로 예열)
    model.predict(np.zeros((64, 64, 3), np.uint8), imgsz=imgsz, device=cfg.device, verbose=False)
    comp_names = sorted({c for v in variants for c in v["comps"]}, key=lambda c: (c != "full", c))
    lab_dir = out / "labels" / args.split
    recs: dict[str, list[tuple[dict, set]]] = {v["name"]: [] for v in variants}   # 변형 -> [(이미지 결과, 도메인 집합)]
    secs = {v["name"]: 0.0 for v in variants}
    t_start = time.perf_counter()
    for n, it in enumerate(items, 1):
        img = imread(it["img"])
        if img is None:
            print(f"  읽지 못해 건너뜀: {it['img']}")
            continue
        h, w = img.shape[:2]
        comps, ctime = {}, {}
        for c in comp_names:
            comps[c], ctime[c] = run_component(model, img, c, args, imgsz, cfg.device)
        gt = read_gt(lab_dir / f"{Path(it['img']).stem}.txt", w, h)
        doms = {k for k, (_, keep) in DOMAINS.items() if keep(it)}
        for v in variants:
            d = compose(v, comps, w, h)
            used = ["full"] if v["big"] and max(w, h) <= BIG_SIDE else v["comps"]
            secs[v["name"]] += sum(ctime[c] for c in used)
            recs[v["name"]].append((eval_image(merge(d, MAP_IOU, MAP_MAXDET, 0.0), merge(d, 0.6, 100, args.op_conf),
                                               gt, w, h, imgsz, known), doms))
        if n % 100 == 0 or n == len(items):
            print(f"  {n}/{len(items)}장 ({time.perf_counter() - t_start:.0f}초)", flush=True)

    result = {"at": datetime.now().isoformat(timespec="seconds"), "weights": weights, "imgsz": imgsz,
              "split": args.split, "overlap": args.overlap, "edge_px": args.edge_px, "images": len(items), "domains": {}}
    for key, (label, _) in DOMAINS.items():
        rows = []
        for v in variants:
            sub = [r for r, doms in recs[v["name"]] if key in doms]
            if not sub:
                break
            rows.append((v["name"], aggregate(sub, known), secs[v["name"]] / max(len(items), 1)))
        if not rows:
            continue
        print_domain(label, rows, names, known, args.op_conf)
        result["domains"][key] = {n: {k: a[k] for k in ("images", "boxes", "map_known", "map50_known", "map_all",
                                                         "per_class", "op", "recall_by_size")} | {"sec_per_img": s}
                                  for n, a, s in rows}
    print("\n  초/장은 전체 이미지 평균 추론 시간(디코딩·NMS 제외)이다. 'big' 변형은 작은 이미지에서 타일을 안 쓰므로 그만큼 줄어든다.")

    if not args.no_check_val and not args.limit and "full" in comp_names:
        listing = out / f"{args.split}_all.txt"
        listing.write_text("\n".join(str(it["img"]) for it in items if DOMAINS["all"][1](it)) + "\n", encoding="utf-8")
        res = model.val(data=str(write_yaml(out / "data_all.yaml", out, listing.name, names)), split="val", imgsz=imgsz,
                        device=cfg.device, workers=0, plots=False, verbose=False,
                        project=str(out.parent), name=f"{out.name}_val", exist_ok=True)
        mine = next((a for n, a, _ in [(v["name"], aggregate([r for r, d in recs[v["name"]] if "all" in d], known), 0)
                                       for v in variants] if n == "full"), None)
        if mine:
            print(f"\n교차 확인 (전체, 8클래스 평균): model.val mAP50-95 {res.box.map:.4f} / mAP50 {res.box.map50:.4f}"
                  f"   이 평가기의 full {mine['map_all']:.4f}   차이 {mine['map_all'] - res.box.map:+.4f}")
            result["check_val"] = {"val_map": float(res.box.map), "val_map50": float(res.box.map50), "mine_full": mine["map_all"]}

    (out / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n결과: {out / 'results.json'}")
    if not args.keep:
        shutil.rmtree(out / "images", ignore_errors=True)
        print("복사한 이미지는 지웠다 (--keep 으로 남긴다).")


if __name__ == "__main__":      # Windows spawn (CLAUDE.md 함정 7)
    main()
