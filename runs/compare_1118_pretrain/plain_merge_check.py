"""통째 자동 라벨링의 차량 겹침 정리(2026-11-20)를 진짜 DB 의 사본·운영 가중치로 끝까지 돌려 본다 — 저장된 박스로 확인.

실제 DB 를 sqlite backup 으로 두 벌 뜬 뒤(원본은 읽기 전용으로 열기만), 같은 승인 이미지에 `AutoLabeler.run`(통째)을
predict.class_iou 0(끔)과 config 값(0.6)으로 한 번씩 돌려 저장된 박스를 견준다: 켠 쪽이 끈 쪽에서 몇 개만 빠진 것인가,
빠진 수가 report.merged 와 같은가, 켠 쪽에 클래스가 다른 차량 박스가 IoU 문턱 이상 겹친 쌍이 남았는가, 빠진 박스와 남은
박스의 클래스 짝. 사본은 인자로 준 폴더에만 쓴다(그 사본의 승인 이미지는 예측으로 덮이고 검수 대기가 된다). 출력에 파일
이름은 없다.
  python -I plain_merge_check.py <임시 폴더> [--limit N]
"""
import argparse
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(r"C:\local_auto_labeling")
sys.path.insert(0, str(REPO / "src"))
from autolabel import tiling  # noqa: E402
from autolabel.config import Config  # noqa: E402
from autolabel.predictor import AutoLabeler  # noqa: E402
from autolabel.store import Store  # noqa: E402


def copy_db(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    s = sqlite3.connect(f"file:{src.as_posix()}?mode=ro", uri=True)
    d = sqlite3.connect(str(dst))
    s.backup(d)
    d.close()
    s.close()


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def saved(store: Store, image_id: int) -> list[tuple]:
    """(클래스, conf, x1, y1, x2, y2 정규화) — IoU 는 축마다 늘여도 그대로라 정규화 좌표로 잰다."""
    return [(b.cls, round(b.conf, 6), round(b.cx - b.w / 2, 6), round(b.cy - b.h / 2, 6), round(b.cx + b.w / 2, 6),
             round(b.cy + b.h / 2, 6)) for b in store.get_boxes(image_id)]


def cross_pairs(bx, group, thr) -> int:
    return sum(1 for i in range(len(bx)) for j in range(i + 1, len(bx))
               if bx[i][0] != bx[j][0] and bx[i][0] in group and bx[j][0] in group and iou(bx[i][2:], bx[j][2:]) >= thr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("work", type=Path)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    src = Config.load(REPO / "config.yaml").db
    print(f"원본 DB(읽기 전용으로 사본만 뜸): {src}")
    res = {}
    for tag, class_iou in (("끔", 0.0), ("켬", None)):
        db = args.work / f"labeling_{'off' if class_iou == 0 else 'on'}.db"
        copy_db(src, db)
        cfg = Config.load(REPO / "config.yaml")
        cfg.raw["data"]["db"] = str(db)
        cfg.raw.setdefault("backup", {})["keep"] = 0
        if class_iou is not None:
            cfg.raw["predict"]["class_iou"] = class_iou
        store = Store(db)
        ids = [r["id"] for r in store.list_images(status="approved", limit=10**9)]
        if args.limit:
            ids = ids[:args.limit]
        report = AutoLabeler(cfg, store).run(ids)
        group = tiling.vehicle_ids(store.get_classes())
        boxes = {i: saved(store, i) for i in ids}
        res[tag] = (report, boxes, group, store.get_classes())
        print(f"[{tag}] class_iou {report.class_iou} · 처리 {report.processed}장 · 박스 {report.boxes} · 뺀 것 {report.merged} · "
              f"오류 {len(report.errors)}", flush=True)
    (r_off, b_off, group, names), (r_on, b_on, _, _) = res["끔"], res["켬"]
    gone_total, not_subset, pairs = 0, 0, Counter()
    for i in b_off:
        off, on = Counter(b_off[i]), Counter(b_on[i])
        if on - off:
            not_subset += 1                       # 켠 쪽에만 있는 박스 — 있으면 안 된다
        gone = list((off - on).elements())
        gone_total += len(gone)
        for g in gone:                            # 빠진 박스를 지운 것: 켠 쪽에 남은, 클래스가 다른 차량 박스 중 가장 많이 겹친 것
            cand = [(iou(g[2:], k[2:]), k) for k in b_on[i] if k[0] != g[0] and k[0] in group]
            if cand:
                best = max(cand)
                pairs[(names[g[0]], names[best[1][0]], best[0] >= 0.6, best[1][1] >= g[1])] += 1
    for tag, (_, bx, _, _) in res.items():
        print(f"[{tag}] 클래스가 다른 차량 박스 쌍: IoU ≥ 0.6 {sum(cross_pairs(v, group, 0.6) for v in bx.values())} · "
              f"≥ 0.5 {sum(cross_pairs(v, group, 0.5) for v in bx.values())} · 그런 쌍이 있는 이미지 "
              f"{sum(1 for v in bx.values() if cross_pairs(v, group, 0.6))}장")
    print(f"켠 쪽에만 있는 박스가 든 이미지: {not_subset}장 (0 이어야 한다) · 끈 쪽에서 빠진 박스 {gone_total}개 = report.merged "
          f"{r_on.merged} → {'같음' if gone_total == r_on.merged else '다름'}")
    print("빠진 박스 ← 남은 박스(클래스), IoU ≥ 0.6, 남은 쪽 conf ≥ 빠진 쪽:",
          {f"{a}←{b}{'' if ok else ' (IoU<0.6!)'}{'' if hi else ' (conf 낮음!)'}": n for (a, b, ok, hi), n in pairs.most_common()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
