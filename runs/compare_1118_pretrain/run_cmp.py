"""사전학습 전부(A) / backbone 만(B) / 없음(C) — 같은 yolo26m 구조·같은 인자로 학습해 견준다.

인자는 프로젝트 config.yaml 의 학습 설정과 같다(640·batch 10·optimizer auto·cos_lr·box 7.5·cls 1.5·dfl 1.5·close_mosaic 10·
증강 값). 다른 것: cache 를 끄고(같은 픽셀이다) 그림·중간 체크포인트를 쓰지 않는다. 끝난 실행은 건너뛴다(done.json).
결과는 results.jsonl 에 한 줄씩 — Ultralytics val(mAP)과, conf 0.25 에서 찾은·클래스까지 맞은·헛 박스 수(cmp_common.count).

  python run_cmp.py --jobs A.0.100,C.0.100,B.0.300        # 갈래.fold.epoch 을 차례로
"""
import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))    # python -I 로 돌려도 옆의 cmp_common 을 찾게
import cmp_common as cc  # noqa: E402


def curve(csv_path: Path) -> dict:
    rows = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8")))
    key = next(k for k in rows[0] if k.strip() == "metrics/mAP50-95(B)")
    tkey = next(k for k in rows[0] if k.strip() == "time")
    vals = [float(r[key]) for r in rows]
    times = [float(r[tkey]) for r in rows]
    steps = sorted(b - a for a, b in zip(times, times[1:]))
    best = max(range(len(vals)), key=lambda i: vals[i])
    marks = {str(e): round(vals[e - 1], 4) for e in (10, 25, 50, 100, 150, 200, 300) if e <= len(vals)}
    return {"epochs_run": len(vals), "best_epoch": best + 1, "best_map": round(vals[best], 4),
            "last_map": round(vals[-1], 4), "at": marks,
            "epoch_sec_median": round(steps[len(steps) // 2], 2) if steps else None,
            "epoch_sec_max": round(steps[-1], 2) if steps else None}


def run_one(arm: str, fold: int, epochs: int, imgsz: int, batch: int, seed: int = 0) -> None:
    from ultralytics import YOLO

    # P2S = P2 곁가지 구조에 사전학습 가중치를 역할대로 얹은 것(make_p2side_init.py), P2N = Ultralytics 의 yolo26m-p2.yaml 에
    # 기본 로더(이름으로만 맞춤)로 얹은 것
    sources = {"A": str(cc.REPO / "weights" / "yolo26m.pt"), "B": str(cc.ROOT / "backbone_only.pt"), "C": "yolo26m.yaml",
               "P2S": str(cc.ROOT / "p2" / "p2side_init.pt"), "P2N": "yolo26m-p2.yaml",
               # O = Objects365 v1 사전학습 가중치 그대로(이름이 같은 6클래스만 분류 줄이 옮겨진다),
               # OM = 같은 가중치의 클래스 이름을 우리 이름에 맞춘 사본(make_o365_mapped.py — pickup·cone·responder 까지 9클래스)
               "O": str(cc.ROOT / "dl_o365" / "yolo26m-objv1-150.pt"), "OM": str(cc.ROOT / "o365_mapped.pt"),
               # F = 운영 가중치(COCO 에서 출발해 옛 데이터 학습 3,725장·8클래스로 학습한 yolo26m)를 파운데이션으로
               "F": str(cc.REPO / "models" / "best_y26m_8cls.pt"),
               # FZ11·FZ23 = F 와 같은 출발점인데 앞쪽을 얼린다(아래 freeze) — 옛 데이터로 배운 특징을 건드리지 않게
               "FZ11": str(cc.REPO / "models" / "best_y26m_8cls.pt"), "FZ23": str(cc.REPO / "models" / "best_y26m_8cls.pt"),
               # FZ11X1·FZ11X2 = FZ11 과 같은데 학습 쪽에만 옛 데이터의 responder·cone 이미지를 더한다(make_extra.py —
               # X1 = 두 클래스 박스가 든 옛 이미지 전부, X2 = 그중 카메라가 적힌 프레임만). 검증 목록은 그대로다.
               "FZ11X1": str(cc.REPO / "models" / "best_y26m_8cls.pt"), "FZ11X2": str(cc.REPO / "models" / "best_y26m_8cls.pt")}
    # Ultralytics 의 freeze=N 은 층 0..N-1 의 파라미터를 고정하고 그 층들의 BatchNorm 통계도 멈춘다(trainer._model_train).
    # 11 = backbone(층 0-10)만 얼림, 23 = backbone·neck(층 0-22)을 얼려 Detect(층 23)만 학습
    freeze = {"FZ11": 11, "FZ23": 23, "FZ11X1": 11, "FZ11X2": 11}
    extra = {"FZ11X1": "X1", "FZ11X2": "X2"}
    runs = cc.ROOT / "runs"
    tag = f"e{epochs}"
    name = f"{arm}_f{fold}_{tag}" + (f"_s{seed}" if seed else "")
    out = runs / name
    if (out / "done.json").exists():
        print(f"[건너뜀] {name} — 이미 끝났다", flush=True)
        return
    data = str(cc.ROOT / "folds" / (f"f{fold}_{extra[arm]}.yaml" if arm in extra else f"f{fold}.yaml"))
    train_images = sum(1 for x in (cc.ROOT / "folds" / (f"f{fold}_train_{extra[arm]}.txt" if arm in extra else f"f{fold}_train.txt"))
                       .read_text(encoding="utf-8").split("\n") if x.strip())
    val_files = cc.fold_files(fold, "val")
    kw = dict(data=data, epochs=epochs, batch=batch, imgsz=imgsz, device=0, patience=120, workers=4, seed=seed,
              project=str(runs), name=name, exist_ok=True, plots=False, pretrained=(arm != "C"), amp=True, cos_lr=True,
              cache=False, box=7.5, cls=1.5, dfl=1.5, close_mosaic=10, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, fliplr=0.5,
              flipud=0.0, save_period=-1)
    if arm in freeze:
        kw["freeze"] = freeze[arm]
    print(f"=== START {name}", flush=True)
    t0 = time.time()
    model = YOLO(sources[arm])
    if arm == "P2N":
        model.load(str(cc.REPO / "weights" / "yolo26m.pt"))
    model.train(**kw)
    rec = {"name": name, "arm": arm, "fold": fold, "tag": tag, "epochs": epochs, "imgsz": imgsz, "batch": batch, "seed": seed,
           "freeze": freeze.get(arm, 0), "extra": extra.get(arm), "train_images": train_images,
           "train_seconds": round(time.time() - t0, 1), "val_images": len(val_files), **curve(out / "results.csv")}
    for which in ("last", "best"):
        m = YOLO(str(out / "weights" / f"{which}.pt"))
        r = m.val(data=data, split="val", imgsz=imgsz, batch=8, device=0, iou=0.6, plots=False, verbose=False,
                  project=str(runs), name=name + "_val", exist_ok=True, workers=0)
        c = cc.count(m, val_files, cc.SELF_OK, imgsz)
        rec[which] = {"map": round(float(r.box.map), 4), "map50": round(float(r.box.map50), 4), **c}
        print(f"RESULT {name} {which}: mAP50-95 {rec[which]['map']} · {cc.brief(c)}", flush=True)
    with open(cc.ROOT / "results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (out / "done.json").write_text(json.dumps({"seconds": rec["train_seconds"]}), encoding="utf-8")
    print(f"=== DONE {name} {rec['train_seconds']}s best_epoch {rec['best_epoch']} "
          f"epoch_sec_median {rec['epoch_sec_median']}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True, help="갈래.fold.epoch 을 쉼표로 (예: A.0.100,C.0.300)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=10)
    args = ap.parse_args()

    os.chdir(cc.ROOT / "work")          # AMP 확인이 찾는 yolo26n.pt 가 여기 있다(내려받지 않게)
    failed = 0
    for job in args.jobs.split(","):
        arm, fold, epochs, *rest = job.split(".")
        seed = int(rest[0]) if rest else 0      # 넷째 칸은 seed (없으면 0)
        for attempt in (1, 2):          # 두 줄기가 같은 라벨 캐시를 동시에 건드려 죽는 드문 경우를 한 번 다시 해 본다
            try:
                run_one(arm, int(fold), int(epochs), args.imgsz, args.batch, seed)
                break
            except Exception:
                print(f"[실패 {attempt}/2] {job}\n{traceback.format_exc()}", flush=True)
                time.sleep(20)
        else:
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
