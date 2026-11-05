"""모델 크기를 바꿔 짧게 돌려보고 본 학습과 같은 지점에서 비교한다.

물음은 하나다 — **용량이 병목인가, 해상도가 병목인가.**

본 학습(yolo26n @640)의 손실 곡선은 용량 부족보다 암기 쪽을 가리킨다.
epoch 184 에서 val/cls_loss 가 바닥(1.5198)을 치고 올라가는데(230 에서 1.5424)
train/cls_loss 는 계속 떨어진다(1.4011 -> 1.1513). 그렇다면 파라미터를 3.9배로
늘려도(2.57M -> 10.01M) 그 속도만 빨라질 수 있다.

추측으로 17시간을 걸기 전에 3.5시간으로 확인하자는 것이 이 스크립트다.

    python tools/probe_scale.py                    # yolo26s, 60 epoch, batch 16
    python tools/probe_scale.py --scale m          # 더 큰 것
    python tools/probe_scale.py --resume           # 중단된 프로브 이어하기

**비교가 성립하려면 첫 60 epoch 이 본 학습과 같은 조건이어야 한다.** 그래서:

- **`epochs` 는 300 으로 두고 60 에서 끊는다.** `cos_lr` 스케줄이 총 epoch 수에
  묶여 있어서, `epochs=60` 으로 주면 60 epoch 만에 lr 이 끝까지 내려가 버린다.
  본 학습의 1~60 구간과 전혀 다른 lr 을 밟게 된다.
- **`optimizer` 를 MuSGD 로 못박는다.** `auto` 로 두면 iterations 로 고르는데
  (`ceil(len/max(batch,nbs)) * epochs`), batch 16 / epochs 60 은 4,440 < 10,000 이라
  **AdamW 로 넘어간다.** 본 학습은 22,200 이라 MuSGD(lr 0.01) 였다. 단 auto 는
  고르면서 `warmup_bias_lr` 을 0 으로 바꾸므로, 이름으로 못박은 여기서는 기본 0.1 로
  남는다 — 본 학습과 다른 유일한 값이다 (warmup 3 epoch 동안 bias 의 학습률).
- **`close_mosaic` 을 0 으로 둔다.** 본 학습은 15/300 이라 285 에서 켜지고 259 에서
  끝나 한 번도 발동하지 않았다. 60 epoch 짜리에 15 를 그대로 주면 45 에서 켜져
  없던 조건이 생긴다.
- **데이터셋을 다시 만들지 않는다.** 같은 val 이어야 비교가 성립한다.

`models/best.pt` 는 건드리지 않는다. 결과는 `runs/probe_{scale}{imgsz}` 에만 남는다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # tools/ -> 저장소 루트

from autolabel.config import Config, pretrained_path, resolve
from autolabel.store import Store
from autolabel.trainer import (
    attach_callbacks,
    ensure_resume_dataset,
    resume_point_in,
    trim_results_csv,
)

# 본 학습에서 optimizer=auto 가 고른 값. 이름으로 지정해야 lr0/momentum 이 먹는다.
BASE_LR = 0.01
BASE_MOMENTUM = 0.9
SCHEDULE_EPOCHS = 300      # 본 학습과 같은 코사인 스케줄을 밟기 위한 총 epoch


def parse_args():
    p = argparse.ArgumentParser(
        description="모델 크기를 바꿔 짧게 돌려보고 본 학습과 같은 지점에서 비교한다.")
    p.add_argument("--config", default=None, help="설정 파일 (기본: 저장소의 config.yaml)")
    p.add_argument("--scale", default="s", choices=list("nsmlx"),
                   help="yolo26 크기 (기본 s)")
    p.add_argument("--imgsz", type=int, default=None,
                   help="기본은 config 의 model.imgsz")
    p.add_argument("--batch", type=int, default=16,
                   help="yolo26s @640 은 이 카드에서 16 이 상한이다 (실측 4.47GiB). "
                        "24 는 6.61GiB 로 공유 메모리 페이징이 시작돼 3배 느려진다")
    p.add_argument("--stop-at", type=int, default=60,
                   help="이 epoch 까지만 돌린다 (기본 60)")
    p.add_argument("--baseline", default="runs/train",
                   help="비교 대상 학습 폴더 (기본: 본 학습)")
    p.add_argument("--resume", nargs="?", const="auto", default=None, metavar="CKPT",
                   help="중단된 프로브를 이어한다")
    p.add_argument("--close-mosaic", type=int, default=None,
                   help=f"이어 학습 전용. 마지막 N epoch 은 mosaic 을 끈다 "
                        f"(스케줄 {SCHEDULE_EPOCHS} 기준이므로 30 이면 270 에서 발동)")
    p.add_argument("--patience", type=int, default=None,
                   help="이어 학습 전용. 개선 없이 N epoch 지나면 조기 종료. "
                        "close_mosaic 보다 먼저 끊기지 않도록 넉넉히 줄 것")
    return p.parse_args()


def read_row(csv: Path, epoch: int) -> dict | None:
    """results.csv 에서 해당 epoch 행을 뽑는다."""
    if not csv.exists():
        return None
    lines = [ln for ln in csv.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) < 2:
        return None
    head = [h.strip() for h in lines[0].split(",")]
    for ln in lines[1:]:
        cells = ln.split(",")
        if int(float(cells[0])) == epoch:
            return {h: c for h, c in zip(head, cells)}
    return None


def compare(run_dir: Path, baseline: Path, epoch: int, scale: str) -> None:
    """같은 epoch 에서 프로브와 기준선을 나란히 보여준다."""
    keys = [
        ("metrics/precision(B)", "P"),
        ("metrics/recall(B)", "R"),
        ("metrics/mAP50(B)", "mAP50"),
        ("metrics/mAP50-95(B)", "mAP50-95"),
        ("train/cls_loss", "train/cls"),
        ("val/cls_loss", "val/cls"),
        ("train/box_loss", "train/box"),
        ("val/box_loss", "val/box"),
    ]
    a = read_row(baseline / "results.csv", epoch)
    b = read_row(run_dir / "results.csv", epoch)
    if not a or not b:
        print(f"\n비교할 행이 없습니다 (epoch {epoch}). "
              f"기준선={bool(a)} 프로브={bool(b)}")
        return

    print("\n" + "=" * 58)
    print(f"epoch {epoch} 시점 비교")
    print("=" * 58)
    # 머리글은 --baseline 폴더 이름으로 (2026-11-05 — 'yolo26n(기준)' 이 박혀 있어 기준선을 바꿔도 그대로였다)
    print(f"{'':<12}{baseline.name + '(기준)':>15}{'yolo26' + scale:>13}{'차이':>12}")
    print("-" * 58)
    for key, label in keys:
        if key not in a or key not in b:
            continue
        x, y = float(a[key]), float(b[key])
        print(f"{label:<12}{x:>15.5f}{y:>13.5f}{y - x:>+12.5f}")
    print("=" * 58)
    print("mAP 는 높을수록, 손실은 낮을수록 좋습니다.\n"
          "mAP50-95 가 뚜렷이 높으면 용량이 병목이라는 뜻이고,\n"
          "비슷하거나 낮으면 해상도(960 학습) 쪽으로 가는 것이 맞습니다.")


def do_resume(args, cfg: Config, run_dir: Path) -> int:
    from ultralytics import YOLO

    point = None
    if args.resume != "auto":
        p = Path(args.resume)
        point = resume_point_in(p if p.is_dir() else p.parent.parent)
    else:
        point = resume_point_in(run_dir)
    if point is None:
        print(f"이어 학습할 체크포인트가 없습니다: {run_dir}")
        return 1

    print(point.summary(), flush=True)
    ensure_resume_dataset(cfg, Store(cfg.db), point.run_dir)   # 없으면 DB 로 다시 만든다
    trimmed = trim_results_csv(point.run_dir, point.done)
    if trimmed:
        print(f"results.csv 에서 {trimmed}행을 잘라냈습니다 (원본은 results.csv.bak).")

    model = YOLO(str(point.ckpt))
    state = {"epoch": point.epoch}

    def _track(trainer) -> None:
        state["epoch"] = trainer.epoch

    model.add_callback("on_train_epoch_start", _track)
    attach_callbacks(model, lambda: state["epoch"] + 1 >= args.stop_at)

    # 이어 학습에서 바꿀 수 있는 것은 check_resume() 의 허용 목록뿐이다
    # (engine/trainer.py). close_mosaic 과 patience 는 그 안에 있어서, 프로브를
    # 본 학습으로 연장할 때 마무리 구간 설정을 여기서 넣을 수 있다.
    kw = dict(resume=str(point.ckpt), device=cfg.device,
              workers=int(cfg.get("train.workers", 4)), batch=args.batch)
    if args.close_mosaic is not None:
        kw["close_mosaic"] = args.close_mosaic
        fires = SCHEDULE_EPOCHS - args.close_mosaic
        print(f"close_mosaic={args.close_mosaic} → epoch {fires} 부터 mosaic 을 끕니다.")
    if args.patience is not None:
        kw["patience"] = args.patience
        print(f"patience={args.patience} 로 조기 종료를 켭니다.")

    model.train(**kw)
    compare(point.run_dir, resolve(args.baseline), args.stop_at, args.scale)
    return 0


def main() -> int:
    args = parse_args()
    cfg = Config.load(args.config)
    imgsz = args.imgsz or cfg.imgsz
    run_dir = resolve("runs") / f"probe_{args.scale}{imgsz}"

    if args.resume:
        return do_resume(args, cfg, run_dir)

    data_yaml = cfg.dataset / "data.yaml"
    if not data_yaml.exists():
        print(f"{cfg.dataset} 가 없습니다. 본 학습과 같은 데이터셋이 있어야 "
              "비교가 성립합니다. 먼저 'labeler.py train' 을 돌리세요.")
        return 1

    n_train = len(list((cfg.dataset / "labels" / "train").glob("*.txt")))
    n_val = len(list((cfg.dataset / "labels" / "val").glob("*.txt")))
    ov = cfg.get("train.overrides") or {}

    print(f"""
모델        : yolo26{args.scale}.pt  (기준선 {args.baseline})
데이터      : {data_yaml}  (train {n_train} / val {n_val}, 다시 만들지 않음)
해상도      : {imgsz}
배치        : {args.batch}
epoch       : {args.stop_at} 에서 중단  (스케줄은 {SCHEDULE_EPOCHS} 기준)
옵티마이저  : MuSGD  lr0={BASE_LR}  momentum={BASE_MOMENTUM}
증강        : 본 학습과 동일, close_mosaic 0 (60 epoch 안에 발동하지 않도록)
결과        : {run_dir}
""", flush=True)

    from ultralytics import YOLO

    model = YOLO(pretrained_path(f"yolo26{args.scale}.pt"))   # weights/ 에 있으면 거기서
    state = {"epoch": 0}

    def _track(trainer) -> None:
        state["epoch"] = trainer.epoch

    model.add_callback("on_train_epoch_start", _track)
    # stop_at epoch 을 마치면 끊는다. 검증·저장은 마치고 나가며,
    # final_eval() 이 last.pt 를 훼손하기 전에 resume.pt 스냅샷도 남는다.
    attach_callbacks(model, lambda: state["epoch"] + 1 >= args.stop_at)

    results = model.train(
        data=str(data_yaml),
        epochs=SCHEDULE_EPOCHS,
        imgsz=imgsz,
        batch=args.batch,
        device=cfg.device,
        workers=int(cfg.get("train.workers", 4)),
        seed=int(cfg.get("train.seed", 0)),
        patience=SCHEDULE_EPOCHS,      # 조기 종료 없음. 끊는 것은 위 콜백이 한다
        project=str(resolve("runs")),
        name=run_dir.name,
        exist_ok=False,
        plots=True,
        optimizer="MuSGD",
        lr0=BASE_LR,
        lrf=float(ov.get("lrf", 0.01)),
        momentum=BASE_MOMENTUM,
        cos_lr=True,
        mosaic=float(ov.get("mosaic", 1.0)),
        close_mosaic=0,
        box=float(ov.get("box", 7.5)),
        cls=float(ov.get("cls", 1.5)),
        dfl=float(ov.get("dfl", 1.5)),   # 0 금지 (박스 거리 손실 게인)
        hsv_h=float(ov.get("hsv_h", 0.015)),
        hsv_s=float(ov.get("hsv_s", 0.7)),
        hsv_v=float(ov.get("hsv_v", 0.4)),
        fliplr=float(ov.get("fliplr", 0.5)),
        flipud=float(ov.get("flipud", 0.0)),
        amp=True,
        cache=ov.get("cache", "disk"),   # Windows 에서 ram 금지
        save_period=20,
    )

    compare(Path(results.save_dir), resolve(args.baseline), args.stop_at, args.scale)
    print(f"\n결과 폴더 : {results.save_dir}")
    print("models/best.pt 는 건드리지 않았습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
