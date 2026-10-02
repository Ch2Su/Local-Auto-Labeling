"""해상도를 올리고 mosaic 을 끈 채 짧게 미세조정한다.

두 가지를 한 번에 회수하려는 스크립트다.

1. **놓친 close_mosaic 구간.** 본 학습은 `patience: 75` 때문에 epoch 259 에서 조기
   종료됐고, `close_mosaic` 이 발동하는 285 에 도달하지 못했다. 마지막에 mosaic 을
   끄고 도는 구간은 보통 mAP50-95 를 한 번 더 밀어올린다.
2. **해상도.** 640 으로 학습한 가중치를 해상도만 960 으로 바꿔 검증해도
   mAP50-95 가 0.7050 -> 0.7193 으로 오른다. 960 으로 학습까지 하면 더 오를 여지가
   있다. 640 입력에서 박스의 23.3% 가 32px 미만이라 근거도 맞는다.

    python tools/finetune.py                 # 960 / batch 16 / 30 epoch
    python tools/finetune.py --imgsz 640     # 해상도는 그대로 두고 mosaic 만 끄고 싶을 때
    python tools/finetune.py --resume        # 중단된 미세조정을 이어한다
    python tools/finetune.py --promote       # 끝나면 models/best.pt 까지 교체 (이전 것은 best.prev.pt)

`models/best.pt` 는 기본적으로 건드리지 않는다. `models/best_ft{imgsz}.pt` 로
저장한 뒤 옛 가중치와 나란히 검증해 보여주고, 교체는 사람이 정한다.

주의할 점 몇 가지 (전부 이 프로젝트에서 실제로 물린 것들이다)

- **`if __name__ == "__main__":` 가드가 반드시 있어야 한다.** Windows 는 DataLoader
  워커를 spawn 으로 띄우면서 이 파일을 다시 import 한다. 가드가 없으면 죽거나
  CPU 0% 로 조용히 멈춘다.
- **`optimizer` 를 이름으로 지정해야 한다.** `auto` 로 두면 `lr0` 와 `momentum` 을
  통째로 무시하고(engine/trainer.py 의 경고 문구 그대로), 게다가 이 스크립트처럼
  iterations 가 10,000 아래면 MuSGD 가 아니라 AdamW 를 골라 본 학습과 다른 체제가
  된다. 본 학습은 MuSGD(lr 0.01) 였다. (auto 는 `warmup_bias_lr` 도 0 으로 바꾸는데,
  여기서는 `warmup_epochs=1` 이라 영향이 더 작다.)
- **데이터셋 폴더(config 의 data.dataset)를 다시 만들지 않는다.** 본 학습과 같은 val 을 써야 전후 비교가
  성립한다. 검수가 늘어 새 데이터로 돌리려면 `--rebuild` 를 준다.
- 중간에 Ctrl+C 로 끊으면 `final_eval()` 이 돌지 않아 `last.pt` 가 온전하게 남는다.
  거기서 이어 붙일 수 있다.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # tools/ -> 저장소 루트

from autolabel.config import Config, resolve
from autolabel.store import Store
from autolabel.trainer import (
    ResumePoint,
    build_dataset,
    ensure_resume_dataset,
    promote_weights,
    resume_point_in,
    trim_results_csv,
)

# 본 학습의 기준값. lr 은 여기서 --lr-scale 을 곱해 쓴다.
BASE_LR = 0.01          # optimizer=auto 가 MuSGD 에 준 값
BASE_MOMENTUM = 0.9     # 같은 이유로 config 의 0.937 이 아니다
REF_BATCH, REF_IMGSZ, REF_VRAM_GB = 40, 640, 6.3   # 실측 기준점
FIXED_VRAM_GB = 1.2     # 배치와 무관한 몫 (컨텍스트 + 가중치 + 옵티마이저)
DEFAULT_BATCH = 16      # 960 에서 640/40 과 비슷한 메모리


def estimate_vram(batch: int, imgsz: int) -> float:
    """VRAM 대략치. 활성값은 batch x imgsz^2 에 비례한다고 본다."""
    act = (REF_VRAM_GB - FIXED_VRAM_GB) * (batch / REF_BATCH) * (imgsz / REF_IMGSZ) ** 2
    return FIXED_VRAM_GB + act


def parse_args():
    p = argparse.ArgumentParser(
        description="해상도를 올리고 mosaic 을 끈 채 짧게 미세조정한다.")
    p.add_argument("--config", default=None, help="설정 파일 (기본: 저장소의 config.yaml)")
    p.add_argument("--weights", default=None,
                   help="출발 가중치 (기본: config 의 model.weights)")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--batch", type=int, default=None,
                   help="메모리는 batch x imgsz^2 에 비례한다. 960 이면 16 이 640/40 과 비슷하다"
                        f" (기본 {DEFAULT_BATCH})")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--resume", nargs="?", const="auto", default=None, metavar="CKPT",
                   help="중단된 미세조정을 이어한다. 경로를 주면 그 체크포인트, "
                        "안 주면 runs/finetune<imgsz> 에서 알아서 찾는다")
    p.add_argument("--lr-scale", type=float, default=0.1,
                   help=f"본 학습 lr({BASE_LR})에 곱할 값")
    p.add_argument("--patience", type=int, default=None,
                   help="기본은 --epochs (즉 조기 종료 없음)")
    p.add_argument("--rebuild", action="store_true",
                   help="데이터셋 폴더를 승인 라벨로 다시 만든다 (val 이 바뀔 수 있음)")
    p.add_argument("--promote", action="store_true",
                   help="끝나면 models/best.pt 까지 교체")
    p.add_argument("--no-eval", action="store_true", help="끝난 뒤 비교 검증 생략")
    return p.parse_args()


def prepare_dataset(cfg: Config, store: Store, rebuild: bool) -> Path:
    data_yaml = cfg.dataset / "data.yaml"
    if rebuild or not data_yaml.exists():
        print("데이터셋을 새로 만듭니다...", flush=True)
        report = build_dataset(cfg, store)
        print(report.summary())
        return report.data_yaml

    n = sum(1 for s in ("train", "val", "test")
            for _ in (cfg.dataset / "labels" / s).glob("*.txt"))
    approved = store.counts().get("approved", 0)
    print(f"기존 데이터셋을 그대로 씁니다: {cfg.dataset} ({n}장)")
    if n != approved:
        print(f"  [알림] 현재 승인은 {approved}장이라 {approved - n:+d}장 차이가 납니다.\n"
              f"         본 학습과 같은 val 로 비교하려는 것이므로 의도된 동작입니다.\n"
              f"         새 검수분까지 넣으려면 --rebuild 를 주세요.")
    return data_yaml


def compare(weights: dict[str, Path], data_yaml: Path, cfg: Config,
            sizes: tuple[int, ...]) -> None:
    """옛/새 가중치를 해상도별로 검증해 나란히 보여준다."""
    from ultralytics import YOLO

    print("\n비교 검증 중...", flush=True)
    rows = []
    for label, path in weights.items():
        if not path.exists():
            continue
        for imgsz in sizes:
            r = YOLO(str(path)).val(
                data=str(data_yaml), imgsz=imgsz, batch=8, device=cfg.device,
                workers=0, verbose=False, plots=False,
            )
            rows.append((label, imgsz, r.box.mp, r.box.mr, r.box.map50, r.box.map))

    print("\n" + "=" * 66)
    print(f"{'가중치':<14}{'imgsz':>7}{'P':>9}{'R':>9}{'mAP50':>9}{'mAP50-95':>12}")
    print("=" * 66)
    base = None
    for label, imgsz, mp, mr, m50, m95 in rows:
        if base is None:
            base = m95
        mark = f"   ({m95 - base:+.4f})" if m95 != base else ""
        print(f"{label:<14}{imgsz:>7}{mp:>9.4f}{mr:>9.4f}{m50:>9.4f}{m95:>12.4f}{mark}")
    print("=" * 66)
    print("괄호는 첫 줄(기존 가중치) 대비 mAP50-95 차이입니다.")


def find_resume(imgsz: int, explicit: str) -> ResumePoint | None:
    """이어 학습할 체크포인트를 찾는다.

    `--resume 경로` 면 그 폴더를, 그냥 `--resume` 이면 runs/finetune{imgsz}* 중
    가장 최근 것을 본다. runs/train(본 학습)은 건드리지 않는다 — 이 스크립트로
    이어갈 대상이 아니다.
    """
    if explicit != "auto":
        p = Path(explicit)
        if p.is_dir():
            return resume_point_in(p)
        if not p.exists():
            return None
        from autolabel.trainer import _csv_last_epoch, _read_ckpt_meta

        meta = _read_ckpt_meta(p)
        if meta is None:
            return None
        run_dir = p.parent.parent
        point = ResumePoint(p, meta[0], meta[1], run_dir, p.name, data=meta[2])
        point.lost = max(0, _csv_last_epoch(run_dir) - point.done)
        return point

    runs = resolve("runs")
    dirs = sorted(
        (d for d in runs.glob(f"finetune{imgsz}*") if (d / "weights").is_dir()),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    for d in dirs:
        point = resume_point_in(d)
        if point is not None:
            return point
    return None


def do_resume(args, cfg: Config, store: Store) -> int:
    """중단된 미세조정을 이어한다.

    하이퍼파라미터는 체크포인트에 담긴 것을 그대로 쓴다. Ultralytics 의
    check_resume() 이 imgsz/batch/device/workers 등 몇 개만 덮어쓰기를 허용하므로
    (engine/trainer.py), 여기서 lr 이나 증강을 다시 넘겨도 무시된다.
    """
    from ultralytics import YOLO

    point = find_resume(args.imgsz, args.resume)
    if point is None:
        print(
            f"이어 학습할 체크포인트가 없습니다 (runs/finetune{args.imgsz}*).\n"
            "  정상 종료된 학습의 last.pt/best.pt 는 final_eval() 이 옵티마이저 상태를\n"
            "  지워 버려서 이어갈 수 없습니다. 그 경우 새로 돌리세요."
        )
        return 1

    data_yaml = cfg.dataset / "data.yaml"
    print(point.summary(), flush=True)
    ensure_resume_dataset(cfg, store, point.run_dir)   # 없으면 DB 로 다시 만든다
    trimmed = trim_results_csv(point.run_dir, point.done)
    if trimmed:
        print(f"results.csv 에서 {point.done} epoch 이후의 {trimmed}행을 잘라냈습니다 "
              f"(원본은 results.csv.bak).", flush=True)

    kw = dict(resume=str(point.ckpt), device=cfg.device,
              workers=int(cfg.get("train.workers", 4)))
    if args.batch is not None:
        kw["batch"] = args.batch
        print(f"배치를 {args.batch} 로 바꿔 이어갑니다 (VRAM 절약).", flush=True)

    # resume 에는 파일 경로를 넘겨야 한다. True 로 두면 Ultralytics 가
    # get_latest_run() 으로 아무 last.pt 나 집는다.
    results = YOLO(str(point.ckpt)).train(**kw)
    return finish(args, cfg, results, data_yaml)


def finish(args, cfg: Config, results, data_yaml: Path) -> int:
    """새 가중치를 models/ 로 내보내고 옛 가중치와 비교해 보여준다."""
    run_dir = Path(results.save_dir)
    best = run_dir / "weights" / "best.pt"
    src = Path(args.weights) if args.weights else cfg.weights
    out = cfg.weights.with_name(f"best_ft{args.imgsz}.pt")
    if best.exists():
        shutil.copy2(best, out)
    print(f"\n결과 폴더 : {run_dir}")
    print(f"새 가중치 : {out}")

    if not args.no_eval:
        sizes = (cfg.imgsz,) if args.imgsz == cfg.imgsz else (cfg.imgsz, args.imgsz)
        compare({"기존": src, "미세조정": out}, data_yaml, cfg, sizes)

    if args.promote:
        # 학습 버튼과 같은 규칙: 기존 운영 가중치를 best.prev.pt 로 남기고 교체한다.
        prev = promote_weights(out, cfg)
        print(f"\n{cfg.weights} 를 교체했습니다."
              + (f" 이전 것은 {prev}" if prev else ""))
    else:
        print("\n마음에 들면 아래로 교체하세요 (자동 라벨링이 쓰는 가중치입니다):")
        print(f"  copy \"{out}\" \"{cfg.weights}\"")
    return 0


def main() -> int:
    args = parse_args()
    cfg = Config.load(args.config)
    store = Store(cfg.db)

    if args.resume:
        return do_resume(args, cfg, store)

    src = Path(args.weights) if args.weights else cfg.weights
    if not src.exists():
        print(f"출발 가중치가 없습니다: {src}")
        return 1

    data_yaml = prepare_dataset(cfg, store, args.rebuild)
    batch = args.batch if args.batch is not None else DEFAULT_BATCH
    lr0 = BASE_LR * args.lr_scale
    patience = args.patience if args.patience is not None else args.epochs
    ov = cfg.get("train.overrides") or {}

    print(f"""
출발 가중치 : {src}
데이터      : {data_yaml}
해상도      : {args.imgsz}   (본 학습 {cfg.imgsz})
배치        : {batch}   (본 학습 {cfg.get('train.batch')})
epoch       : {args.epochs}  (patience {patience})
옵티마이저  : MuSGD  lr0={lr0}  momentum={BASE_MOMENTUM}
증강        : mosaic 0.0 (끔), close_mosaic 0
VRAM 예상   : 약 {estimate_vram(batch, args.imgsz):.1f} GB
""", flush=True)

    from ultralytics import YOLO

    model = YOLO(str(src))
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        device=cfg.device,
        workers=int(cfg.get("train.workers", 4)),
        seed=int(cfg.get("train.seed", 0)),
        patience=patience,
        project=str(resolve("runs")),
        name=f"finetune{args.imgsz}",
        exist_ok=False,
        plots=True,
        # 옵티마이저를 이름으로 못박아야 lr0/momentum 이 먹는다 (모듈 설명 참고).
        optimizer="MuSGD",
        lr0=lr0,
        lrf=float(ov.get("lrf", 0.01)),
        momentum=BASE_MOMENTUM,
        cos_lr=True,
        warmup_epochs=1.0,
        # 회수하려는 구간 그 자체. 처음부터 끝까지 mosaic 없이 돈다.
        mosaic=0.0,
        close_mosaic=0,
        # 손실 가중치와 색/반전 증강은 본 학습과 동일하게 유지한다.
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
        save_period=10,
    )

    return finish(args, cfg, results, data_yaml)


if __name__ == "__main__":
    raise SystemExit(main())
