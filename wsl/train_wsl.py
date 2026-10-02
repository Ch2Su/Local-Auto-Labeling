"""WSL(Linux)에서 돌리는 학습 스크립트. Windows 프로젝트와 독립적이다.

    # 1) Windows 에서 만든 데이터셋을 WSL 홈으로 옮긴다 (한 번, 또는 데이터가 바뀔 때마다)
    bash /mnt/e/local_auto_labeling/wsl/sync_dataset.sh

    # 2) 학습
    ~/miniconda3/envs/yolo_final/bin/python /mnt/e/local_auto_labeling/wsl/train_wsl.py

    # 3) 멈추기 — 아무 창에서나 (다음 배치에서 멈추고 resume.pt 를 남긴다)
    touch ~/new_env/yolo/runs_wsl/STOP
    #    Windows 탐색기라면 \\\\wsl.localhost\\Ubuntu\\home\\chisu\\new_env\\yolo\\runs_wsl 에
    #    STOP 이라는 빈 파일을 만들면 된다.

    # 4) 이어하기
    ~/miniconda3/envs/yolo_final/bin/python /mnt/e/local_auto_labeling/wsl/train_wsl.py --resume

    # 인자 하나만 바꿔 보기 / 실제로 무엇이 넘어가는지만 확인
    ... train_wsl.py --hyp flipud=0.5 mosaic=0.8
    ... train_wsl.py --dry-run

## 인자는 어디서 오나

- **웹 UI 'WSL 학습' 버튼** — `--bare --hyp ...` 로 부른다. `--hyp` 는 Windows 학습
  버튼과 **같은 인자**(`trainer.train_args()`: config.yaml 의 train.* 와 train.overrides)
  이고, `--bare` 는 아래 DEFAULTS 를 쓰지 말라는 뜻이다. 그래서 config 를 고치면 두
  버튼이 똑같이 바뀐다. config 에 없는 키는 양쪽 다 Ultralytics 기본값이다
  (예: optimizer=auto — 지금 데이터에서는 MuSGD lr 0.01 을 고른다).
- **직접 실행** — 아래 DEFAULTS 를 쓴다. `--hyp` 와 `--epochs` 등이 그 위를 덮는다.

WSL 쪽이 따로 정하는 것은 data·project·name(WSL 경로)과 workers·cache 뿐이다.
WSL 은 Linux 라 **`cache: ram` 을 쓸 수 있다.** Windows 는 spawn 이라 RAM 캐시를
워커마다 통째로 피클하다 죽지만(CLAUDE.md 함정 2), Linux 는 fork 라 공짜다. 원래
쓰던 스크립트의 `cache=True` 는 이 점에서 옳았다. 워커도 fork 라 싸므로 늘린다.

DEFAULTS 는 Windows 에서 실측으로 정한 값이다.

- **`dfl=1.5`** — 원래 스크립트(`train_yolo_v3_1.py`)는 0.0 이었다. 이름과 달리 이 값은
  yolo26 에서 **L1 박스 거리 손실의 게인**이라 0 이면 좌표 정밀도 감독이 꺼진다.
  WSL 의 8.4.50 소스에서도 확인했다 (`utils/loss.py`: reg_max==1 이면 l1_loss 가
  loss[2] 가 되고, 이어서 `loss[2] *= self.hyp.dfl`). 8.4.50 의 default.yaml 주석은
  여전히 "distribution focal loss gain" 이라고만 적혀 있어 오해하기 쉽다.
- **yolo26s** — 같은 조건 A/B 에서 n 보다 mAP50-95 +0.079 (60 epoch).
- **patience 120 / close_mosaic 30** — 75/15 는 close_mosaic(285)에 한 번도 못 닿았다.
- **optimizer 를 MuSGD 로 못박는다** — `auto` 는 iterations 가 10,000 이하면 AdamW 로
  바꾼다. epoch 이나 batch 를 줄이면 조용히 다른 체제가 된다. 단 auto 는 고르면서
  `warmup_bias_lr` 도 0 으로 바꾸므로(engine/trainer.py `build_optimizer`), 이름으로
  못박으면 그 값이 기본 0.1 로 남는다. 버튼 학습(auto)과 여기서만 다르다.

## 중단이 안전한 이유

학습이 끝나면 Ultralytics 의 `final_eval()` 이 `strip_optimizer()` 로 last.pt 에서
옵티마이저 상태를 지운다(8.4.50 에서도 동일). 그러면 이어 학습이 불가능해진다.
정지 파일로 멈추면 그 직전(`on_fit_epoch_end`, save_model() 바로 뒤)에 last.pt 를
resume.pt 로 복사해 둔다. 이게 유일한 기회다.

Ctrl+C 는 즉시 종료다. DataLoader 워커가 신호를 같이 받아 먼저 죽기 때문에
배치 경계까지 갈 수 없다. 그래도 final_eval 이 돌지 않아 last.pt 는 온전하므로
`--resume` 으로 이어갈 수 있다(진행 중이던 epoch 만 잃는다).
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

HOME = Path.home()
DEFAULT_DATA = HOME / "datasets" / "visible" / "data.yaml"
DEFAULT_PROJECT = HOME / "new_env" / "yolo" / "runs_wsl"
# 이 파일은 /mnt/e/local_auto_labeling/wsl/ 에 있다. 사전학습 가중치는 저장소의 weights/
# 것을 쓴다 — 이름만 주면 Ultralytics 가 실행한 폴더에서 찾다가 없으면 새로 받는다.
REPO = Path(__file__).resolve().parents[1]
_W = REPO / "weights" / "yolo26s.pt"
DEFAULT_MODEL = str(_W) if _W.exists() else "yolo26s.pt"

# 직접 실행할 때의 Ultralytics train() 인자. 웹 UI 버튼은 --bare 로 이걸 끄고 config.yaml
# 로 만든 인자를 --hyp 로 넘긴다 (모듈 설명 참고).
DEFAULTS = dict(
    epochs=300, imgsz=640, batch=16, patience=120, close_mosaic=30,
    seed=0, pretrained=True,
    optimizer="MuSGD", lr0=0.01, lrf=0.01, momentum=0.9,
    cos_lr=True, box=7.5, cls=1.5,
    dfl=1.5,                 # 0 금지 — yolo26 에서는 L1 박스 거리 손실의 게인
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, fliplr=0.5, flipud=0.0,
    amp=True, save_period=20,
    # `time` 은 넣지 말 것 — Ultralytics 에서 '최대 학습 시간(시간 단위)' 이고 epochs 를
    # 덮어쓴다. 여기 time=0.01 이 남아 있어서, 직접 실행하면 0.01시간(36초)이 지난 배치에서
    # 멈추게 돼 있었다(2026-09-23 코드 검토로 발견 — 8.4.142 engine/trainer.py 의
    # `args.time` 확인. 웹 UI 버튼과 비교 학습은 --bare 라 이 값을 쓰지 않았다).
)
# 이 스크립트가 정하는 것. --hyp 로 줘도 무시한다.
FIXED = ("data", "device", "workers", "cache", "project", "name", "exist_ok", "plots", "resume")


def parse_args():
    p = argparse.ArgumentParser(description="WSL 학습 (정지 파일로 정상 중단 가능)")
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--name", default="visible", help="runs_wsl/<name> 에 남는다")
    p.add_argument("--project", default=str(DEFAULT_PROJECT))
    p.add_argument("--epochs", type=int, default=None, help=f"기본 {DEFAULTS['epochs']}")
    p.add_argument("--imgsz", type=int, default=None, help=f"기본 {DEFAULTS['imgsz']}")
    p.add_argument("--batch", type=int, default=None,
                   help="기본 16. yolo26s@640 은 8GB 에서 16~20 이 상한이다 "
                        "(24 부터 공유 메모리 페이징이 시작돼 몇 배 느려진다)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--cache", default="ram", choices=["ram", "disk", "false"])
    p.add_argument("--patience", type=int, default=None, help=f"기본 {DEFAULTS['patience']}")
    p.add_argument("--close-mosaic", type=int, default=None,
                   help=f"기본 {DEFAULTS['close_mosaic']}")
    p.add_argument("--hyp", nargs="*", default=[], metavar="KEY=VALUE",
                   help="Ultralytics train() 인자를 더 준다 (예: --hyp flipud=0.5 mosaic=0.8). "
                        "DEFAULTS 보다 우선이고 --epochs 등보다는 뒤다")
    p.add_argument("--bare", action="store_true",
                   help="DEFAULTS 를 쓰지 않는다 — Ultralytics 기본값 + --hyp 만. 웹 UI 버튼이 "
                        "Windows 학습과 같은 인자로 돌리려고 쓴다")
    p.add_argument("--dry-run", action="store_true",
                   help="최종 인자를 Ultralytics 로 검사해 보여주고 끝낸다 (학습하지 않음)")
    p.add_argument("--resume", action="store_true", help="runs_wsl/<name> 에서 이어한다")
    p.add_argument("--stop-at", type=int, default=None, metavar="N",
                   help="N epoch 을 마치면 정지 파일처럼 멈춘다(resume.pt 를 남긴다). epochs 는 "
                        "그대로라 lr 스케줄이 본 학습과 같고, 나중에 --resume 으로 이어갈 수 있다")
    return p.parse_args()


def parse_hyp(items: list[str]) -> dict:
    """`KEY=VALUE` 조각을 dict 로. 값은 JSON 으로 읽고(숫자·true·null), 안 되면 문자열."""
    out = {}
    for it in items:
        k, sep, v = it.partition("=")
        if not sep or not k:
            raise SystemExit(f"--hyp 는 KEY=VALUE 형식이어야 합니다: {it!r}")
        try:
            out[k] = json.loads(v)
        except ValueError:
            out[k] = v
    return out


def build_kwargs(args, data: Path, project: Path) -> tuple[dict, list[str]]:
    """model.train() 에 넘길 인자와, --hyp 로 줬지만 무시한 키를 돌려준다."""
    kw = {} if args.bare else dict(DEFAULTS)
    kw.update(parse_hyp(args.hyp))
    for key in ("epochs", "imgsz", "batch", "patience", "close_mosaic"):
        v = getattr(args, key)
        if v is not None:
            kw[key] = v
    ignored = sorted(k for k in kw if k in FIXED)
    kw.update(
        data=str(data),
        device=0,
        workers=args.workers,
        cache=False if args.cache == "false" else args.cache,
        project=str(project),
        name=args.name,
        exist_ok=False,          # 같은 이름이 있으면 name2 로. 덮어쓰지 않는다
        plots=True,
    )
    return kw, ignored


# --------------------------------------------------------------------- 중단

def attach(model, stop_file: Path, state: dict, stop_at: int | None = None) -> None:
    """정지 파일을 보고 멈추게 하고, 멈출 때 resume.pt 스냅샷을 남긴다.

    stop_at 을 주면 그 epoch 을 마친 뒤 같은 방식으로 멈춘다. 설정끼리 짧게 비교할 때
    쓴다 — epochs=60 으로 주면 cos_lr 이 60 epoch 만에 바닥까지 내려가 본 학습의 1~60
    구간과 다른 학습이 된다(tools/probe_scale.py 와 같은 이유).
    """
    if stop_at:
        def _stop_at(trainer) -> None:
            # on_fit_epoch_end 는 검증·저장 뒤, `if self.stop: break` 바로 앞에서 불린다
            # (8.4.142 engine/trainer.py). 아래 _snapshot 보다 먼저 등록해야 스냅샷이 남는다.
            if not trainer.stop and trainer.epoch + 1 >= stop_at:
                trainer.stop = True
                state["stopped"] = True
                print(f"\n[중단] --stop-at {stop_at}: epoch {trainer.epoch + 1} 을 마쳐 멈춥니다.",
                      flush=True)

        model.add_callback("on_fit_epoch_end", _stop_at)

    def _check(trainer) -> None:
        if not trainer.stop and stop_file.exists():
            trainer.stop = True
            state["stopped"] = True
            print(f"\n[중단] {stop_file} 가 생겼습니다. 다음 배치에서 멈추고 "
                  "검증·저장합니다.", flush=True)

    # 배치마다 봐야 반응이 빠르다. 진행 중이던 epoch 의 나머지는 건너뛰고 그 epoch 을
    # 완료로 기록한다(검증·저장은 한다). 300 epoch 에서 한 epoch 일부는 무시할 만하다.
    model.add_callback("on_train_batch_end", _check)
    model.add_callback("on_train_epoch_end", _check)

    def _snapshot(trainer) -> None:
        # save_model() 직후이자 final_eval() 이전 — last.pt 가 온전한 마지막 순간.
        # **사람이 멈춘 때만** 남긴다. trainer.stop 은 조기 종료(patience)에도 서므로
        # 그걸 보면 끝난 학습이 '이어하기' 대상으로 뜬다.
        if state["snapped"] or not state["stopped"]:
            return
        if trainer.epoch + 1 >= trainer.epochs:
            return  # 끝까지 간 학습은 이어 붙일 것이 없다
        last = Path(trainer.last)
        if last.exists():
            dst = last.with_name("resume.pt")
            shutil.copy2(last, dst)
            state["snapped"] = True
            state["snapshot"] = dst
            print(f"[중단] 이어 학습용 스냅샷: {dst}", flush=True)

    model.add_callback("on_fit_epoch_end", _snapshot)


# ------------------------------------------------------------------ 이어하기

def ckpt_meta(path: Path) -> tuple[int, int] | None:
    """이어 학습 가능한 체크포인트면 (완료 epoch 0-기반, 총 epoch). 아니면 None."""
    import torch

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if not isinstance(ckpt, dict):
        return None
    epoch = int(ckpt.get("epoch", -1))
    if epoch < 0 or ckpt.get("optimizer") is None:
        return None  # final_eval 이 지나간 파일 (strip 됨)
    total = int((ckpt.get("train_args") or {}).get("epochs", 0))
    if total and epoch + 1 >= total:
        return None  # 이미 끝까지 간 학습
    return epoch, total


def find_resume(run_dir: Path) -> tuple[Path, int, int] | None:
    """resume.pt(정지 파일로 멈춘 것) > last.pt(강제 종료로 살아남은 것) 순."""
    best = None
    for name in ("resume.pt", "last.pt"):
        p = run_dir / "weights" / name
        if not p.exists():
            continue
        m = ckpt_meta(p)
        if m and (best is None or m[0] > best[1]):
            best = (p, m[0], m[1])
    return best


# ---------------------------------------------------------------------- 메인

def main() -> int:
    args = parse_args()
    project = Path(args.project).expanduser()
    stop_file = project / "STOP"

    # 인자부터 만든다. --dry-run 은 여기서 끝나야 한다 — 아래에서 STOP 을 지우므로,
    # 돌고 있는 학습을 멈추려고 만든 STOP 을 확인용 실행이 치워 버리면 안 된다.
    if not args.resume:
        data = Path(args.data).expanduser()
        if not data.exists():
            print(f"데이터셋이 없습니다: {data}\n"
                  "  먼저 bash /mnt/e/local_auto_labeling/wsl/sync_dataset.sh")
            return 1
        kw, ignored = build_kwargs(args, data, project)
        if args.dry_run:
            from ultralytics.cfg import get_cfg

            get_cfg(overrides=kw)    # 이름·타입이 틀리면 여기서 에러가 난다
            print("KWARGS=" + json.dumps(kw, ensure_ascii=False, sort_keys=True))
            return 0
    elif args.dry_run:
        print("--dry-run 은 새 학습에만 쓸 수 있습니다 (이어하기는 체크포인트의 인자를 쓴다).")
        return 2

    project.mkdir(parents=True, exist_ok=True)
    if stop_file.exists():
        stop_file.unlink()  # 지난 학습의 잔재. 두면 시작하자마자 멈춘다.

    from ultralytics import YOLO

    state = {"stopped": False, "snapped": False, "snapshot": None}
    t0 = time.time()

    if args.resume:
        run_dir = project / args.name
        found = find_resume(run_dir)
        if found is None:
            print(f"이어 학습할 체크포인트가 없습니다: {run_dir}/weights\n"
                  "  정상 종료된 학습의 last.pt 는 옵티마이저가 지워져 이어갈 수 없습니다.")
            return 1
        ckpt, epoch, total = found
        print(f"이어 학습: {ckpt}\n  {epoch+1}/{total} epoch 완료 → {epoch+2} 부터")
        model = YOLO(str(ckpt))
        attach(model, stop_file, state, args.stop_at)
        # 이어 학습에서 바꿀 수 있는 것은 check_resume() 허용 목록뿐이다
        # (8.4.50: imgsz batch device close_mosaic save_period workers cache patience ...).
        kw = dict(resume=str(ckpt), device=0, workers=args.workers)
        if args.batch is not None:
            kw["batch"] = args.batch
        results = model.train(**kw)
    else:
        opt = kw.get("optimizer", "auto")
        shown = {"data", "project", "name", "workers", "cache", "device", "exist_ok", "plots",
                 "epochs", "imgsz", "batch", "patience", "close_mosaic", "optimizer", "lr0",
                 "momentum"}
        rest = " ".join(f"{k}={v}" for k, v in kw.items() if k not in shown)
        g = {k: kw.get(k, "기본값") for k in ("imgsz", "batch", "epochs", "patience",
                                              "close_mosaic", "lr0", "momentum")}
        if opt == "auto":
            opt_note = "(iterations 로 MuSGD/AdamW 를 고른다)"
        else:
            opt_note = f"lr0={g['lr0']}  momentum={g['momentum']}"
        if args.bare:
            source = "config.yaml (웹 UI 버튼 — Windows 학습과 같은 인자)"
        else:
            source = "이 스크립트 기본값" + (" + --hyp" if args.hyp else "")
        print(f"""
모델        : {args.model}
데이터      : {data}
해상도/배치 : {g['imgsz']} / {g['batch']}
epoch       : {g['epochs']}  (patience {g['patience']}, close_mosaic {g['close_mosaic']})
옵티마이저  : {opt}  {opt_note}
캐시/워커   : {args.cache} / {args.workers}
그 밖       : {rest}
인자 출처   : {source}
결과        : {project / args.name}
""", flush=True)
        if ignored:
            print(f"[알림] --hyp 의 {ignored} 는 이 스크립트가 정하는 값이라 무시했습니다.")
        if float(kw.get("dfl", 1.5)) == 0:
            print("[경고] dfl=0 이면 박스 거리 손실(yolo26 에서는 L1)이 꺼집니다. "
                  "검출은 되지만 좌표 정밀도(mAP50-95)가 떨어집니다. 1.5 로 두세요.")
        model = YOLO(args.model)
        attach(model, stop_file, state, args.stop_at)
        results = model.train(**kw)

    # 멈추는 데 쓴 STOP 은 지운다. 남겨 두면 '아직 멈추는 중인가' 헷갈린다.
    # (다음 실행도 시작할 때 지우지만, 끝난 상태를 깔끔하게 두는 편이 낫다)
    if stop_file.exists():
        stop_file.unlink()

    run_dir = Path(results.save_dir)
    best = run_dir / "weights" / "best.pt"
    hours = (time.time() - t0) / 3600

    # 사람이 멈춘 게 아니면 남은 resume.pt 를 치운다. 중단 → 이어하기 → 완주 순서면
    # 첫 중단의 스냅샷이 남아, 끝난 학습이 '이어하기' 대상으로 계속 뜬다.
    stale = run_dir / "weights" / "resume.pt"
    if not state["stopped"] and stale.exists():
        stale.unlink()
        print(f"완주했으므로 이전 중단 때의 스냅샷을 지웠습니다: {stale}")
    print("\n" + "=" * 64)
    print(("중단됨 — 이어하려면 --resume" if state["stopped"] else "학습 완료")
          + f"   ({hours:.1f}시간)")
    print(f"결과 폴더 : {run_dir}")
    if state["snapshot"]:
        print(f"스냅샷    : {state['snapshot']}")
    if best.exists():
        print(f"best.pt   : {best}")
        print("\nWindows 로 가져오려면 (운영 가중치를 바로 덮지 않도록 다른 이름으로):")
        print(f"  cp {best} /mnt/e/local_auto_labeling/models/best_wsl_{args.name}.pt")
    print("=" * 64)
    # 웹 UI 'WSL 학습' 버튼이 읽는 줄. Ultralytics 는 같은 이름이 있으면 name2, name3
    # 으로 바꾸므로 결과 폴더를 호출한 쪽이 추측할 수 없다. 형식을 바꾸지 말 것.
    print(f"RESULT_DIR={run_dir}")
    print(f"STOPPED={1 if state['stopped'] else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
