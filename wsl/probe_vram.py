"""모델 크기 × 해상도 × batch 별 학습 VRAM 을 WSL 에서 잰다.

'8GB 카드에서 어디까지 무겁게 돌릴 수 있나'를 추측 대신 실측으로 정하려고 만들었다.
학습은 WSL 에서 하므로(CLAUDE.md, Windows 보다 약 4배 빠름) 여기서 잰다.

    ~/miniconda3/envs/yolo_final/bin/python /mnt/e/local_auto_labeling/wsl/probe_vram.py
    ... probe_vram.py --models s m l x --imgsz 640 960 1280        # 기본 격자
    ... probe_vram.py --models s-p2 m-p2 --imgsz 640 960           # 작은 객체용 P2 헤드

**사전학습 가중치를 받지 않는다.** `yolo26m.yaml` 처럼 구조만으로 모델을 만든다 — VRAM 은
구조·해상도·batch 로 정해지고 가중치 값과는 무관하다. 실제로 학습할 때만 `yolo26m.pt` 등이
필요하다. AMP 확인(check_amp)이 `yolo26n.pt` 를 찾으므로 프로젝트 `weights/` 의 것을 작업
폴더로 복사해 둔다(없으면 Ultralytics 가 인터넷에서 받으려 한다).

**설정마다 별도 프로세스로** 실제 학습을 `--steps` 배치만큼 돌리고 멈춘다. CUDA 캐시가
앞 설정에서 이어지지 않게 하려는 것이다. 본 학습과 같게 AMP·MuSGD 를 쓰고, RAM 캐시는
끈다(설정마다 5GB 를 채우면 느리다 — VRAM 과는 무관).

**기록:** torch 가 잡은 최대 VRAM(reserved), 실제 텐서 최대(allocated), 그 시점 GPU 전체
사용량(다른 프로세스 포함), 마지막 10 스텝 평균 ms/step. 8GB 를 넘기면 WSL 도 Windows 처럼
OOM 대신 공유 메모리로 넘길 수 있어서(함정 9), ms/step 이 튀는지도 같이 본다.

**건너뛰기:** batch 8 이 실패(OOM·한도 초과)하면 그 해상도는 batch 4 만 더 재고, 그 모델의
더 높은 해상도는 재지 않는다(더 무거울 뿐이다).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
PROJECT = HERE.parents[1]                      # /mnt/e/local_auto_labeling
WORKDIR = Path("/tmp/vram_probe")
LIMIT_FRAC = 0.94                              # 이보다 많이 쓰면 '한도 초과' 로 본다


def parse_args():
    p = argparse.ArgumentParser(description="모델·해상도·batch 별 학습 VRAM 을 잰다.")
    p.add_argument("--models", nargs="+", default=["s", "m", "l", "x"],
                   help="yolo26 뒤에 붙는 이름: n s m l x, s-p2 등")
    p.add_argument("--imgsz", nargs="+", type=int, default=[640, 960, 1280])
    p.add_argument("--batches", nargs="+", type=int, default=[8, 16], help="작은 것부터")
    p.add_argument("--steps", type=int, default=30, help="설정마다 돌릴 배치 수")
    p.add_argument("--data", default=str(Path.home() / "datasets/visible/data.yaml"))
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out", default=str(PROJECT / "runs/vram_probe/results.jsonl"))
    p.add_argument("--one", nargs=3, metavar=("MODEL", "IMGSZ", "BATCH"), help=argparse.SUPPRESS)
    return p.parse_args()


def probe_one(model: str, imgsz: int, batch: int, args) -> dict:
    """자식 프로세스: 한 설정을 steps 배치만큼 학습시키고 기록을 돌려준다."""
    import torch
    from ultralytics import YOLO

    class Done(Exception):
        pass

    rec = {"model": model, "imgsz": imgsz, "batch": batch}
    stamps: list[float] = []

    def on_batch_end(trainer):
        torch.cuda.synchronize()
        stamps.append(time.perf_counter())
        if len(stamps) >= args.steps:
            free, total = torch.cuda.mem_get_info()
            n = min(10, len(stamps) - 1)
            rec.update(
                reserved=round(torch.cuda.max_memory_reserved() / 2**30, 2),
                allocated=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                device_used=round((total - free) / 2**30, 2),
                total=round(total / 2**30, 2),
                ms_per_step=round(1000 * (stamps[-1] - stamps[-1 - n]) / n),
            )
            raise Done

    m = YOLO(f"yolo26{model}.yaml")
    rec["params_m"] = round(sum(p.numel() for p in m.model.parameters()) / 1e6, 1)
    m.add_callback("on_train_batch_end", on_batch_end)
    try:
        m.train(data=args.data, imgsz=imgsz, batch=batch, epochs=300, optimizer="MuSGD",
                lr0=0.01, momentum=0.9, workers=args.workers, cache=False, amp=True,
                val=False, plots=False, pretrained=False, project=str(WORKDIR / "runs"),
                name="p", exist_ok=True, verbose=False)
        rec["status"] = "ended"
    except Done:
        # torch 가 잡은 양(reserved)에는 CUDA 컨텍스트(약 1GiB)가 빠져 있어 GPU 전체로 본다.
        over = rec["device_used"] > LIMIT_FRAC * rec["total"]
        rec["status"] = "over" if over else "ok"
    except torch.OutOfMemoryError:
        rec["status"] = "oom"
    return rec


def run_child(model: str, imgsz: int, batch: int, args) -> dict:
    cmd = [sys.executable, str(HERE), "--one", model, str(imgsz), str(batch),
           "--steps", str(args.steps), "--data", args.data, "--workers", str(args.workers)]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=WORKDIR, capture_output=True, text=True)
    for line in (proc.stdout + proc.stderr).splitlines():
        if "Download" in line:                     # 가중치·폰트를 인터넷에서 받으면 알린다
            print(f"  [다운로드 감지] {line.strip()[:160]}", flush=True)
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            rec = json.loads(line[7:])
            break
    else:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
        oom = any("out of memory" in l.lower() for l in (proc.stderr or "").splitlines())
        rec = {"model": model, "imgsz": imgsz, "batch": batch,
               "status": "oom" if oom else "error", "error": " | ".join(tail)[-300:]}
    rec["wall_s"] = round(time.time() - t0)
    return rec


def fmt(r: dict) -> str:
    if r["status"] in ("ok", "over"):
        return (f"{r['model']:>5} {r['imgsz']:>5} b{r['batch']:<3} {r['reserved']:>6.2f} GiB "
                f"(텐서 {r['allocated']:.2f}, GPU 전체 {r['device_used']:.2f}/{r['total']:.2f}) "
                f"{r['ms_per_step']:>5} ms/step  {'<- 한도 초과' if r['status'] == 'over' else ''}")
    return f"{r['model']:>5} {r['imgsz']:>5} b{r['batch']:<3} {r['status'].upper()}  {r.get('error', '')}"


def main():
    args = parse_args()
    if args.one:
        model, imgsz, batch = args.one[0], int(args.one[1]), int(args.one[2])
        print("RESULT " + json.dumps(probe_one(model, imgsz, batch, args)), flush=True)
        return

    WORKDIR.mkdir(parents=True, exist_ok=True)
    # check_amp 는 작업 폴더의 weights/yolo26n.pt 를 찾고, 없으면 GitHub 에서 받는다
    # (2026-09-18 시험에서 작업 폴더 바로 아래에 둔 것은 못 찾고 받아 갔다).
    amp_ref = PROJECT / "weights/yolo26n.pt"
    (WORKDIR / "weights").mkdir(exist_ok=True)
    if amp_ref.exists() and not (WORKDIR / "weights/yolo26n.pt").exists():
        shutil.copy2(amp_ref, WORKDIR / "weights/yolo26n.pt")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"데이터 {args.data} / {args.steps} 배치씩 / 결과 {out}", flush=True)

    for model in args.models:
        for imgsz in args.imgsz:
            failed_small = False
            for batch in sorted(args.batches):
                r = run_child(model, imgsz, batch, args)
                with open(out, "a", encoding="utf-8") as f:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                print(fmt(r) + f"  [{r['wall_s']}s]", flush=True)
                if r["status"] != "ok":
                    failed_small = batch == min(args.batches)
                    break
            if failed_small:
                if 4 < min(args.batches):
                    r = run_child(model, imgsz, 4, args)
                    with open(out, "a", encoding="utf-8") as f:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    print(fmt(r) + f"  [{r['wall_s']}s]", flush=True)
                print(f"  -> {model} 는 {imgsz} 에서 batch {min(args.batches)} 도 안 돼서 더 높은 해상도는 건너뛴다",
                      flush=True)
                break

    shutil.rmtree(WORKDIR / "runs", ignore_errors=True)
    print("끝", flush=True)


if __name__ == "__main__":
    main()
