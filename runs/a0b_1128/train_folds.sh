#!/usr/bin/env bash
# A0b — fold 마다 yolo26m 1280 b5 를 학습한다(make_folds.py 가 만든 세트, 레시피는 runs/aihub_split_1127/m1280.sh 와 같고
# save_period 만 뺐다). 평가는 last.pt 로 한다(val = 그 fold — 보기만, best.pt 는 그 fold 로 고른 것이라 쓰지 않는다).
#   bash train_folds.sh <GPU 번호> <fold> [<fold> …]      예) bash train_folds.sh 1 0 2 4
#   MODEL=<출발 가중치> PREFIX=<이름 앞머리> bash train_folds.sh …
#     기본은 COCO weights/yolo26m.pt · a0b. A0b′(같은 날 밤)는 운영 m 에서 출발:
#     MODEL=/mnt/c/local_auto_labeling/models/best_y26m_8cls.pt PREFIX=a0bp (8클래스 → 10클래스 — 분류 마지막 층만 새로 시작한다)
# 결과: ~/new_env/yolo/runs_cmp/<PREFIX>_f<N>/ (운영 가중치는 그대로), 로그 ~/new_env/yolo/runs_cmp/<PREFIX>_f<N>.log
set -euo pipefail
GPU="$1"; shift
export CUDA_VISIBLE_DEVICES="$GPU" YOLO_AUTOINSTALL=false
REPO=/mnt/c/local_auto_labeling
MODEL="${MODEL:-$REPO/weights/yolo26m.pt}"
PREFIX="${PREFIX:-a0b}"
PY="$HOME/miniconda3/envs/yolo_final/bin/python"
RUNS="$HOME/new_env/yolo/runs_cmp"
HYP="epochs=100 batch=5 imgsz=1280 patience=120 seed=0 pretrained=true amp=true cos_lr=true box=7.5 cls=1.5 dfl=1.5 close_mosaic=10 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 fliplr=0.5 flipud=0.0"
[ -f "$MODEL" ] || { echo "출발 가중치가 없습니다: $MODEL"; exit 1; }
mkdir -p "$RUNS"
for f in "$@"; do
  SRC="/mnt/c/datasets/a0b_1128/f$f/dataset"
  D="$HOME/datasets/a0b_f$f"
  bash "$REPO/wsl/sync_dataset.sh" "$SRC" "$D" > /dev/null
  echo "fold $f 시작 (GPU $GPU, $PREFIX, $(basename "$MODEL")) $(date +%H:%M:%S)"
  if "$PY" "$REPO/wsl/train_wsl.py" --project "$RUNS" --name "${PREFIX}_f$f" --data "$D/data.yaml" \
       --model "$MODEL" --workers 8 --cache ram --bare --hyp $HYP > "$RUNS/${PREFIX}_f$f.log" 2>&1; then
    echo "fold $f 끝 $(date +%H:%M:%S) — $(ls "$RUNS/${PREFIX}_f$f/weights/")"
  else
    echo "fold $f 실패 — $RUNS/${PREFIX}_f$f.log"; tail -5 "$RUNS/${PREFIX}_f$f.log"; exit 1
  fi
done
