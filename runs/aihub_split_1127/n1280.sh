#!/usr/bin/env bash
# yolo26n 1280 batch 20 — 새 GPU(1번), 고정을 푼 첫 분할(사본에서 미리 돌린 것)의 학습 세트. 운영 가중치는 건드리지 않는다.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1 YOLO_AUTOINSTALL=false
SRC=/mnt/c/Users/chisu/AppData/Local/Temp/claude/3b97b6605d596f29/2e54f18b-e378-4fcb-8689-43102bf14dc5/scratchpad/probe/ds_n1280
bash /mnt/c/local_auto_labeling/wsl/sync_dataset.sh "$SRC" "$HOME/datasets/n1280" | tail -4
HYP="epochs=100 batch=20 imgsz=1280 patience=120 seed=0 pretrained=true amp=true cos_lr=true box=7.5 cls=1.5 dfl=1.5 close_mosaic=10 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 fliplr=0.5 flipud=0.0 save_period=20"
ARGS=(--project "$HOME/new_env/yolo/runs_cmp" --name n1280_b20 --data "$HOME/datasets/n1280/data.yaml"
      --model /mnt/c/local_auto_labeling/weights/yolo26n.pt --workers 8 --cache ram --bare --hyp $HYP)
PY="$HOME/miniconda3/envs/yolo_final/bin/python"
"$PY" /mnt/c/local_auto_labeling/wsl/train_wsl.py "${ARGS[@]}" --dry-run
[ "${1:-}" = "--go" ] && "$PY" /mnt/c/local_auto_labeling/wsl/train_wsl.py "${ARGS[@]}"
