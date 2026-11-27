#!/usr/bin/env bash
# yolo26m 1280 batch 5 — n1280 과 같은 분할(사본에서 미리 돌린 것)·같은 인자, 출발은 COCO yolo26m.pt. 새 GPU(1번). 운영 가중치는 그대로.
# 끝나면 n1280 과 같은 세 세트(전체·직접 모은 것·AI Hub)로 test 를 잰다.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1 YOLO_AUTOINSTALL=false
SRC=/mnt/c/Users/chisu/AppData/Local/Temp/claude/3b97b6605d596f29/2e54f18b-e378-4fcb-8689-43102bf14dc5/scratchpad/probe/ds_n1280
D="$HOME/datasets/n1280"
bash /mnt/c/local_auto_labeling/wsl/sync_dataset.sh "$SRC" "$D" | tail -2
HYP="epochs=100 batch=5 imgsz=1280 patience=120 seed=0 pretrained=true amp=true cos_lr=true box=7.5 cls=1.5 dfl=1.5 close_mosaic=10 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 fliplr=0.5 flipud=0.0 save_period=20"
ARGS=(--project "$HOME/new_env/yolo/runs_cmp" --name m1280_b5 --data "$D/data.yaml"
      --model /mnt/c/local_auto_labeling/weights/yolo26m.pt --workers 8 --cache ram --bare --hyp $HYP)
PY="$HOME/miniconda3/envs/yolo_final/bin/python"
"$PY" /mnt/c/local_auto_labeling/wsl/train_wsl.py "${ARGS[@]}" --dry-run
[ "${1:-}" = "--go" ] || exit 0
"$PY" /mnt/c/local_auto_labeling/wsl/train_wsl.py "${ARGS[@]}"
R="$HOME/new_env/yolo/runs_cmp/m1280_b5/weights"
cp "$R/best.pt" /tmp/m1280_best.pt; cp "$R/last.pt" /tmp/m1280_last.pt
"$PY" /mnt/c/local_auto_labeling/wsl/eval_test.py --weights /tmp/m1280_best.pt /tmp/m1280_last.pt \
  --data all="$D/data.yaml" own="$D/data_test_own.yaml" aihub="$D/data_test_aihub.yaml" --imgsz 1280 \
  --out /mnt/c/local_auto_labeling/runs/aihub_split_1127/eval_n1280.jsonl --note "m1280_b5 첫 분할 사본(고정 푼 AI Hub)" 2>&1 | grep "^RESULT"
