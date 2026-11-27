#!/usr/bin/env bash
# n1280 의 test — 전체·직접 모은 것·AI Hub 를 따로. 운영 가중치(m 640, 8클래스)도 같은 세트로(640·1280).
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1 YOLO_AUTOINSTALL=false
S=/mnt/c/Users/chisu/AppData/Local/Temp/claude/3b97b6605d596f29/2e54f18b-e378-4fcb-8689-43102bf14dc5/scratchpad/probe/n1280_sets
D="$HOME/datasets/n1280"
cp "$S"/test_own.txt "$S"/test_aihub.txt "$S"/data_test_own.yaml "$S"/data_test_aihub.yaml "$D"/
PY="$HOME/miniconda3/envs/yolo_final/bin/python"
OUT=/mnt/c/local_auto_labeling/runs/aihub_split_1127/eval_n1280.jsonl
R="$HOME/new_env/yolo/runs_cmp/n1280_b20/weights"
cp "$R/best.pt" /tmp/n1280_best.pt; cp "$R/last.pt" /tmp/n1280_last.pt
SETS=(all="$D/data.yaml" own="$D/data_test_own.yaml" aihub="$D/data_test_aihub.yaml")
"$PY" /mnt/c/local_auto_labeling/wsl/eval_test.py --weights /tmp/n1280_best.pt /tmp/n1280_last.pt --data "${SETS[@]}" --imgsz 1280 --out "$OUT" --note "n1280_b20 첫 분할 사본(고정 푼 AI Hub)" 2>&1 | grep -v "━" | tail -40
"$PY" /mnt/c/local_auto_labeling/wsl/eval_test.py --weights /mnt/c/local_auto_labeling/models/best.pt --data "${SETS[@]}" --imgsz 640 1280 --out "$OUT" --note "운영 m 8클래스, 같은 test" 2>&1 | grep -v "━" | tail -40
