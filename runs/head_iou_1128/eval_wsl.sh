#!/usr/bin/env bash
# 2026-11-28 — YOLO26 의 두 추론 헤드(one2many+NMS = Ultralytics 기본, one2one = nms=False)와 IoU 문턱별 AP 를 같은 test 로.
# 세트: 첫 분할 사본의 test(전체·직접 모은 것·AI Hub — runs/aihub_split_1127) · newtest8(직접 모은 119장, 2560×1440).
# 가중치: m1280 best.pt(COCO 에서 201장, 10클래스) · 운영 m(models/best.pt, 8클래스). 새 GPU(1번). 결과는 eval.jsonl 에 덧붙는다.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1 YOLO_AUTOINSTALL=false
HERE=/mnt/c/local_auto_labeling
PY="$HOME/miniconda3/envs/yolo_final/bin/python"
OUT="$HERE/runs/head_iou_1128/eval.jsonl"
D="$HOME/datasets/n1280"
N8="$HOME/datasets/lal_newtest8/dataset/data.yaml"
cp "$HOME/new_env/yolo/runs_cmp/m1280_b5/weights/best.pt" /tmp/m1280_best.pt
cp "$HERE/models/best.pt" /tmp/prod_m8.pt
SPLIT=(all="$D/data.yaml" own="$D/data_test_own.yaml" aihub="$D/data_test_aihub.yaml")
"$PY" "$HERE/wsl/eval_test.py" --weights /tmp/m1280_best.pt --data "${SPLIT[@]}" --imgsz 1280 --heads nms one2one \
  --out "$OUT" --note "m1280_b5 — 두 헤드" 2>&1 | grep -E "^RESULT|알림|건너뜀|Error|error"
"$PY" "$HERE/wsl/eval_test.py" --weights /tmp/prod_m8.pt --data "${SPLIT[@]}" newtest8="$N8" --imgsz 1280 --heads nms one2one \
  --out "$OUT" --note "운영 m 8클래스 — 두 헤드" 2>&1 | grep -E "^RESULT|알림|건너뜀|Error|error"
