#!/bin/bash
# Runpod A40 에서 모델·해상도 비교 (2026-09-18). runs/compare_0918/run_compare.sh 의 pod 판.
# 인자는 웹 UI 'WSL 학습' 버튼과 같다(wsl_train.hyp_tokens). 바꾸는 것은 모델·imgsz·batch 뿐.
# 300 epoch 스케줄로 60 epoch 에서 멈춘다(--stop-at) — 이긴 설정은 --resume 으로 300 까지.
#
#   setsid nohup bash /workspace/pod_compare.sh [이름...] > /workspace/lal/runs/compare_a40/progress.log 2>&1 &
#   touch /workspace/runs_cmp/STOP                          # 지금 것만 멈춤
#   touch /workspace/runs_cmp/HALT /workspace/runs_cmp/STOP # 전부 멈춤 (이 스크립트를 kill 하지 말 것)
set -u
export YOLO_CONFIG_DIR=/tmp/Ultralytics
P=python3
T=/workspace/lal/wsl/train_wsl.py
W=/workspace/lal/weights
PROJ=/workspace/runs_cmp
OUT=/workspace/lal/runs/compare_a40
DATA=/root/datasets/visible/data.yaml
STOP_AT=${STOP_AT:-60}
# ram 이 기본. 1280 은 ram 캐시(train 17GB + val 2.7GB)에 워커 8개가 얹혀 pod 의 RAM 한도 46.6GiB 를
# 넘어 커널에 죽었다(09-18 a40_m1280). 그때는 CACHE=disk — 원본 크기 .npy 를 /root 에 약 17GB 쓴다.
CACHE=${CACHE:-ram}
HYP="epochs=300 patience=120 seed=0 pretrained=true amp=true cos_lr=true box=7.5 cls=1.5 dfl=1.5 close_mosaic=30 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 fliplr=0.5 flipud=0.0 save_period=20"

run() {   # 이름 가중치 imgsz batch
  local name=$1 weights=$2 imgsz=$3 batch=$4
  if [ ! -f "$W/$weights" ]; then
    echo "=== $name 건너뜀 — $W/$weights 없음 $(date '+%F %T')"
    return
  fi
  echo "=== $name 시작 $(date '+%F %T')  ($weights / $imgsz / b$batch / stop-at $STOP_AT / cache $CACHE)"
  "$P" "$T" --project "$PROJ" --name "$name" --data "$DATA" --model "$W/$weights" \
    --workers 8 --cache "$CACHE" --bare --hyp $HYP --imgsz "$imgsz" --batch "$batch" \
    --stop-at "$STOP_AT" > "$OUT/$name.log" 2>&1
  local dir
  dir=$(grep '^RESULT_DIR=' "$OUT/$name.log" | tail -1 | cut -d= -f2)
  if [ -n "$dir" ] && [ -f "$dir/weights/best.pt" ]; then
    cp "$dir/weights/best.pt" "$OUT/${name}_best.pt"
    cp "$dir/results.csv" "$OUT/${name}_results.csv"
    echo "=== $name 끝 $(date '+%F %T')  $dir"
  else
    echo "=== $name 실패 $(date '+%F %T') — $OUT/$name.log 확인"
  fi
}

declare -A CFG=(
  [a40_s640]="yolo26s.pt 640 20"       # 대조: 지금 운영 설정(visible-3)과 같은 모델·해상도·batch
  [a40_l960]="yolo26l.pt 960 24"
  [a40_m1280]="yolo26m.pt 1280 16"
  [a40_x960]="yolo26x.pt 960 16"
)
ORDER=(a40_s640 a40_l960 a40_m1280 a40_x960)
[ $# -gt 0 ] && ORDER=("$@")

mkdir -p "$OUT" "$PROJ"
rm -f "$PROJ/HALT"
for name in "${ORDER[@]}"; do
  if [ -f "$PROJ/HALT" ]; then
    rm -f "$PROJ/HALT"
    echo "=== HALT — $name 부터 시작하지 않고 멈춤 $(date '+%F %T')"
    break
  fi
  run "$name" ${CFG[$name]}
done
echo "ALLDONE $(date '+%F %T')"
