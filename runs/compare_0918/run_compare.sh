#!/bin/bash
# 모델 크기·해상도 비교 학습 (2026-09-18). WSL 에서 파일로 돌린다 — wsl -- bash -c 로 넘기면
# $변수가 바깥에서 먼저 풀린다(CLAUDE.md).
#
# 인자는 웹 UI 'WSL 학습' 버튼과 같다(wsl_train.hyp_tokens). 바꾸는 것은 모델·imgsz·batch 뿐.
# 300 epoch 스케줄로 60 epoch 에서 멈춘다(--stop-at 60) — lr 이 본 학습의 1~60 구간과 같고,
# 이긴 설정은 --resume 으로 300 까지 이어갈 수 있다.
#
# 결과는 runs_wsl 이 아니라 runs_cmp 에 둔다. runs_wsl 에 두면 웹 UI 'WSL 이어하기' 가 이걸
# 집어 가고, 그걸 누르면 끝날 때 models/best.pt 가 교체된다.
#
#   bash /mnt/e/local_auto_labeling/runs/compare_0918/run_compare.sh [이름...]   # 없으면 전부
#   touch ~/new_env/yolo/runs_cmp/STOP                                          # 지금 것만 멈춤
#   touch ~/new_env/yolo/runs_cmp/HALT ~/new_env/yolo/runs_cmp/STOP             # 전부 멈춤
#
# **이 스크립트를 kill 하지 말 것.** 2026-09-18 에 순서 스크립트를 pkill 하자 wsl 세션이
# 닫히며 학습 프로세스도 같이 죽었다(정지 파일을 보기도 전에, resume.pt 없이). last.pt 는
# 온전해서 이어갈 수는 있지만, 멈출 때는 위처럼 HALT+STOP 으로 할 것.
#   이어하기: ... train_wsl.py --project ~/new_env/yolo/runs_cmp --name <이름> --resume --stop-at 60
set -u
P=/home/chisu/miniconda3/envs/yolo_final/bin/python
T=/mnt/e/local_auto_labeling/wsl/train_wsl.py
PROJ=${PROJ:-/home/chisu/new_env/yolo/runs_cmp}
OUT=/mnt/e/local_auto_labeling/runs/compare_0918
STOP_AT=${STOP_AT:-60}
HYP="epochs=300 patience=120 seed=0 pretrained=true amp=true cos_lr=true box=7.5 cls=1.5 dfl=1.5 close_mosaic=30 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 fliplr=0.5 flipud=0.0 save_period=20"

run() {   # 이름 가중치 imgsz batch
  local name=$1 weights=$2 imgsz=$3 batch=$4
  echo "=== $name 시작 $(date '+%F %T')  ($weights / $imgsz / b$batch / stop-at $STOP_AT)"
  "$P" "$T" --project "$PROJ" --name "$name" --data /home/chisu/datasets/visible/data.yaml \
    --model "/mnt/e/local_auto_labeling/weights/$weights" --workers 8 --cache ram --bare \
    --hyp $HYP --imgsz "$imgsz" --batch "$batch" --stop-at "$STOP_AT" > "$OUT/$name.log" 2>&1
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
  [cmp_s640]="yolo26s.pt 640 20"
  [cmp_m640]="yolo26m.pt 640 10"
  [cmp_s960]="yolo26s.pt 960 8"
  [cmp_l640]="yolo26l.pt 640 8"
)
ORDER=(cmp_s640 cmp_m640 cmp_s960 cmp_l640)
[ $# -gt 0 ] && ORDER=("$@")

mkdir -p "$OUT"
for name in "${ORDER[@]}"; do
  if [ -f "$PROJ/HALT" ]; then       # 설정 사이에서만 본다 — 학습 도중에는 STOP 이 할 일
    rm -f "$PROJ/HALT"
    echo "=== HALT — $name 부터 시작하지 않고 멈춤 $(date '+%F %T')"
    break
  fi
  run "$name" ${CFG[$name]}
done
echo "ALLDONE $(date '+%F %T')"
