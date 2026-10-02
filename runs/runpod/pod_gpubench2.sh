#!/bin/bash
# 2차 측정 (2026-09-20). 1차(pod_gpubench.sh)에는 두 가지 문제가 있었다.
#  1) train 480장이라 l960 b24(20 스텝/epoch)·s640 b20(24 스텝/epoch)은 30 스텝을 채우며 epoch
#     경계를 넘었고, 마지막 10 스텝 평균에 로더 재시작이 섞였다. -> 하드링크로 960장으로 늘린다.
#  2) 프로브는 cache=False 라 로더가 병목이면 GPU 속도가 아니라 디코딩 속도를 잰다
#     (A40 이 이전 실측 885 대신 1,275 ms/step). -> 본 학습과 같은 cache=ram 으로 잰다.
set -eu
export YOLO_CONFIG_DIR=/tmp/Ultralytics
D=/root/datasets/probe
cd "$D/images/train"; for f in *.*; do case "$f" in dup_*) continue;; esac; [ -e "dup_$f" ] || ln "$f" "dup_$f"; done
cd "$D/labels/train"; for f in *.txt; do case "$f" in dup_*) continue;; esac; [ -e "dup_$f" ] || ln "$f" "dup_$f"; done
rm -f "$D/labels/train.cache" "$D/labels/val.cache"

cd /root/bench
sed 's/cache=False/cache="ram"/' probe_vram.py > probe_ram.py      # 본 학습과 같은 조건
NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
CLK=$(nvidia-smi --query-gpu=clocks.max.sm,power.limit --format=csv,noheader | tr '\n' ' ')
W=8
echo "GPU2 $NAME / train $(ls "$D/images/train" | wc -l)장 / workers $W / $CLK"

for cfg in "x 960 16" "l 960 24" "s 640 20"; do
  set -- $cfg
  python3 probe_ram.py --one "$1" "$2" "$3" --steps 30 --workers "$W" \
      --data "$D/data.yaml" > "/root/bench/$1_$2_$3_v2.log" 2>&1 || true
  if grep -q '^RESULT ' "/root/bench/$1_$2_$3_v2.log"; then
    grep '^RESULT ' "/root/bench/$1_$2_$3_v2.log" | sed "s|^RESULT |BENCH2 $NAME |"
  else
    echo "BENCH2 $NAME 실패 $1 $2 b$3 — $(tail -2 "/root/bench/$1_$2_$3_v2.log" | tr '\n' ' ' | cut -c1-200)"
  fi
done
echo BENCH2DONE
