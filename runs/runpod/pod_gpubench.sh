#!/bin/bash
# GPU 속도 실측 (2026-09-20). pod 마다 같은 작은 묶음으로 같은 설정 3개를 30 스텝씩 돌린다.
#   scp probe_subset.tar pod_gpubench.sh root@<ip>:/root/  &&  bash /root/pod_gpubench.sh
# A40 실측(전체 데이터셋): x960 b16 885 / l960 b24 790 / s640 b20 173 ms/step.
set -eu
export YOLO_CONFIG_DIR=/tmp/Ultralytics
mkdir -p /root/bench /root/datasets
tar -xf /root/probe_subset.tar -C /root/bench            # probe/ + probe_vram.py + weights/
[ -d /root/datasets/probe ] || mv /root/bench/probe /root/datasets/probe
python3 -c "import ultralytics" 2>/dev/null || pip install --break-system-packages -q ultralytics==8.4.142
cd /root/bench

NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
CPUS=$(nproc)
W=$(( CPUS < 8 ? CPUS : 8 ))
echo "GPU $NAME / vCPU $CPUS / workers $W / $(python3 -c 'import torch;print(torch.__version__)')"

for cfg in "x 960 16" "l 960 24" "s 640 20"; do
  set -- $cfg
  python3 probe_vram.py --one "$1" "$2" "$3" --steps 30 --workers "$W" \
      --data /root/datasets/probe/data.yaml > "/root/bench/$1_$2_$3.log" 2>&1 || true
  if grep -q '^RESULT ' "/root/bench/$1_$2_$3.log"; then
    grep '^RESULT ' "/root/bench/$1_$2_$3.log" | sed "s|^RESULT |BENCH $NAME |"
  else
    echo "BENCH $NAME 실패 $1 $2 b$3 — $(tail -2 "/root/bench/$1_$2_$3.log" | tr '\n' ' ' | cut -c1-200)"
  fi
done
echo BENCHDONE
