#!/bin/bash
# Runpod pod 에서: 묶음을 풀고 A40 VRAM·속도를 잰다 (2026-09-18).
# /workspace 는 네트워크 파일시스템이라 데이터셋은 로컬 컨테이너 디스크(/root)에 푼다.
# 컨테이너 디스크는 pod 를 멈추면 지워지므로, 다시 켜면 이 스크립트를 다시 돌리면 된다.
set -eu
export YOLO_CONFIG_DIR=/tmp/Ultralytics

mkdir -p /workspace/lal /root/datasets
tar -xf /workspace/code.tar -C /workspace                 # -> /workspace/lal/{wsl,weights}
if [ ! -f /root/datasets/visible/data.yaml ]; then
  tar -xf /workspace/dataset.tar -C /root/datasets        # -> /root/datasets/dataset
  mv /root/datasets/dataset /root/datasets/visible
fi
# data.yaml 의 path 를 pod 경로로 (Windows 에서 만든 파일이라 Windows 경로가 들어 있다)
sed -i 's#^path:.*#path: /root/datasets/visible#' /root/datasets/visible/data.yaml
head -3 /root/datasets/visible/data.yaml
for s in train val test; do echo "$s: $(ls /root/datasets/visible/images/$s | wc -l)"; done

python3 -c "import ultralytics" 2>/dev/null || pip install --break-system-packages -q ultralytics==8.4.142

mkdir -p /workspace/lal/runs/vram_probe
cd /workspace/lal
# 48GB 라 batch 16 부터. 넘치면 probe_vram 이 batch 4 로 한 번 더 재고 더 높은 해상도는 건너뛴다.
setsid nohup python3 wsl/probe_vram.py \
  --data /root/datasets/visible/data.yaml \
  --models s m l x --imgsz 640 960 1280 --batches 16 32 --steps 30 --workers 8 \
  --out /workspace/lal/runs/vram_probe/results_a40.jsonl \
  > /workspace/lal/runs/vram_probe/probe_a40.log 2>&1 < /dev/null &
echo "측정 시작 pid $!"
