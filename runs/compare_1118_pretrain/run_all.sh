#!/usr/bin/env bash
# 두 줄기를 나란히 돌린다(yolo26m 640 batch 10 은 줄기마다 VRAM 약 6GB — 16GB 에 둘까지).
# 앞쪽 세 실행(A.0 B.0 C.0 — 100 epoch)이 첫 비교이고, 그 뒤 fold 1·2 와 긴 일정(300 epoch)이 이어진다.
cd "$(dirname "$0")" || exit 1
PY=/c/local_auto_labeling/.venv/Scripts/python.exe
export YOLO_OFFLINE=true
"$PY" -B run_cmp.py --jobs A.0.100,C.0.100,A.1.100,B.1.100,C.1.100,C.0.300 > stream_a.log 2>&1 &
PA=$!
sleep 60
"$PY" -B run_cmp.py --jobs B.0.100,A.2.100,B.2.100,C.2.100,B.0.300 > stream_b.log 2>&1 &
PB=$!
wait $PA; EA=$?
wait $PB; EB=$?
echo "streams done: a=$EA b=$EB"
date
