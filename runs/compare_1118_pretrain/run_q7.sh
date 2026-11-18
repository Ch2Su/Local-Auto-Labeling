#!/usr/bin/env bash
# 얼린 갈래를 25 epoch 일정으로(11-19 사용자 승인 '25 재보기') - 100 epoch 일정에서 검증 mAP 가 25 epoch 뒤로 내려가서.
#   a = FZ11 3묶음 / b = FZ23 3묶음, 두 줄기 동시(3.3GB + 2.2GB). 일정이 총 epoch 에 묶이니(cos_lr·warmup 3·close_mosaic 10)
#   '100 epoch 일정의 25번째 epoch' 과 같은 것이 아니라 25 epoch 일정 그 자체다.
cd "$(dirname "$0")" || exit 1
PY=/c/local_auto_labeling/.venv/Scripts/python.exe
export YOLO_OFFLINE=true
busy() {
  powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { \$_.CommandLine -match 'run_cmp\.py' }).Count" 2>/dev/null | tr -d '\r\n '
}
if [ "$(busy)" != "0" ]; then echo "run_cmp.py 가 이미 돌고 있다 - 시작하지 않는다"; exit 2; fi
mkdir q7.lock 2>/dev/null || { echo "q7.lock 이 있다 - 이미 떠 있거나 죽은 흔적이다"; exit 2; }
cat /proc/$$/winpid > q7.winpid 2>/dev/null       # 멈출 때: taskkill /PID <이 값> /T /F
echo "시작"; date
"$PY" -I -B run_cmp.py --jobs FZ11.0.25,FZ11.1.25,FZ11.2.25 > stream_q7a.log 2>&1 &
PA=$!
sleep 45
"$PY" -I -B run_cmp.py --jobs FZ23.0.25,FZ23.1.25,FZ23.2.25 > stream_q7b.log 2>&1 &
PB=$!
wait $PA; EA=$?
wait $PB; EB=$?
echo "25 epoch 얼린 갈래 끝: fz11=$EA fz23=$EB"; date
rmdir q7.lock
