#!/usr/bin/env bash
# 11-19 아침에 다시 띄운 대기열. 11-18 19:14 에 PC 가 꺼지면서 run_q4.sh 의 3단계(P2N)가 묶음 0 의 18 epoch 에서 끊겼고,
# 그 뒤를 기다리던 run_q5.sh(얼린 갈래)는 시작하지 못했다. 남은 9개를 이 스크립트 하나로 차례대로 돌린다.
#   1단계  두 줄기 동시(각 6GB 이하):  a = FZ11 3묶음 (backbone 층 0-10 얼림)  /  b = FZ23 3묶음 (층 0-22 얼림, Detect 만 학습)
#   2단계  혼자(8.8GB): 기본 P2(yolo26m-p2.yaml) 3묶음
cd "$(dirname "$0")" || exit 1
PY=/c/local_auto_labeling/.venv/Scripts/python.exe
export YOLO_OFFLINE=true
busy() {
  powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { \$_.CommandLine -match 'run_cmp\.py' }).Count" 2>/dev/null | tr -d '\r\n '
}
if [ "$(busy)" != "0" ]; then echo "run_cmp.py 가 이미 돌고 있다 - 시작하지 않는다"; exit 2; fi
mkdir q6.lock 2>/dev/null || { echo "q6.lock 이 있다 - 이미 떠 있거나 죽은 흔적이다"; exit 2; }
cat /proc/$$/winpid > q6.winpid 2>/dev/null       # 멈출 때: taskkill /PID <이 값> /T /F
echo "시작"; date
"$PY" -I -B run_cmp.py --jobs FZ11.0.100,FZ11.1.100,FZ11.2.100 > stream_q6a.log 2>&1 &
PA=$!
sleep 60
"$PY" -I -B run_cmp.py --jobs FZ23.0.100,FZ23.1.100,FZ23.2.100 > stream_q6b.log 2>&1 &
PB=$!
wait $PA; EA=$?
wait $PB; EB=$?
echo "1단계(얼린 갈래) 끝: fz11=$EA fz23=$EB"; date
"$PY" -B run_cmp.py --jobs P2N.0.100,P2N.1.100,P2N.2.100 > stream_q6c_p2n.log 2>&1
echo "2단계(P2N) 끝: $?"; date
rmdir q6.lock
