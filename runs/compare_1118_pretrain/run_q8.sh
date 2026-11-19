#!/usr/bin/env bash
# 옛 데이터의 responder·cone 이미지를 학습 쪽에만 더해 보기(11-19, 사용자가 보관 폴더를 이 PC 로 가져옴).
#   a = FZ11X2 3묶음(카메라 프레임만 더함 - 짧다) 다음 FZ11X1 묶음 2 / b = FZ11X1 묶음 0·1(전부 더함). 두 줄기 동시(3.3GB 씩).
#   25 epoch 일정 - 견줄 기준은 이미 돌린 FZ11 25 epoch(같은 인자·같은 검증 목록). 끝나면 conf·크기별 세기까지 돌린다.
cd "$(dirname "$0")" || exit 1
PY=/c/local_auto_labeling/.venv/Scripts/python.exe
export YOLO_OFFLINE=true
busy() {
  powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { \$_.CommandLine -match 'run_cmp\.py' }).Count" 2>/dev/null | tr -d '\r\n '
}
if [ "$(busy)" != "0" ]; then echo "run_cmp.py 가 이미 돌고 있다 - 시작하지 않는다"; exit 2; fi
mkdir q8.lock 2>/dev/null || { echo "q8.lock 이 있다 - 이미 떠 있거나 죽은 흔적이다"; exit 2; }
cat /proc/$$/winpid > q8.winpid 2>/dev/null       # 멈출 때: taskkill /PID <이 값> /T /F
echo "시작"; date
"$PY" -I -B run_cmp.py --jobs FZ11X2.0.25,FZ11X2.1.25,FZ11X2.2.25,FZ11X1.2.25 > stream_q8a.log 2>&1 &
PA=$!
sleep 60
"$PY" -I -B run_cmp.py --jobs FZ11X1.0.25,FZ11X1.1.25 > stream_q8b.log 2>&1 &
PB=$!
wait $PA; EA=$?
wait $PB; EB=$?
echo "학습 끝: a=$EA b=$EB"; date
"$PY" -I -B sweep_eval.py --sets FZ11X2:25,FZ11X1:25 > stream_q8_sweep.log 2>&1
echo "세기 끝: $?"; date
rmdir q8.lock
