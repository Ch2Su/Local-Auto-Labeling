#!/usr/bin/env bash
# 검수 사본에서 사용자가 승인한 프레임을 학습 쪽에만 더해 보기(11-20, 사용자 '39장 넣었을 때 결과 측정').
#   a = FZ11R(사용자가 검수한 라벨) seed 0·1·2 다음 기준 FZ11 seed 1 / b = FZ11U(같은 프레임의 검수 전 라벨) seed 0·1·2 다음 기준
#   FZ11 seed 2. 두 줄기 동시. 25 epoch 일정 - 기준 FZ11 seed 0 은 이미 있다(같은 인자·같은 검증 목록). 끝나면 conf·크기별 세기.
cd "$(dirname "$0")" || exit 1
PY=/c/local_auto_labeling/.venv/Scripts/python.exe
export YOLO_OFFLINE=true
busy() {
  powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { \$_.CommandLine -match 'run_cmp\.py' }).Count" 2>/dev/null | tr -d '\r\n '
}
if [ "$(busy)" != "0" ]; then echo "run_cmp.py 가 이미 돌고 있다 - 시작하지 않는다"; exit 2; fi
mkdir q9.lock 2>/dev/null || { echo "q9.lock 이 있다 - 이미 떠 있거나 죽은 흔적이다"; exit 2; }
cat /proc/$$/winpid > q9.winpid 2>/dev/null       # 멈출 때: taskkill /PID <이 값> /T /F
jobs_for() {   # 갈래 이름 + seed 들 -> 갈래.묶음.25[.seed] 목록
  local arm=$1; shift; local out=""
  for s in "$@"; do for f in 0 1 2; do
    if [ "$s" = 0 ]; then out="$out,$arm.$f.25"; else out="$out,$arm.$f.25.$s"; fi
  done; done
  echo "${out#,}"
}
echo "시작"; date
"$PY" -I -B run_cmp.py --jobs "$(jobs_for FZ11R 0 1 2),$(jobs_for FZ11 1)" > stream_q9a.log 2>&1 &
PA=$!
sleep 60
"$PY" -I -B run_cmp.py --jobs "$(jobs_for FZ11U 0 1 2),$(jobs_for FZ11 2)" > stream_q9b.log 2>&1 &
PB=$!
wait $PA; EA=$?
wait $PB; EB=$?
echo "학습 끝: a=$EA b=$EB"; date
"$PY" -I -B sweep_eval.py --sets FZ11R:25,FZ11U:25 > stream_q9_sweep.log 2>&1
echo "세기 끝: $?"; date
rmdir q9.lock
