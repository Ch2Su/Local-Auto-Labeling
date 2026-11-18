#!/usr/bin/env bash
# 11-18 18시에 다시 짠 대기열. 그 전 사고: TaskStop 으로 멈춘 줄 알았던 run_o365.sh 가 살아 있어 run_q2.sh 와 겹쳐 돌았다
# (학습 4개 동시 -> VRAM 이 공유 메모리로 넘쳐 epoch 5.5초가 100초로, OM 갈래는 두 프로세스가 같은 폴더에 썼다).
# 전부 트리째 죽이고(taskkill /T) 남은 갈래를 이 스크립트 하나로 차례대로 돌린다 — 다른 스크립트의 로그를 기다리는 사슬을 두지 않는다.
#   1단계  두 줄기 동시(각 약 6GB):  a = F.1 F.2 O.0 O.1   /   b = OM.0 OM.1 OM.2 O.2
#   2단계  혼자(11.6GB): P2 곁가지를 seed 1 로 3묶음
#   3단계  혼자(8.7GB):  기본 P2(yolo26m-p2.yaml) 3묶음
# 내려받은 가중치를 읽는 1단계는 python 을 -I(격리)로 부른다.
cd "$(dirname "$0")" || exit 1
PY=/c/local_auto_labeling/.venv/Scripts/python.exe
export YOLO_OFFLINE=true
mkdir q4.lock 2>/dev/null || { echo "q4.lock 이 있다 - 이미 떠 있거나 죽은 흔적이다. 확인하고 지운 뒤 다시"; exit 2; }
cat /proc/$$/winpid > q4.winpid 2>/dev/null       # 멈출 때: taskkill /PID <이 값> /T /F
echo "시작"; date
"$PY" -I -B run_cmp.py --jobs F.1.100,F.2.100,O.0.100,O.1.100 > stream_q4a.log 2>&1 &
PA=$!
sleep 60
"$PY" -I -B run_cmp.py --jobs OM.0.100,OM.1.100,OM.2.100,O.2.100 > stream_q4b.log 2>&1 &
PB=$!
wait $PA; EA=$?
wait $PB; EB=$?
echo "1단계(F, O365) 끝: a=$EA b=$EB"; date
"$PY" -B run_cmp.py --jobs P2S.0.100.1,P2S.1.100.1,P2S.2.100.1 > stream_q4c_p2s_s1.log 2>&1
echo "2단계(P2S seed 1) 끝: $?"; date
"$PY" -B run_cmp.py --jobs P2N.0.100,P2N.1.100,P2N.2.100 > stream_q4d_p2n.log 2>&1
echo "3단계(P2N) 끝: $?"; date
rmdir q4.lock
