#!/usr/bin/env bash
# ② 시험 — P2 곁가지(P2S)를 같은 3묶음·같은 인자로, 이어서 견줄 것 둘.
# 1) P2S 3묶음: VRAM 11.6GB 라 혼자 돈다.
# 2) A 를 seed 1 로 다시(같은 갈래가 seed 만으로 얼마나 흔들리는지 — 두 줄기, 줄기마다 약 6GB).
# 3) P2N 3묶음: Ultralytics 의 yolo26m-p2.yaml 에 기본 로더로 얹은 것(VRAM 8.7GB, 혼자).
cd "$(dirname "$0")" || exit 1
PY=/c/local_auto_labeling/.venv/Scripts/python.exe
export YOLO_OFFLINE=true
"$PY" -B run_cmp.py --jobs P2S.0.100,P2S.1.100,P2S.2.100 > stream_p2s.log 2>&1
echo "P2S done: $?"; date
"$PY" -B run_cmp.py --jobs A.0.100.1,A.2.100.1 > stream_a_s1a.log 2>&1 &
PA=$!
sleep 60
"$PY" -B run_cmp.py --jobs A.1.100.1 > stream_a_s1b.log 2>&1 &
PB=$!
wait $PA; EA=$?
wait $PB; EB=$?
echo "A seed1 done: a=$EA b=$EB"; date
"$PY" -B run_cmp.py --jobs P2N.0.100,P2N.1.100,P2N.2.100 > stream_p2n.log 2>&1
echo "P2N done: $?"; date
