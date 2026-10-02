#!/bin/bash
# 앞선 비교(pod_compare.sh, PID 를 인자로)가 끝나기를 기다렸다가, RAM 한도로 죽은 a40_m1280 을
# disk 캐시로 다시 돌린다 (2026-09-18). 끝나면 progress.log 에 RETRYDONE 을 찍는다.
#   setsid nohup bash /workspace/pod_retry.sh <PID> >> /workspace/lal/runs/compare_a40/progress.log 2>&1 < /dev/null &
# 멈추려면 pod_compare.sh 와 같다: touch /workspace/runs_cmp/HALT (+ STOP)
PID=$1
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
if [ -f /workspace/runs_cmp/HALT ]; then
  echo "=== HALT — a40_m1280 재시도 안 함 $(date '+%F %T')"
else
  rm -rf /workspace/runs_cmp/a40_m1280          # 죽은 시도의 빈 폴더 (args.yaml 뿐)
  # 도는 중인 pod_compare.sh 를 덮어쓰면 bash 가 읽던 자리가 어긋나므로, 고친 판은 _v2 로 올렸다
  CACHE=disk bash /workspace/pod_compare_v2.sh a40_m1280
fi
echo "RETRYDONE $(date '+%F %T')"
