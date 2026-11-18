#!/usr/bin/env bash
# 대기열을 지켜본다(watch_q4.sh 에 로그 이름 앞머리와 포기 표시 파일을 인자로 더한 것).
# results.jsonl 에 <정규식>에 맞는 줄이 <개수>개 모이거나, 실행이 두 번 실패하거나, 포기 표시 파일이 생기거나,
# 느려지면(VRAM 이 공유 메모리로 넘침 - 끝난 epoch 의 진행 줄이 it/s 가 아니라 s/it 로 찍히는 것이 3번 연속) 끝난다.
#   bash watch_runs.sh '"name": "FZ(11|23)_f[012]_e100"' 6 330 stream_q5 q5.failed
cd "$(dirname "$0")" || exit 1
pat="$1"; need="$2"; max="${3:-240}"; pre="${4:-stream_q4}"; giveup="${5:-}"
n=0; slow=0; why="시간 초과"
while [ "$n" -lt "$max" ]; do
  c=$(grep -c -E "$pat" results.jsonl 2>/dev/null)
  if [ "${c:-0}" -ge "$need" ]; then why="결과 ${c}개 모임"; break; fi
  if grep -q "실패 2/2" "$pre"*.log 2>/dev/null; then why="두 번 실패한 실행이 있다"; break; fi
  if [ -n "$giveup" ] && [ -e "$giveup" ]; then why="대기열이 시작하지 않고 끝났다($giveup)"; break; fi
  s=0
  for f in "$pre"*.log; do
    [ -f "$f" ] || continue
    if [ $(( $(date +%s) - $(stat -c %Y "$f") )) -lt 30 ]; then        # 지금 쓰이고 있는 로그만
      last=$(tail -c 12000 "$f" | sed 's/\x1b\[[0-9;]*[A-Za-z]//g' | tr '\r' '\n' | grep -E "^ +[0-9]+/100 .* 12/12 " | tail -1)
      case "$last" in *s/it*) s=1 ;; esac
    fi
  done
  if [ "$s" -eq 1 ]; then slow=$((slow+1)); else slow=0; fi
  if [ "$slow" -ge 3 ]; then why="느려졌다(끝난 epoch 의 진행 줄이 s/it)"; break; fi
  sleep 20; n=$((n+1))
done
date; echo "$why (기다린 시간 $((n*20))초)"
sed 's/\x1b\[[0-9;]*[A-Za-z]//g' "$pre"*.log 2>/dev/null | tr '\r' '\n' | grep -E "Remapped|RESULT .* last|=== DONE|=== START|실패|건너뜀" | tail -30
nvidia-smi --query-gpu=memory.used --format=csv,noheader
