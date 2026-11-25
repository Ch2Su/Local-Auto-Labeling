#!/usr/bin/env bash
# 네 갈래를 차례로 (한 갈래 = 한 프로세스 — VRAM 최대치를 따로 재려고). 저장소 폴더에서: bash runs/sam_click_bench/run_all.sh
set -e
cd "$(dirname "$0")/../.."
for v in t s sam31 tcrop; do
  echo "=== $v $(date +%H:%M:%S)"
  .venv/Scripts/python.exe -I runs/sam_click_bench/bench.py --variant "$v" 2>&1 | grep -v "deprecated"
done
echo "=== done $(date +%H:%M:%S)"
