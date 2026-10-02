#!/bin/bash
# A40 비교 가중치를 로컬 GPU 로 test 전체·드론·비CCTV 에 잰다. 사용: bash eval_local.sh 이름:imgsz ...
cd /e/local_auto_labeling
PY=.venv/Scripts/python.exe
for spec in "$@"; do
  n=${spec%%:*}; sz=${spec##*:}
  w=runs/compare_a40/${n}_best.pt
  for dom in all drone noncctv; do
    case $dom in
      all) extra=();;
      drone) extra=(--like stabilized_);;
      noncctv) extra=(--like 'data\labeled');;
    esac
    echo "##### $n $dom imgsz $sz"
    $PY tools/test_baseline.py --weights "$w" --imgsz "$sz" --out "runs/test_baseline/20260918_${n}_${dom}" "${extra[@]}" 2>&1 \
      | grep -aE "^(test|가중치|mAP50-95|  [a-z]+ )" 
  done
done
echo EVALDONE
