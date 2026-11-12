#!/bin/bash
# Runpod 에서 '큰 모델 x 큰 해상도' 비교 (2026-11-12, 사용자 요청 — 라벨링용 모델을 x 로, 큰 해상도로).
# 데이터는 tools/build_subset_dataset.py 로 만든 두 묶음이다:
#   oldcctv8  옛 데이터의 CCTV 승인 이미지 (responder·cone 든 이미지를 뺀 8클래스) — train 2,290 / val 523 / test 441
#   newtest8  11-10 뒤에 새로 모은 풀 프레임 승인 이미지 119장 — 전부 test (어느 모델도 본 적 없다, 가릴 영역은 칠했다)
# 인자는 지금 웹 UI 학습 버튼과 같다(wsl_train.hyp_tokens — 100 epoch, close_mosaic 10). 바꾸는 것은 모델·imgsz 뿐이고
# **batch 는 모든 설정이 8 로 같다** — x1280 이 48GB 에 들어가는 최대가 8 이고, batch 가 다르면 BatchNorm 통계가 달라져
# 모델·해상도의 효과와 섞인다(nbs=64 누적이 옵티마이저 쪽 유효 batch 는 맞춰 준다).
#
# pod 준비 (docs/runpod.md):
#   /workspace/lal/{wsl/train_wsl.py,wsl/eval_test.py,weights/yolo26?.pt,models/best_y26m_8cls.pt}
#   tar -xf oldcctv8.tar -C /root/datasets/oldcctv8 ; tar -xf newtest8.tar -C /root/datasets/newtest8   (각각 dataset/ 이 풀린다)
#   bash /workspace/lal/runs/runpod/pod_xres.sh prep
# 실행:
#   setsid nohup bash /workspace/lal/runs/runpod/pod_xres.sh [이름...] > /workspace/xres_progress.log 2>&1 &
#   touch /workspace/runs_xres/STOP                            # 지금 것만 멈춤
#   touch /workspace/runs_xres/HALT /workspace/runs_xres/STOP  # 전부 멈춤 (이 스크립트를 kill 하지 말 것)
#   bash /workspace/lal/runs/runpod/pod_xres.sh eval           # 끝난 가중치 전부 + 지금 운영 모델을 두 test 에 잰다
set -u
export YOLO_CONFIG_DIR=/tmp/Ultralytics
P=python3
LAL=/workspace/lal
T=$LAL/wsl/train_wsl.py
E=$LAL/wsl/eval_test.py
W=$LAL/weights
PROJ=/workspace/runs_xres
OUT=$LAL/runs/xres_1112
OLD=/root/datasets/oldcctv8/dataset
NEW=/root/datasets/newtest8/dataset
BATCH=${BATCH:-8}
# ram 이 기본. 09-18 에 m1280 이 ram 캐시(train 4,708장 17GB + val)와 워커 8개로 pod 의 RAM 한도 46.6GiB 를 넘어 죽었다 —
# 이 세트는 train 2,290장이라 절반이지만, 죽으면 CACHE=disk 로 다시(원본 크기 .npy 를 /root 에 쓴다).
CACHE=${CACHE:-ram}
HYP="epochs=100 patience=120 seed=0 pretrained=true amp=true cos_lr=true box=7.5 cls=1.5 dfl=1.5 close_mosaic=10 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 fliplr=0.5 flipud=0.0 save_period=20"

declare -A CFG=(
  [m640]="yolo26m.pt 640"       # 대조: 지금 운영 설정의 모델·해상도
  [x640]="yolo26x.pt 640"       # 모델만 키움
  [m1280]="yolo26m.pt 1280"     # 해상도만 키움
  [x1280]="yolo26x.pt 1280"     # 둘 다 — 사용자가 보려는 것
  [x960]="yolo26x.pt 960"       # 09-18 A40 비교와 같은 설정
  [l1280]="yolo26l.pt 1280"
)

prep() {
  # 이미지의 Pillow 11.0.0 은 확장자만 .png 인 JPEG 을 '손상' 으로 빼고 AVIF 를 못 읽는다 (docs/runpod.md)
  pip install --break-system-packages -q -U pillow
  for d in "$OLD" "$NEW"; do
    if [ ! -f "$d/data.yaml" ]; then echo "없음: $d/data.yaml"; exit 1; fi
    sed -i "s|^path:.*|path: $d|" "$d/data.yaml"      # 만든 PC 의 Windows 경로가 적혀 있다
    find "$d/labels" -name '*.cache' -delete
    echo "$d: $(find "$d/images" -type f | wc -l) images"
  done
  mkdir -p "$OUT" "$PROJ"
  ls -la "$W" "$LAL/models" 2>/dev/null
  $P -c "import ultralytics, torch, PIL; print('ultralytics', ultralytics.__version__, 'torch', torch.__version__, 'pillow', PIL.__version__, 'cuda', torch.cuda.is_available())"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  free -g | head -2
}

run() {   # 이름 가중치 imgsz
  local name=$1 weights=$2 imgsz=$3
  if [ ! -f "$W/$weights" ]; then
    echo "=== $name 건너뜀 — $W/$weights 없음 $(date '+%F %T')"
    return
  fi
  echo "=== $name 시작 $(date '+%F %T')  ($weights / $imgsz / b$BATCH / cache $CACHE)"
  "$P" "$T" --project "$PROJ" --name "$name" --data "$OLD/data.yaml" --model "$W/$weights" \
    --workers 8 --cache "$CACHE" --bare --hyp $HYP --imgsz "$imgsz" --batch "$BATCH" > "$OUT/$name.log" 2>&1
  local dir
  dir=$(grep '^RESULT_DIR=' "$OUT/$name.log" | tail -1 | cut -d= -f2)
  if [ -n "$dir" ] && [ -f "$dir/weights/best.pt" ]; then
    cp "$dir/weights/best.pt" "$OUT/${name}_best.pt"
    cp "$dir/results.csv" "$OUT/${name}_results.csv"
    cp "$dir/args.yaml" "$OUT/${name}_args.yaml"
    echo "=== $name 끝 $(date '+%F %T')  $dir"
  else
    echo "=== $name 실패 $(date '+%F %T') — $OUT/$name.log 확인"
  fi
}

evaluate() {
  # 옛 test 는 학습한 해상도로(가중치 이름의 숫자), 새 풀 프레임 test 는 여러 입력 크기로 잰다.
  # 운영 모델(best_y26m_8cls — 옛 데이터 전체 도메인으로 학습한 m640)은 두 세트 모두 같은 크기들로 잰다.
  local out="$OUT/summary.jsonl"
  for w in "$OUT"/*_best.pt; do
    [ -f "$w" ] || continue
    local sz
    sz=$(basename "$w" | sed -E 's/^[a-z]+([0-9]+)_best\.pt$/\1/')
    "$P" "$E" --weights "$w" --data "oldcctv8=$OLD/data.yaml" --imgsz "$sz" --out "$out" --note "train imgsz $sz"
    "$P" "$E" --weights "$w" --data "newtest8=$NEW/data.yaml" --imgsz 640 1280 1920 2560 --out "$out" --note "train imgsz $sz"
  done
  if [ -f "$LAL/models/best_y26m_8cls.pt" ]; then
    "$P" "$E" --weights "$LAL/models/best_y26m_8cls.pt" --data "oldcctv8=$OLD/data.yaml" --imgsz 640 1280 --out "$out" --note "operating model"
    "$P" "$E" --weights "$LAL/models/best_y26m_8cls.pt" --data "newtest8=$NEW/data.yaml" --imgsz 640 1280 1920 2560 --out "$out" --note "operating model"
  fi
  echo "EVALDONE $(date '+%F %T')  $out"
}

case "${1:-}" in
  prep) prep; exit 0 ;;
  eval) evaluate; exit 0 ;;
esac

ORDER=(m640 x1280)
[ $# -gt 0 ] && ORDER=("$@")
mkdir -p "$OUT" "$PROJ"
rm -f "$PROJ/HALT"
for name in "${ORDER[@]}"; do
  if [ -z "${CFG[$name]:-}" ]; then
    echo "=== 모르는 이름: $name (아는 것: ${!CFG[*]})"
    continue
  fi
  if [ -f "$PROJ/HALT" ]; then
    rm -f "$PROJ/HALT"
    echo "=== HALT — $name 부터 시작하지 않고 멈춤 $(date '+%F %T')"
    break
  fi
  run "$name" ${CFG[$name]}
done
evaluate
echo "ALLDONE $(date '+%F %T')"
