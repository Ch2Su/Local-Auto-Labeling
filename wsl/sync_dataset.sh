#!/usr/bin/env bash
# Windows 프로젝트가 만든 YOLO 데이터셋을 WSL 홈으로 복사한다 (이미지 + 라벨만).
#
#   bash /mnt/e/local_auto_labeling/wsl/sync_dataset.sh                  # 가시광
#   bash /mnt/e/local_auto_labeling/wsl/sync_dataset.sh <원본 dataset> <WSL 폴더>   # 다른 프로젝트
#
# 왜 복사하나
#   - /mnt/e 는 Windows 드라이브를 거쳐 읽어서 느리다. WSL 자체 파일시스템이 빠르다.
#   - **DB 는 가져가지 않는다.** data/labeling.db 는 SQLite 인데 /mnt/e 경로로는
#     파일 잠금이 제대로 안 먹는다. Windows 검수 서버가 떠 있는 동안 WSL 이 같은 DB 를
#     쓰면 손상될 수 있다. 학습에는 이미지와 라벨 .txt 만 있으면 된다.
#
# 빼는 것
#   *.npy    Windows 의 cache: disk 산출물(14GB). WSL 은 cache: ram 을 쓴다.
#   *.cache  라벨 캐시. 안에 Windows 절대경로가 들어 있어 WSL 에서는 쓸모없다.
#
# 데이터셋 폴더(data/<프로젝트>/dataset)는 Windows 쪽 학습이 만든다. 한 번도 안 돌렸으면 없다.

set -euo pipefail

# 기본값은 가시광. 다른 프로젝트는 첫 인자로 준다 (웹 UI 의 WSL 학습 버튼은 config 의
# data.dataset 을 그대로 넘긴다). 2026-09-21 에 data/dataset -> data/visible/dataset.
SRC="${1:-/mnt/e/local_auto_labeling/data/visible/dataset}"
DST="${2:-$HOME/datasets/visible}"

if [ ! -f "$SRC/data.yaml" ]; then
  echo "데이터셋이 없습니다: $SRC/data.yaml"
  echo "  Windows 에서 labeler.py train 을 한 번 돌리면 만들어집니다."
  exit 1
fi

mkdir -p "$DST"

# images/ 와 labels/ 를 각각 따로 맞춘다. --delete 를 **그 하위 폴더에만** 걸어서,
# 분할이 바뀌어 사라진 파일은 지우고 그대로인 파일은 건너뛴다. 데이터셋 폴더는
# 원본 수정 시각을 보존해 복사되므로(shutil.copy2) 두 번째부터는 거의 안 옮긴다.
# DST 자체에 --delete 를 걸지 않는 이유: 경로를 잘못 주면 엉뚱한 곳을 지운다.
echo "복사 중: $SRC  ->  $DST"
for sub in images labels; do
  mkdir -p "$DST/$sub"
  rsync -a --delete --info=progress2 --exclude='*.npy' --exclude='*.cache' \
        "$SRC/$sub"/ "$DST/$sub"/
done
cp "$SRC/data.yaml" "$DST/data.yaml"

# data.yaml 의 path 는 Windows 절대경로(E:\...)로 적혀 있다. WSL 경로로 바꾼다.
sed -i "s|^path:.*|path: $DST|" "$DST/data.yaml"

echo
echo "완료"
for s in train val test; do
  [ -d "$DST/images/$s" ] || continue
  ni=$(find "$DST/images/$s" -type f | wc -l)
  nl=$(find "$DST/labels/$s" -type f -name '*.txt' | wc -l)
  echo "  $s : 이미지 $ni / 라벨 $nl"
done
du -sh "$DST" | awk '{print "  용량 : "$1}'
echo "  data.yaml path -> $(grep '^path:' "$DST/data.yaml")"
