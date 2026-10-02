"""YOLO 포맷 라벨 입출력과 이미지 스캔."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

from PIL import Image

# Ultralytics 의 IMG_FORMATS 와 맞춘 목록에서, 이 환경에서 실제로 열리는 것만 담았다.
# .avif 는 Pillow 11.3+ 에 코덱이 내장돼 있다 (Windows 12.3.0 / WSL 12.1.1 둘 다 확인).
# .heic·.heif 는 Ultralytics 목록엔 있지만 이 Pillow 에 HEIF 코덱이 없어서 뺐다 —
# 넣으려면 `pillow-heif` 를 먼저 깔 것. 여기 없는 확장자는 scan 이 **조용히 건너뛴다.**
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".avif"}


@dataclass
class Box:
    """YOLO 정규화 좌표(0~1) 박스. cx, cy는 중심점."""

    cls: int
    cx: float
    cy: float
    w: float
    h: float
    conf: float | None = None
    origin: str = "auto"  # auto | human | seed

    def clamped(self) -> "Box":
        """이미지 밖으로 나간 박스를 [0,1] 안으로 잘라 넣는다."""
        x1 = max(0.0, self.cx - self.w / 2)
        y1 = max(0.0, self.cy - self.h / 2)
        x2 = min(1.0, self.cx + self.w / 2)
        y2 = min(1.0, self.cy + self.h / 2)
        return Box(
            cls=self.cls,
            cx=(x1 + x2) / 2,
            cy=(y1 + y2) / 2,
            w=max(0.0, x2 - x1),
            h=max(0.0, y2 - y1),
            conf=self.conf,
            origin=self.origin,
        )

    def is_valid(self, min_side: float = 1e-4) -> bool:
        return self.w > min_side and self.h > min_side and self.cls >= 0

    def to_dict(self) -> dict:
        return asdict(self)


def _row_to_box(parts: list[str], origin: str) -> Box:
    """라벨 한 줄을 Box 로 바꾼다. 박스 행과 세그멘테이션 폴리곤 행을 모두 받는다.

    열 구성:
        5열       cls cx cy w h                  -> 바운딩 박스
        6열       cls cx cy w h conf             -> 바운딩 박스 + 신뢰도 (이 도구가 쓰는 형식)
        7열 이상  cls x1 y1 x2 y2 x3 y3 ...      -> 폴리곤. 정점의 min/max 로 박스화

    Ultralytics 와 같은 기준(7열 이상 = 폴리곤)을 쓰되, Ultralytics 가 한 파일에
    두 형식이 섞이면 파일째 버리는 것과 달리 행 단위로 각각 처리한다.
    COCO 계열 데이터는 일부 객체만 폴리곤을 갖는 경우가 흔해서, 파일째 버리면
    멀쩡한 이미지까지 통째로 사라진다.
    """
    cls = int(float(parts[0]))
    vals = [float(v) for v in parts[1:]]

    if len(vals) >= 6:  # 정점 3개 이상 = 폴리곤
        if len(vals) % 2 != 0:
            raise ValueError(f"폴리곤 좌표 개수가 홀수입니다 ({len(vals)}개)")
        xs, ys = vals[0::2], vals[1::2]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        return Box(cls, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, None, origin)

    cx, cy, w, h = vals[:4]
    conf = vals[4] if len(vals) >= 5 else None
    return Box(cls, cx, cy, w, h, conf, origin)


def read_yolo_txt(path: Path, origin: str = "seed") -> list[Box]:
    """YOLO .txt 한 개를 읽어 박스 목록으로. 폴리곤 행은 외접 사각형으로 변환한다."""
    boxes: list[Box] = []
    if not path.exists():
        return boxes
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 5:
            raise ValueError(f"{path}:{lineno} 열이 5개 미만입니다: {line!r}")
        try:
            box = _row_to_box(parts, origin).clamped()
        except ValueError as exc:
            raise ValueError(f"{path}:{lineno} 파싱 실패 ({exc}): {line!r}") from exc
        if box.is_valid():
            boxes.append(box)
    return boxes


def write_yolo_txt(path: Path, boxes: list[Box]) -> None:
    """YOLO .txt 로 저장. 박스가 없으면 빈 파일(=배경 샘플)로 남긴다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{b.cls} {b.cx:.6f} {b.cy:.6f} {b.w:.6f} {b.h:.6f}"
        for b in boxes
        if b.is_valid()
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def iter_images(root: Path) -> Iterator[Path]:
    """디렉터리를 재귀 탐색하며 이미지 파일을 정렬된 순서로 내놓는다."""
    if not root.exists():
        return
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def image_size(path: Path) -> tuple[int, int]:
    """픽셀 데이터를 디코딩하지 않고 헤더만 읽어 크기를 구한다.

    **EXIF 회전을 적용하지 않은 원시 크기**다. DB 의 width/height 가 이 값이고
    `labeler.py check` 도 이 값과 견준다. 화면·모델이 보는 크기는 `display_size`.
    """
    with Image.open(path) as im:
        return im.size


_EXIF_ORIENTATION = 0x0112


def display_size(path: Path) -> tuple[int, int]:
    """EXIF 회전을 적용한 뒤의 (가로, 세로) — 브라우저와 Ultralytics(학습·추론·SAM)가 보는 크기.

    EXIF Orientation 이 5~8(90도 회전)이면 원시 크기와 가로·세로가 뒤바뀐다. 이 데이터에
    49장이 있고 그중 정사각이 아닌 4장(id 1050·1126·1692·3181)에서 DB 크기(원시)와
    어긋난다. 라벨은 회전을 적용한 화면 기준 정규화 좌표다 (2026-09-23 확인).
    """
    with Image.open(path) as im:
        w, h = im.size
        if im.getexif().get(_EXIF_ORIENTATION, 1) in (5, 6, 7, 8):
            return h, w
    return w, h


def label_path_for(image: Path, images_root: Path, labels_root: Path) -> Path:
    """이미지에 대응하는 YOLO .txt 경로를 찾는다.

    두 가지 레이아웃을 모두 지원한다.

    1) 경로에 'images' 폴더가 있으면 마지막 것을 'labels' 로 바꾼다.
       Ultralytics/Roboflow 관례라 아래 두 형태가 모두 그대로 처리된다.
           data/seed/images/x.jpg        -> data/seed/labels/x.txt
           data/seed/train/images/x.jpg  -> data/seed/train/labels/x.txt
       train/valid/test 로 나뉜 내보내기를 seed 루트 하나로 지정해도 동작한다.

    2) 'images' 폴더가 없거나 1)의 결과가 실제로 없으면,
       images_root 기준 상대경로를 labels_root 아래로 그대로 옮긴다.
    """
    parts = list(image.parts)
    for i in range(len(parts) - 2, -1, -1):  # 파일명 바로 앞부터 거슬러 올라간다
        if parts[i].lower() == "images":
            swapped = Path(*parts[:i], "labels", *parts[i + 1 :]).with_suffix(".txt")
            if swapped.exists():
                return swapped
            break

    try:
        rel = image.relative_to(images_root)
    except ValueError:
        rel = Path(image.name)
    return (labels_root / rel).with_suffix(".txt")
