"""COCO JSON 어노테이션 읽기.

Roboflow 의 YOLO 세그 내보내기는 폴리곤을 한 줄에 하나만 담기 때문에, 가려져서
조각난 객체는 가장 큰 조각만 남고 나머지가 잘린다. COCO 내보내기의 `bbox` 는
어노테이션 전체로부터 계산된 값이라 그런 손실이 없다.

폴더를 주면 하위의 `_annotations.coco.json` 을 모두 찾아 읽는다
(Roboflow 의 train/valid/test 분할 구조를 그대로 넣으면 된다).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .yolo_io import Box

ANNOTATION_NAME = "_annotations.coco.json"


@dataclass
class CocoRecord:
    """이미지 한 장과 그 박스들."""

    path: Path
    width: int
    height: int
    boxes: list[Box]


@dataclass
class CocoReport:
    files: list[Path] = field(default_factory=list)
    images: int = 0
    boxes: int = 0
    missing_images: list[str] = field(default_factory=list)
    unknown_categories: dict[str, int] = field(default_factory=dict)
    dropped_invalid: int = 0

    def summary(self) -> str:
        lines = [
            f"COCO 파일 {len(self.files)}개에서 이미지 {self.images}장 / 박스 {self.boxes}개",
        ]
        if self.unknown_categories:
            items = ", ".join(f"{k}({v})" for k, v in sorted(self.unknown_categories.items()))
            lines.append(f"  클래스 목록에 없어 버린 카테고리: {items}")
        if self.missing_images:
            lines.append(
                f"  JSON 에 있으나 파일이 없는 이미지 {len(self.missing_images)}장"
                f" (예: {self.missing_images[:2]})"
            )
        if self.dropped_invalid:
            lines.append(f"  크기가 0 이라 버린 박스 {self.dropped_invalid}개")
        return "\n".join(lines)


def find_annotation_files(root: Path) -> list[Path]:
    """폴더 아래의 COCO 어노테이션 파일들을 찾는다. 파일을 직접 줘도 된다."""
    root = Path(root)
    if root.is_file():
        return [root]
    found = sorted(root.rglob(ANNOTATION_NAME))
    if found:
        return found
    # 이름이 다른 경우를 위한 차선책
    return sorted(p for p in root.rglob("*.json") if "annotation" in p.name.lower())


_WIN_ILLEGAL = '<>:"/\\|?*'


def _image_dirs(base: Path) -> list[Path]:
    """이미지가 있을 만한 폴더들. base 는 json 이 놓인 폴더다.

    Roboflow 내보내기는 json 을 이미지와 같은 폴더에 둔다(train/valid/test 각각).
    다만 사람이 정리하다 보면 json 만 `coco/` 로 모으고 이미지는 `images/` 에
    두는 형태가 흔하다(이 저장소의 `data/seed` 가 그렇게 되어 있다). 둘 다 받는다.

        seed2/train/_annotations.coco.json + 같은 폴더의 이미지   -> base
        seed2/_annotations.coco.json      + seed2/images/         -> base/images
        seed2/coco/train_annotations.json + seed2/images/         -> base.parent/images
    """
    out, seen = [], set()
    for d in (base, base / "images", base.parent / "images"):
        key = str(d)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def _resolve_image(base: Path, file_name: str) -> Path | None:
    """JSON 의 file_name 에 해당하는 실제 파일을 찾는다.

    Windows 는 < > : " / \\ | ? * 를 파일명에 쓸 수 없어서, 압축을 풀 때 이런 글자가
    '_' 등으로 치환된다. 그러면 JSON 의 이름과 디스크의 이름이 어긋난다.
    정확한 이름이 없으면 금지문자를 '_' 로 바꾼 이름도 찾아본다.
    """
    names = [file_name]
    if any(ch in file_name for ch in _WIN_ILLEGAL):
        names.append("".join("_" if ch in _WIN_ILLEGAL else ch for ch in file_name))

    for d in _image_dirs(base):
        for name in names:
            path = d / name
            if path.exists():
                return path
    return None


def category_names(files: list[Path]) -> list[str]:
    """COCO 카테고리에서 클래스 이름을 순서대로 뽑는다.

    Roboflow 는 0번에 프로젝트 이름 같은 더미 상위 카테고리를 넣고 실제 클래스를
    1번부터 매기는 일이 잦다. supercategory 가 'none' 인 항목은 그 더미로 보고 건너뛴다.
    """
    for f in files:
        data = json.loads(Path(f).read_text(encoding="utf-8"))
        cats = sorted(data.get("categories", []), key=lambda c: c["id"])
        names = [
            str(c["name"])
            for c in cats
            if str(c.get("supercategory", "")).lower() != "none" or len(cats) == 1
        ]
        if names:
            return names
    return []


def read_coco(root: Path, class_index: dict[str, int]) -> tuple[list[CocoRecord], CocoReport]:
    """COCO JSON 들을 읽어 이미지별 박스 목록으로.

    class_index 는 {클래스 이름(소문자): 인덱스}. 여기에 없는 카테고리는 버리고 보고한다.
    """
    report = CocoReport()
    records: list[CocoRecord] = []

    for jf in find_annotation_files(Path(root)):
        report.files.append(jf)
        data = json.loads(jf.read_text(encoding="utf-8"))
        base = jf.parent

        cats = {c["id"]: str(c["name"]) for c in data.get("categories", [])}
        by_image: dict[int, list[Box]] = {}

        for a in data.get("annotations", []):
            name = cats.get(a["category_id"], "")
            cls = class_index.get(name.strip().lower())
            if cls is None:
                report.unknown_categories[name] = report.unknown_categories.get(name, 0) + 1
                continue
            by_image.setdefault(a["image_id"], []).append((cls, a["bbox"]))  # type: ignore[arg-type]

        for im in data.get("images", []):
            path = _resolve_image(base, im["file_name"])
            if path is None:
                report.missing_images.append(im["file_name"])
                continue
            w, h = int(im["width"]), int(im["height"])
            boxes: list[Box] = []
            for cls, bbox in by_image.get(im["id"], []):  # type: ignore[misc]
                x, y, bw, bh = (float(v) for v in bbox)
                box = Box(
                    cls,
                    (x + bw / 2) / w,
                    (y + bh / 2) / h,
                    bw / w,
                    bh / h,
                    None,
                    "seed",
                ).clamped()
                if box.is_valid():
                    boxes.append(box)
                else:
                    report.dropped_invalid += 1
            records.append(CocoRecord(path=path, width=w, height=h, boxes=boxes))
            report.images += 1
            report.boxes += len(boxes)

    return records, report
