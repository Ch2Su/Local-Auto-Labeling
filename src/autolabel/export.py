"""검수 완료된 라벨을 YOLO 포맷으로 내보낸다.

**폴더는 프로젝트마다 따로다** (기본 `export/<프로젝트>`, 2026-09-24). 예전 기본값 `export` 는
가시광·열화상·화재가 같이 써서, 차례로 내보내면 라벨이 한 폴더에 섞이고 `classes.txt`·
`data.yaml` 이 덮어써졌다. 다시 내보낼 때 지난번 파일(그 사이 빠진 이미지의 라벨)이 남아
있던 것도 같은 날 고쳤다 — 이 함수가 만든 폴더(표식 `export_info.json`)만 비우고, 표식 없는
폴더가 비어 있지 않으면 **쓰지 않고 멈춘다** (엉뚱한 폴더를 지우지 않게).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .config import Config, resolve
from .store import Store
from .yolo_io import write_yolo_txt

MARKER = "export_info.json"


def default_export_dir(cfg: Config) -> Path:
    """`export/<프로젝트>` — 프로젝트 이름은 DB 가 든 폴더 이름(예: `data/visible/labeling.db` → visible)."""
    return resolve("export") / cfg.db.parent.name


@dataclass
class ExportReport:
    out_dir: Path
    images: int = 0
    boxes: int = 0
    copied_images: int = 0
    cleared: bool = False       # 지난번 내보내기를 비우고 다시 썼나

    def summary(self) -> str:
        lines = [
            f"내보낸 위치 : {self.out_dir}" + ("  (지난번 내보내기를 비우고 새로 씀)" if self.cleared else ""),
            f"라벨 파일   : {self.images}개 (박스 {self.boxes}개)",
        ]
        if self.copied_images:
            lines.append(f"복사한 이미지: {self.copied_images}장")
        return "\n".join(lines)


def _prepare(out_dir: Path, cfg: Config) -> bool:
    """쓸 폴더를 준비하고 표식을 먼저 남긴다. 지난번 내보내기를 비웠으면 True.

    이 함수가 만든 폴더(표식 파일이 있음)만 비운다. 표식 없이 무언가 들어 있는 폴더는
    다른 용도일 수 있어 건드리지 않고 멈춘다. 표식을 **쓰기 전에** 남기는 이유: 끝에만 쓰면
    첫 내보내기가 도중에 죽었을 때 표식 없는 폴더가 남아 다음 내보내기가 거절된다.
    """
    cleared = False
    if out_dir.exists() and any(out_dir.iterdir()):
        if not (out_dir / MARKER).exists():
            raise RuntimeError(
                f"{out_dir} 에 이미 다른 파일이 있습니다. 내보내기는 빈 폴더나 예전 내보내기 폴더에만 "
                "씁니다 — 다른 폴더를 고르거나 그 폴더를 비우세요."
            )
        for sub in ("labels", "images"):
            if (out_dir / sub).exists():
                # 못 지우면 멈춘다 — 지난번 라벨이 섞여 남으면 지금 승인되지 않은 이미지가 따라간다.
                shutil.rmtree(out_dir / sub)
        cleared = True
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_marker(out_dir, {"db": str(cfg.db), "finished": False})
    return cleared


def _write_marker(out_dir: Path, info: dict) -> None:
    info = {**info, "at": datetime.now().isoformat(timespec="seconds")}
    (out_dir / MARKER).write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def export_yolo(
    cfg: Config,
    store: Store,
    out_dir: Path | None = None,
    *,
    with_images: bool = False,
    statuses: tuple[str, ...] = ("approved",),
) -> ExportReport:
    """승인된 이미지의 라벨을 out_dir/labels 에 YOLO .txt 로 저장한다.

    out_dir 를 주지 않으면 `export/<프로젝트>`. 같은 이름의 이미지가 여러 폴더에 있을 수
    있으므로 경로 해시를 붙여 구분한다. with_images=True 면 out_dir/images 에 이미지도 함께
    복사해 그대로 학습에 쓸 수 있다.
    """
    import hashlib

    out_dir = Path(out_dir) if out_dir else default_export_dir(cfg)
    cleared = _prepare(out_dir, cfg)
    labels_dir = out_dir / "labels"
    images_dir = out_dir / "images"
    labels_dir.mkdir(parents=True, exist_ok=True)
    if with_images:
        images_dir.mkdir(parents=True, exist_ok=True)

    report = ExportReport(out_dir=out_dir, cleared=cleared)
    classes = store.get_classes()

    seen: set[str] = set()
    for status in statuses:
        for row in store.list_images(status=status, limit=10**9):
            src = Path(row["path"])
            stem = src.stem
            # Windows 는 파일 이름의 대소문자를 가리지 않는다 — IMG_1 과 img_1 은 같은 .txt 다.
            if stem.lower() in seen:
                stem = f"{stem}__{hashlib.sha1(str(src).encode()).hexdigest()[:8]}"
            seen.add(stem.lower())

            boxes = store.get_boxes(int(row["id"]))
            write_yolo_txt(labels_dir / f"{stem}.txt", boxes)
            report.images += 1
            report.boxes += len(boxes)

            if with_images and src.exists():
                shutil.copy2(src, images_dir / f"{stem}{src.suffix}")
                report.copied_images += 1

    (out_dir / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    (out_dir / "data.yaml").write_text(
        "path: .\ntrain: images\nval: images\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(classes)),
        encoding="utf-8",
    )
    _write_marker(out_dir, {
        "db": str(cfg.db), "finished": True, "statuses": list(statuses),
        "images": report.images, "boxes": report.boxes, "with_images": with_images,
    })
    return report
