"""검수 완료된 라벨을 YOLO 포맷으로 내보낸다.

**폴더는 프로젝트마다 따로다** (기본 `export/<프로젝트>`, 2026-09-24). 예전 기본값 `export` 는
가시광·열화상·화재가 같이 써서, 차례로 내보내면 라벨이 한 폴더에 섞이고 `classes.txt`·
`data.yaml` 이 덮어써졌다. 다시 내보낼 때 지난번 파일(그 사이 빠진 이미지의 라벨)이 남아
있던 것도 같은 날 고쳤다 — 이 함수가 만든 폴더(표식 `export_info.json`)만 비우고, 표식 없는
폴더가 비어 있지 않으면 **쓰지 않고 멈춘다** (엉뚱한 폴더를 지우지 않게).

**저장소 안에 내보내면 그 폴더에 `.gitignore`(`*`)를 둔다** (2026-11-01). 자동 커밋 훅이 턴마다
`git add -A` 를 하므로, 기본값(`export/`, 무시됨)이 아닌 폴더로 내보낸 라벨·이미지 수천 개가 커밋되려
했다 — 그만큼이면 훅이 60초 제한에도 걸린다.

**data.yaml 은 DB 의 분할을 따른다** (2026-11-05). 예전에는 늘 `train: images / val: images` 라 test 이미지까지
학습에 섞였다. 폴더(images/·labels/)는 그대로 두고 분할별 목록(train.txt·val.txt·test.txt — 줄마다 `./images/…`,
Ultralytics 가 목록 파일 기준으로 푼다)을 써서 data.yaml 이 그것을 가리킨다. 분할이 아직 없는 이미지는 목록에 넣지 않고
수만 알린다. `path:` 는 적지 않는다 — 없으면 Ultralytics 가 data.yaml 이 있는 폴더를 기준으로 삼는다(`path: .` 는 실행한
폴더 기준이었다). 표식이 **다른 DB 의 것이면** 비우지 않고 멈춘다(다른 프로젝트의 내보내기를 지우지 않게).
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from .config import ROOT, Config, resolve
from .masks import split_boxes, write_masked
from .split import SPLITS
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
    missing: list[str] = field(default_factory=list)       # 이미지를 함께 내보내는데 원본이 없어 뺀 것(라벨도 안 씀)
    splits: dict[str, int] = field(default_factory=dict)    # 분할 목록(train.txt …)에 넣은 장수. 비면 분할 없음
    unassigned: int = 0         # DB 에 분할이 아직 없어 목록에서 뺀 장수 (다음 학습·split 때 배정된다)
    masked: int = 0             # 가릴 영역을 칠해 내보낸 이미지 (masks.py — 이미지를 함께 내보낼 때만)
    mask_dropped: int = 0       # 그 영역에 반 넘게 가려져 라벨에서 뺀 박스
    masked_raw: int = 0         # 영역이 있지만 칠하지 않고 DB 박스 그대로 쓴 이미지 (라벨만 내보냄·test)

    def summary(self) -> str:
        lines = [
            f"내보낸 위치 : {self.out_dir}" + ("  (지난번 내보내기를 비우고 새로 씀)" if self.cleared else ""),
            f"라벨 파일   : {self.images}개 (박스 {self.boxes}개)",
        ]
        if self.copied_images:
            lines.append(f"복사한 이미지: {self.copied_images}장")
        if self.missing:
            lines.append(f"  원본이 없어 뺀 이미지 {len(self.missing)}장 — 라벨도 쓰지 않았습니다 (예: {self.missing[:2]})")
        if self.splits:
            lines.append("분할 목록   : " + ", ".join(f"{s}.txt {n}장" for s, n in self.splits.items()))
            if self.unassigned:
                lines.append(f"  분할이 아직 없는 {self.unassigned}장은 목록에 넣지 않았습니다 "
                             "(라벨은 labels/ 에 있다 — 다음 학습이나 labeler.py split 때 배정된다)")
        elif self.images:
            lines.append("  DB 에 분할(train·val·test)이 없어 data.yaml 의 train·val 이 모두 images/ 입니다")
        if self.masked:
            lines.append(f"가릴 영역  : {self.masked}장을 회색(114)으로 칠해 내보냈고, 반 넘게 가려진 박스 "
                         f"{self.mask_dropped}개는 라벨에서 뺐습니다")
        if self.masked_raw:
            lines.append(f"  가릴 영역이 있는 {self.masked_raw}장은 칠하지 않고 DB 박스 그대로 썼습니다 "
                         "(이미지를 함께 내보내지 않았거나 test 라서)")
        return "\n".join(lines)


def _same_file(a: str | Path, b: str | Path) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def _prepare(out_dir: Path, cfg: Config) -> bool:
    """쓸 폴더를 준비하고 표식을 먼저 남긴다. 지난번 내보내기를 비웠으면 True.

    이 함수가 만든 폴더(표식 파일이 있음)만 비운다. 표식 없이 무언가 들어 있는 폴더는
    다른 용도일 수 있어 건드리지 않고 멈춘다. 표식을 **쓰기 전에** 남기는 이유: 끝에만 쓰면
    첫 내보내기가 도중에 죽었을 때 표식 없는 폴더가 남아 다음 내보내기가 거절된다.
    """
    cleared = False
    if out_dir.exists() and any(out_dir.iterdir()):
        marker = out_dir / MARKER
        if not marker.exists():
            raise RuntimeError(
                f"{out_dir} 에 이미 다른 파일이 있습니다. 내보내기는 빈 폴더나 예전 내보내기 폴더에만 "
                "씁니다 — 다른 폴더를 고르거나 그 폴더를 비우세요."
            )
        # 표식이 있어도 **다른 DB(프로젝트)의 내보내기면** 비우지 않는다 (2026-11-05) — 예전에는 표식만 보고 지웠다.
        try:
            prev_db = json.loads(marker.read_text(encoding="utf-8")).get("db")
        except (OSError, ValueError, AttributeError):
            prev_db = None
        if not prev_db or not _same_file(prev_db, cfg.db):
            raise RuntimeError(
                f"{out_dir} 는 다른 DB 의 내보내기입니다 (표식의 DB: {prev_db or '읽지 못함'}, 지금 DB: {cfg.db}). "
                "다른 프로젝트의 결과를 지우지 않게 멈췄습니다 — 다른 폴더를 고르거나, 필요 없는 폴더면 직접 비우세요."
            )
        for sub in ("labels", "images"):
            if (out_dir / sub).exists():
                # 못 지우면 멈춘다 — 지난번 라벨이 섞여 남으면 지금 승인되지 않은 이미지가 따라간다.
                shutil.rmtree(out_dir / sub)
        for s in SPLITS:                 # 지난번 분할 목록 — 이번에 그 분할이 없으면 남아서 헷갈린다
            (out_dir / f"{s}.txt").unlink(missing_ok=True)
        # 이 폴더로 학습하면 Ultralytics 가 labels.cache 를 남긴다. 캐시 열쇠가 파일 경로·크기 합이라, 클래스 번호만
        # 바뀐 라벨(크기가 같다)로 다시 내보내면 옛 라벨을 그대로 읽는다 — 같이 지운다.
        (out_dir / "labels.cache").unlink(missing_ok=True)
        cleared = True
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_marker(out_dir, {"db": str(cfg.db), "finished": False})
    if out_dir.resolve().is_relative_to(ROOT) and not (out_dir / ".gitignore").exists():
        (out_dir / ".gitignore").write_text("# labeler.py export 가 만든 폴더 — git 에 담지 않는다\n*\n", encoding="utf-8")
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
    복사해 그대로 학습에 쓸 수 있다 — 원본이 없는 이미지는 라벨도 쓰지 않는다(이미지 없는 라벨만
    남으면 그 폴더로 학습할 때 어긋난다). 분할은 DB 의 images.split 을 그대로 목록 파일로 쓴다(머리말).
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
    lists: dict[str, list[str]] = {s: [] for s in SPLITS}

    seen: set[str] = set()
    masks = store.masks_by_image()      # 가릴 영역 (masks.py) — 이미지를 함께 내보낼 때만 칠한다
    # 박스는 연결 하나로 한꺼번에 읽는다 — 예전에는 이미지마다 get_boxes() 가 연결을 새로 열었다(2026-11-05).
    for row, boxes in store.approved_with_boxes(tuple(statuses)):
        src = Path(row["path"])
        if with_images and not src.exists():
            report.missing.append(str(src))
            continue
        stem = src.stem
        # Windows 는 파일 이름의 대소문자를 가리지 않는다 — IMG_1 과 img_1 은 같은 .txt 다.
        if stem.lower() in seen:
            stem = f"{stem}__{hashlib.sha1(str(src).encode()).hexdigest()[:8]}"

        suffix = src.suffix
        rects = masks.get(row["id"])
        if rects and (not with_images or row.get("split") == "test"):
            # 라벨만 내보내면 받는 쪽은 원본을 쓴다 — 칠하지 않은 그림에서 박스만 빼면 그 사람들이 배경이 된다.
            # test 는 실제 화면 그대로(masks.py). 둘 다 DB 박스를 그대로 쓴다.
            report.masked_raw += 1
            rects = None
        if with_images:                  # 이미지를 먼저 — 복사가 안 되면 라벨도 쓰지 않는다
            try:
                if rects:
                    suffix = ".jpg"
                    write_masked(src, images_dir / f"{stem}{suffix}", rects)
                else:
                    shutil.copy2(src, images_dir / f"{stem}{suffix}")
            except FileNotFoundError:
                report.missing.append(str(src))
                continue
            report.copied_images += 1
            if rects:
                boxes, dropped = split_boxes(boxes, rects)
                report.masked += 1
                report.mask_dropped += len(dropped)
        seen.add(stem.lower())
        write_yolo_txt(labels_dir / f"{stem}.txt", boxes)
        report.images += 1
        report.boxes += len(boxes)
        if row.get("split") in lists:
            lists[row["split"]].append(f"./images/{stem}{suffix}")
        else:
            report.unassigned += 1

    (out_dir / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    head = ["# labeler.py export 가 만든 파일 — path 는 일부러 적지 않는다(없으면 이 파일이 있는 폴더가 기준이다)."]
    if any(lists.values()):
        for s in SPLITS:
            if s != "test" or lists[s]:
                (out_dir / f"{s}.txt").write_text("".join(f"{p}\n" for p in lists[s]), encoding="utf-8")
                report.splits[s] = len(lists[s])
        spec: dict = {s: f"{s}.txt" for s in report.splits}
        head.append("# train·val·test 는 DB 의 분할(images.split) 그대로다. test 는 학습·best.pt 고르기에 쓰지 말 것.")
        if report.unassigned:
            head.append(f"# 분할이 아직 없는 {report.unassigned}장은 목록에 없다(라벨은 labels/ 에 있다).")
    else:
        spec = {"train": "images", "val": "images"}
        head.append("# DB 에 분할(train·val·test)이 없어 train·val 이 모두 images/ 다 — val 점수는 학습한 이미지로 잰 값이 된다.")
    if not with_images:
        head.append("# 이미지를 함께 내보내지 않았다(images/ 가 비어 있다) — 그대로는 학습할 수 없다.")
    if report.masked:
        head.append(f"# 가릴 영역이 있는 {report.masked}장은 그 자리를 회색(114)으로 칠한 JPEG 이고, 반 넘게 가려진 박스는 라벨에서 뺐다.")
    if report.masked_raw:
        head.append(f"# 가릴 영역이 있는 {report.masked_raw}장은 칠하지 않고 DB 박스 그대로다(라벨만 내보냈거나 test).")
    (out_dir / "data.yaml").write_text(
        "\n".join(head) + "\n"
        + yaml.safe_dump({**spec, "names": dict(enumerate(classes))}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    _write_marker(out_dir, {
        "db": str(cfg.db), "finished": True, "statuses": list(statuses),
        "images": report.images, "boxes": report.boxes, "with_images": with_images,
        "splits": report.splits, "unassigned": report.unassigned, "missing": len(report.missing),
    })
    return report
