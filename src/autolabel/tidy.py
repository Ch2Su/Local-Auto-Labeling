"""검수를 마친 이미지를 자동 라벨링 폴더 밖으로 옮겨 정리한다.

data/unlabeled 는 '아직 처리할 것'만 남아 있어야 남은 작업량이 보인다. 승인된
이미지를 그대로 두면 자동 라벨링을 돌릴 때마다 폴더가 계속 불어난다.

옮길 때 라벨 .txt 도 함께 쓴다. 그러면 data/labeled 자체가 완결된 YOLO 데이터셋이
되어, 나중에 DB를 다시 만들더라도 검수 결과가 사라지지 않는다.

**절대 덮어쓰지 않는다** (2026-09-27 에 고침). 예전에는 이름이 겹치면 `원래이름__<원래 경로의
해시>` 로 한 번만 바꿨는데, 같은 이름(`images (1).jpg`)으로 받은 사진이 **같은 자리**
(`unlabeled/images (1).jpg`)를 두 번 거쳐 오면 해시까지 같아져서, 두 번째 정리가 첫 번째 사진
파일을 덮어썼다(Windows 의 `shutil.move` 는 대상이 있으면 복사로 넘어가 덮어쓴다). 이어서 DB
경로 갱신이 경로 중복으로 실패해 파일만 옮겨진 행이 '파일 없음' 이 됐고, 함께 도는 prune 이 그
행(라벨 포함)을 지웠다 — 6639·6641 의 원본 사진을 잃고 6676·6686 의 행이 사라졌다. 지금은
이미지·라벨 이름이 둘 다 비어 있을 때까지 번호를 붙이고, DB 갱신에 실패하면 파일을 되돌린다.
"""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .config import Config
from .store import Store
from .yolo_io import write_yolo_txt

ProgressFn = Callable[[int, int, str], None]


@dataclass
class TidyReport:
    moved: int = 0
    boxes: int = 0
    skipped_missing: list[str] = field(default_factory=list)
    renamed: int = 0
    dest: Path | None = None
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.moved and not self.errors:
            return "옮길 이미지가 없습니다. (자동 라벨링 폴더에 승인된 이미지가 없음)"
        lines = [
            f"옮긴 이미지 : {self.moved}장 (라벨 박스 {self.boxes}개)",
            f"옮긴 위치   : {self.dest}",
        ]
        if self.renamed:
            lines.append(f"  이름 충돌로 뒤에 해시를 붙인 파일 {self.renamed}개")
        if self.skipped_missing:
            lines.append(
                f"  원본이 사라져 건너뛴 이미지 {len(self.skipped_missing)}장"
                f" (예: {self.skipped_missing[:2]})"
            )
        if self.errors:
            lines.append(f"  오류 {len(self.errors)}건: {self.errors[:3]}")
        return "\n".join(lines)


@dataclass
class PurgeReport:
    """휴지통 비우기 결과."""

    files_deleted: int = 0
    rows_deleted: int = 0
    boxes: int = 0
    already_gone: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.rows_deleted and not self.errors:
            return "휴지통이 비어 있습니다."
        lines = [
            f"휴지통을 비웠습니다: 이미지 {self.rows_deleted}장 삭제 "
            f"(원본 파일 {self.files_deleted}개, 라벨 박스 {self.boxes}개)"
        ]
        if self.already_gone:
            lines.append(f"  원본이 이미 없던 항목 {self.already_gone}개")
        if self.errors:
            lines.append(f"  파일을 지우지 못함 {len(self.errors)}건: {self.errors[:3]}")
        return "\n".join(lines)


def purge_trashed(
    cfg: Config,
    store: Store,
    *,
    delete_files: bool = True,
    progress: ProgressFn | None = None,
) -> PurgeReport:
    """휴지통(trashed) 이미지를 영구 삭제한다.

    원본 파일과 DB 항목(라벨 포함)을 모두 지운다. 되돌릴 수 없다.
    파일 삭제에 실패해도 DB 항목은 지우지 않아, 다시 시도할 수 있게 남겨둔다.
    """
    report = PurgeReport()
    rows = store.list_images(status="trashed", limit=10**9)
    if not rows:
        return report

    total = len(rows)
    victims: list[int] = []
    for n, row in enumerate(rows, 1):
        src = Path(row["path"])
        if progress:
            progress(n, total, src.name)
        if delete_files:
            if src.exists():
                try:
                    src.unlink()
                    report.files_deleted += 1
                except OSError as exc:
                    report.errors.append(f"{src.name}: {exc}")
                    continue  # 파일이 남았으면 DB 항목도 남겨 재시도 가능하게
            else:
                report.already_gone += 1
        victims.append(int(row["id"]))
        report.boxes += int(row["n_boxes"])

    report.rows_deleted = store.delete_images(victims)
    return report


@dataclass
class PruneReport:
    checked: int = 0
    missing: int = 0
    deleted: int = 0
    boxes: int = 0
    by_status: dict = field(default_factory=dict)
    samples: list[str] = field(default_factory=list)
    aborted: str | None = None

    def summary(self) -> str:
        if self.aborted:
            return f"정리를 중단했습니다: {self.aborted}"
        if not self.missing:
            return f"원본이 사라진 항목 없음 ({self.checked}장 확인)"
        detail = ", ".join(f"{k} {v}" for k, v in sorted(self.by_status.items()))
        verb = "지웠습니다" if self.deleted else "지울 대상입니다"
        lines = [
            f"원본 파일이 없는 DB 항목 {self.missing}개를 {verb} (박스 {self.boxes}개)",
            f"  상태별: {detail}",
        ]
        if self.samples:
            lines.append(f"  예: {self.samples[:3]}")
        return "\n".join(lines)


@dataclass
class CheckReport:
    """DB 가 기억하는 것과 디스크의 실제 파일이 맞는지 본 결과."""

    checked: int = 0
    missing: list[tuple[int, str]] = field(default_factory=list)
    unreadable: list[tuple[int, str, str]] = field(default_factory=list)
    resized: list[tuple[int, str, tuple[int, int], tuple[int, int]]] = field(
        default_factory=list
    )
    fixed: int = 0

    @property
    def ok(self) -> bool:
        return not (self.missing or self.unreadable or self.resized)

    def summary(self) -> str:
        lines = [
            f"이미지 {self.checked}건 확인",
            f"  파일 없음   : {len(self.missing)}",
            f"  열리지 않음 : {len(self.unreadable)}",
            f"  크기 불일치 : {len(self.resized)}"
            + (f" (그중 {self.fixed}건 DB 갱신)" if self.fixed else ""),
        ]
        if self.missing:
            lines.append("\n[파일 없음]  labeler.py prune 으로 정리할 수 있습니다")
            lines += [f"  id={i}  {p}" for i, p in self.missing[:10]]
        if self.unreadable:
            lines.append("\n[열리지 않음]  저장이 깨졌을 수 있습니다")
            lines += [f"  id={i}  {p}\n        {e}" for i, p, e in self.unreadable[:10]]
        if self.resized:
            lines.append("\n[크기 불일치]  DB -> 실제")
            for i, p, was, now in self.resized[:10]:
                same_ratio = abs(was[0] / max(was[1], 1) - now[0] / max(now[1], 1)) < 1e-3
                note = ("종횡비 같음 — 학습에는 영향 없음"
                        if same_ratio else "종횡비 다름 — 박스가 틀어집니다")
                lines.append(f"  id={i}  {was[0]}x{was[1]} -> {now[0]}x{now[1]}  {note}")
                lines.append(f"        {Path(p).name}")
            if not self.fixed:
                lines.append("  --fix-size 로 DB 를 실제 크기에 맞출 수 있습니다")
        for kind, items in (("파일 없음", self.missing),
                            ("열리지 않음", self.unreadable),
                            ("크기 불일치", self.resized)):
            if len(items) > 10:
                lines.append(f"  ({kind} 외 {len(items) - 10}건 더 있음)")
        if self.ok:
            lines.append("\n이상 없음.")
        return "\n".join(lines)


def check_images(cfg: Config, store: Store, *, fix_size: bool = False) -> CheckReport:
    """DB 의 이미지 정보가 실제 파일과 맞는지 전수 확인한다.

    이미지를 밖에서 편집하고 나면(예: 개인정보 블러 처리) 세 가지가 어긋날 수 있다.

    - **경로**: 편집 툴이 확장자를 바꿔 저장하면(.png -> .jpg) DB 참조가 끊긴다.
    - **픽셀 크기**: 자르기 등으로 크기가 달라지면, 박스는 정규화 좌표라 값은
      그대로여도 검수 UI 가 그리는 위치가 어긋난다(app.js 가 DB 의 width/height 로
      화면 좌표를 계산한다). 종횡비까지 달라지면 박스 자체가 틀어진다.
    - **파일 손상**: 저장이 깨지면 학습 중에야 드러난다.

    fix_size=True 면 크기 불일치를 DB 쪽에 맞춘다. 박스 값은 건드리지 않는다 —
    정규화 좌표라 종횡비가 유지되는 한 그대로 유효하다.
    """
    from PIL import Image

    report = CheckReport()
    for row in store.list_images(status="all", limit=10**9):
        report.checked += 1
        path = Path(row["path"])
        if not path.exists():
            report.missing.append((row["id"], str(path)))
            continue
        try:
            with Image.open(path) as im:
                size = im.size
        except Exception as exc:
            report.unreadable.append((row["id"], str(path), f"{type(exc).__name__}: {exc}"))
            continue
        was = (int(row["width"]), int(row["height"]))
        if size != was:
            report.resized.append((row["id"], str(path), was, size))
            if fix_size:
                store.update_size(row["id"], size[0], size[1])
                report.fixed += 1
    return report


def prune_missing(
    cfg: Config,
    store: Store,
    *,
    dry_run: bool = False,
    max_ratio: float = 0.5,
) -> PruneReport:
    """이미지 파일이 사라진 DB 항목을 지운다.

    사람이 탐색기에서 이미지를 지우면 DB에는 그대로 남아 목록과 개수가 어긋난다.

    안전장치: 전체의 max_ratio 를 넘게 사라졌으면 지우지 않고 멈춘다. 드라이브가
    연결되지 않았거나 경로 설정이 틀린 상황에서 DB를 통째로 날리는 것을 막는다.
    """
    report = PruneReport()
    victims: list[int] = []

    for row in store.list_images(status="all", limit=10**9):
        report.checked += 1
        if Path(row["path"]).exists():
            continue
        report.missing += 1
        report.boxes += int(row["n_boxes"])
        st = row["status"]
        report.by_status[st] = report.by_status.get(st, 0) + 1
        if len(report.samples) < 5:
            report.samples.append(row["name"])
        victims.append(int(row["id"]))

    if not victims:
        return report

    if report.checked and report.missing / report.checked > max_ratio:
        report.aborted = (
            f"전체 {report.checked}장 중 {report.missing}장이 사라진 것으로 나옵니다"
            f" ({report.missing / report.checked:.0%}). 드라이브 연결이나 config.yaml 의"
            " 경로 설정을 먼저 확인하세요. 정말 지우려면 --force 를 쓰세요."
        )
        return report

    if not dry_run:
        report.deleted = store.delete_images(victims)
    return report


def _free_name(img_dir: Path, lbl_dir: Path, src: Path) -> tuple[str, Path]:
    """이미지·라벨이 **둘 다 비어 있는** 이름을 고른다. (stem, 이미지 경로)

    라벨까지 보는 이유: `a.jpg` 와 `a.png` 는 이미지 이름은 달라도 라벨이 둘 다 `a.txt` 다.
    겹치면 `원래이름__<원래 경로 해시>` 를 붙이고, 그것도 차 있으면 `_2`, `_3` … 을 더 붙인다
    (같은 경로를 두 번 거쳐 온 파일은 해시가 같다 — 머리말 참고).
    """
    stem, suffix = src.stem, src.suffix
    free = lambda s: not (img_dir / f"{s}{suffix}").exists() and not (lbl_dir / f"{s}.txt").exists()
    if free(stem):
        return stem, img_dir / f"{stem}{suffix}"
    base = f"{stem}__{hashlib.sha1(str(src).encode('utf-8')).hexdigest()[:8]}"
    for n in itertools.count(1):
        s = base if n == 1 else f"{base}_{n}"
        if free(s):
            return s, img_dir / f"{s}{suffix}"
    raise AssertionError("unreachable")


def tidy_approved(
    cfg: Config,
    store: Store,
    *,
    statuses: tuple[str, ...] = ("approved",),
    progress: ProgressFn | None = None,
) -> TidyReport:
    """자동 라벨링 폴더 안의 승인된 이미지를 data/labeled 로 옮긴다.

    이미지 파일을 옮기고, 같은 이름의 라벨 .txt 를 써주고, DB 의 경로를 갱신한다.
    시드 폴더의 이미지는 건드리지 않는다.
    """
    dest_root = cfg.labeled
    img_dir = dest_root / "images"
    lbl_dir = dest_root / "labels"
    report = TidyReport(dest=dest_root)

    unlabeled_root = cfg.unlabeled.resolve()
    targets = []
    for status in statuses:
        for row in store.list_images(status=status, limit=10**9):
            path = Path(row["path"])
            try:
                path.relative_to(unlabeled_root)
            except ValueError:
                continue  # 자동 라벨링 폴더 밖(시드 등)은 그대로 둔다
            targets.append(row)

    if not targets:
        return report

    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    total = len(targets)

    for n, row in enumerate(targets, 1):
        src = Path(row["path"])
        if progress:
            progress(n, total, src.name)
        if not src.exists():
            report.skipped_missing.append(src.name)
            continue

        # 이미지·라벨 이름이 둘 다 비어 있는 곳으로만 옮긴다 (머리말 — 덮어쓰기 사고).
        stem, dst = _free_name(img_dir, lbl_dir, src)
        lbl = lbl_dir / f"{stem}.txt"

        boxes = store.get_boxes(int(row["id"]))
        try:
            if dst.exists():      # _free_name 직후라 있을 수 없지만, 덮어쓰기만은 절대 안 된다
                raise FileExistsError(dst)
            shutil.move(str(src), str(dst))
        except Exception as exc:
            report.errors.append(f"{src.name}: {exc}")
            continue
        try:
            store.update_path(int(row["id"]), dst)
        except Exception as exc:
            # DB 를 못 바꾸면 파일을 제자리로 돌린다. 파일만 옮겨지면 그 행은 '파일 없음' 이 되고
            # 뒤이어 도는 prune 이 행을 라벨째 지운다 (09-27 에 실제로 그렇게 두 장을 잃었다).
            try:
                shutil.move(str(dst), str(src))
            except Exception as exc2:
                # 되돌리지도 못했다 — 파일은 dst, DB 는 옛 경로다. 라벨을 파일 옆에 남기고 여기서
                # 멈춘다(예외가 나가면 웹 UI 작업의 뒤이은 prune 도 돌지 않는다).
                with contextlib.suppress(OSError):
                    write_yolo_txt(lbl, boxes)
                raise RuntimeError(
                    f"{src.name}: DB 경로 갱신({exc})과 되돌리기({exc2})가 모두 실패했습니다. 파일은 "
                    f"{dst} 에 있습니다 — prune 을 돌리기 전에 DB 경로(id {row['id']})를 고칠 것."
                ) from exc2
            report.errors.append(f"{src.name}: DB 경로를 못 바꿔 되돌렸습니다 ({exc})")
            continue
        report.moved += 1
        report.boxes += len(boxes)
        if stem != src.stem:
            report.renamed += 1
        try:
            write_yolo_txt(lbl, boxes)
        except OSError as exc:
            # 라벨 .txt 는 DB 의 사본이라 못 써도 라벨은 남는다. 정리 전체를 멈추지는 않는다.
            report.errors.append(f"{src.name}: 라벨 파일을 못 썼습니다 ({exc})")

    return report
