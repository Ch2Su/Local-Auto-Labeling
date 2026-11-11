"""카메라 키 바꾸기 — 이미지가 있는 카메라도 (2026-11-11, 사용자 요청. 검수 UI 영상 캡처 탭의 '고치기…').

키는 세 곳에 있다. 셋을 함께 바꾼다:
- `cameras.yaml` 의 키 줄 (provenance.edit_camera — 부르는 쪽이 `edit_yaml` 로 넘긴다)
- 카메라 폴더 `unlabeled/<키>/`·`labeled/images/<키>/`·`labeled/labels/<키>/` 와 그 안 파일 이름의 앞머리 `<키>_`
  (capture.frame_name·notime_name, extract_frames, import_roadplus 가 그렇게 짓는다. 앞머리가 다른 이름은 그대로 옮기기만 한다)
- DB `images.camera`·`path`·`name`, `mask_templates.camera`

순서 — 중간에 꺼져도 **DB 행이 없는 파일을 가리키는 때가 없게** 한다(그러면 정리(tidy) 뒤의 prune 이 그 행을 라벨째 지운다):
1. 계획(`plan_rename`): 옮길 파일마다 새 경로. 새 카메라 폴더에 파일이 있거나, DB 에 새 키 이미지가 있거나, 새 경로가 서로
   겹치거나 DB 의 다른 행이 쓰면 아무것도 안 하고 RenameBlocked.
2. DB 를 `backups/<DB 이름>_camrename_<시각>.db` 로 뜬다(하루 사본 정리 대상이 아니다).
3. 새 경로에 **하드링크**를 만든다 — 같은 파일에 이름을 하나 더 다는 것이라 바로 끝나고, 수정 시각·크기가 같아 썸네일 캐시
   (`thumbs/<id>_<크기>_<수정 시각>_<바이트>.jpg`)도 그대로 맞는다. 하드링크가 안 되는 디스크면 복사한다. 하나라도 실패하면
   만든 것을 지우고 멈춘다(아무것도 안 바뀜).
4. DB 를 한 트랜잭션으로 고친다(store.rename_camera_rows — 그사이 경로가 바뀐 행이 있으면 아무것도 안 바꾼다).
5. cameras.yaml 을 고친다(`edit_yaml`). 실패하면 DB 를 되돌리고 3 의 이름을 지운 뒤 그 예외를 그대로 낸다.
6. 옛 이름을 지운다(빈 옛 폴더까지). 다른 프로그램이 열어 둬 못 지운 것은 돌려준다 — 옛 폴더는 등록이 빠져 scan 이 멈추니
   손으로 지울 것(같은 파일의 다른 이름이라 지워도 잃는 것이 없다).
3·6 도중에 꺼지면 한쪽 폴더에 같은 파일의 이름이 하나 더 남을 뿐이다 — 등록 안 된 폴더라 scan 이 멈추고 알린다.

다른 작업(학습·자동 라벨링·정리 등 — 파일 경로를 읽는다)과 겹치지 않게 서버가 작업 큐(jobs)에서 돌리고, 넣기·스캔과 같은
잠금(capman.commit_lock → ingest_lock)을 쥔다. updated_at 은 바꾸지 않는다(store.rename_camera_rows).
"""

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .provenance import is_camera_key

_COPY_MARGIN = 1 << 30          # 하드링크가 안 돼 복사할 때, 복사한 뒤에도 남아야 하는 공간


class RenameBlocked(Exception):
    """바꿀 수 없는 까닭(화면에 그대로) — 아무것도 바꾸지 않았다."""


class RenameStuck(RuntimeError):
    """cameras.yaml 을 못 고친 뒤 DB 도 되돌리지 못했다 — 손으로 맞춰야 한다(메시지에 방법). 그 밖의 실패는 다 되돌린다."""


@dataclass
class RenamePlan:
    old: str
    new: str
    files: list[tuple[Path, Path]] = field(default_factory=list)     # 디스크의 파일 전부 (지금, 새)
    moves: list[tuple[int, str, str]] = field(default_factory=list)  # 경로가 바뀌는 DB 행 (id, 지금, 새)
    camera_rows: int = 0         # camera=old 인 행
    kept: int = 0                # camera=old 인데 카메라 폴더 밖이라 경로는 그대로 둔 행
    renamed: int = 0             # 이름 앞머리를 바꾼 파일
    roots: list[tuple[Path, Path]] = field(default_factory=list)     # (옛 폴더, 새 폴더)


def camera_roots(cfg, key: str) -> list[Path]:
    """그 카메라의 폴더 — 미라벨(scan 이 키를 읽는 곳)과 정리(tidy)가 옮기는 이미지·라벨."""
    labeled = cfg.labeled.resolve()
    return [cfg.unlabeled.resolve() / key, labeled / "images" / key, labeled / "labels" / key]


def new_name(name: str, old: str, new: str) -> str:
    """`<옛 키>_…` → `<새 키>_…`. 앞머리가 다르면 그대로('_' 까지 봐서 cam-1 이 cam-10_… 을 건드리지 않는다)."""
    return new + name[len(old):] if name.startswith(old + "_") else name


def _nk(path) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


def _has_files(d: Path) -> bool:
    return d.is_dir() and any(p.is_file() for p in d.rglob("*"))


def plan_rename(cfg, store, old: str, new: str) -> RenamePlan:
    """옮길 것을 센다 — 아무것도 바꾸지 않는다. 막히면 RenameBlocked."""
    if not is_camera_key(new) or new == old:
        raise RenameBlocked(f"새 키 '{new}' 를 쓸 수 없습니다 — 영문 소문자·숫자·'-' 만, 지금 키와 달라야 합니다")
    plan = RenamePlan(old, new)
    plan.roots = list(zip(camera_roots(cfg, old), camera_roots(cfg, new)))
    for _, n in plan.roots:
        if _has_files(n):
            raise RenameBlocked(f"새 카메라 폴더 {n} 에 이미 파일이 있습니다 — 그 파일을 먼저 옮기거나 다른 키를 고르세요")
    rows = store.image_paths()
    clash = sum(1 for _, _, cam in rows if cam == new)
    if clash:
        raise RenameBlocked(f"DB 에 이미 '{new}' 카메라 이미지가 {clash}장 있습니다(등록 파일에서 빠진 카메라) — 다른 키를 고르세요")

    for o, n in plan.roots:
        if not o.is_dir():
            continue
        for p in sorted(o.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(o)
            name = new_name(p.name, old, new)
            plan.files.append((p, n / rel.parent / name))
            plan.renamed += name != p.name

    by_key = {_nk(src): dst for src, dst in plan.files}
    pairs = [(_nk(o) + os.sep, n) for o, n in plan.roots]
    for i, path, cam in rows:
        plan.camera_rows += cam == old
        dst = by_key.get(_nk(path))
        if dst is None:                       # 파일이 이미 없는 행 — 경로만 같은 규칙으로 (prune 판단은 그대로 '파일 없음')
            norm = os.path.normpath(path)
            for prefix, n in pairs:
                if os.path.normcase(norm).startswith(prefix):
                    rel = Path(norm[len(prefix):])
                    dst = n / rel.parent / new_name(rel.name, old, new)
                    break
        if dst is not None:
            plan.moves.append((i, path, str(dst)))
        elif cam == old:
            plan.kept += 1

    # 새 경로가 겹치면(같은 이름이 둘 — 앞머리 없는 'new_…' 파일이 옛 폴더에 있던 때 등) 막는다
    taken: dict[str, str] = {}
    for src, dst in plan.files:
        k = _nk(dst)
        if k in taken:
            raise RenameBlocked(f"새 이름이 겹칩니다: {src.name} 와 {Path(taken[k]).name} → {dst.name}")
        if os.path.lexists(dst):
            raise RenameBlocked(f"새 경로에 이미 파일이 있습니다: {dst}")
        taken[k] = str(src)
    moving = {i for i, _, _ in plan.moves}
    others = {_nk(p) for i, p, _ in rows if i not in moving}
    for i, _, dst in plan.moves:
        if _nk(dst) in others:
            raise RenameBlocked(f"새 경로를 DB 의 다른 이미지가 이미 씁니다: {dst} (id {i} 를 옮길 자리)")
    return plan


def _link_or_copy(src: Path, dst: Path) -> bool:
    """dst 에 src 의 하드링크를 만든다. 안 되는 디스크면 복사(덮지 않게 'xb', 수정 시각도) — 복사했으면 True."""
    try:
        os.link(src, dst)
        return False
    except FileExistsError:
        raise
    except OSError:
        pass
    try:
        with open(src, "rb") as fi, open(dst, "xb") as fo:
            shutil.copyfileobj(fi, fo, 1 << 20)
        shutil.copystat(src, dst)
    except FileExistsError:
        raise
    except BaseException:
        dst.unlink(missing_ok=True)
        raise
    return True


def _unlink(path: Path, tries: int = 20) -> bool:
    """지운다 — Windows 에서 화면이 그 파일을 받아 가는 중(열려 있다)이면 잠깐 기다렸다 다시. 못 지웠으면 False."""
    for k in range(tries):
        try:
            os.unlink(path)
            return True
        except FileNotFoundError:
            return True
        except PermissionError:
            if k < tries - 1:
                time.sleep(0.05)
    return False


def _prune_empty(root: Path) -> None:
    """root 아래의 빈 폴더와 root 자신(비었으면)을 지운다."""
    if not root.is_dir():
        return
    for d in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        try:
            d.rmdir()
        except OSError:
            pass
    try:
        root.rmdir()
    except OSError:
        pass


def _drop_new(plan: RenamePlan, made: list[Path]) -> None:
    """3 에서 만든 새 이름을 지운다(옛 이름은 그대로라 잃는 것이 없다)."""
    for p in made:
        _unlink(p)
    for _, n in plan.roots:
        _prune_empty(n)


def apply_rename(cfg, store, plan: RenamePlan, *, edit_yaml: Callable[[], object],
                 log: Callable[[str], None] = print) -> dict:
    """계획대로 바꾼다(머리말의 순서). edit_yaml 은 cameras.yaml 의 키를 바꾸는 함수 — 그 결과를 'yaml' 로 돌려준다."""
    t0 = time.monotonic()
    backup = store.backup_to(cfg.db.parent / "backups" / f"{cfg.db.stem}_camrename_{datetime.now():%Y%m%d_%H%M%S}.db")
    log(f"DB 사본: {backup.name}")
    made: list[Path] = []
    copied = 0
    try:
        for k, (src, dst) in enumerate(plan.files, 1):
            dst.parent.mkdir(parents=True, exist_ok=True)
            was_copy = _link_or_copy(src, dst)
            made.append(dst)
            if was_copy:
                copied += 1
                if copied == 1:       # 하드링크가 안 되는 디스크 — 남은 것을 다 복사해도 공간이 남는지
                    rest = sum(s.stat().st_size for s, _ in plan.files[k:])
                    free = shutil.disk_usage(dst.parent).free
                    if free - rest < _COPY_MARGIN:
                        raise OSError(f"하드링크가 안 되는 디스크라 복사해야 하는데 공간이 모자랍니다(남은 {free / 1e9:.1f}GB, "
                                      f"복사할 {rest / 1e9:.1f}GB): {dst.parent}")
            if k % 500 == 0:
                log(f"새 이름 {k}/{len(plan.files)}")
    except BaseException:
        _drop_new(plan, made)
        raise
    try:
        undo = store.rename_camera_rows(plan.moves, plan.old, plan.new)
    except BaseException:
        _drop_new(plan, made)
        raise
    try:
        yaml_result = edit_yaml()
    except BaseException as exc:
        try:
            store.undo_camera_rows(plan.moves, undo, plan.old)
        except Exception as exc2:
            # DB 는 새 키·새 경로, 등록 파일은 옛 키다. 파일은 옛·새 이름이 다 있어 잃은 것은 없다 — 새 이름을 남기고 알린다
            raise RenameStuck(f"cameras.yaml 을 고치지 못했고({exc}) DB 를 되돌리지도 못했습니다({exc2}). 파일은 옛·새 폴더에 다 "
                               f"있습니다 — {plan.old} 의 등록 키를 손으로 {plan.new} 로 바꾸거나 DB 사본 {backup} 을 "
                               "되살릴 것") from exc
        _drop_new(plan, made)
        raise
    left: list[Path] = []
    for src, _ in plan.files:
        if not _unlink(src):
            left.append(src)
    for o, _ in plan.roots:
        _prune_empty(o)
    secs = time.monotonic() - t0
    log(f"{plan.old} → {plan.new}: 파일 {len(plan.files)}개(이름 바꿈 {plan.renamed}, 복사 {copied}), DB 경로 "
        f"{len(plan.moves)}장·카메라 {len(undo['camera_ids'])}장, {secs:.1f}초" + (f", 못 지운 옛 이름 {len(left)}개" if left else ""))
    return {"old": plan.old, "new": plan.new, "files": len(plan.files), "renamed": plan.renamed, "copied": copied,
            "db_paths": len(plan.moves), "db_camera": len(undo["camera_ids"]), "templates": len(undo["template_ids"]),
            "kept": plan.kept, "left": [str(p) for p in left[:5]], "left_n": len(left), "backup": str(backup),
            "seconds": round(secs, 2), "yaml": yaml_result}
