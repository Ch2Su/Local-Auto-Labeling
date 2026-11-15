"""고른 이미지만 다른 카메라로 옮기기 (2026-11-15, 사용자 요청 — 이미지 보기 오른쪽 칸의 '카메라 옮기기').

카메라 키 바꾸기(camera_rename.py)는 카메라를 **통째로** 새 키로 바꾼다. 이것은 **몇 장만** 이미 등록된 다른 카메라로
옮긴다 — 같은 카메라가 화면을 돌린 동안과 제자리로 돌아온 뒤를 다른 키로 나눠 두는데 한 키로 몰아 넣었을 때처럼.
한 장마다 키가 든 곳을 함께 바꾼다:
- 파일: `unlabeled/<옛 키>/…/<옛 키>_….jpg` → `unlabeled/<새 키>/…/<새 키>_….jpg`(정리된 것은 `labeled/images/<키>/` 와 그 짝
  `labeled/labels/<키>/<이름>.txt` 도). 카메라 폴더 밖에 있는 이미지는 경로를 그대로 두고 DB 의 카메라만 바꾼다.
- DB `images.camera`·`path`·`name`, 촬영 시각(새 카메라의 시간대로 다시 센다 — 다시 셀 수 없으면 그대로), 그리고 **묶음
  `grp` 를 비운다** — 묶음은 카메라째라(grouping.py) 옛 카메라의 묶음 키를 달고 가면 안 된다. 비워 두면 다음 분할이 묶음을
  전부 다시 센다(split.ensure_splits 의 need_group).
- 그 이미지에서 만든 가릴 영역 템플릿의 `camera`(보여 주기만 하는 값).

막는 것(MoveBlocked — 아무것도 안 바꾼다):
- 새 카메라가 등록돼 있지 않다 / 없는 이미지 / 출처(카메라)가 없는 이미지(legacy — 경로·source 규칙이 달라 여기서 다루지 않는다).
- **분할** — 분할이 이미 정해진 이미지를 옮기면 그 카메라가 두 분할에 걸치게 되는 때(새 카메라의 고정 분할이나 이미 있는
  이미지의 분할과 다르다). 분할은 split.py 만 바꾸고 정한 분할은 옮기지 않는다 — 학습한 이미지가 test 카메라로 들어가면
  평가가 샌다. 미배정 이미지는 막지 않는다(다음 분할 때 새 카메라의 묶음을 따라간다).
- 가릴 영역이 있는 이미지를 test 로 고정한 카메라로(server.mask_block 과 같은 규칙).
- 새 경로에 이미 파일이 있거나 DB 의 다른 이미지가 쓰거나, 옮길 것끼리 새 이름이 겹친다.

순서는 camera_rename.py 와 같다 — **DB 행이 없는 파일을 가리키는 때가 없게**: 새 이름을 하드링크로 만들고(안 되는 디스크면
복사) → DB 를 한 트랜잭션으로 고치고(store.move_camera_rows — 그사이 경로·카메라가 바뀐 행이 있으면 아무것도 안 바꾼다) →
옛 이름을 지운다. 앞의 둘 중 하나가 실패하면 만든 이름을 지운다(바뀐 것 없음). 옛 이름을 못 지우면(다른 프로그램이 열어 둠)
알린다 — 그 파일은 옛 카메라 폴더에 남아 다음 scan 이 새 이미지로 보려 하지만 바이트가 같아 등록하지 않는다(dupcheck.py).
몇 장짜리라 DB 를 통째로 뜨지 않고, 옮긴 것(id·옛/새 경로·옛/새 카메라·옛 촬영 시각·옛 묶음)을
`<DB 폴더>/backups/camera_moves.jsonl` 에 한 줄씩 남긴다. updated_at 은 바꾸지 않는다(라벨·그림이 그대로다).

서버가 작업 큐(jobs)에서 돌리고 넣기·스캔과 같은 잠금을 쥔다(카메라 키 바꾸기와 같은 이유 — 파일 경로를 읽는 학습·자동
라벨링·정리와 겹치지 않게).
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .camera_rename import _link_or_copy, _nk, _unlink, camera_roots, new_name
from .provenance import capture_time

LOG_NAME = "camera_moves.jsonl"


class MoveBlocked(Exception):
    """옮길 수 없는 까닭(화면에 그대로) — 아무것도 바꾸지 않았다."""


@dataclass
class MoveItem:
    id: int
    old: str                    # 지금 카메라
    src: str                    # 지금 경로 (DB 글자 그대로)
    dst: str                    # 새 경로 — 카메라 폴더 밖이면 src 그대로
    captured_at: str | None     # 지금 값
    grp: str | None
    has_file: bool = True


@dataclass
class MovePlan:
    new: str
    items: list[MoveItem] = field(default_factory=list)              # 카메라가 바뀌는 이미지
    files: list[tuple[Path, Path]] = field(default_factory=list)     # 디스크에서 옮길 파일 (지금, 새) — 이미지와 라벨 .txt
    same: int = 0               # 이미 그 카메라라 건너뛴 이미지
    kept: int = 0               # 카메라 폴더 밖이라 경로는 그대로 둔 이미지
    no_file: int = 0            # 파일이 없는 행(경로만 같은 규칙으로 바꾼다 — '파일 없음' 판단은 그대로)
    renamed: int = 0            # 이름 앞머리를 바꾼 이미지

    def summary(self) -> dict:
        """화면의 확인 문구에 쓰는 수 — 이름은 앞의 몇 개만."""
        return {"camera": self.new, "moved": len(self.items), "same": self.same, "kept": self.kept,
                "no_file": self.no_file, "renamed": self.renamed, "from": dict(Counter(it.old for it in self.items)),
                "names": [[Path(it.src).name, Path(it.dst).name] for it in self.items[:3]]}


def _ids_text(ids: list[int], limit: int = 8) -> str:
    return ", ".join(str(i) for i in ids[:limit]) + (f" … 모두 {len(ids)}장" if len(ids) > limit else "")


def _split_text(c: dict) -> str:
    return " · ".join(f"{s} {n}장" for s, n in sorted(c.items()))


def _target(cfg, path: str, old: str, new: str) -> tuple[Path, Path | None, Path | None] | None:
    """옛 카메라 폴더 안의 이미지면 (새 경로, 라벨 지금 경로, 라벨 새 경로) — 라벨은 정리된 이미지에만. 폴더 밖이면 None."""
    norm = os.path.normpath(path)
    un_o, img_o, lbl_o = camera_roots(cfg, old)
    un_n, img_n, lbl_n = camera_roots(cfg, new)
    for o, n, labels in ((un_o, un_n, None), (img_o, img_n, (lbl_o, lbl_n))):
        prefix = _nk(o) + os.sep
        if not os.path.normcase(norm).startswith(prefix):
            continue
        rel = Path(norm[len(prefix):])
        name = new_name(rel.name, old, new)
        dst = n / rel.parent / name
        if labels is None:
            return dst, None, None
        return dst, labels[0] / rel.parent / f"{rel.stem}.txt", labels[1] / rel.parent / f"{Path(name).stem}.txt"
    return None


def plan_move(cfg, store, ids: list[int], new: str, cameras: dict) -> MovePlan:
    """옮길 것을 센다 — 아무것도 바꾸지 않는다. 막히면 MoveBlocked. cameras 는 load_cameras 의 결과."""
    target = cameras.get(new)
    if target is None:
        raise MoveBlocked(f"등록돼 있지 않은 카메라입니다: {new} — 영상 캡처 탭의 '＋ 새로' 로 먼저 등록하세요")
    ids = list(dict.fromkeys(int(i) for i in ids))
    rows = store.images_for_move(ids)
    gone = [i for i in ids if i not in rows]
    if gone:
        raise MoveBlocked(f"없는 이미지가 있습니다(id {_ids_text(gone)}) — 목록을 새로 받은 뒤 다시 고르세요")
    bare = [i for i in ids if not rows[i]["camera"]]
    if bare:
        raise MoveBlocked(f"출처(카메라)가 없는 이미지는 옮길 수 없습니다(id {_ids_text(bare)}) — 카메라가 있는 이미지만 고르세요")

    plan = MovePlan(new)
    moving = [rows[i] for i in ids if rows[i]["camera"] != new]
    plan.same = len(ids) - len(moving)

    # 분할 — 정해진 분할은 옮기지 않는다(split.py). 옮긴 뒤 새 카메라가 한 분할에만 있어야 한다
    have = Counter(r["split"] for r in moving if r["split"])
    if have:
        there = store.camera_splits(new)
        if target.split and set(have) != {target.split}:
            raise MoveBlocked(f"분할이 이미 정해진 이미지({_split_text(have)})를 {target.split} 로 고정한 카메라 '{new}' 로 옮길 수 "
                              "없습니다 — 정한 분할은 옮기지 않아 그 카메라가 두 분할에 걸칩니다.")
        if len(set(have) | set(there)) > 1:
            raise MoveBlocked(f"옮기면 카메라 '{new}' 가 두 분할에 걸칩니다(옮길 이미지: {_split_text(have)}"
                              + (f", 그 카메라: {_split_text(there)}" if there else "") + ") — 분할이 정해진 이미지는 같은 "
                              "분할의 카메라로만 옮길 수 있습니다(학습한 이미지가 평가 쪽으로 새지 않게).")
    if target.split == "test":
        masked = [r["id"] for r in moving if r["masked"]]
        if masked:
            raise MoveBlocked(f"가릴 영역이 있는 이미지(id {_ids_text(masked)})는 test 로 고정한 카메라 '{new}' 로 옮길 수 없습니다 — "
                              "test 이미지에는 가릴 영역을 둘 수 없습니다(먼저 지울 것).")

    for r in moving:
        got = _target(cfg, r["path"], r["camera"], new)
        if got is None:
            plan.kept += 1
            plan.items.append(MoveItem(r["id"], r["camera"], r["path"], r["path"], r["captured_at"], r["grp"],
                                       has_file=Path(r["path"]).is_file()))
            continue
        dst, lbl_src, lbl_dst = got
        has_file = Path(r["path"]).is_file()
        plan.items.append(MoveItem(r["id"], r["camera"], r["path"], str(dst), r["captured_at"], r["grp"], has_file=has_file))
        plan.renamed += dst.name != Path(r["path"]).name
        if has_file:
            plan.files.append((Path(r["path"]), dst))
        else:
            plan.no_file += 1
        if lbl_src is not None and lbl_src.is_file():
            plan.files.append((lbl_src, lbl_dst))

    # 새 경로가 겹치거나 이미 있으면 막는다 (다른 카메라에서 온 같은 시각의 프레임은 같은 이름이 된다)
    taken: dict[str, str] = {}
    for src, dst in plan.files:
        k = _nk(dst)
        if k in taken:
            raise MoveBlocked(f"새 이름이 겹칩니다: {src.name} 와 {Path(taken[k]).name} → {dst.name}")
        if os.path.lexists(dst):
            raise MoveBlocked(f"새 경로에 이미 파일이 있습니다: {dst}")
        taken[k] = str(src)
    mine = {it.id for it in plan.items}
    others = {_nk(p) for i, p, _ in store.image_paths() if i not in mine}
    seen: dict[str, int] = {}
    for it in plan.items:
        if it.dst == it.src:
            continue
        k = _nk(it.dst)
        if k in others:
            raise MoveBlocked(f"새 경로를 DB 의 다른 이미지가 이미 씁니다: {it.dst} (id {it.id} 를 옮길 자리)")
        if k in seen:
            raise MoveBlocked(f"새 이름이 겹칩니다: id {seen[k]} 와 id {it.id} → {Path(it.dst).name}")
        seen[k] = it.id
    return plan


def _prune_up(start: Path, roots: list[Path]) -> None:
    """start 부터 위로, 빈 폴더를 지운다 — 카메라 폴더(roots)까지만(그 위는 건드리지 않는다)."""
    tops = {_nk(r) for r in roots}
    d = start
    while any(_nk(d) == t or _nk(d).startswith(t + os.sep) for t in tops):
        try:
            d.rmdir()
        except OSError:
            return
        d = d.parent


def _drop_new(cfg, plan: MovePlan, made: list[Path]) -> None:
    """만든 새 이름을 지운다(옛 이름은 그대로라 잃는 것이 없다)."""
    roots = camera_roots(cfg, plan.new)
    for p in made:
        _unlink(p)
        _prune_up(p.parent, roots)


def apply_move(cfg, store, plan: MovePlan, cameras: dict, *, log: Callable[[str], None] = print) -> dict:
    """계획대로 옮긴다(머리말의 순서). 돌려주는 것: 화면이 줄을 고칠 rows(id·name·camera·captured_at)와 수."""
    t0 = time.monotonic()
    if not plan.items:
        return {**plan.summary(), "rows": [], "copied": 0, "left": [], "left_n": 0, "templates": 0, "seconds": 0.0}
    tz = cameras[plan.new].tz
    made: list[Path] = []
    copied = 0
    try:
        for src, dst in plan.files:
            dst.parent.mkdir(parents=True, exist_ok=True)
            copied += _link_or_copy(src, dst)
            made.append(dst)
        rows = []
        for it in plan.items:
            # 새 이름으로 다시 센다(파일 이름의 시각 → EXIF). 시간대가 같으면 같은 값이고, 다시 셀 수 없으면 그대로 둔다
            at = capture_time(Path(it.dst), tz) or it.captured_at
            rows.append((it.id, it.src, it.dst, it.old, plan.new, at))
        templates = store.move_camera_rows(rows)
    except BaseException:
        _drop_new(cfg, plan, made)
        raise

    # 여기부터는 DB 가 이미 새 경로다 — 예외를 내지 않는다(옛 이름이 남는 것은 알리기만 한다)
    left: list[Path] = []
    olds = sorted({it.old for it in plan.items})
    old_roots = [r for k in olds for r in camera_roots(cfg, k)]
    for src, _ in plan.files:
        try:
            gone = _unlink(src)
        except OSError:
            gone = False
        if gone:
            _prune_up(src.parent, old_roots)
        else:
            left.append(src)
    _write_log(cfg, plan, rows)
    secs = time.monotonic() - t0
    log(f"카메라 옮기기 → {plan.new}: {len(plan.items)}장({', '.join(f'{k} {n}' for k, n in plan.summary()['from'].items())}), "
        f"파일 {len(plan.files)}개(이름 바꿈 {plan.renamed}, 복사 {copied}), {secs:.2f}초"
        + (f", 못 지운 옛 이름 {len(left)}개" if left else ""))
    return {**plan.summary(), "copied": copied, "templates": templates,
            "rows": [{"id": i, "name": Path(dst).name, "camera": new, "captured_at": at} for i, _, dst, _, new, at in rows],
            "left": [str(p) for p in left[:5]], "left_n": len(left), "seconds": round(secs, 2)}


def _write_log(cfg, plan: MovePlan, rows: list[tuple]) -> None:
    """옮긴 것을 남긴다 — 되돌릴 때 볼 것(옛 경로·카메라·촬영 시각·묶음). 못 써도 옮기기는 끝난 것이다."""
    at = datetime.now().isoformat(timespec="seconds")
    before = {it.id: it for it in plan.items}
    try:
        path = cfg.db.parent / "backups" / LOG_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8", newline="\n") as f:
            for i, src, dst, old, new, cap in rows:
                f.write(json.dumps({"at": at, "id": i, "from": old, "to": new, "path_before": src, "path_after": dst,
                                    "captured_before": before[i].captured_at, "captured_after": cap,
                                    "grp_before": before[i].grp}, ensure_ascii=False) + "\n")
    except OSError as exc:
        print(f"[주의] 카메라 옮기기 기록을 남기지 못했습니다({exc}) — 옮기기는 끝났습니다.")
