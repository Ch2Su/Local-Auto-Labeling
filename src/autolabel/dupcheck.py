"""내용이 같은 이미지(바이트가 같은 파일) 찾기 — 중복 유입 방지 (2026-09-23).

**왜 필요한가.** `ingest` 는 해석된 절대경로로만 '이미 있다' 를 가른다. 로보플로에서 프로젝트를
통째로 다시 내보내면 같은 사진이 새 `.rf.<해시>` 이름으로 새 폴더에 들어오고, 전부 별개로
등록된다 → 학습 데이터 중복, 그리고 같은 사진이 train 과 val·test 에 동시에 들어가는 누수.

**키는 파일 내용 해시다** (실측 근거는 아래). 파일명은 69개 스템이 충돌하는데
내용이 서로 달랐고, dHash 거리 0 은 1,631쌍이 걸렸는데 바이트 중복은 0개(= 전부 오탐, 같은 CCTV
의 인접 프레임)였다. 로보플로 내보내기는 전처리를 켜지 않아 재내보내기해도 바이트가 같다.

**싸게 하는 법.** 먼저 파일 크기로 모으고, 크기가 겹치는 것만 md5 를 구한다. 해시는 DB 에 두지
않는다 — 밖에서 편집(블러)하면 낡은 값이 남기 때문이다. 크기 조회는 6천여 장에 1초 안팎이다.

**박스가 다른 중복**(로보플로에서 라벨을 고쳐 다시 내보낸 경우)은 건너뛸지 갱신할지 사람이
정해야 한다. 그래서 scan 은 **등록하지 않고 알리기만** 한다 — 박스가 다른 것을 따로 센다.
"""

from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .yolo_io import Box


def file_md5(path: Path | str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _size(path: str) -> int | None:
    try:
        return os.stat(path).st_size
    except OSError:
        return None


class DupIndex:
    """DB 에 등록된 이미지의 (파일 크기 -> [행]) 색인. md5 는 크기가 겹칠 때만 구해 기억한다.

    행은 {id, path, status, split, n_boxes} dict. 이번 스캔에서 새로 받은 파일도 `add` 로 넣으면
    새 파일끼리의 중복도 잡는다.
    """

    def __init__(self, rows: list[dict]):
        self.by_size: dict[int, list[dict]] = defaultdict(list)
        self._md5: dict[str, str] = {}
        for r in rows:
            s = _size(r["path"])
            if s is not None:
                self.by_size[s].append(r)

    def md5(self, path: str) -> str:
        if path not in self._md5:
            self._md5[path] = file_md5(path)
        return self._md5[path]

    def find(self, path: str) -> dict | None:
        """내용이 같은 기존 행. 없으면 None. 같은 경로 자신은 짝으로 치지 않는다."""
        s = _size(path)
        if s is None or s not in self.by_size:
            return None
        for r in self.by_size[s]:
            if os.path.normcase(r["path"]) == os.path.normcase(path):
                continue
            if self.md5(r["path"]) == self.md5(path):
                return r
        return None

    def add(self, path: str, row: dict) -> None:
        s = _size(path)
        if s is not None:
            self.by_size[s].append(row)


def boxes_equal(a: list[Box], b: list[Box]) -> bool:
    """클래스와 좌표(소수 4자리)가 순서와 무관하게 같은가."""
    key = lambda bs: sorted((x.cls, round(x.cx, 4), round(x.cy, 4), round(x.w, 4), round(x.h, 4))
                            for x in bs)
    return key(a) == key(b)


@dataclass
class DupReport:
    groups: list[list[dict]] = field(default_factory=list)      # DB 안의 중복 묶음
    folder: list[tuple[str, dict]] = field(default_factory=list)  # (새 파일, 같은 기존 행)
    folder_checked: int = 0
    folder_known: int = 0      # 이미 경로로 등록된 파일

    def summary(self) -> str:
        lines = []
        if self.groups is not None:
            n = sum(len(g) for g in self.groups)
            lines.append(f"DB 안에서 내용이 같은 이미지: {len(self.groups)}묶음 / {n}장")
            for g in self.groups[:30]:
                splits = {r["split"] or "미배정" for r in g}
                leak = " [분할이 갈림 — 누수]" if len(splits) > 1 else ""
                lines.append(f"  - {len(g)}장{leak}")
                for r in g:
                    lines.append(f"      id {r['id']:>6}  {r['status']:<9} {r['split'] or '미배정':<6}"
                                 f" 박스 {r['n_boxes']:>3}  {r['path']}")
            if len(self.groups) > 30:
                lines.append(f"  (외 {len(self.groups) - 30}묶음)")
        if self.folder_checked:
            lines.append(f"\n폴더 검사: {self.folder_checked}장 중 이미 등록 {self.folder_known}장, "
                         f"새 파일 {self.folder_checked - self.folder_known}장 중 기존과 내용이 같은 것 "
                         f"{len(self.folder)}장")
            for p, r in self.folder[:30]:
                lines.append(f"  {p}\n      = id {r['id']} ({r['status']}, {r['split'] or '미배정'}) {r['path']}")
            if len(self.folder) > 30:
                lines.append(f"  (외 {len(self.folder) - 30}장)")
        return "\n".join(lines)


def find_duplicates(rows: list[dict]) -> list[list[dict]]:
    """행들 가운데 파일 내용이 같은 묶음. 크기가 겹치는 것만 해시한다."""
    idx = DupIndex(rows)
    groups = []
    for same_size in idx.by_size.values():
        if len(same_size) < 2:
            continue
        by_hash: dict[str, list[dict]] = defaultdict(list)
        for r in same_size:
            by_hash[idx.md5(r["path"])].append(r)
        groups += [g for g in by_hash.values() if len(g) > 1]
    groups.sort(key=lambda g: min(r["id"] for r in g))
    return groups


def check(store, folder: Path | None = None) -> DupReport:
    """DB 안의 중복과, (주면) 폴더의 새 파일이 기존 이미지와 같은지 본다. 아무것도 쓰지 않는다."""
    from .yolo_io import iter_images

    rows = [{k: r[k] for k in ("id", "path", "status", "split", "n_boxes")}
            for r in store.list_images(status="all", limit=10**9)]
    rep = DupReport(groups=find_duplicates(rows))
    if folder is not None:
        known = {os.path.normcase(r["path"]) for r in rows}
        idx = DupIndex(rows)
        for img in iter_images(Path(folder)):
            p = str(img.resolve())
            rep.folder_checked += 1
            if os.path.normcase(p) in known:
                rep.folder_known += 1
                continue
            hit = idx.find(p)
            if hit is not None:
                rep.folder.append((p, hit))
            else:
                idx.add(p, {"id": None, "path": p, "status": "(이 폴더)", "split": None, "n_boxes": 0})
    return rep
