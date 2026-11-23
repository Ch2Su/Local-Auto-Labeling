"""작업 배치 — 한 번에 넣은 묶음을 따로 보고 검수한 뒤 데이터셋에 넣는다 (2026-11-23, 사용자 요청).

**무엇인가.** 이미지를 넣는 길(영상 캡처의 넣기·웹 이미지 올리기·폴더 스캔)마다, 그 한 번이 새로 등록한 미라벨 이미지를
'작업 N'(DB `work_batches` 한 줄 + `images.work_batch`)으로 묶는다. 화면의 `작업 배치` 보기가 배치마다 상태별 장수를
보여 주고, 들어가면 라벨링·이미지 보기가 그 배치로 걸러진다(목록 필터 하나 — `store._list_where` 의 work).

**데이터셋에 넣기.** '데이터셋' 은 지금까지처럼 `status='approved'` 하나다(학습·분할·내보내기·tools 가 이것만 읽는다).
배치마다 두 방식 가운데 하나를 고른다(`work_batches.hold`):
  - hold 0 '바로 추가' — 검수하는 대로 승인(Space = approved). 배치가 없던 때와 같다.
  - hold 1 '모아서 한꺼번에' — 검수한 장은 `reviewed`('검수 끝')로 두고, 배치의 '데이터셋에 추가' 가 reviewed 를 전부
    approved 로 올린다(`store.add_reviewed_to_dataset`). 검수 끝인 장에서 다시 승인하면 그 장만 올라간다.
**사람이 보지 않은 이미지는 어느 길로도 승인되지 않는다** — 한꺼번에 올리는 것도 reviewed 만이다.

**배치가 생기는 때.** `ingest_as_batch` 가 `ingest.ingest` 에 `Intake` 를 넘긴다. Intake 는 새 미라벨 이미지가 **처음 나올 때**
배치를 만든다 — 새 이미지가 없는 스캔은 배치를 만들지 않는다. 영상 캡처·이미지 올리기는 '이번에 옮긴 파일'(mine)만 그
배치에 넣고, 같은 스캔에 함께 걸린 다른 새 파일(사람이 폴더에 따로 둔 것)은 '폴더 스캔' 배치로 따로 묶는다. 시드(곧바로
승인으로 들어간다)와 `tools/import_*` 가 직접 넣는 이미지는 배치에 넣지 않는다('배치 없음').

이미 있던 이미지를 배치로 묶는 것은 `tools/work_batch.py`(기본은 보기만).
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

from .config import Config
from .ingest import IngestReport, ingest
from .store import Store

KIND_NAMES = {"capture": "영상 캡처", "upload": "이미지 올리기", "scan": "폴더 스캔", "manual": "묶음"}


def path_key(path: str | Path) -> str:
    """'이번에 옮긴 파일인가' 를 볼 때의 경로 키 — ingest 가 등록하는 경로(해석된 절대경로)와 같은 꼴로, Windows 처럼
    대소문자를 가리지 않게."""
    return os.path.normcase(str(Path(path).resolve()))


class Intake:
    """한 번의 넣기가 새로 등록하는 미라벨 이미지에 작업 배치 번호를 준다 (`ingest(work=)`).

    kind: 이 넣기의 종류(capture·upload·scan). mine 을 주면(경로들) 그 파일만 이 넣기의 배치('main')에 넣고, 나머지 새
    파일은 '폴더 스캔' 배치('other')로 따로 묶는다. mine 이 None 이면(폴더 스캔) 새 파일 전부가 main 이다.
    배치는 그 몫의 새 이미지가 처음 나올 때 만든다. 끝나면 `finish()` — 만들었는데 한 장도 못 넣은 배치는 지운다.
    """

    def __init__(self, store: Store, kind: str, *, mine: Iterable[str | Path] | None = None,
                 note: str | None = None, name: str | None = None, hold: bool = False):
        self.store, self.kind, self.note, self.name, self.hold = store, kind, note, name, hold
        self.mine = None if mine is None else {path_key(p) for p in mine}
        self.made: dict[str, int] = {}          # 'main' | 'other' -> 배치 번호

    def __call__(self, path: str) -> int:
        which = "main" if self.mine is None or path_key(path) in self.mine else "other"
        if which not in self.made:
            if which == "main":
                self.made[which] = self.store.create_work_batch(self.kind, name=self.name, note=self.note,
                                                                hold=self.hold)
            else:
                self.made[which] = self.store.create_work_batch("scan")
        return self.made[which]

    def finish(self) -> dict:
        """{'main': {id, n} | None, 'other': {id, n} | None} — n 은 그 배치에 실제로 들어간 장수. 0장이면 배치를 지운다
        (스캔이 도중에 멈췄거나, 같은 경로가 그사이 등록돼 넣지 못했다)."""
        out: dict = {"main": None, "other": None}
        for which, batch_id in self.made.items():
            row = self.store.get_work_batch(batch_id)
            n = row["total"] if row else 0
            if n:
                out[which] = {"id": batch_id, "n": n}
            else:
                self.store.delete_work_batch(batch_id, only_if_empty=True)
        return out


def ingest_as_batch(cfg: Config, store: Store, kind: str, *, mine: Iterable[str | Path] | None = None,
                    note: str | None = None, name: str | None = None, hold: bool = False,
                    force: bool = False) -> IngestReport:
    """`ingest.ingest` 를 돌리되 새 미라벨 이미지를 작업 배치로 묶는다. 만든 배치는 report.work 에 남는다
    (`Intake.finish` 의 꼴). **스캔 잠금(server 의 ingest_lock)은 부르는 쪽이 쥔다** — ingest 를 부르는 다른 길과 같다."""
    intake = Intake(store, kind, mine=mine, note=note, name=name, hold=hold)
    report = None
    try:
        report = ingest(cfg, store, force=force, work=intake)
    finally:
        made = intake.finish()          # 도중에 멈췄어도 빈 배치는 남기지 않는다
        if report is not None:
            report.work = made
    return report


def batch_title(row: dict) -> str:
    """로그·요약 줄에 쓰는 한 줄 이름 — '작업 7 (영상 캡처)'. 사람이 붙인 이름·note(영상 파일 이름)는 넣지 않는다:
    장소가 드러날 수 있는 글이라 화면에만 쓴다(카메라 장소와 같은 규칙)."""
    return f"작업 {row['id']} ({KIND_NAMES.get(row['kind'], row['kind'])})"
