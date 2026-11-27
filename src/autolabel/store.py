"""SQLite 기반 라벨/검수 상태 저장소.

이미지 한 장의 상태는 다음 중 하나:
  unlabeled : 아직 모델 추론을 돌리지 않음
  pending   : 자동 라벨이 붙었고 사람 검수 대기
  reviewed  : 검수 끝 — 사람이 봤지만 아직 데이터셋에 넣지 않음 (2026-11-23, 작업 배치의 '모아서 한꺼번에')
  approved  : 검수 완료 -> 학습 데이터로 사용 ('데이터셋' 은 이 상태 하나뿐이다)
  skipped   : 사람이 보류/제외
  trashed   : 휴지통. 학습에서 빠지며, 비우기 전까지는 되돌릴 수 있다

**reviewed 를 approved 와 따로 둔 이유.** 학습·분할·내보내기·`tools/` 가 전부 `status='approved'` 만 읽는다 — '검수는
끝났지만 아직 넣지 않은' 이미지를 approved 에 섞으면 그 전부를 고쳐야 한다. pending 에 표시만 다는 방식은
`label --all`(미라벨·검수 대기를 예측으로 덮는다)이 검수 끝낸 박스를 덮는다.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

from .masks import MAX_MASKS, as_mask, drops, from_row, is_poly, row_values, same_rects, to_wire
from .masks import clean as clean_masks
from .masks import merge as merge_masks
from .masks import subtract as subtract_masks
from .yolo_io import Box

STATUSES = ("unlabeled", "pending", "reviewed", "approved", "skipped", "trashed")

# 작업 배치가 어디서 왔나 (work_batches.kind): 영상 캡처의 넣기 · 웹 이미지 올리기 · 폴더 스캔 · 이미 있던 이미지를 묶음
WORK_KINDS = ("capture", "upload", "scan", "manual")
WORK_NAME_MAX = 60       # 작업 배치 이름

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- id 는 반드시 AUTOINCREMENT. 없으면 rowid 별칭이 되어 마지막 행을 지웠을 때
-- 같은 id 가 다음 이미지에 다시 붙는다. 웹 UI 가 이미지를 id 로만 가리키므로
-- (/api/images/{id}/file), 재사용되면 브라우저가 캐시해 둔 '예전 이미지'를
-- 새 이미지 자리에 보여준다. 실제로 겪었다.
CREATE TABLE IF NOT EXISTS images (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    path          TEXT    NOT NULL UNIQUE,
    name          TEXT    NOT NULL,
    width         INTEGER NOT NULL,
    height        INTEGER NOT NULL,
    source        TEXT    NOT NULL,
    status        TEXT    NOT NULL,
    uncertainty   REAL    NOT NULL DEFAULT 0,
    model_version TEXT,
    updated_at    TEXT    NOT NULL,
    -- 학습 배정 ('train'/'val'/'test', 아직 안 정했으면 NULL). split.py 가 정한다.
    -- 여기 두는 이유: 예전처럼 경로 해시로 매번 계산하면 tidy 로 파일을 옮길 때
    -- 배정이 뒤집혀 학습끼리 점수를 비교할 수 없다.
    split         TEXT,
    -- 같은 장면 묶음 키 (grouping.py). 묶음은 통째로 한 split 에만 들어간다.
    grp           TEXT,
    -- 8x8 dHash 16진수. 묶음 계산용이라 한 번 구해 두고 다시 쓴다.
    dhash         TEXT,
    -- 출처 (2026-11-03, docs/provenance.md). camera 는 cameras.yaml 의 키, captured_at 은 시간대를 붙인
    -- ISO 8601. 처음 등록할 때만 쓰고 재스캔이 덮지 않는다. legacy 이미지는 둘 다 NULL(모름).
    camera        TEXT,
    captured_at   TEXT,
    -- 촬영 형태 (2026-11-03): 'cctv' | 'noncctv' | 'drone', 모르면 NULL. legacy 는 출처가 없어 사람이 검수
    -- UI 에서 고른다(DOMAINS). 경로·source 로 가르던 도메인(test_baseline)에 CCTV 아닌 사진이 섞여 있었다.
    domain        TEXT,
    -- 작업 배치 (2026-11-23, 아래 work_batches). 한 번에 넣은 묶음 — NULL = 배치 없음(그 전에 넣은 이미지·도구가 넣은 것).
    -- 라벨이 아니라 updated_at·변경 이력은 건드리지 않는다. 배치를 지우면 NULL 이 된다(이미지는 남는다).
    work_batch    INTEGER REFERENCES work_batches(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_images_status ON images(status, uncertainty DESC, id);

-- 작업 배치 (2026-11-23, 사용자 요청 — workbatch.py). 한 번에 넣은 묶음(영상 캡처의 넣기·웹 이미지 올리기·폴더 스캔)을
-- '작업 N' 으로 따로 보고, 그 안에서 자동/수동 라벨링·검수를 한 뒤 데이터셋(approved)에 넣는다.
-- hold = 1 이면 검수한 장을 곧바로 approved 로 올리지 않고 reviewed('검수 끝')로 모아 두었다가 한꺼번에 올린다
-- (화면이 Space 의 뜻을 이 값으로 정한다 — 서버는 어느 쪽 상태든 받는다). note 는 무엇에서 왔나(영상 파일 이름 등) —
-- 화면에만 쓴다. id 는 AUTOINCREMENT(지운 번호가 새 배치에 붙으면 열려 있던 화면이 엉뚱한 배치를 가리킨다 — 태그와 같은 이유).
-- images.work_batch 는 이 표보다 먼저 만들어지지만 SQLite 는 참조하는 표가 나중에 생겨도 된다.
CREATE TABLE IF NOT EXISTS work_batches (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    kind       TEXT    NOT NULL,
    name       TEXT,
    note       TEXT,
    hold       INTEGER NOT NULL DEFAULT 0,
    created_at TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS boxes (
    id       INTEGER PRIMARY KEY,
    image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    cls      INTEGER NOT NULL,
    cx       REAL    NOT NULL,
    cy       REAL    NOT NULL,
    w        REAL    NOT NULL,
    h        REAL    NOT NULL,
    conf     REAL,
    origin   TEXT    NOT NULL DEFAULT 'auto'
);
CREATE INDEX IF NOT EXISTS idx_boxes_image ON boxes(image_id);

CREATE TABLE IF NOT EXISTS classes (
    idx  INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

-- 박스 변경 이력 (2026-09-23). 박스를 저장할 때마다 '바뀌기 전' 과 '바뀐 뒤' 를 남긴다.
-- 09-21 에 박스 2개가 경고 없이 사라졌을 때 백업과 대조해서야 찾았다 — 그 대조를
-- 이 표 하나로 대신한다. boxes 는 저장 때마다 지우고 다시 넣으므로(행 id 가 바뀐다)
-- 박스 목록을 통째로 JSON 으로 둔다: [[cls, cx, cy, w, h, conf, origin], ...]
-- 2026-11-11 부터 새 줄의 before·after·masks_before·masks_after 는 '#<hist_lists.id>' 다(아래 hist_lists). 그 전 줄은
-- JSON 글자를 그대로 갖고 있고 둘 다 읽는다 — 읽을 때는 _hist_bodies·_hist_text 를 거칠 것.
CREATE TABLE IF NOT EXISTS box_history (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    at       TEXT    NOT NULL,
    source   TEXT    NOT NULL,     -- ui | predict | restore | bulk | revert | crop | mask
    batch    TEXT,                 -- 일괄 수정 한 번을 묶는 키 (같은 자리 일괄 수정)
    before   TEXT    NOT NULL,
    after    TEXT    NOT NULL,
    n_before INTEGER NOT NULL,
    n_after  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_hist_image ON box_history(image_id, id);
CREATE INDEX IF NOT EXISTS idx_hist_batch ON box_history(batch);

-- 변경 이력의 박스·영역 목록 (2026-11-11, 사용자 요청 — 이력 줄이기). 같은 목록(같은 JSON 글자)은 한 번만 두고
-- box_history 가 '#id' 로 가리킨다. 저장마다 박스 목록 전후를 통째로 남기던 때는 박스 95개 장 하나가 32분 작업(저장
-- 338번)에 8.3MB 였다 — 한 줄의 '전' 은 앞 줄의 '후' 와 같고, 영역만 바꾼 저장은 박스 전후가 같아서다. 지울 때는
-- delete_images 가 아무 줄도 가리키지 않는 목록을 함께 지운다. hash 는 body 의 sha1 앞 16자 — 찾는 데만 쓰고 같은지는
-- body 로 본다. id 는 AUTOINCREMENT(지운 번호를 다시 쓰지 않게 — 함정 11 과 같은 이유).
CREATE TABLE IF NOT EXISTS hist_lists (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    hash TEXT    NOT NULL,
    body TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_hist_lists_hash ON hist_lists(hash);

-- 태그 (2026-11-03). 비슷한 이미지끼리 사람이 이름을 붙여 묶는다(같은 카메라·크롭 방식 등 — legacy 는
-- 출처가 없어 grp 만으로는 못 가른다). 한 이미지에 여러 개 붙일 수 있다. 라벨이 아니라서 updated_at·
-- 변경 이력은 건드리지 않는다. id 는 AUTOINCREMENT — 지운 태그의 번호가 새 태그에 붙으면 열려 있던
-- 화면이 엉뚱한 태그를 가리킨다(함정 11 과 같은 이유).
CREATE TABLE IF NOT EXISTS tags (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT    NOT NULL UNIQUE,
    created_at TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS image_tags (
    image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    tag_id   INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (image_id, tag_id)
);
CREATE INDEX IF NOT EXISTS idx_image_tags_tag ON image_tags(tag_id, image_id);

-- 가릴 영역 (2026-11-09, masks.py). 학습 세트를 만들 때만 회색(114)으로 칠할 사각형·다각형 — 원본 그림은 그대로 둔다.
-- 좌표는 박스와 같은 화면(EXIF 회전 적용) 기준 정규화 xyxy. 박스처럼 저장 때마다 통째로 바꾸고, 바뀌면
-- box_history 의 masks_before·masks_after 에 전후를 남긴다(박스만 바뀐 이력은 NULL).
-- poly (2026-11-19, 다각형): 꼭짓점 [[x, y], …] JSON — 이때 x1~y2 는 그 바깥 사각형이다. 사각형은 NULL.
-- 이미 있는 DB 에는 Store 가 열 때 ALTER TABLE 로 더한다(아래 __init__ — 사용자 결정). 읽기·쓰기는 masks.from_row·row_values.
CREATE TABLE IF NOT EXISTS masks (
    id       INTEGER PRIMARY KEY,
    image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    x1       REAL    NOT NULL,
    y1       REAL    NOT NULL,
    x2       REAL    NOT NULL,
    y2       REAL    NOT NULL,
    poly     TEXT
);
CREATE INDEX IF NOT EXISTS idx_masks_image ON masks(image_id);

-- 가릴 영역 템플릿 (2026-11-10, 사용자 요청). 고정 카메라 화면처럼 같은 자리를 여러 장에 덮을 때 이름(예: 카메라A_1)을
-- 붙여 둔 영역 묶음. 적용하면 이미지의 masks 로 **복사**된다 — 템플릿을 고치거나 지워도 이미 적용한 이미지는 그대로다.
-- rects 는 masks 와 같은 화면 기준 정규화 좌표의 JSON 목록 — 사각형 [x1, y1, x2, y2], 다각형 [[x, y], …](2026-11-19,
-- masks.to_wire). width·height 는 만든 이미지의 화면 크기 — 비율이 다른 그림에
-- 잘못 적용하는 것을 막는 데 쓴다. id 는 AUTOINCREMENT(지운 번호가 새 템플릿에 붙지 않게 — 태그와 같은 이유).
CREATE TABLE IF NOT EXISTS mask_templates (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT    NOT NULL UNIQUE,
    rects      TEXT    NOT NULL,
    width      INTEGER,
    height     INTEGER,
    image_id   INTEGER,
    camera     TEXT,
    created_at TEXT    NOT NULL,
    updated_at TEXT    NOT NULL
);
"""

# 이력에 남기는 박스 한 개의 모양. 순서를 바꾸면 옛 이력을 못 읽는다.
_HIST_FIELDS = ("cls", "cx", "cy", "w", "h", "conf", "origin")


def _box_key(b) -> tuple:
    """박스가 '바뀌었나' 를 가르는 값. 부동소수 잡음(1e-9)은 같은 것으로 본다."""
    get = (lambda k: b[k]) if not isinstance(b, Box) else (lambda k: getattr(b, k))
    conf = get("conf")
    return (int(get("cls")), *(round(float(get(k)), 7) for k in ("cx", "cy", "w", "h")),
            None if conf is None else round(float(conf), 6), get("origin"))


def _boxes_json(rows) -> str:
    """박스 목록의 이력 모양. 좌표는 7자리·conf 는 6자리로 반올림하고 빈칸 없이 쓴다(2026-11-11 — 글자가 반쯤으로
    준다). _box_key 와 같은 자리라 이력에서 되돌린 박스도 '바뀌었나' 판정에서는 원래 박스와 같다(반올림을 두 번 해도
    값이 그대로다). 차이는 정규화 5e-8 — 2560px 그림에서 0.0001px 이다."""
    out = []
    for r in rows:
        cls, cx, cy, w, h, conf, origin = ([r[k] for k in _HIST_FIELDS] if not isinstance(r, Box)
                                           else [getattr(r, k) for k in _HIST_FIELDS])
        out.append([cls, *(round(float(v), 7) for v in (cx, cy, w, h)),
                    None if conf is None else round(float(conf), 6), origin])
    return json.dumps(out, separators=(",", ":"))


def _boxes_from_json(text: str) -> list[Box]:
    return [Box(*vals) for vals in json.loads(text)]


def _masks_json(rects) -> str:
    """가릴 영역 목록의 이력 모양: 사각형 [x1, y1, x2, y2], 다각형 [[x, y], …](2026-11-19)의 목록 — 7자리로 반올림, 빈칸
    없이(2026-11-11). 반올림으로 생기는 차(5e-8 까지)는 masks.SAME 안이라, 이력에서 되읽은 영역은 원래 영역과 같은 자리로
    본다(자릿수를 줄이면 SAME 도 볼 것)."""
    return json.dumps([[[round(float(p[0]), 7), round(float(p[1]), 7)] for p in r] if is_poly(r)
                       else [round(float(v), 7) for v in r] for r in rects], separators=(",", ":"))


def _masks_from_json(text: str | None) -> list | None:
    """이력의 영역 목록 → 화면·API 가 쓰는 JSON 꼴(사각형 [x1, y1, x2, y2], 다각형 [[x, y], …]). None 은 None."""
    return None if text is None else [to_wire(as_mask(r)) for r in json.loads(text)]


# masks 표를 읽는 열 — 행은 masks.from_row(*행) 으로 영역이 된다 (2026-11-19 다각형: poly 가 있으면 다각형)
_MASK_COLS = "x1, y1, x2, y2, poly"


def _read_masks(con: sqlite3.Connection, image_id: int) -> list:
    """이 이미지의 가릴 영역 — 넣은 순서대로(masks.py 의 꼴: 사각형 튜플·다각형 점 튜플)."""
    return [from_row(*r) for r in con.execute(
        f"SELECT {_MASK_COLS} FROM masks WHERE image_id=? ORDER BY id", (image_id,))]


def _approved_masks(con: sqlite3.Connection) -> dict[int, list]:
    """승인 이미지의 가릴 영역 {이미지 id: [영역, …]} — 박스를 읽은 그 연결(같은 읽기 시점)로 읽는다."""
    return _masks_where(con, " WHERE i.status='approved'", ())


def _masks_where(con: sqlite3.Connection, where: str, args) -> dict[int, list]:
    """`images i{where}` 에 드는 이미지의 가릴 영역 {이미지 id: [영역, …]} — where 는 목록 조건(`_list_where`)의 꼴이다."""
    out: dict[int, list] = {}
    for r in con.execute(
            "SELECT m.image_id, m.x1, m.y1, m.x2, m.y2, m.poly FROM masks m"
            f" WHERE m.image_id IN (SELECT i.id FROM images i{where}) ORDER BY m.image_id, m.id", list(args)):
        out.setdefault(int(r[0]), []).append(from_row(*tuple(r)[1:]))
    return out


def _masked_out(row: sqlite3.Row, rects: dict[int, list], *, key: str = "image_id") -> bool:
    """이 박스 줄(cls·cx·cy·w·h 와 이미지 id 열 `key`)이 라벨 파일에서 빠지는가 — 그 이미지의 영역이 반 이상 덮는다.
    판정은 학습 세트·내보내기·데이터 점검과 같은 `masks.drops` 하나다(따로 적지 말 것 — masks.py 머리말)."""
    mine = rects.get(int(row[key]))
    return bool(mine) and drops(Box(int(row["cls"]), row["cx"], row["cy"], row["w"], row["h"]), mine)


# 변경 이력의 목록 (2026-11-11 — 위 hist_lists). 새 줄은 '#id', 그 전 줄은 JSON 글자 그대로다.

def _hist_put(con: sqlite3.Connection, body: str | None) -> str | None:
    """목록 글자를 hist_lists 에 (같은 글자가 있으면 그것을) 두고 '#id' 를 돌려준다. None 은 None(영역이 그대로인 줄).

    열린 트랜잭션 안에서 부른다 — 이력 줄과 함께 들어가고 함께 되돌아간다.
    """
    if body is None:
        return None
    h = hashlib.sha1(body.encode("utf-8")).hexdigest()[:16]
    r = con.execute("SELECT id FROM hist_lists WHERE hash=? AND body=?", (h, body)).fetchone()
    lid = r[0] if r is not None else con.execute(
        "INSERT INTO hist_lists(hash, body) VALUES (?,?)", (h, body)).lastrowid
    return f"#{lid}"


_HIST_COLS = ("before", "after", "masks_before", "masks_after")


def _hist_bodies(con: sqlite3.Connection, rows) -> dict[int, str]:
    """이력 줄들이 가리키는 목록 글자 {id: body} — 한 번에(500개씩) 읽는다."""
    ids = sorted({int(v[1:]) for r in rows for k in _HIST_COLS if k in r.keys()
                  and isinstance(v := r[k], str) and v.startswith("#")})
    out: dict[int, str] = {}
    for k in range(0, len(ids), 500):
        part = ids[k:k + 500]
        out.update((int(i), b) for i, b in con.execute(
            f"SELECT id, body FROM hist_lists WHERE id IN ({','.join('?' * len(part))})", part))
    return out


def _hist_text(bodies: dict[int, str], value: str | None) -> str | None:
    """이력 칸의 값을 JSON 글자로 — '#id' 면 목록을 찾아 주고, 옛 줄의 JSON·None 은 그대로."""
    if value is None or not value.startswith("#"):
        return value
    try:
        return bodies[int(value[1:])]
    except KeyError:
        raise RuntimeError(f"변경 이력이 가리키는 목록 {value} 이 없습니다 (hist_lists).") from None


def _hist_gc(con: sqlite3.Connection) -> int:
    """아무 이력 줄도 가리키지 않는 목록을 지운다(이미지를 지운 뒤). 지운 수."""
    refs = " UNION ".join(f"SELECT CAST(substr({c}, 2) AS INTEGER) FROM box_history WHERE {c} LIKE '#%'"
                          for c in _HIST_COLS)
    return con.execute(f"DELETE FROM hist_lists WHERE id NOT IN ({refs})").rowcount


def _iou_xywh(a, b) -> float:
    """정규화 (cx, cy, w, h) 두 개의 IoU."""
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0

_write_lock = threading.Lock()


NO_CAMERA = "__none__"   # 카메라 필터에서 '출처 없음(legacy)' 을 뜻하는 값
DOMAINS = ("cctv", "noncctv", "drone")   # images.domain 에 넣을 수 있는 값 (NULL = 미분류)
NO_DOMAIN = "__none__"   # 도메인 필터에서 '미분류' 를 뜻하는 값
NO_SPLIT = "__none__"    # 분할 필터에서 '미배정'(다음 학습·split 때 배정)을 뜻하는 값
NO_TAG = "__none__"      # 태그 필터에서 '태그 없음' 을 뜻하는 값
NO_WORK = "__none__"     # 작업 배치 필터에서 '배치 없음' 을 뜻하는 값 (2026-11-23)
LIVE = "live"            # 목록의 상태 필터에서 '휴지통만 뺀 전부' — 'all' 은 휴지통도 넣는다 (데이터 점검의 범위 all, 2026-11-15)
TAG_NAME_MAX = 40
TEMPLATE_NAME_MAX = 40   # 가릴 영역 템플릿 이름


class TemplateExists(ValueError):
    """같은 이름의 가릴 영역 템플릿이 있다 — 덮어쓸지 사람에게 물어야 한다(서버 409)."""


@dataclass(frozen=True)
class TagFilter:
    """목록의 태그 조건 (2026-11-04 사용자 요청 — 여러 개를 넣고 빼기).

    `inc` 는 붙어 있어야 할 태그 — `match_any` 가 거짓이면 **전부**(클래스 필터와 같은 교집합), 참이면 하나라도.
    `exc` 는 하나라도 붙어 있으면 빼는 태그. `none` 이면 태그가 하나도 없는 것만이고 `inc`·`exc` 는 보지 않는다.
    """
    inc: tuple[int, ...] = ()
    exc: tuple[int, ...] = ()
    match_any: bool = False
    none: bool = False

    @classmethod
    def of(cls, tag: int | str | TagFilter | None) -> TagFilter | None:
        """예전 꼴(태그 id 하나, NO_TAG)도 받는다."""
        if tag is None or isinstance(tag, TagFilter):
            return tag
        if tag == NO_TAG:
            return cls(none=True)
        return cls(inc=(int(tag),))


# 목록이 한 장마다 받는 열: 박스 수와 붙은 태그 id(쉼표로 이은 문자열, 없으면 NULL)
_LIST_COLS = (
    "i.*, (SELECT COUNT(*) FROM boxes b WHERE b.image_id=i.id) AS n_boxes,"
    " (SELECT GROUP_CONCAT(t.tag_id) FROM image_tags t WHERE t.image_id=i.id) AS tag_ids"
)


def clean_tag_name(name: str) -> str:
    """태그 이름 다듬기. 앞뒤 공백을 떼고, 비었거나 너무 길면 ValueError."""
    name = " ".join(str(name or "").split())
    if not name:
        raise ValueError("태그 이름을 넣어 주세요.")
    if len(name) > TAG_NAME_MAX:
        raise ValueError(f"태그 이름은 {TAG_NAME_MAX}자까지입니다.")
    return name


def clean_template_name(name: str) -> str:
    """가릴 영역 템플릿 이름 다듬기 (태그 이름과 같은 규칙). 비었거나 너무 길면 ValueError."""
    name = " ".join(str(name or "").split())
    if not name:
        raise ValueError("템플릿 이름을 넣어 주세요 (예: 카메라A_1).")
    if len(name) > TEMPLATE_NAME_MAX:
        raise ValueError(f"템플릿 이름은 {TEMPLATE_NAME_MAX}자까지입니다.")
    return name


def clean_work_name(name: str | None) -> str | None:
    """작업 배치 이름 다듬기 — 앞뒤·겹친 공백을 떼고, 비면 None(이름 없음 — 화면이 종류·시각으로 적는다). 너무 길면 ValueError."""
    name = " ".join(str(name or "").split())
    if len(name) > WORK_NAME_MAX:
        raise ValueError(f"작업 배치 이름은 {WORK_NAME_MAX}자까지입니다.")
    return name or None


def _list_where(
    status: str | None, classes: list[int] | None, empty: bool, camera: str | None = None,
    domain: str | None = None, split: str | None = None, tag: int | str | TagFilter | None = None,
    work: int | str | None = None,
) -> tuple[str, list]:
    """목록·개수가 함께 쓰는 WHERE 절. 둘이 어긋나면 묶음 수가 틀어진다.

    클래스 조건은 '전부 가진 것'(교집합)이다 — 고른 클래스 중 **서로 다른 것 몇 개가
    이 이미지에 있는지**를 세어 고른 개수와 같은지 본다. 'A 나 B' 로 하려면 `IN` 만으로
    충분하지만, 검수에서 원하는 것은 '둘 다 있는 장면' 쪽이었다 (2026-09-21 사용자 요청).
    """
    parts: list[str] = []
    args: list = []
    if status == LIVE:               # 휴지통만 뺀 전부 — health_snapshot(approved_only=False) 와 같은 조건이어야 한다
        parts.append("i.status != 'trashed'")
    elif status and status != "all":
        parts.append("i.status=?")
        args.append(status)
    if camera == NO_CAMERA:          # 카메라(출처, docs/provenance.md) — 다른 조건과 AND
        parts.append("i.camera IS NULL")
    elif camera:
        parts.append("i.camera=?")
        args.append(camera)
    if domain == NO_DOMAIN:
        parts.append("i.domain IS NULL")
    elif domain:
        parts.append("i.domain=?")
        args.append(domain)
    if split == NO_SPLIT:            # 분할(train·val·test) — 보기만 한다. 바꾸는 것은 split.py 뿐
        parts.append("i.split IS NULL")
    elif split:
        parts.append("i.split=?")
        args.append(split)
    if work == NO_WORK:              # 작업 배치(한 번에 넣은 묶음, 2026-11-23) — 다른 조건과 AND
        parts.append("i.work_batch IS NULL")
    elif work is not None and work != "":
        parts.append("i.work_batch=?")
        args.append(int(work))
    tf = TagFilter.of(tag)           # 태그 — 넣을 것(전부/하나라도)·뺄 것, 또는 하나도 없는 것 (TagFilter)
    if tf is not None and tf.none:
        parts.append("NOT EXISTS (SELECT 1 FROM image_tags t WHERE t.image_id=i.id)")
    elif tf is not None:
        has = "EXISTS (SELECT 1 FROM image_tags t WHERE t.image_id=i.id AND t.tag_id IN ({}))"
        if tf.inc and (tf.match_any or len(tf.inc) == 1):
            parts.append(has.format(",".join("?" * len(tf.inc))))
            args += list(tf.inc)
        else:                        # 전부 — 태그마다 따로 묻는다
            for t in tf.inc:
                parts.append(has.format("?"))
                args.append(t)
        if tf.exc:
            parts.append("NOT " + has.format(",".join("?" * len(tf.exc))))
            args += list(tf.exc)
    if empty:
        parts.append("NOT EXISTS (SELECT 1 FROM boxes b WHERE b.image_id=i.id)")
    elif classes:
        want = sorted(set(classes))
        marks = ",".join("?" * len(want))
        parts.append(
            f"(SELECT COUNT(DISTINCT b.cls) FROM boxes b"
            f"  WHERE b.image_id=i.id AND b.cls IN ({marks})) = ?"
        )
        args += [*want, len(want)]
    return (" WHERE " + " AND ".join(parts)) if parts else "", args


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def copy_db(src: Path, dst: Path) -> Path:
    """DB 를 dst 로 뜬다(sqlite backup — 다른 연결이 쓰는 중이어도 일관된 사본). **다 뜬 뒤에야 dst 이름이 생긴다**
    (2026-11-23 Gemini 리뷰): 같은 폴더의 `<dst 이름>.tmp` 에 뜨고 연결을 닫은 뒤 바꿔 끼우고, 도중에 실패하면(디스크가 참·
    잠금) 그 임시 파일을 지운다. 전에는 dst 에 바로 떠서 덜 뜬(0바이트일 수도 있는) 파일이 남았다 — 하루 사본이면 그날의
    사본으로 쳐져 그날은 다시 뜨지 않았고, 보관 개수(keep)도 한 칸 차지했다. src 가 없으면 FileNotFoundError(빈 DB 를 만들지
    않는다). 임시 파일 이름(`….db.tmp`)은 하루 사본의 이름 꼴(`….db`)에 걸리지 않는다."""
    src, dst = Path(src), Path(dst)
    if not src.is_file():
        raise FileNotFoundError(f"DB 가 없습니다: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    try:
        con = sqlite3.connect(src, timeout=30.0)
        try:
            out = sqlite3.connect(tmp)
            try:
                con.backup(out)
            finally:
                out.close()
        finally:
            con.close()
        os.replace(tmp, dst)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass                                # 못 지우면 둔다 — `.tmp` 는 사본으로 치지 않는다
        raise
    return dst


def needs_work_batch_migration(db_path: Path) -> bool:
    """이미지가 든 DB 인데 작업 배치 열(images.work_batch)이 아직 없는가 — 읽기 전용으로만 본다(아무것도 만들지 않는다).
    DB 파일이 없거나 비었거나 images 표가 없으면(새 DB) False."""
    db_path = Path(db_path)
    try:
        if not db_path.is_file() or db_path.stat().st_size == 0:
            return False
    except OSError:
        return False

    def look(con: sqlite3.Connection) -> bool:
        try:
            cols = {r[1] for r in con.execute("PRAGMA table_info(images)")}
            if not cols or "work_batch" in cols:
                return False
            return con.execute("SELECT 1 FROM images LIMIT 1").fetchone() is not None
        finally:
            con.close()

    try:
        return look(sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True, timeout=30.0))
    except sqlite3.OperationalError:
        # 읽기 전용으로 못 여는 자리(WAL 색인을 만들 수 없는 폴더 등) — 보통 연결로 읽기만 한다
        return look(sqlite3.connect(db_path, timeout=30.0))


class Store:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # 작업 배치 (2026-11-23, 사용자 결정 — '서버를 새 코드로 켤 때 자동으로, 직전에 사본'). 표 work_batches 와 열
        # images.work_batch 를 더하기 **전에** DB 를 통째로 떠 둔다(`<DB 폴더>/backups/<이름>_before_workbatch_<시각>.db` —
        # 하루 사본의 이름 꼴이 아니라 keep 정리에 지워지지 않는다). 사본을 못 뜨면 DB 를 바꾸지 않고 멈춘다.
        # 이미지가 없는 DB(새 DB·테스트)는 뜨지 않는다. 그 사본의 경로를 migrated_from 에 남긴다(서버가 알린다).
        self.migrated_from: Path | None = None
        if needs_work_batch_migration(self.db_path):
            dst = (self.db_path.parent / "backups"
                   / f"{self.db_path.stem}_before_workbatch_{datetime.now():%Y%m%d_%H%M%S}.db")
            try:
                self.migrated_from = copy_db(self.db_path, dst)
            except Exception as exc:
                raise RuntimeError(
                    f"작업 배치 열을 더하기 전에 DB 사본을 뜨지 못해 DB 를 바꾸지 않았습니다 ({dst}): "
                    f"{type(exc).__name__}: {exc}") from exc
        with self.connect() as con:
            con.executescript(_SCHEMA)
            # 이미 있는 DB 에는 CREATE TABLE IF NOT EXISTS 가 열을 더해 주지 않는다.
            # ALTER TABLE ADD COLUMN 은 기존 행을 다시 쓰지 않아 안전하다.
            cols = {r["name"] for r in con.execute("PRAGMA table_info(images)")}
            for col in ("split", "grp", "dhash", "camera", "captured_at", "domain"):
                if col not in cols:
                    con.execute(f"ALTER TABLE images ADD COLUMN {col} TEXT")
            # 작업 배치 (2026-11-23). 옛 코드는 이 열과 work_batches 표를 모르고 지나간다(넣는 이미지는 배치 없음).
            # 참조가 붙은 열은 기본값이 NULL 일 때만 더할 수 있다 — 그래서 NULL = 배치 없음이다.
            if "work_batch" not in cols:
                con.execute("ALTER TABLE images ADD COLUMN work_batch INTEGER"
                            " REFERENCES work_batches(id) ON DELETE SET NULL")
            # 열이 생긴 뒤에야 만들 수 있어 _SCHEMA 밖에 둔다 (옛 DB 는 executescript 때 아직 열이 없다)
            con.execute("CREATE INDEX IF NOT EXISTS idx_images_work ON images(work_batch)")
            # 가릴 영역의 전후 (2026-11-09). NULL = 그 저장에서 영역은 그대로였다.
            hcols = {r["name"] for r in con.execute("PRAGMA table_info(box_history)")}
            for col in ("masks_before", "masks_after"):
                if col not in hcols:
                    con.execute(f"ALTER TABLE box_history ADD COLUMN {col} TEXT")
            # 가릴 영역의 다각형 꼭짓점 (2026-11-19, 사용자 결정 — 열 때 더한다). NULL = 사각형. 옛 코드는 이 열을 모르고
            # x1~y2(바깥 사각형)만 읽는다 — 옛 코드로 그 장의 영역을 다시 저장하면 다각형이 바깥 사각형으로 바뀐다.
            if "poly" not in {r["name"] for r in con.execute("PRAGMA table_info(masks)")}:
                con.execute("ALTER TABLE masks ADD COLUMN poly TEXT")

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.db_path, timeout=30.0)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys=ON")
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    # ------------------------------------------------------------------ 클래스

    def get_classes(self) -> list[str]:
        with self.connect() as con:
            rows = con.execute("SELECT idx, name FROM classes ORDER BY idx").fetchall()
        return [r["name"] for r in rows]

    def set_classes(self, names: list[str], *, drop_extra: bool = False) -> None:
        """클래스 목록을 앞에서부터 교체한다. 인덱스는 목록 순서를 그대로 따른다.

        기본값은 **목록보다 뒤에 있는 클래스를 지우지 않는다.** 검수 UI 의 '클래스 추가'는
        DB 에만 쓰는데(`add_class`), 예전에는 scan 이 config.yaml 의 목록으로 통째로
        덮어써서 추가한 클래스가 조용히 사라졌다 — 박스는 남아 UI 에 'cls_8' 로 보였다
        (2026-09-20 실제로 겪음). 목록을 진짜로 줄이려면 `drop_extra=True`.
        """
        with _write_lock, self.connect() as con:
            if drop_extra:
                con.execute("DELETE FROM classes")
            else:
                con.execute("DELETE FROM classes WHERE idx < ?", (len(names),))
            con.executemany(
                "INSERT INTO classes(idx, name) VALUES (?, ?)",
                list(enumerate(names)),
            )

    def add_class(self, name: str) -> int:
        """새 클래스를 맨 뒤에 추가하고 인덱스를 돌려준다. 이미 있으면 기존 인덱스."""
        name = name.strip()
        if not name:
            raise ValueError("클래스 이름이 비어 있습니다.")
        with _write_lock, self.connect() as con:
            row = con.execute("SELECT idx FROM classes WHERE name=?", (name,)).fetchone()
            if row:
                return int(row["idx"])
            nxt = con.execute(
                "SELECT COALESCE(MAX(idx)+1, 0) AS n FROM classes"
            ).fetchone()["n"]
            con.execute("INSERT INTO classes(idx, name) VALUES (?, ?)", (int(nxt), name))
            return int(nxt)

    # ------------------------------------------------------------------ 이미지

    def get_image(self, image_id: int) -> dict | None:
        with self.connect() as con:
            row = con.execute("SELECT * FROM images WHERE id=?", (image_id,)).fetchone()
        return dict(row) if row else None

    def image_sizes(self, image_ids: list[int]) -> dict[int, dict]:
        """{id: {id, path, width, height}} — 여러 장을 한 연결로 읽는다(없는 id 는 빠진다). 크기는 DB 의 원시 크기다."""
        ids = list(dict.fromkeys(image_ids))
        out: dict[int, dict] = {}
        with self.connect() as con:
            for k in range(0, len(ids), 500):
                part = ids[k:k + 500]
                for r in con.execute("SELECT id, path, width, height FROM images"
                                     f" WHERE id IN ({','.join('?' * len(part))})", part):
                    out[int(r["id"])] = dict(r)
        return out

    def delete_images(self, image_ids: list[int]) -> int:
        """이미지와 그 박스를 DB에서 지운다. (boxes·변경 이력은 ON DELETE CASCADE)

        이력이 가리키던 목록(hist_lists, 2026-11-11)은 FOREIGN KEY 가 없어 따라 지워지지 않는다 — 같은 트랜잭션에서
        아무 이력 줄도 가리키지 않게 된 목록을 지운다(다른 이미지와 같은 목록이면 남는다).
        """
        if not image_ids:
            return 0
        with _write_lock, self.connect() as con:
            con.executemany("DELETE FROM images WHERE id=?", [(i,) for i in image_ids])
            _hist_gc(con)
        return len(image_ids)

    # 파일을 지우거나 옮기는 작업(휴지통 비우기·정리)은 시작할 때 목록을 한 번 읽고 한참 돈다 — 그사이 사람이 그 이미지를
    # 휴지통에서 되살리거나 승인을 풀 수 있다. 아래 둘은 **쓰기 잠금 안에서** 그 행이 아직 그 상태·그 경로인지 보고, 맞을 때만
    # 파일 일(부르는 쪽이 준 함수)과 DB 일을 함께 한다 (2026-11-27, Gemini 리뷰 묶음 8). 파일 일이 도는 동안 다른 쓰기(상태
    # 바꾸기 포함)는 기다린다 — 한 장의 지우기·옮기기만큼이다.

    def purge_row(self, image_id: int, path: str, unlink: Callable[[], None]) -> bool:
        """휴지통의 한 장을 지운다 — 그 행이 지금도 휴지통이고 경로가 `path` 일 때만 `unlink()` 를 부르고 행을 지운다(True).
        그새 되살렸거나 경로가 바뀌었으면 아무것도 하지 않는다(False). `unlink` 가 예외를 내면 행은 그대로 남는다(다시 시도).
        이력이 가리키던 목록은 지우지 않는다 — 다 끝난 뒤 `gc_history` 를 한 번 부를 것."""
        with _write_lock, self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute("SELECT status, path FROM images WHERE id=?", (image_id,)).fetchone()
            if row is None or row["status"] != "trashed" or row["path"] != path:
                return False
            unlink()
            con.execute("DELETE FROM images WHERE id=?", (image_id,))
        return True

    def gc_history(self) -> int:
        """아무 이력 줄도 가리키지 않는 목록을 지운다(`purge_row` 로 여러 장을 지운 뒤 한 번). 지운 수."""
        with _write_lock, self.connect() as con:
            return _hist_gc(con)

    def move_row(self, image_id: int, src: str, dst: Path, statuses: tuple[str, ...],
                 move: Callable[[], None]) -> list[Box] | None:
        """한 장의 파일을 옮기고 경로를 고친다 — 그 행이 지금도 `statuses` 중 하나이고 경로가 `src` 일 때만 `move()` 를 부르고
        path·name 을 `dst` 로 바꾼다. 돌려주는 것은 **그 시점의** 박스(라벨 파일에 쓸 것), 그새 상태·경로가 바뀌었으면 None
        (아무것도 안 한다). `move` 나 DB 쓰기가 예외를 내면 DB 는 그대로다 — 파일을 이미 옮겼으면 되돌리는 것은 부르는 쪽 몫."""
        p = str(Path(dst).resolve())
        with _write_lock, self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute("SELECT status, path FROM images WHERE id=?", (image_id,)).fetchone()
            if row is None or row["status"] not in statuses or row["path"] != src:
                return None
            boxes = [Box(**dict(r)) for r in con.execute(
                "SELECT cls, cx, cy, w, h, conf, origin FROM boxes WHERE image_id=? ORDER BY id", (image_id,))]
            move()
            con.execute("UPDATE images SET path=?, name=?, updated_at=? WHERE id=?", (p, Path(p).name, _now(), image_id))
        return boxes

    def delete_missing(self, rows: list[tuple[int, str]]) -> int:
        """파일이 없어진 행을 지운다(prune) — (id, 그때 본 경로)마다 **경로가 그대로인 행만**. 그새 다른 곳으로 옮겨진 행
        (경로가 바뀐 행)은 남긴다. 지운 수."""
        if not rows:
            return 0
        with _write_lock, self.connect() as con:
            n = sum(con.execute("DELETE FROM images WHERE id=? AND path=?", (i, path)).rowcount for i, path in rows)
            _hist_gc(con)
        return n

    def update_path(self, image_id: int, new_path: Path) -> None:
        """이미지 파일을 옮긴 뒤 DB의 경로를 갱신한다. 라벨과 검수 이력은 그대로 따라간다."""
        p = str(Path(new_path).resolve())
        with _write_lock, self.connect() as con:
            con.execute(
                "UPDATE images SET path=?, name=?, updated_at=? WHERE id=?",
                (p, Path(p).name, _now(), image_id),
            )

    def update_size(self, image_id: int, width: int, height: int) -> None:
        """이미지를 편집해 픽셀 크기가 달라졌을 때 DB를 맞춘다.

        박스는 정규화 좌표라 값 자체는 그대로 두면 된다. 다만 검수 UI 가 화면 좌표를
        계산할 때 여기 적힌 width/height 를 쓰므로(app.js boxRect), 어긋나면 박스가
        엉뚱한 자리에 그려진다.
        """
        with _write_lock, self.connect() as con:
            con.execute(
                "UPDATE images SET width=?, height=?, updated_at=? WHERE id=?",
                (int(width), int(height), _now(), image_id),
            )

    def set_status(self, image_id: int, status: str) -> None:
        if status not in STATUSES:
            raise ValueError(f"알 수 없는 상태: {status}")
        with _write_lock, self.connect() as con:
            con.execute(
                "UPDATE images SET status=?, updated_at=? WHERE id=?",
                (status, _now(), image_id),
            )

    def counts(self) -> dict[str, int]:
        with self.connect() as con:
            rows = con.execute(
                "SELECT status, COUNT(*) AS n FROM images GROUP BY status"
            ).fetchall()
        out = {s: 0 for s in STATUSES}
        for r in rows:
            out[r["status"]] = int(r["n"])
        out["total"] = sum(out[s] for s in STATUSES)
        return out

    def list_images(
        self,
        *,
        status: str | None = None,
        limit: int = 200,
        offset: int = 0,
        order: str = "uncertainty",
        ids: list[int] | None = None,
        classes: list[int] | None = None,
        empty: bool = False,
        camera: str | None = None,
        domain: str | None = None,
        split: str | None = None,
        tag: int | str | TagFilter | None = None,
        work: int | str | None = None,
    ) -> list[dict]:
        """검수 대상 목록. 기본은 불확실도 높은 순(= 사람이 볼 가치가 큰 순).

        `ids` 를 주면 상태·정렬과 무관하게 그 이미지들만 **준 순서대로** 돌려준다
        (검수 UI 의 'id 로 열기'). 없는 id 는 빠진다.

        `classes` 는 **그 클래스를 전부 가진** 이미지만 남긴다(교집합). `empty` 는 박스가
        하나도 없는 이미지만 남기며, 둘을 같이 주면 `empty` 가 이긴다 — 둘의 교집합은
        언제나 빈 결과라 실수를 조용히 통과시키지 않는다 (2026-09-21 클래스 필터).
        """
        if ids is not None:
            wanted = list(dict.fromkeys(ids))
            if not wanted:
                return []
            marks = ",".join("?" * len(wanted))
            with self.connect() as con:
                rows = con.execute(
                    f"SELECT {_LIST_COLS} FROM images i WHERE i.id IN ({marks})",
                    wanted,
                ).fetchall()
            by_id = {r["id"]: dict(r) for r in rows}
            return [by_id[i] for i in wanted if i in by_id][offset:offset + limit]

        if order == "rare":
            return self._list_rare(status, classes, empty, limit, offset, camera, domain, split, tag, work)
        if order == "tag":
            return self._list_by_tag(status, classes, empty, limit, offset, camera, domain, split, tag, work)

        order_sql = {
            "uncertainty": "i.uncertainty DESC, i.id ASC",
            "name": "i.name ASC",
            "recent": "i.updated_at DESC, i.id DESC",
        }.get(order, "i.uncertainty DESC, i.id ASC")

        where, args = _list_where(status, classes, empty, camera, domain, split, tag, work)
        sql = f"SELECT {_LIST_COLS} FROM images i{where} ORDER BY {order_sql} LIMIT ? OFFSET ?"
        args += [limit, offset]

        with self.connect() as con:
            rows = con.execute(sql, args).fetchall()
        return [dict(r) for r in rows]

    def class_rarity(self) -> dict[int, dict]:
        """클래스마다 **승인 이미지에서 나온 장면 묶음 수**와 그걸로 만든 희소도(0~1].

        박스 수가 아니라 묶음 수로 센다. van 은 박스가 1,966개로 많아 보이지만 83% 가 고정
        카메라 한 대의 같은 두 대라, 묶음으로는 233개로 가장 적다(2026-09-23).
        희소도 = (가장 드문 클래스의 묶음 수) / (이 클래스의 묶음 수). 가장 드문 것이 1 이다.
        아직 승인 박스가 없는 클래스는 1 로 둔다. 묶음이 없는 이미지는 한 장을 한 묶음으로 센다.
        **가릴 영역이 반 이상 덮은 박스는 세지 않는다 (2026-11-25)** — 라벨 파일에 없는 박스다(`approved_class_counts`
        와 같은 판정 `masks.drops`). 넣고 세면 대부분이 영역 아래에 있는 클래스가 실제보다 흔해 보인다.
        """
        with self.connect() as con:
            con.execute("BEGIN")                # 박스와 영역을 같은 시점으로 읽는다
            rows = con.execute(
                "SELECT b.image_id AS image_id, b.cls AS cls, b.cx AS cx, b.cy AS cy, b.w AS w, b.h AS h,"
                " COALESCE(i.grp, 'id' || i.id) AS g"
                " FROM boxes b JOIN images i ON i.id = b.image_id WHERE i.status='approved'"
            ).fetchall()
            rects = _approved_masks(con)
            n_classes = con.execute("SELECT COUNT(*) FROM classes").fetchone()[0]
        seen: dict[int, set[str]] = {}
        for r in rows:
            if not _masked_out(r, rects):
                seen.setdefault(int(r["cls"]), set()).add(r["g"])
        groups = {c: len(g) for c, g in seen.items()}
        least = min(groups.values(), default=1)
        return {c: {"groups": groups.get(c, 0),
                    "weight": (least / groups[c]) if groups.get(c) else 1.0}
                for c in range(max(n_classes, max(groups, default=-1) + 1))}

    def _list_rare(self, status, classes, empty, limit: int, offset: int,
                   camera: str | None = None, domain: str | None = None,
                   split: str | None = None, tag: int | str | TagFilter | None = None,
                   work: int | str | None = None) -> list[dict]:
        """'드문 클래스 먼저' 정렬. 이미지 점수 = 박스마다 (신뢰도 × 그 클래스의 희소도) 중 최댓값.

        자동 라벨이 van·pickup 처럼 드문 클래스를 확신하는 이미지가 맨 앞에 온다 — 검수할
        이미지가 많을 때 **다른 장면의 드문 클래스**부터 채우려는 것이다. 사람이 그린 박스
        (conf 없음)는 신뢰도 1 로 본다. 점수가 같으면 불확실한 순.
        목록 묶음(100개씩…)과 맞도록 같은 조건의 전체를 점수로 줄 세운 뒤 잘라 준다.
        **가릴 영역이 반 이상 덮은 박스는 점수에 넣지 않는다 (2026-11-27, Gemini 리뷰 묶음 7)** — 라벨 파일에 없을 박스다
        (희소도 `class_rarity` 와 같은 판정). 넣고 매기면 템플릿으로 덮어 둔 자리의 헛 박스 하나로 그 이미지가 맨 앞에 온다.
        이미지·박스·영역은 한 시점으로 읽는다(희소도는 그 앞에 따로 읽는다 — 가중치일 뿐이라 어긋나도 순서만 조금 다르다).
        """
        rarity = self.class_rarity()
        where, args = _list_where(status, classes, empty, camera, domain, split, tag, work)
        with self.connect() as con:
            con.execute("BEGIN")
            rows = con.execute(f"SELECT {_LIST_COLS} FROM images i{where}", args).fetchall()
            found = con.execute(
                "SELECT bx.image_id AS image_id, bx.cls AS cls, bx.conf AS conf,"
                " bx.cx AS cx, bx.cy AS cy, bx.w AS w, bx.h AS h FROM boxes bx"
                f" WHERE bx.image_id IN (SELECT i.id FROM images i{where})", args,
            ).fetchall()
            rects = _masks_where(con, where, args)
        boxes = [(r["image_id"], r["cls"], r["conf"]) for r in found if not _masked_out(r, rects)]
        best: dict[int, tuple[float, int]] = {}
        for image_id, cls, conf in boxes:
            w = rarity.get(cls, {"weight": 1.0})["weight"]
            s = (1.0 if conf is None else float(conf)) * w
            if s > best.get(image_id, (-1.0, -1))[0]:
                best[image_id] = (s, cls)
        out = []
        for r in rows:
            d = dict(r)
            s, c = best.get(d["id"], (0.0, None))
            d["rare_score"] = round(s, 4)
            d["rare_cls"] = c
            d["rare_groups"] = rarity[c]["groups"] if c is not None and c in rarity else None
            out.append(d)
        out.sort(key=lambda d: (-d["rare_score"], -d["uncertainty"], d["id"]))
        return out[offset:offset + limit]

    def _list_by_tag(self, status, classes, empty, limit: int, offset: int,
                     camera: str | None = None, domain: str | None = None,
                     split: str | None = None, tag: int | str | TagFilter | None = None,
                     work: int | str | None = None) -> list[dict]:
        """'태그 순' 정렬 (2026-11-10 사용자 요청). 이미지의 태그를 **넓은 것부터** 이어 붙인 이름으로 줄 세운다.

        넓은 것 = 붙은 이미지가 많은 것(`list_tags` 의 n, 같으면 이름순) — 태그는 나라·도시·장소, 영상 종류처럼 넓고
        좁은 것이 섞여 붙어서, 넓은 것부터 견주면 같은 나라 안에 같은 도시·장소가 모인다(좁은 것부터 견주면 한 장소의
        이미지가 다른 태그 하나 더 붙은 것만으로 멀리 갈린다). 같은 태그끼리는 이름순(연번·촬영 시각), 태그 없는 이미지는
        맨 뒤. 목록 묶음(100개씩…)과 맞도록 같은 조건의 전체를 줄 세운 뒤 잘라 준다('드문 클래스 먼저' 와 같다).
        """
        tags = {t["id"]: (t["n"], t["name"].casefold()) for t in self.list_tags()}
        where, args = _list_where(status, classes, empty, camera, domain, split, tag, work)
        with self.connect() as con:
            rows = [dict(r) for r in con.execute(f"SELECT {_LIST_COLS} FROM images i{where}", args).fetchall()]

        def key(d: dict) -> tuple:
            ids = [int(t) for t in d["tag_ids"].split(",")] if d.get("tag_ids") else []
            info = sorted((tags.get(t, (0, str(t))) for t in ids), key=lambda x: (-x[0], x[1]))
            return (not info, [name for _, name in info], d["name"].casefold(), d["id"])

        rows.sort(key=key)
        return rows[offset:offset + limit]

    def count_images(
        self,
        *,
        status: str | None = None,
        classes: list[int] | None = None,
        empty: bool = False,
        camera: str | None = None,
        domain: str | None = None,
        split: str | None = None,
        tag: int | str | TagFilter | None = None,
        work: int | str | None = None,
    ) -> int:
        """`list_images` 와 같은 조건으로 개수만 센다.

        클래스로 거르면 상태별 개수(`counts()`)를 쓸 수 없어서 따로 센다. 목록 UI 는
        이 수로 묶음(100개씩…)을 나누므로 필터와 정확히 같은 조건이어야 한다.
        """
        where, args = _list_where(status, classes, empty, camera, domain, split, tag, work)
        with self.connect() as con:
            return int(con.execute(f"SELECT COUNT(*) FROM images i{where}", args).fetchone()[0])

    def domain_counts(self) -> dict[str | None, int]:
        """도메인별 이미지 수(휴지통 제외). 미분류는 키 None."""
        with self.connect() as con:
            rows = con.execute(
                "SELECT domain, COUNT(*) AS n FROM images WHERE status != 'trashed' GROUP BY domain"
            ).fetchall()
        return {r["domain"]: int(r["n"]) for r in rows}

    def split_counts(self) -> dict[str | None, int]:
        """분할별 이미지 수(휴지통 제외). 미배정은 키 None."""
        with self.connect() as con:
            rows = con.execute(
                "SELECT split, COUNT(*) AS n FROM images WHERE status != 'trashed' GROUP BY split"
            ).fetchall()
        return {r["split"]: int(r["n"]) for r in rows}

    # ── 태그 (2026-11-03) ── 라벨이 아니라 updated_at·변경 이력은 건드리지 않는다

    def list_tags(self) -> list[dict]:
        """태그 목록(이름순)과 붙은 이미지 수(휴지통 제외)."""
        with self.connect() as con:
            rows = con.execute(
                "SELECT t.id, t.name, (SELECT COUNT(*) FROM image_tags it JOIN images i ON i.id=it.image_id"
                "  WHERE it.tag_id=t.id AND i.status != 'trashed') AS n"
                " FROM tags t ORDER BY t.name COLLATE NOCASE, t.id"
            ).fetchall()
        return [dict(r) for r in rows]

    def untagged_count(self) -> int:
        """태그가 하나도 없는 이미지 수(휴지통 제외)."""
        with self.connect() as con:
            return int(con.execute(
                "SELECT COUNT(*) FROM images i WHERE i.status != 'trashed'"
                " AND NOT EXISTS (SELECT 1 FROM image_tags t WHERE t.image_id=i.id)").fetchone()[0])

    def create_tag(self, name: str) -> int:
        """새 태그. 이름이 겹치면 ValueError (대소문자는 가린다)."""
        name = clean_tag_name(name)
        with _write_lock, self.connect() as con:
            if con.execute("SELECT 1 FROM tags WHERE name=?", (name,)).fetchone():
                raise ValueError(f"같은 이름의 태그가 있습니다: {name}")
            return int(con.execute("INSERT INTO tags(name, created_at) VALUES(?, ?)",
                                   (name, _now())).lastrowid)

    def rename_tag(self, tag_id: int, name: str) -> bool:
        """태그 이름을 바꾼다. 없는 태그면 False, 다른 태그와 이름이 겹치면 ValueError."""
        name = clean_tag_name(name)
        with _write_lock, self.connect() as con:
            other = con.execute("SELECT id FROM tags WHERE name=?", (name,)).fetchone()
            if other and int(other["id"]) != tag_id:
                raise ValueError(f"같은 이름의 태그가 있습니다: {name}")
            return con.execute("UPDATE tags SET name=? WHERE id=?", (name, tag_id)).rowcount > 0

    def delete_tag(self, tag_id: int) -> int | None:
        """태그를 지운다(붙어 있던 것도 함께 떨어진다 — 이미지는 그대로). 떨어진 이미지 수, 없는 태그면 None."""
        with _write_lock, self.connect() as con:
            if not con.execute("SELECT 1 FROM tags WHERE id=?", (tag_id,)).fetchone():
                return None
            n = int(con.execute("SELECT COUNT(*) FROM image_tags WHERE tag_id=?", (tag_id,)).fetchone()[0])
            con.execute("DELETE FROM image_tags WHERE tag_id=?", (tag_id,))
            con.execute("DELETE FROM tags WHERE id=?", (tag_id,))
            return n

    def image_tag_ids(self, image_id: int) -> list[int]:
        with self.connect() as con:
            return [int(r[0]) for r in con.execute(
                "SELECT tag_id FROM image_tags WHERE image_id=? ORDER BY tag_id", (image_id,))]

    def set_image_tag(self, image_ids: list[int], tag_id: int, on: bool) -> int:
        """이미지들에 태그를 붙이거나(on) 뗀다. 실제로 바뀐 이미지 수. 없는 태그면 ValueError.

        없는 이미지 id(다른 창에서 지운 것 등)는 건너뛴다 (2026-11-05) — `INSERT OR IGNORE` 는 FK 위반을 무시하지
        않아서, 예전에는 한 장만 없어도 통째로 롤백되고 /api/bulk/tag 가 500 이었다. 그래서 있는 이미지에서만 붙인다.
        """
        if not image_ids:
            return 0
        with _write_lock, self.connect() as con:
            if not con.execute("SELECT 1 FROM tags WHERE id=?", (tag_id,)).fetchone():
                raise ValueError("없는 태그입니다(다른 창에서 지웠을 수 있습니다).")
            sql = ("INSERT OR IGNORE INTO image_tags(image_id, tag_id) SELECT id, ? FROM images WHERE id=?" if on
                   else "DELETE FROM image_tags WHERE tag_id=? AND image_id=?")
            before = con.total_changes
            con.executemany(sql, [(tag_id, i) for i in image_ids])
            return con.total_changes - before

    def set_domain(self, image_ids: list[int], domain: str | None) -> int:
        """도메인을 정한다(None = 미분류로 되돌림). 라벨이 아니라서 updated_at 은 건드리지 않는다 —
        '최근 수정순' 목록이 분류하는 동안 뒤섞이지 않게. 바꾼 행 수를 돌려준다."""
        if domain is not None and domain not in DOMAINS:
            raise ValueError(f"알 수 없는 도메인: {domain}")
        if not image_ids:
            return 0
        with _write_lock, self.connect() as con:
            cur = con.executemany("UPDATE images SET domain=? WHERE id=?", [(domain, i) for i in image_ids])
            return cur.rowcount

    def camera_counts(self, with_trashed: bool = False) -> dict[str | None, int]:
        """카메라별 이미지 수(기본은 휴지통 제외). 출처 없는 이미지는 키 None.
        with_trashed — 카메라 등록을 지워도 되는지 볼 때(2026-11-10): 휴지통 이미지도 그 카메라를 가리킨다."""
        where = "" if with_trashed else "WHERE status != 'trashed'"
        with self.connect() as con:
            rows = con.execute(f"SELECT camera, COUNT(*) AS n FROM images {where} GROUP BY camera").fetchall()
        return {r["camera"]: int(r["n"]) for r in rows}

    def camera_image_stats(self) -> dict[str, dict]:
        """카메라 등록을 고치거나 지울 때 보는 수 (2026-11-10, 휴지통 포함): {카메라: {db 이미지 수, assigned 분할이 정해진
        수, masked 가릴 영역이 있는 수}}. 분할이 정해진 이미지가 있으면 고정 분할을 바꿀 수 없다(split.py 는 정한 분할을
        옮기지 않아 같은 카메라가 두 분할로 갈린다)."""
        with self.connect() as con:
            rows = con.execute(
                "SELECT camera, COUNT(*) AS n, SUM(split IS NOT NULL) AS assigned,"
                " SUM(EXISTS(SELECT 1 FROM masks m WHERE m.image_id = images.id)) AS masked"
                " FROM images WHERE camera IS NOT NULL GROUP BY camera"
            ).fetchall()
        return {r["camera"]: {"db": int(r["n"]), "assigned": int(r["assigned"] or 0), "masked": int(r["masked"] or 0)}
                for r in rows}

    def camera_images(self, camera: str) -> list[dict]:
        """그 카메라의 이미지 (id, path, captured_at) — 휴지통 포함. 카메라 시간대를 고친 뒤 촬영 시각을 다시 셀 때 쓴다."""
        with self.connect() as con:
            rows = con.execute("SELECT id, path, captured_at FROM images WHERE camera=? ORDER BY id", (camera,)).fetchall()
        return [dict(r) for r in rows]

    def set_captured_at(self, pairs: list[tuple[int, str | None]]) -> int:
        """촬영 시각만 바꾼다 — 라벨이 아니라 updated_at 은 건드리지 않는다(set_domain 과 같은 이유, 자동 라벨링의
        (상태, updated_at) 확인도 깨지 않는다). 바꾼 행 수."""
        if not pairs:
            return 0
        with _write_lock, self.connect() as con:
            cur = con.executemany("UPDATE images SET captured_at=? WHERE id=?", [(v, i) for i, v in pairs])
            return cur.rowcount

    def image_paths(self) -> list[tuple[int, str, str | None]]:
        """모든 이미지의 (id, path, camera) — 카메라 키 바꾸기(camera_rename.py)가 옮길 행과 겹치는 경로를 찾는다."""
        with self.connect() as con:
            return [(int(r[0]), r[1], r[2]) for r in con.execute("SELECT id, path, camera FROM images ORDER BY id")]

    def rename_camera_rows(self, moves: list[tuple[int, str, str]], old: str, new: str) -> dict:
        """카메라 키 바꾸기 (2026-11-11, camera_rename.py) — 한 트랜잭션: moves (id, 지금 경로, 새 경로)의 path·name,
        camera=old 인 행과 가릴 영역 템플릿의 camera. 지금 경로가 그사이 바뀐 행이 하나라도 있으면 아무것도 안 바꾸고
        RuntimeError. updated_at 은 그대로다 — 라벨·그림이 아니라 이름만 바뀐다(set_captured_at 과 같은 이유: 최근 수정순·
        자동 라벨링의 (상태, updated_at) 확인을 흔들지 않는다. 썸네일 캐시는 파일의 수정 시각·크기로 맞춘다).
        돌려주는 것: 되돌릴 때 쓰는 {camera_ids, template_ids}."""
        with _write_lock, self.connect() as con:
            con.execute("BEGIN IMMEDIATE")      # 읽고 쓰는 사이에 다른 연결이 끼지 않게 처음부터 쓰기 잠금
            for i, src, dst in moves:
                cur = con.execute("UPDATE images SET path=?, name=? WHERE id=? AND path=?", (dst, Path(dst).name, i, src))
                if cur.rowcount != 1:
                    raise RuntimeError(f"이미지 id {i} 의 경로가 그사이 바뀌었습니다 — 아무것도 바꾸지 않았습니다. 다시 하세요.")
            cams = [int(r[0]) for r in con.execute("UPDATE images SET camera=? WHERE camera=? RETURNING id", (new, old))]
            tpls = [int(r[0]) for r in con.execute("UPDATE mask_templates SET camera=? WHERE camera=? RETURNING id",
                                                   (new, old))]
        return {"camera_ids": cams, "template_ids": tpls}

    def undo_camera_rows(self, moves: list[tuple[int, str, str]], undo: dict, old: str) -> None:
        """rename_camera_rows 를 되돌린다(그 뒤 cameras.yaml 을 못 고쳤을 때) — 경로는 새 경로인 행만 옛 경로로."""
        with _write_lock, self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            con.executemany("UPDATE images SET path=?, name=? WHERE id=? AND path=?",
                            [(src, Path(src).name, i, dst) for i, src, dst in moves])
            con.executemany("UPDATE images SET camera=? WHERE id=?", [(old, i) for i in undo["camera_ids"]])
            con.executemany("UPDATE mask_templates SET camera=? WHERE id=?", [(old, i) for i in undo["template_ids"]])

    def images_for_move(self, image_ids: list[int]) -> dict[int, dict]:
        """고른 이미지만 다른 카메라로 옮길 때(camera_move.py, 2026-11-15) 보는 것 — {id: {id, path, camera, split,
        captured_at, grp, masked 가릴 영역 수}}. 없는 id 는 빠진다. 휴지통 이미지도 본다(그 파일도 카메라 폴더에 있다)."""
        ids = list(dict.fromkeys(image_ids))
        out: dict[int, dict] = {}
        with self.connect() as con:
            for k in range(0, len(ids), 500):
                part = ids[k:k + 500]
                for r in con.execute(
                        "SELECT id, path, camera, split, captured_at, grp,"
                        " (SELECT COUNT(*) FROM masks m WHERE m.image_id = images.id) AS masked"
                        f" FROM images WHERE id IN ({','.join('?' * len(part))})", part):
                    out[int(r["id"])] = dict(r)
        return out

    def camera_splits(self, camera: str) -> dict[str, int]:
        """그 카메라 이미지의 분할별 장수(휴지통 포함, 미배정은 뺀다) — 분할이 정해진 이미지를 옮겨 넣어도 되는지 본다."""
        with self.connect() as con:
            rows = con.execute("SELECT split, COUNT(*) AS n FROM images WHERE camera=? AND split IS NOT NULL"
                               " GROUP BY split", (camera,)).fetchall()
        return {r["split"]: int(r["n"]) for r in rows}

    def move_camera_rows(self, rows: list[tuple[int, str, str, str, str, str | None]]) -> int:
        """고른 이미지를 다른 카메라로 (2026-11-15, camera_move.py) — 한 트랜잭션: (id, 지금 경로, 새 경로, 지금 카메라,
        새 카메라, 새 촬영 시각)마다 path·name·camera·captured_at 을 바꾸고 **grp 를 비운다**(묶음은 카메라째라 옛 카메라의
        묶음 키를 달고 가면 안 된다 — 비워 두면 다음 분할이 묶음을 다시 센다). 지금 경로·카메라가 그사이 바뀐 행이 하나라도
        있으면 아무것도 안 바꾸고 RuntimeError. updated_at 은 그대로다(rename_camera_rows 와 같은 이유). 그 이미지에서 만든
        가릴 영역 템플릿의 camera 도 따라 바꾼다 — 돌려주는 것은 그렇게 바꾼 템플릿 수."""
        if not rows:
            return 0
        with _write_lock, self.connect() as con:
            con.execute("BEGIN IMMEDIATE")      # 읽고 쓰는 사이에 다른 연결이 끼지 않게 처음부터 쓰기 잠금
            templates = 0
            for i, src, dst, old, new, captured_at in rows:
                cur = con.execute("UPDATE images SET path=?, name=?, camera=?, captured_at=?, grp=NULL"
                                  " WHERE id=? AND path=? AND camera=?", (dst, Path(dst).name, new, captured_at, i, src, old))
                if cur.rowcount != 1:
                    raise RuntimeError(f"이미지 id {i} 의 경로나 카메라가 그사이 바뀌었습니다 — 아무것도 바꾸지 않았습니다. 다시 하세요.")
                templates += con.execute("UPDATE mask_templates SET camera=? WHERE image_id=? AND camera=?",
                                         (new, i, old)).rowcount
        return templates

    def backup_to(self, dst: Path) -> Path:
        """DB 를 dst 로 뜬다(`copy_db` — 서버가 쓰는 중이어도 일관된 사본, 다 뜬 뒤에야 dst 가 생긴다). 카메라 키 바꾸기처럼
        행을 많이 바꾸기 전에."""
        return copy_db(self.db_path, dst)

    def next_image_id(self, current_id: int, status: str = "pending") -> int | None:
        """현재 이미지 다음으로 검수할 이미지. 목록과 같은 정렬 기준을 쓴다."""
        with self.connect() as con:
            cur = con.execute(
                "SELECT uncertainty FROM images WHERE id=?", (current_id,)
            ).fetchone()
            if cur is None:
                return None
            row = con.execute(
                "SELECT id FROM images WHERE status=?"
                " AND (uncertainty < ? OR (uncertainty = ? AND id > ?))"
                " ORDER BY uncertainty DESC, id ASC LIMIT 1",
                (status, cur["uncertainty"], cur["uncertainty"], current_id),
            ).fetchone()
            if row:
                return int(row["id"])
            # 끝까지 갔으면 맨 앞으로 되돌아간다(현재 이미지는 제외).
            row = con.execute(
                "SELECT id FROM images WHERE status=? AND id<>?"
                " ORDER BY uncertainty DESC, id ASC LIMIT 1",
                (status, current_id),
            ).fetchone()
        return int(row["id"]) if row else None

    # ------------------------------------------------------------ 작업 배치
    # 한 번에 넣은 묶음 (2026-11-23, 사용자 요청 — workbatch.py 머리말). 표 work_batches + images.work_batch.
    # '일괄 수정 묶음'(box_history.batch·아래 batches())과는 다른 것이다 — 그래서 이름을 work 로 쓴다.

    def create_work_batch(self, kind: str, *, name: str | None = None, note: str | None = None,
                          hold: bool = False) -> int:
        """새 작업 배치를 만들고 번호를 돌려준다. 이미지는 넣을 때(ingest_batch 의 work_batch)나 set_work_batch 로 붙인다."""
        if kind not in WORK_KINDS:
            raise ValueError(f"알 수 없는 작업 배치 종류: {kind}")
        name = clean_work_name(name)
        note = " ".join(str(note or "").split())[:200] or None
        with _write_lock, self.connect() as con:
            cur = con.execute("INSERT INTO work_batches(kind, name, note, hold, created_at) VALUES (?,?,?,?,?)",
                              (kind, name, note, int(bool(hold)), _now()))
            return int(cur.lastrowid)

    @staticmethod
    def _work_rows(con: sqlite3.Connection, where: str = "", args: tuple = ()) -> list[dict]:
        """배치 줄 + 상태별 장수 + 카메라별 장수(휴지통 제외, 많은 것부터). 새 배치가 앞이다."""
        rows = [dict(r) for r in con.execute(
            f"SELECT id, kind, name, note, hold, created_at FROM work_batches{where} ORDER BY id DESC", args)]
        by_id = {r["id"]: r for r in rows}
        for r in rows:
            r["hold"] = bool(r["hold"])
            r["counts"] = {s: 0 for s in STATUSES}
            r["cameras"] = []
        for b, status, n in con.execute(
                "SELECT work_batch, status, COUNT(*) FROM images WHERE work_batch IS NOT NULL"
                " GROUP BY work_batch, status"):
            if b in by_id and status in by_id[b]["counts"]:
                by_id[b]["counts"][status] = int(n)
        for b, camera, n in con.execute(
                "SELECT work_batch, camera, COUNT(*) AS n FROM images"
                " WHERE work_batch IS NOT NULL AND status != 'trashed'"
                " GROUP BY work_batch, camera ORDER BY n DESC, camera"):
            if b in by_id:
                by_id[b]["cameras"].append({"key": camera, "n": int(n)})
        for r in rows:
            r["total"] = sum(r["counts"].values())
        return rows

    def list_work_batches(self) -> list[dict]:
        """작업 배치 전부(새것부터): {id, kind, name, note, hold, created_at, counts{상태: 장수}, total, cameras[{key, n}]}.
        이미지가 한 장도 안 남은 배치(전부 휴지통에서 비워졌다)도 싣는다 — 숨기는 것은 화면이 한다."""
        with self.connect() as con:
            con.execute("BEGIN")            # 줄과 장수가 같은 시점이게 (health_snapshot 과 같다)
            return self._work_rows(con)

    def get_work_batch(self, batch_id: int) -> dict | None:
        with self.connect() as con:
            con.execute("BEGIN")
            rows = self._work_rows(con, " WHERE id=?", (int(batch_id),))
        return rows[0] if rows else None

    def work_batch_brief(self, batch_id: int) -> dict | None:
        """배치 한 줄만, 장수 없이 — {id, kind, name, hold}. 검수 화면이 이미지를 열 때마다(↑·↓ 를 누르고 있으면 초에 수십 번)
        부르므로 이미지 표를 세지 않는다."""
        with self.connect() as con:
            r = con.execute("SELECT id, kind, name, hold FROM work_batches WHERE id=?", (int(batch_id),)).fetchone()
        return None if r is None else {"id": int(r["id"]), "kind": r["kind"], "name": r["name"], "hold": bool(r["hold"])}

    def unbatched_counts(self) -> dict[str, int]:
        """배치 없는 이미지의 상태별 장수 ({상태: n} + total)."""
        out = {s: 0 for s in STATUSES}
        with self.connect() as con:
            for status, n in con.execute(
                    "SELECT status, COUNT(*) FROM images WHERE work_batch IS NULL GROUP BY status"):
                if status in out:
                    out[status] = int(n)
        out["total"] = sum(out[s] for s in STATUSES)
        return out

    def update_work_batch(self, batch_id: int, *, name: str | None = None, set_name: bool = False,
                          hold: bool | None = None) -> bool:
        """이름(set_name 이 참일 때만 — 빈 이름은 '이름 없음')과 hold 를 고친다. 없는 배치면 False."""
        fields, args = [], []
        if set_name:
            fields.append("name=?")
            args.append(clean_work_name(name))
        if hold is not None:
            fields.append("hold=?")
            args.append(int(bool(hold)))
        with _write_lock, self.connect() as con:
            if not fields:
                return con.execute("SELECT 1 FROM work_batches WHERE id=?", (int(batch_id),)).fetchone() is not None
            cur = con.execute(f"UPDATE work_batches SET {', '.join(fields)} WHERE id=?", (*args, int(batch_id)))
            return cur.rowcount > 0

    def set_work_batch(self, image_ids: list[int], batch_id: int | None) -> int:
        """이미지들을 그 배치로 옮긴다(None = 배치 없음). 바꾼 행 수. 라벨이 아니라 updated_at 은 건드리지 않는다
        (set_domain 과 같은 이유 — 자동 라벨링의 (상태, updated_at) 확인도 깨지 않는다). 없는 배치면 ValueError."""
        ids = list(dict.fromkeys(int(i) for i in image_ids))
        if not ids:
            return 0
        with _write_lock, self.connect() as con:
            if batch_id is not None and con.execute(
                    "SELECT 1 FROM work_batches WHERE id=?", (int(batch_id),)).fetchone() is None:
                raise ValueError(f"작업 배치 {batch_id} 가 없습니다.")
            cur = con.executemany("UPDATE images SET work_batch=? WHERE id=?", [(batch_id, i) for i in ids])
            return cur.rowcount

    def work_batch_image_ids(self, batch_id: int, statuses: tuple[str, ...]) -> list[int]:
        """그 배치에서 주어진 상태인 이미지 id(번호순) — 배치만 자동 라벨링할 때."""
        if not statuses:
            return []
        marks = ",".join("?" * len(statuses))
        with self.connect() as con:
            return [int(r[0]) for r in con.execute(
                f"SELECT id FROM images WHERE work_batch=? AND status IN ({marks}) ORDER BY id",
                (int(batch_id), *statuses))]

    def add_reviewed_to_dataset(self, batch_id: int) -> int:
        """그 배치의 '검수 끝'(reviewed)을 전부 승인(approved — 데이터셋)으로 올린다. 올린 장수.

        **reviewed 만 올린다** — 검수 대기·미라벨은 사람이 보지 않은 것이라 그대로 둔다(자동 승인은 없다).
        한 트랜잭션이라 도중에 사람이 '검수 끝' 으로 넘긴 장은 통째로 들어가거나 통째로 다음 번에 들어간다.
        """
        with _write_lock, self.connect() as con:
            cur = con.execute(
                "UPDATE images SET status='approved', updated_at=? WHERE work_batch=? AND status='reviewed'",
                (_now(), int(batch_id)))
            return cur.rowcount

    def delete_work_batch(self, batch_id: int, *, only_if_empty: bool = False) -> bool:
        """배치를 지운다 — 이미지는 남고 '배치 없음' 이 된다(FOREIGN KEY ON DELETE SET NULL). only_if_empty 면 이미지가
        한 장이라도 있을 때 지우지 않는다(넣기가 한 장도 못 넣은 배치를 걷어 낼 때). 지웠으면 True."""
        with _write_lock, self.connect() as con:
            if only_if_empty and con.execute(
                    "SELECT 1 FROM images WHERE work_batch=? LIMIT 1", (int(batch_id),)).fetchone():
                return False
            return con.execute("DELETE FROM work_batches WHERE id=?", (int(batch_id),)).rowcount > 0

    # -------------------------------------------------------------------- 박스

    def get_boxes(self, image_id: int) -> list[Box]:
        with self.connect() as con:
            rows = con.execute(
                "SELECT cls, cx, cy, w, h, conf, origin FROM boxes"
                " WHERE image_id=? ORDER BY id",
                (image_id,),
            ).fetchall()
        return [Box(**dict(r)) for r in rows]

    # ------------------------------------------------------------ 가릴 영역

    def get_masks(self, image_id: int) -> list:
        """이 이미지의 가릴 영역 — 넣은 순서대로. 사각형은 정규화 xyxy 튜플, 다각형은 꼭짓점 튜플의 튜플 (masks.py)."""
        with self.connect() as con:
            return _read_masks(con, image_id)

    def masks_by_image(self) -> dict[int, list]:
        """영역이 있는 이미지만 {id: [영역, ...]} — 학습 세트·내보내기가 한 번에 읽는다."""
        out: dict[int, list] = {}
        with self.connect() as con:
            for r in con.execute(f"SELECT image_id, {_MASK_COLS} FROM masks ORDER BY image_id, id"):
                out.setdefault(int(r["image_id"]), []).append(from_row(*tuple(r)[1:]))
        return out

    def set_masks(self, image_id: int, masks: list, *, expect: tuple[str, str] | None = None,
                  source: str = "mask") -> bool:
        """가릴 영역만 바꾼다 — 박스·상태·파일은 그대로 (tools/mask_images.py). 영역 전후는 변경 이력에 남는다.

        expect(status, updated_at)를 주면 지금 값이 다를 때 아무것도 쓰지 않고 False(도는 사이 검수 화면에서 바뀐 장).
        학습 라벨이 달라지는 손이라 updated_at 은 새로 찍는다(검수 화면의 저장과 같다).
        """
        rects = clean_masks(masks)
        with _write_lock, self.connect() as con:
            row = con.execute("SELECT status, updated_at FROM images WHERE id=?", (image_id,)).fetchone()
            if row is None or (expect is not None and (row["status"], row["updated_at"]) != tuple(expect)):
                return False
            boxes = [Box(**dict(r)) for r in con.execute(
                "SELECT cls, cx, cy, w, h, conf, origin FROM boxes WHERE image_id=? ORDER BY id", (image_id,))]
            self._replace_boxes(con, image_id, boxes, source=source, batch=None, masks=rects)
            con.execute("UPDATE images SET updated_at=? WHERE id=?", (_now(), image_id))
        return True

    # ------------------------------------------------- 가릴 영역 템플릿 (2026-11-10)

    @staticmethod
    def _template_row(r: sqlite3.Row) -> dict:
        out = dict(r)
        out["rects"] = [to_wire(as_mask(x)) for x in json.loads(r["rects"])]     # 사각형 [x1, y1, x2, y2]·다각형 [[x, y], …]
        return out

    def list_mask_templates(self) -> list[dict]:
        """가릴 영역 템플릿 전부(이름순)."""
        with self.connect() as con:
            rows = con.execute("SELECT * FROM mask_templates ORDER BY name COLLATE NOCASE, id").fetchall()
        return [self._template_row(r) for r in rows]

    def get_mask_template(self, template_id: int) -> dict | None:
        with self.connect() as con:
            r = con.execute("SELECT * FROM mask_templates WHERE id=?", (template_id,)).fetchone()
        return self._template_row(r) if r else None

    def save_mask_template(self, name: str, rects: list, *, width: int | None, height: int | None,
                           image_id: int | None = None, camera: str | None = None,
                           overwrite: bool = False) -> tuple[int, bool]:
        """영역 묶음을 이름으로 남긴다. (템플릿 id, 덮어썼나).

        같은 이름이 있으면 overwrite 가 아닌 한 TemplateExists(사람에게 물을 일) — 덮어써도 이미 적용한 이미지의
        영역은 그대로다(복사해 넣었으므로). 영역이 없거나 한 장 한도(MAX_MASKS)를 넘으면 ValueError.
        """
        name = clean_template_name(name)
        rects = clean_masks(rects)
        if not rects:
            raise ValueError("템플릿에 넣을 가릴 영역이 없습니다 — 먼저 M 으로 영역을 그리세요.")
        if len(rects) > MAX_MASKS:
            raise ValueError(f"가릴 영역은 {MAX_MASKS}개까지입니다.")
        now = _now()
        with _write_lock, self.connect() as con:
            row = con.execute("SELECT id FROM mask_templates WHERE name=?", (name,)).fetchone()
            if row and not overwrite:
                raise TemplateExists(f"같은 이름의 템플릿이 있습니다: {name}")
            # 템플릿은 반올림하지 않는다 — 이력(_masks_json, 11-11 부터 7자리)과 달리 이미지에 그대로 복사될 값이다
            vals = (json.dumps([to_wire(r) for r in rects]), width, height, image_id, camera)
            if row:
                con.execute("UPDATE mask_templates SET rects=?, width=?, height=?, image_id=?, camera=?,"
                            " updated_at=? WHERE id=?", (*vals, now, row["id"]))
                return int(row["id"]), True
            return int(con.execute(
                "INSERT INTO mask_templates(name, rects, width, height, image_id, camera, created_at, updated_at)"
                " VALUES (?,?,?,?,?,?,?,?)", (name, *vals, now, now)).lastrowid), False

    def rename_mask_template(self, template_id: int, name: str) -> bool:
        """이름을 바꾼다. 없는 템플릿이면 False, 다른 템플릿과 이름이 겹치면 TemplateExists."""
        name = clean_template_name(name)
        with _write_lock, self.connect() as con:
            other = con.execute("SELECT id FROM mask_templates WHERE name=?", (name,)).fetchone()
            if other and int(other["id"]) != template_id:
                raise TemplateExists(f"같은 이름의 템플릿이 있습니다: {name}")
            return con.execute("UPDATE mask_templates SET name=?, updated_at=? WHERE id=?",
                               (name, _now(), template_id)).rowcount > 0

    def delete_mask_template(self, template_id: int) -> bool:
        """템플릿만 지운다 — 이미 적용한 이미지의 영역은 그대로."""
        with _write_lock, self.connect() as con:
            return con.execute("DELETE FROM mask_templates WHERE id=?", (template_id,)).rowcount > 0

    def apply_mask_template(self, image_ids: list[int], rects: list, *, mode: str, batch: str) -> dict:
        """템플릿 영역을 이미지들에 더하거나(add) 뺀다(remove) — 한 트랜잭션, 이미지마다 이력(source='bulk', 같은 batch).

        더하기는 이미 같은 좌표의 영역이 있으면 그것은 건너뛰고(masks.merge), 빼기는 템플릿과 좌표가 같은 영역만 뺀다
        (masks.subtract — 손으로 옮긴 영역은 그 이미지의 것이라 남긴다). 박스는 그대로이고, 학습 라벨이 달라지는 손이라
        updated_at 은 새로 찍는다(set_masks 와 같다). test 이미지에도 더한다 — 그 영역은 칠하지 않고 채점에서 뺀다(masks.py,
        2026-11-24 — 그 전에는 서버가 넘긴 blocked 로 test 를 건너뛰었다). 묶음째 되돌리기는 revert_batch(영역도 되돌린다).
        {changed, unchanged, full, missing} 장수를 돌려준다.
        """
        if mode not in ("add", "remove"):
            raise ValueError(f"알 수 없는 적용 방식: {mode}")
        rects = clean_masks(rects)
        out = {"changed": 0, "unchanged": 0, "full": 0, "missing": 0}
        with _write_lock, self.connect() as con:
            now = _now()
            for image_id in dict.fromkeys(image_ids):          # 같은 id 가 두 번 와도 한 번만
                row = con.execute("SELECT split, camera FROM images WHERE id=?", (image_id,)).fetchone()
                if row is None:
                    out["missing"] += 1
                    continue
                cur = _read_masks(con, image_id)
                new = merge_masks(cur, rects) if mode == "add" else subtract_masks(cur, rects)
                if same_rects(new, cur):
                    out["unchanged"] += 1
                    continue
                if len(new) > MAX_MASKS:
                    out["full"] += 1
                    continue
                boxes = [Box(**dict(r)) for r in con.execute(
                    "SELECT cls, cx, cy, w, h, conf, origin FROM boxes WHERE image_id=? ORDER BY id", (image_id,))]
                self._replace_boxes(con, image_id, boxes, source="bulk", batch=batch, masks=new)
                con.execute("UPDATE images SET updated_at=? WHERE id=?", (now, image_id))
                out["changed"] += 1
        return out

    def set_boxes(
        self,
        image_id: int,
        boxes: list[Box],
        *,
        uncertainty: float | None = None,
        model_version: str | None = None,
        source: str = "ui",
        batch: str | None = None,
        masks: list | None = None,
    ) -> tuple[int, int]:
        """해당 이미지의 박스를 통째로 교체하고 (바뀌기 전 개수, 바뀐 뒤 개수) 를 돌려준다.

        내용이 달라졌으면 `box_history` 에 전후를 남긴다. **바뀌기 전이 비어 있으면 남기지
        않는다** — 되돌릴 것이 없고, 자동 라벨링이 미라벨 수천 장에 쓰는 몫이 쌓일 뿐이다.
        masks 를 주면 가릴 영역도 같은 트랜잭션에서 그것으로 바꾼다(None 이면 그대로 둔다 — masks.py).
        """
        new = [b for b in boxes if b.is_valid()]
        with _write_lock, self.connect() as con:
            n_old = self._replace_boxes(con, image_id, new, source=source, batch=batch,
                                        masks=None if masks is None else clean_masks(masks))
            fields = ["updated_at=?"]
            args: list = [_now()]
            if uncertainty is not None:
                fields.append("uncertainty=?")
                args.append(float(uncertainty))
            if model_version is not None:
                fields.append("model_version=?")
                args.append(model_version)
            args.append(image_id)
            con.execute(f"UPDATE images SET {', '.join(fields)} WHERE id=?", args)
        return n_old, len(new)

    def apply_prediction(
        self,
        image_id: int,
        boxes: list[Box],
        *,
        expect: tuple[str, str],
        uncertainty: float,
        model_version: str,
    ) -> bool:
        """자동 라벨링 결과로 박스를 바꾸고 검수 대기(pending)로 둔다 — **작업을 시작할 때 본 뒤로 아무도 손대지
        않았을 때만** (2026-11-05). expect 는 시작 때의 (status, updated_at). 지금 값이 다르면 아무것도 쓰지 않고
        False 를 돌려준다.

        자동 라벨링은 수천 장에 몇십 분씩 돈다. 예전에는 시작 때 고른 이미지를 상태를 다시 보지 않고 덮어써서, 그
        사이 사람이 승인·휴지통·직접 수정한 이미지가 예측 박스와 '검수 대기' 로 되돌아갔다. 확인과 쓰기를 한 잠금·한
        트랜잭션에서 해야 그 틈에 끼어드는 저장이 없다. 박스는 `_replace_boxes` 로 바꿔 이력이 남는다.
        updated_at 은 초 단위라, 시작한 바로 그 초에 한 번 더 고친 것은 가리지 못한다.
        """
        new = [b for b in boxes if b.is_valid()]
        with _write_lock, self.connect() as con:
            row = con.execute("SELECT status, updated_at FROM images WHERE id=?", (image_id,)).fetchone()
            if row is None or (row["status"], row["updated_at"]) != tuple(expect):
                return False
            self._replace_boxes(con, image_id, new, source="predict", batch=None)
            con.execute(
                "UPDATE images SET status='pending', uncertainty=?, model_version=?, updated_at=? WHERE id=?",
                (float(uncertainty), model_version, _now(), image_id),
            )
        return True

    def replace_file(
        self,
        image_id: int,
        path: str,
        width: int,
        height: int,
        boxes: list[Box],
        *,
        expect: tuple[str, str],
        uncertainty: float,
        model_version: str,
        keep_status: bool = False,
        source: str = "predict",
        masks: list | None = None,
    ) -> bool:
        """같은 이미지 행이 다른 파일(새 경로·크기)을 가리키게 하고 박스를 그 파일 기준으로 바꾼다 — 검수 대기로 둔다
        (2026-11-07, VIRAT 장면을 잘라 다시 쓸 때 `tools/import_virat.py --recrop`. id·태그·도메인·카메라는 그대로).

        apply_prediction 처럼 expect(시작 때의 status, updated_at)가 지금과 다르면 아무것도 쓰지 않고 False. 박스는
        `_replace_boxes` 로 바꿔 이전 박스가 이력(source)에 남는다. 파일 쓰기·옛 파일 지우기는 부르는 쪽이 한다(이 함수는 DB 만).
        dhash 는 지운다 — 다음 분할 때 새 파일로 다시 잰다.
        keep_status 면 상태(승인·휴지통 등)를 그대로 둔다 — 잘라 내기만 해서 남는 박스가 사람이 본 그대로일 때
        (2026-11-09 `tools/crop_images.py`, source='crop' — 그 전 이력은 좌표가 달라 서버가 되돌리기를 막는다.
        `tools/mask_images.py` 는 source='mask' — 크기가 같아 그 전 이력도 좌표는 맞는다).
        **가릴 영역이 있는 이미지는 masks(새 그림 기준으로 옮긴 영역)를 함께 줘야 한다** — 안 주면 ValueError(아무것도
        쓰지 않는다). 옛 그림 좌표의 영역이 새 그림의 엉뚱한 자리를 칠하지 않게 (2026-11-09).
        """
        new = [b for b in boxes if b.is_valid()]
        with _write_lock, self.connect() as con:
            row = con.execute("SELECT status, updated_at FROM images WHERE id=?", (image_id,)).fetchone()
            if row is None or (row["status"], row["updated_at"]) != tuple(expect):
                return False
            if masks is None and con.execute("SELECT 1 FROM masks WHERE image_id=? LIMIT 1", (image_id,)).fetchone():
                raise ValueError(f"{image_id}: 가릴 영역이 있는 이미지다 — 새 그림 기준으로 옮긴 영역(masks)을 함께 줘야 한다")
            self._replace_boxes(con, image_id, new, source=source, batch=None,
                                masks=None if masks is None else clean_masks(masks))
            con.execute(
                "UPDATE images SET path=?, name=?, width=?, height=?, dhash=NULL,"
                + ("" if keep_status else " status='pending',")
                + " uncertainty=?, model_version=?, updated_at=? WHERE id=?",
                (path, Path(path).name, int(width), int(height), float(uncertainty), model_version, _now(),
                 image_id),
            )
        return True

    @staticmethod
    def _replace_boxes(con: sqlite3.Connection, image_id: int, new: list[Box], *,
                       source: str, batch: str | None, masks: list | None = None) -> int:
        """열린 트랜잭션 안에서 박스를 통째로 바꾸고 이력을 남긴다. 바뀌기 전 개수를 돌려준다.

        잠금(_write_lock)은 부르는 쪽이 쥐고 있어야 한다 — 이 안에서 다시 잡으면 멈춘다.
        masks(정리된 영역 목록 — 사각형·다각형, masks.clean)를 주면 가릴 영역도 바꾼다. 영역이 바뀌었으면 박스가 그대로여도
        이력을 남긴다(masks_before·masks_after — 학습 라벨이 달라지는 사람의 손이다). 바뀌기 전이 비어 있어도 남긴다.
        """
        old = con.execute(
            "SELECT cls, cx, cy, w, h, conf, origin FROM boxes WHERE image_id=? ORDER BY id",
            (image_id,),
        ).fetchall()
        same = [_box_key(r) for r in old] == [_box_key(b) for b in new]
        boxes_changed = bool(old) and not same
        old_m = None                          # 영역이 바뀔 때만 (바뀌기 전 영역 목록)
        if masks is not None:
            cur = _read_masks(con, image_id)
            if not same_rects(cur, masks):
                old_m = cur
        if boxes_changed or old_m is not None:
            # 목록은 hist_lists 에 한 번만 — 앞 줄의 '후' 와 같은 '전', 영역만 바꾼 저장의 박스 전후는 같은 #id 다 (2026-11-11)
            con.execute(
                "INSERT INTO box_history(image_id, at, source, batch, before, after,"
                " n_before, n_after, masks_before, masks_after) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (image_id, _now(), source, batch, _hist_put(con, _boxes_json(old)), _hist_put(con, _boxes_json(new)),
                 len(old), len(new),
                 None if old_m is None else _hist_put(con, _masks_json(old_m)),
                 None if old_m is None else _hist_put(con, _masks_json(masks))),
            )
        if not same:                          # 박스가 그대로면(영역만 바꾸는 저장·템플릿 적용) 행을 다시 쓰지 않는다
            con.execute("DELETE FROM boxes WHERE image_id=?", (image_id,))
            con.executemany(
                "INSERT INTO boxes(image_id, cls, cx, cy, w, h, conf, origin)"
                " VALUES (?,?,?,?,?,?,?,?)",
                [(image_id, b.cls, b.cx, b.cy, b.w, b.h, b.conf, b.origin) for b in new],
            )
        if old_m is not None:
            con.execute("DELETE FROM masks WHERE image_id=?", (image_id,))
            con.executemany("INSERT INTO masks(image_id, x1, y1, x2, y2, poly) VALUES (?,?,?,?,?,?)",
                            [(image_id, *row_values(r)) for r in masks])
        return len(old)

    # ------------------------------------------------------- 같은 자리 일괄 수정

    def same_place(
        self, image_id: int, ref: tuple[float, float, float, float], *,
        from_cls: list[int] | None, iou_min: float,
    ) -> dict:
        """같은 묶음(grp)의 **다른** 이미지에서 ref 와 같은 자리의 박스를 찾는다.

        고정 카메라 묶음에서 '같은 차' 를 한꺼번에 고치려는 것이다 — van 박스의 83% 가
        고정 CCTV 한 대의 두 자리에서 나온다. 이미지마다 ref 와 가장 많이 겹치는 박스 하나를
        고르고, IoU 가 iou_min 이상이면 짝으로 본다. from_cls 를 주면 그 클래스만 본다
        (오토바이와 탄 사람처럼 크게 겹치는 다른 객체를 끌어오지 않게).
        """
        with self.connect() as con:
            row = con.execute("SELECT grp FROM images WHERE id=?", (image_id,)).fetchone()
            if row is None:
                raise LookupError("이미지를 찾을 수 없습니다.")
            grp = row["grp"]
            if not grp:
                raise ValueError("이 이미지는 아직 묶음이 정해지지 않았습니다 (다음 학습 때 계산됩니다).")
            sql = ("SELECT b.id, b.image_id, b.cls, b.cx, b.cy, b.w, b.h, i.split, i.status"
                   " FROM boxes b JOIN images i ON i.id = b.image_id WHERE i.grp=? AND i.id<>?")
            args: list = [grp, image_id]
            if from_cls:
                sql += f" AND b.cls IN ({','.join('?' * len(from_cls))})"
                args += list(from_cls)
            rows = con.execute(sql, args).fetchall()
            size = con.execute("SELECT COUNT(*) FROM images WHERE grp=?", (grp,)).fetchone()[0]
        best: dict[int, tuple[float, dict]] = {}
        for r in rows:
            v = _iou_xywh(ref, (r["cx"], r["cy"], r["w"], r["h"]))
            if v >= iou_min and v > best.get(r["image_id"], (0.0, None))[0]:
                best[r["image_id"]] = (v, {**dict(r), "iou": v})
        matches = [m for _, m in sorted(best.values(), key=lambda t: t[1]["image_id"])]
        return {"grp": grp, "group_size": int(size), "matches": matches}

    def set_class_bulk(self, box_ids: dict[int, list[int]], to_cls: int, *, batch: str,
                       score: Callable[[list[Box]], float] | None = None) -> int:
        """{이미지 id: [박스 행 id, ...]} 의 박스 클래스를 한 트랜잭션으로 바꾼다.

        사람이 정한 것이므로 origin='human', conf=NULL 로 둔다(검수 UI 에서 클래스를 바꿀 때와
        같다). 이미지마다 이력(source='bulk', 같은 batch)을 남기므로 묶음째 되돌릴 수 있다.
        바뀐 이미지 수를 돌려준다.

        박스 행 id 는 **그 이미지의 것만** 바꾼다 (2026-11-05) — 예전에는 `WHERE id=?` 뿐이라 다른 이미지의 박스 id 가
        섞이면 그 이미지가 이력 없이 바뀌었다. score 를 주면 바뀐 이미지의 검수 우선순위(uncertainty)를 새 박스로
        다시 매긴다 — 서버의 `save_boxes` 와 같은 식(scoring)을 서버가 넘긴다(설정값이 서버에 있다).
        """
        now = _now()
        changed = 0
        with _write_lock, self.connect() as con:
            for image_id, ids in box_ids.items():
                target = set(ids)
                old = con.execute(
                    "SELECT id, cls, cx, cy, w, h, conf, origin FROM boxes"
                    " WHERE image_id=? ORDER BY id", (image_id,),
                ).fetchall()
                new = [Box(to_cls, r["cx"], r["cy"], r["w"], r["h"], None, "human")
                       if r["id"] in target else
                       Box(r["cls"], r["cx"], r["cy"], r["w"], r["h"], r["conf"], r["origin"])
                       for r in old]
                if [_box_key(r) for r in old] == [_box_key(b) for b in new]:
                    continue
                con.execute(
                    "INSERT INTO box_history(image_id, at, source, batch, before, after,"
                    " n_before, n_after) VALUES (?,?,?,?,?,?,?,?)",
                    (image_id, now, "bulk", batch, _hist_put(con, _boxes_json(old)), _hist_put(con, _boxes_json(new)),
                     len(old), len(new)),
                )
                con.executemany("UPDATE boxes SET cls=?, conf=NULL, origin='human' WHERE id=? AND image_id=?",
                                [(to_cls, i, image_id) for i in target])
                if score is not None:
                    con.execute("UPDATE images SET updated_at=?, uncertainty=? WHERE id=?",
                                (now, float(score(new)), image_id))
                else:
                    con.execute("UPDATE images SET updated_at=? WHERE id=?", (now, image_id))
                changed += 1
        return changed

    def revert_batch(self, batch: str, *,
                     score: Callable[[list[Box]], float] | None = None) -> tuple[int, int]:
        """일괄 수정 한 묶음을 되돌린다. (되돌린 이미지 수, 건너뛴 이미지 수)

        **그 뒤에 또 고친 이미지는 건드리지 않는다** — 지금 박스가 일괄 수정 직후와 같을
        때만 그 전으로 돌린다. 되돌린 것도 이력(source='revert', batch='취소: …')에 남는다.
        되돌릴 박스가 클래스 목록에 없는 번호를 쓰면 아무것도 바꾸지 않고 ValueError — 클래스를 합쳐
        뺀 묶음(tools/merge_class.py, 2026-11-02 van → car)은 그 클래스를 먼저 되살려야 되돌린다.
        score 는 `set_class_bulk` 와 같다 — 되돌린 박스로 uncertainty 를 다시 매긴다 (2026-11-05).
        가릴 영역도 바꾼 묶음(가릴 영역 템플릿 적용, 2026-11-10 — 이력에 masks_after 가 있음)은 지금 영역도 적용 직후와
        같을 때만 되돌리고, 영역도 그 전으로 돌린다. **영역만 바꾼 줄(박스 전후가 같음)은 박스를 견주지 않고 지금 박스를
        그대로 둔다** — 템플릿을 넣은 뒤 박스를 검수한 이미지도 영역은 되돌릴 수 있어야 한다(test 이미지도 — 2026-11-24 부터
        test 의 영역은 채점에서 뺄 자리라 막지 않는다).
        """
        skipped = 0
        with _write_lock, self.connect() as con:
            rows = con.execute(
                "SELECT image_id, before, after, masks_before, masks_after FROM box_history"
                " WHERE batch=? AND source='bulk' ORDER BY id", (batch,),
            ).fetchall()
            bodies = _hist_bodies(con, rows)
            todo = []
            for r in rows:
                cur = con.execute(
                    "SELECT cls, cx, cy, w, h, conf, origin FROM boxes WHERE image_id=? ORDER BY id",
                    (r["image_id"],),
                ).fetchall()
                after = _boxes_from_json(_hist_text(bodies, r["after"]))
                before = _boxes_from_json(_hist_text(bodies, r["before"]))
                if r["masks_after"] is not None and [_box_key(b) for b in before] == [_box_key(b) for b in after]:
                    before = [Box(**dict(b)) for b in cur]      # 영역만 바꾼 줄 — 박스는 지금 것을 그대로 둔다
                elif [_box_key(b) for b in cur] != [_box_key(b) for b in after]:
                    skipped += 1
                    continue
                masks_before = None             # None = 그 묶음에서 영역은 그대로였다 — 되돌릴 때도 건드리지 않는다
                if r["masks_after"] is not None:
                    cur_m = _read_masks(con, r["image_id"])
                    if not same_rects(cur_m, _masks_from_json(_hist_text(bodies, r["masks_after"]))):
                        skipped += 1
                        continue
                    masks_before = _masks_from_json(_hist_text(bodies, r["masks_before"]))
                todo.append((r["image_id"], before, masks_before))
            known = {c["idx"] for c in con.execute("SELECT idx FROM classes")}
            gone = sorted({b.cls for _, before, _m in todo for b in before} - known)
            if gone:
                raise ValueError(f"되돌리면 클래스 목록에 없는 {', '.join(map(str, gone))}번 박스가 생깁니다"
                                 " — '클래스 추가' 로 그 클래스를 먼저 되살리세요.")
            for image_id, before, masks_before in todo:
                self._replace_boxes(con, image_id, before, source="revert", batch=f"취소: {batch}",
                                    masks=masks_before)
                if score is not None:
                    con.execute("UPDATE images SET updated_at=?, uncertainty=? WHERE id=?",
                                (_now(), float(score(before)), image_id))
                else:
                    con.execute("UPDATE images SET updated_at=? WHERE id=?", (_now(), image_id))
        return len(todo), skipped

    # ---------------------------------------------------------------- 변경 이력

    @staticmethod
    def _hist_row(r: sqlite3.Row, bodies: dict[int, str], with_boxes: bool = True) -> dict:
        """이력 한 줄 → 화면·되돌리기가 쓰는 모양. bodies 는 _hist_bodies(그 줄들이 가리키는 목록)."""
        out = {k: r[k] for k in ("id", "image_id", "at", "source", "batch", "n_before", "n_after")}
        keys = r.keys()
        # 가릴 영역의 전후 — 그 저장에서 영역이 바뀌었을 때만 목록, 아니면 None (2026-11-09)
        for k in ("masks_before", "masks_after"):
            out[k] = _masks_from_json(_hist_text(bodies, r[k])) if k in keys else None
        if with_boxes:
            out["before"] = [b.to_dict() for b in _boxes_from_json(_hist_text(bodies, r["before"]))]
            out["after"] = [b.to_dict() for b in _boxes_from_json(_hist_text(bodies, r["after"]))]
        return out

    def box_history(self, image_id: int, limit: int = 100) -> list[dict]:
        """이미지 한 장의 박스 변경 이력, 최근 것부터."""
        with self.connect() as con:
            rows = con.execute(
                "SELECT * FROM box_history WHERE image_id=? ORDER BY id DESC LIMIT ?",
                (image_id, limit),
            ).fetchall()
            bodies = _hist_bodies(con, rows)
        return [self._hist_row(r, bodies) for r in rows]

    def crop_entries(self, image_id: int) -> list[int]:
        """이 이미지를 잘라 바꾼 이력(source='crop')의 id. 그보다 앞선 이력의 박스는 자르기 전 화면 좌표라
        지금 그림에 되돌리면 어긋난다 (2026-11-09, `tools/crop_images.py`)."""
        with self.connect() as con:
            return [int(r[0]) for r in con.execute(
                "SELECT id FROM box_history WHERE image_id=? AND source='crop' ORDER BY id", (image_id,))]

    def history_count(self, image_id: int) -> int:
        with self.connect() as con:
            return int(con.execute("SELECT COUNT(*) FROM box_history WHERE image_id=?",
                                   (image_id,)).fetchone()[0])

    def history_entry(self, entry_id: int) -> dict | None:
        with self.connect() as con:
            r = con.execute("SELECT * FROM box_history WHERE id=?", (entry_id,)).fetchone()
            bodies = _hist_bodies(con, [r]) if r else {}
        return self._hist_row(r, bodies) if r else None

    def recent_history(
        self, *, limit: int = 200, shrunk: bool = False, source: str | None = None,
    ) -> list[dict]:
        """여러 이미지에 걸친 최근 변경. 박스 목록은 빼고 개수·이미지 정보만.

        shrunk=True 면 **박스 수가 줄어든 저장만** — 검수 UI 로 여러 장을 고친 뒤 '혹시
        사라진 박스가 없나' 를 보는 용도다 (09-21 에는 백업과 대조해야 했다).
        """
        where, args = [], []
        if shrunk:
            where.append("h.n_after < h.n_before")
        if source:
            where.append("h.source = ?")
            args.append(source)
        sql = (
            "SELECT h.id, h.image_id, h.at, h.source, h.batch, h.n_before, h.n_after,"
            " i.name, i.status, i.split FROM box_history h JOIN images i ON i.id = h.image_id"
            + (" WHERE " + " AND ".join(where) if where else "")
            + " ORDER BY h.id DESC LIMIT ?"
        )
        with self.connect() as con:
            return [dict(r) for r in con.execute(sql, [*args, limit]).fetchall()]

    def batches(self, limit: int = 30) -> list[dict]:
        """일괄 수정 묶음 목록, 최근 것부터 — **일괄 수정(source 'bulk') `limit` 건**과 그 뒤에 남은 다른 줄(되돌리기 'revert' 등).

        한 묶음은 (batch, source) 한 줄이고, 되돌리면 같은 이름 앞에 '취소: ' 가 붙은 revert 줄이 하나 더 생긴다(화면은 그것으로
        '되돌림' 을 표시한다). 2026-11-16 전에는 두 종류를 함께 `limit` 줄로 잘라서, 되돌린 묶음이 많을수록 화면의 '최근 30건' 에
        일괄 수정이 그만큼 덜 보였다(15건을 되돌렸으면 15건만). 이제 일괄 수정을 `limit` 건 고르고, 그보다 새로운 다른 줄은
        전부 싣는다 — 되돌리기는 늘 그 일괄 수정보다 뒤라서 빠지지 않는다.
        """
        with self.connect() as con:
            rows = con.execute(
                "SELECT batch, source, MIN(at) AS at, COUNT(*) AS images,"
                " MIN(id) AS first_id FROM box_history WHERE batch IS NOT NULL"
                " GROUP BY batch, source ORDER BY first_id DESC"
            ).fetchall()
        out, bulk = [], 0
        for r in rows:
            if r["source"] == "bulk":
                if bulk == limit:
                    break                   # 이보다 오래된 일괄 수정과, 그 사이의 되돌리기는 싣지 않는다
                bulk += 1
            out.append(dict(r))
        return out

    def backup_daily(self, keep: int = 14, now: datetime | None = None) -> Path | None:
        """하루 한 번 DB 를 `<DB 폴더>/backups/` 로 떠 둔다. 그날 것이 있으면 건너뛴다.

        검수 서버가 켤 때와, 켜 둔 채 날이 바뀐 뒤 첫 변경 요청 앞에서 부른다(server._DailyBackup, 2026-11-21 — 전에는
        켤 때만). now 는 그 '지금'(없으면 지금 시각)으로, 날짜와 파일 이름을 정한다. sqlite 의 backup() 이라 다른 서버가
        쓰는 중이어도 일관된 사본이 나온다. **가장 최근 `keep` 개만 남기고 그보다 오래된 사본은 지운다**
        (이 함수가 만든 `backups/<DB 이름>_날짜_시각.db` 만 대상이다). keep=0 이면 끈다.
        뜨다 실패하면 아무 파일도 남기지 않는다(`copy_db`, 2026-11-23) — 덜 뜬 파일이 '그날 것' 으로 쳐지지 않게.
        """
        if keep <= 0 or not self.db_path.exists():
            return None
        d = self.db_path.parent / "backups"
        stem = self.db_path.stem
        now = now or datetime.now()
        if any(d.glob(f"{stem}_{now:%Y%m%d}_*.db")):
            return None
        dst = copy_db(self.db_path, d / f"{stem}_{now:%Y%m%d_%H%M%S}.db")
        for old in sorted(d.glob(f"{stem}_????????_??????.db"))[:-keep]:
            old.unlink()
        return dst

    def health_snapshot(self, *, approved_only: bool = True, tags: bool = True) -> dict:
        """데이터 점검(health.py, 2026-11-15)이 읽는 것을 **한 시점에서** 한 번에: 이미지 줄·박스·가릴 영역·태그 수.

        읽기만 한다. approved_only 면 승인 이미지만, 아니면 휴지통만 뺀다. 여러 SELECT 가 같은 상태를 보게 BEGIN 으로
        읽기 트랜잭션을 연다(WAL — 도는 사이 검수 화면이 저장해도 박스와 이미지 줄이 어긋나지 않는다).
        tags=False 면 태그 수를 세지 않는다(빈 값) — 히트맵만 받을 때는 쓰지 않는다.
        """
        where = "i.status = 'approved'" if approved_only else "i.status != 'trashed'"
        with self.connect() as con:
            con.execute("BEGIN")
            images = [dict(r) for r in con.execute(
                "SELECT i.id, i.width, i.height, i.status, i.split, i.camera, i.domain, i.captured_at, i.grp"
                f" FROM images i WHERE {where} ORDER BY i.id")]
            boxes = [tuple(r) for r in con.execute(
                "SELECT b.image_id, b.cls, b.cx, b.cy, b.w, b.h FROM boxes b JOIN images i ON i.id = b.image_id"
                f" WHERE {where} ORDER BY b.image_id, b.id")]
            # (이미지 id, 영역) — 영역은 masks.py 의 꼴(사각형 튜플·다각형 점 튜플, 2026-11-19)
            masks = [(int(r[0]), from_row(*tuple(r)[1:])) for r in con.execute(
                "SELECT m.image_id, m.x1, m.y1, m.x2, m.y2, m.poly FROM masks m JOIN images i ON i.id = m.image_id"
                f" WHERE {where} ORDER BY m.image_id, m.id")]
            tag_rows: list[dict] = []
            untagged = 0
            if tags:
                tag_rows = [{"id": int(r[0]), "name": r[1], "n": int(r[2])} for r in con.execute(
                    "SELECT t.id, t.name, (SELECT COUNT(*) FROM image_tags it JOIN images i ON i.id = it.image_id"
                    f"  WHERE it.tag_id = t.id AND {where}) FROM tags t ORDER BY t.name COLLATE NOCASE, t.id")]
                untagged = int(con.execute(
                    f"SELECT COUNT(*) FROM images i WHERE {where}"
                    " AND NOT EXISTS (SELECT 1 FROM image_tags t WHERE t.image_id = i.id)").fetchone()[0])
        return {"images": images, "boxes": boxes, "masks": masks, "tags": tag_rows, "untagged": untagged}

    def class_box_counts(self) -> dict[int, dict[str, int]]:
        """클래스별 박스 수를 이미지 상태별로 센다.

        반환: {클래스인덱스: {"total": n, "approved": n, "pending": n, ...}}
        """
        with self.connect() as con:
            rows = con.execute(
                "SELECT b.cls AS cls, i.status AS status, COUNT(*) AS n"
                " FROM boxes b JOIN images i ON i.id = b.image_id"
                " GROUP BY b.cls, i.status"
            ).fetchall()
        out: dict[int, dict[str, int]] = {}
        for r in rows:
            cls = int(r["cls"])
            bucket = out.setdefault(cls, {"total": 0})
            bucket[r["status"]] = bucket.get(r["status"], 0) + int(r["n"])
            bucket["total"] += int(r["n"])
        return out

    def max_class_id(self) -> int:
        """등록된 박스에서 가장 큰 클래스 ID. 박스가 없으면 -1."""
        with self.connect() as con:
            row = con.execute("SELECT COALESCE(MAX(cls), -1) AS m FROM boxes").fetchone()
        return int(row["m"])

    def count_updated_since(self, when: datetime, *, status: str | None = None) -> int:
        """when 이후에 박스·상태·경로·크기가 바뀐 이미지 수 (updated_at 기준).

        status 를 주면 지금 그 상태인 것만 센다. DB 에서 지운 행은 흔적이 없으므로
        세지 못한다.
        """
        stamp = when.astimezone(timezone.utc).isoformat(timespec="seconds")
        sql, args = "SELECT COUNT(*) AS n FROM images WHERE updated_at > ?", [stamp]
        if status is not None:
            sql += " AND status = ?"
            args.append(status)
        with self.connect() as con:
            row = con.execute(sql, args).fetchone()
        return int(row["n"])

    def known_statuses(self) -> dict[str, str]:
        """이미 등록된 이미지의 경로 -> 검수 상태."""
        with self.connect() as con:
            rows = con.execute("SELECT path, status FROM images").fetchall()
        return {r["path"]: r["status"] for r in rows}

    def ingest_batch(self, items: list[dict]) -> int:
        """새 이미지 + 박스를 한 트랜잭션으로 몰아 넣는다.

        이미지 한 장마다 연결을 새로 여는 방식은 수천 장에서 몇 분씩 걸린다.
        items 각 항목: {path, source, width, height, status, boxes, uncertainty, model_version}
        (+ 있으면 camera, captured_at, work_batch — 새로 넣을 때만 쓰고, 이미 있는 행에는 쓰지 않는다.
        work_batch 는 작업 배치 번호(work_batches.id) — 없는 번호면 FOREIGN KEY 로 통째로 실패한다)
        """
        if not items:
            return 0
        now = _now()
        with _write_lock, self.connect() as con:
            for it in items:
                # 이미 있는 경로면 크기·출처만 맞추고 **박스는 넣지 않는다.** 예전에는
                # `ON CONFLICT DO UPDATE` 뒤에 lastrowid 를 썼는데, UPDATE 로 끝나면
                # lastrowid 는 바뀌지 않아 **직전에 넣은 다른 이미지의 id** 가 나온다
                # (sqlite 3.50 에서 확인). 그러면 박스가 엉뚱한 이미지에 붙는다.
                cur = con.execute(
                    "INSERT INTO images"
                    "(path, name, width, height, source, status, uncertainty, model_version, updated_at,"
                    " camera, captured_at, work_batch)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)"
                    " ON CONFLICT(path) DO NOTHING",
                    (
                        it["path"],
                        Path(it["path"]).name,
                        it["width"],
                        it["height"],
                        it["source"],
                        it.get("status", "unlabeled"),
                        float(it.get("uncertainty", 0.0)),
                        it.get("model_version"),
                        now,
                        it.get("camera"),
                        it.get("captured_at"),
                        it.get("work_batch"),
                    ),
                )
                if cur.rowcount == 0:
                    con.execute(
                        "UPDATE images SET width=?, height=?, source=? WHERE path=?",
                        (it["width"], it["height"], it["source"], it["path"]),
                    )
                    continue
                image_id = cur.lastrowid
                boxes = it.get("boxes")
                if boxes:
                    con.executemany(
                        "INSERT INTO boxes(image_id, cls, cx, cy, w, h, conf, origin)"
                        " VALUES (?,?,?,?,?,?,?,?)",
                        [
                            (image_id, b.cls, b.cx, b.cy, b.w, b.h, b.conf, b.origin)
                            for b in boxes
                            if b.is_valid()
                        ],
                    )
        return len(items)

    def approved_class_counts(self, *, drop_masked: bool = True) -> list[dict]:
        """분할 계산용: 승인 이미지별
        {id, path, width, height, split, grp, dhash, camera, counts={클래스: 박스 수}}.

        박스가 하나도 없는 배경 이미지도 빠뜨리지 않는다 (LEFT JOIN).
        width·height 는 이름이 숫자뿐인 파일을 묶을 때 쓴다 (grouping.name_series).
        camera 는 같은 카메라를 한 묶음으로 합칠 때 쓴다 (grouping.group_images, 2026-11-03).

        **counts 는 라벨 파일에 실제로 들어가는 박스만 센다 (2026-11-25, Gemini 리뷰 묶음 6).** 가릴 영역이 반 이상 덮은
        박스(`masks.drops` — 학습 세트·내보내기·데이터 점검과 같은 판정)는 DB 에 남아 있어도 뺀다. 전에는 DB 의 박스를
        그대로 세어 계층화가 라벨에 없는 박스로 클래스 비중을 맞췄다 — 그날 DB 는 승인 박스 3,186개 가운데 1,630개가 그런
        박스였고, 사본의 첫 분할에서 val 에 bicycle·scooter 라벨이 하나도 없었다(docs/experiments.md). 영역만 있고 남는
        박스가 없는 이미지는 배경 이미지처럼 counts 가 빈다. drop_masked=False 면 DB 의 박스 전부.
        """
        with self.connect() as con:
            con.execute("BEGIN")                # 박스와 영역을 같은 시점으로 읽는다
            rows = con.execute(
                "SELECT i.id AS id, i.path AS path, i.width AS width, i.height AS height,"
                " i.split AS split, i.grp AS grp, i.camera AS camera, i.dhash AS dhash,"
                " b.cls AS cls, b.cx AS cx, b.cy AS cy, b.w AS w, b.h AS h"
                " FROM images i LEFT JOIN boxes b ON b.image_id = i.id"
                " WHERE i.status='approved' ORDER BY i.id, b.id"
            ).fetchall()
            rects = _approved_masks(con) if drop_masked else {}
        out: dict[int, dict] = {}
        for r in rows:
            it = out.setdefault(
                int(r["id"]),
                {"id": int(r["id"]), "path": r["path"], "width": r["width"],
                 "height": r["height"], "split": r["split"],
                 "grp": r["grp"], "dhash": r["dhash"], "camera": r["camera"], "counts": {}},
            )
            if r["cls"] is not None and not _masked_out(r, rects, key="id"):
                it["counts"][int(r["cls"])] = it["counts"].get(int(r["cls"]), 0) + 1
        return list(out.values())

    def _set_column(self, column: str, pairs: list[tuple[int, str]]) -> None:
        """이미지의 분할 관련 열을 채운다. updated_at 은 건드리지 않는다 — 검수 내용이
        바뀐 게 아니라서, 이어하기의 '바뀐 이미지' 판정에 섞이면 안 된다."""
        if not pairs:
            return
        with _write_lock, self.connect() as con:
            con.executemany(f"UPDATE images SET {column}=? WHERE id=?",
                            [(v, i) for i, v in pairs])

    def set_splits(self, pairs: list[tuple[int, str]]) -> None:
        """{이미지 id: 'train'|'val'|'test'} 배정을 저장한다."""
        self._set_column("split", pairs)

    def set_groups(self, pairs: list[tuple[int, str]]) -> None:
        """같은 장면 묶음 키를 저장한다 (grouping.py)."""
        self._set_column("grp", pairs)

    def set_dhashes(self, pairs: list[tuple[int, str]]) -> None:
        """이미지 dHash 를 저장한다. 한 번 구하면 다시 계산하지 않는다."""
        self._set_column("dhash", pairs)

    def approved_with_boxes(self, statuses: tuple[str, ...] = ("approved",)) -> list[tuple[dict, list[Box]]]:
        """학습 데이터셋 빌드·내보내기용: 승인된 이미지(statuses 를 주면 그 상태들) + 그 박스 전체. 연결 하나로 읽는다."""
        if not statuses:
            return []
        marks = ",".join("?" * len(statuses))
        with self.connect() as con:
            imgs = con.execute(
                f"SELECT * FROM images WHERE status IN ({marks}) ORDER BY id", list(statuses)
            ).fetchall()
            out = []
            for im in imgs:
                rows = con.execute(
                    "SELECT cls, cx, cy, w, h, conf, origin FROM boxes"
                    " WHERE image_id=? ORDER BY id",
                    (im["id"],),
                ).fetchall()
                out.append((dict(im), [Box(**dict(r)) for r in rows]))
        return out
