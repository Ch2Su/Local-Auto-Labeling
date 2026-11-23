"""이미 있는 이미지를 작업 배치 하나로 묶는다 (2026-11-23, 사용자 결정 — 검수 중이던 AI Hub 100장을 배치로).

작업 배치는 넣을 때 생긴다(영상 캡처의 넣기·이미지 넣기·폴더 스캔 — `src/autolabel/workbatch.py`). 그 기능 전에 넣은
이미지와 `tools/import_*` 가 직접 넣은 이미지는 '배치 없음' 이다 — 그 가운데 고른 것을 배치 하나로 묶는 것이 이 도구다.

**고르는 법**(여럿을 주면 모두 맞는 것): `--source`(images.source — 예: aihub_pm), `--camera-prefix`(카메라 키 앞머리 —
예: `ah-`), `--camera`(키 그대로), `--status`, `--ids`(`8327-8426,8500` — 검수 UI 의 'id 로 열기' 와 같은 꼴).
**이미 다른 배치에 든 이미지는 건드리지 않는다**(몇 장인지만 알린다) — 옮기려면 `--move`.

**기본은 보기만 한다** — DB 를 읽기 전용으로 열어, 묶일 이미지 수를 상태·카메라별로 보여 준다. `--apply` 때만 쓴다.
쓰는 것은 `work_batches` 한 줄과 그 이미지들의 `images.work_batch` 뿐이다: 상태·박스·분할·`updated_at` 은 그대로다
(배치는 라벨이 아니다). 되돌리려면 `--dissolve <번호>`(배치만 지운다 — 이미지는 '배치 없음' 으로 돌아간다).

DB 에 아직 작업 배치 열이 없으면(옛 DB) `--apply` 가 `Store` 를 열 때 더한다 — 더하기 직전에 DB 를
`backups/<이름>_before_workbatch_<시각>.db` 로 뜬다(검수 서버를 새 코드로 켤 때와 같다). 보기만 할 때는 더하지 않는다.

검수 서버가 켜져 있어도 된다(배치 목록은 화면이 다음에 받을 때 보인다 — '작업 배치' 보기의 `↻ 새로 고침`).

    python tools/work_batch.py                                           # 배치 목록과 '배치 없음' 수
    python tools/work_batch.py --source aihub_pm --name "AI Hub 개인형 이동장치"       # 무엇이 묶일지 본다
    python tools/work_batch.py --source aihub_pm --name "AI Hub 개인형 이동장치" --apply
    python tools/work_batch.py --dissolve 3 --apply                      # 배치 3 을 푼다(이미지는 그대로)
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # tools/ -> 저장소 루트

from autolabel.config import Config  # noqa: E402
from autolabel.store import STATUSES, WORK_KINDS, Store, clean_work_name  # noqa: E402
from autolabel.workbatch import KIND_NAMES  # noqa: E402

STATUS_NAMES = {"unlabeled": "미라벨", "pending": "검수 대기", "reviewed": "검수 끝", "approved": "승인",
                "skipped": "건너뜀", "trashed": "휴지통"}
IDS_MAX = 100_000


def parse_ids(text: str) -> list[int]:
    """'8327-8426, 8500' -> [8327, …, 8426, 8500]. 쉼표·공백으로 나누고 범위는 `-`. 읽을 수 없으면 ValueError."""
    out: dict[int, None] = {}
    num = lambda s: s.isascii() and s.isdigit() and len(s) <= 18      # '²' 같은 유니코드 숫자·SQLite 가 못 담는 수는 받지 않는다
    for tok in text.replace(",", " ").split():
        a, sep, b = tok.partition("-")
        if not num(a) or (sep and not num(b)):
            raise ValueError(f"id 로 읽을 수 없는 조각: '{tok}' — 숫자나 범위(8327-8426)를 쉼표·공백으로 구분해 주세요.")
        lo, hi = int(a), int(b) if sep else int(a)
        if hi < lo:
            lo, hi = hi, lo
        if len(out) + hi - lo + 1 > IDS_MAX:
            raise ValueError(f"id 는 한 번에 {IDS_MAX:,}개까지입니다.")
        for i in range(lo, hi + 1):
            out[i] = None
    return list(out)


def read_only(db: Path) -> sqlite3.Connection:
    """DB 를 읽기 전용으로 — 보기만 할 때는 열·표를 더하지도, 사본을 뜨지도 않는다."""
    if not db.is_file():
        sys.exit(f"DB 가 없습니다: {db}")
    con = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True, timeout=30.0)
    con.row_factory = sqlite3.Row
    return con


def has_batches(con: sqlite3.Connection) -> bool:
    return "work_batch" in {r[1] for r in con.execute("PRAGMA table_info(images)")}


def select(con: sqlite3.Connection, args, migrated: bool) -> list[sqlite3.Row]:
    """고른 조건에 맞는 이미지 줄(id, status, camera, work_batch) — 번호순."""
    where, vals = [], []
    if args.source:
        where.append("source = ?")
        vals.append(args.source)
    if args.camera:
        where.append("camera = ?")
        vals.append(args.camera)
    if args.camera_prefix:
        where.append("camera IS NOT NULL AND substr(camera, 1, ?) = ?")      # LIKE 는 '_'·'%' 를 달리 읽는다
        vals += [len(args.camera_prefix), args.camera_prefix]
    if args.status:
        where.append(f"status IN ({','.join('?' * len(args.status))})")
        vals += args.status
    col = "work_batch" if migrated else "NULL AS work_batch"
    sql = f"SELECT id, status, camera, {col} FROM images" + (f" WHERE {' AND '.join(where)}" if where else "") + " ORDER BY id"
    rows = con.execute(sql, vals).fetchall()
    if args.id_list is not None:
        wanted = set(args.id_list)
        rows = [r for r in rows if r["id"] in wanted]
    return rows


def show_batches(con: sqlite3.Connection, migrated: bool) -> None:
    if not migrated:
        n = con.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        print(f"이 DB 에는 아직 작업 배치 열이 없습니다(이미지 {n}장 — 전부 '배치 없음'). 검수 서버를 새 코드로 켜거나 이 도구를"
              " --apply 로 돌리면 더합니다(더하기 직전에 DB 를 backups/ 에 뜹니다).")
        return
    rows = con.execute("SELECT id, kind, name, hold, created_at FROM work_batches ORDER BY id").fetchall()
    counts: dict = {}
    for b, status, n in con.execute("SELECT work_batch, status, COUNT(*) FROM images GROUP BY work_batch, status"):
        counts.setdefault(b, Counter())[status] = n
    line = lambda c: " · ".join(f"{STATUS_NAMES.get(s, s)} {c[s]}" for s in STATUSES if c.get(s)) or "이미지 없음"
    print(f"작업 배치 {len(rows)}개")
    for r in rows:
        c = counts.get(r["id"], Counter())
        print(f"  작업 {r['id']:>3}  {KIND_NAMES.get(r['kind'], r['kind']):<8} {'모아서' if r['hold'] else '바로  '} "
              f"{sum(c.values()):>5}장  {line(c)}" + (f"  — {r['name']}" if r["name"] else ""))
    none = counts.get(None, Counter())
    print(f"  배치 없음            {sum(none.values()):>5}장  {line(none)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="설정 파일 (기본: 저장소의 config.yaml)")
    ap.add_argument("--source", help="images.source 가 이것인 이미지 (예: aihub_pm)")
    ap.add_argument("--camera", help="카메라 키가 이것인 이미지")
    ap.add_argument("--camera-prefix", help="카메라 키가 이것으로 시작하는 이미지 (예: ah-)")
    ap.add_argument("--status", action="append", choices=STATUSES, help="이 상태인 이미지 (여러 번 줄 수 있다)")
    ap.add_argument("--ids", help="이미지 id — 8327-8426,8500 처럼")
    ap.add_argument("--name", help="새 배치의 이름 (없어도 된다)")
    ap.add_argument("--hold", action="store_true", help="'모아서 한꺼번에' 로 만든다 (기본은 '바로 추가' — 승인이 곧 데이터셋)")
    ap.add_argument("--move", action="store_true", help="이미 다른 배치에 든 이미지도 새 배치로 옮긴다")
    ap.add_argument("--dissolve", type=int, metavar="번호", help="그 배치를 지운다 — 이미지는 '배치 없음' 으로 돌아간다")
    ap.add_argument("--apply", action="store_true", help="실제로 DB 를 바꾼다 (없으면 보기만)")
    args = ap.parse_args()
    cfg = Config.load(args.config)
    picking = any([args.source, args.camera, args.camera_prefix, args.status, args.ids])
    try:
        name = clean_work_name(args.name)
        args.id_list = parse_ids(args.ids) if args.ids else None       # 읽을 수 없는 id 는 DB 를 열기 전에 알린다
    except ValueError as exc:
        sys.exit(str(exc))

    con = read_only(cfg.db)
    try:
        migrated = has_batches(con)
        if args.dissolve is not None:
            if picking:
                sys.exit("--dissolve 는 고르는 조건(--source 등)과 함께 쓰지 않습니다.")
            row = con.execute("SELECT id, kind, name FROM work_batches WHERE id=?", (args.dissolve,)).fetchone() if migrated else None
            if row is None:
                sys.exit(f"작업 배치 {args.dissolve} 가 없습니다.")
            n = con.execute("SELECT COUNT(*) FROM images WHERE work_batch=?", (args.dissolve,)).fetchone()[0]
            print(f"작업 {row['id']}({KIND_NAMES.get(row['kind'], row['kind'])}" + (f" — {row['name']}" if row["name"] else "")
                  + f") 을 지웁니다 — 이미지 {n}장은 그대로 남고 '배치 없음' 이 됩니다(상태·박스는 바뀌지 않습니다).")
        elif not picking:
            show_batches(con, migrated)
            return 0
        else:
            rows = select(con, args, migrated)
            taken = [r for r in rows if r["work_batch"] is not None]
            chosen = rows if args.move else [r for r in rows if r["work_batch"] is None]
            if not rows:
                sys.exit("조건에 맞는 이미지가 없습니다.")
            by_status = Counter(r["status"] for r in chosen)
            cams = Counter(r["camera"] or "(출처 없음)" for r in chosen)
            print(f"조건에 맞는 이미지 {len(rows)}장 — 새 배치로 묶을 것 {len(chosen)}장"
                  + (f" (id {chosen[0]['id']}–{chosen[-1]['id']})" if chosen else ""))
            print("  상태: " + (" · ".join(f"{STATUS_NAMES.get(s, s)} {by_status[s]}" for s in STATUSES if by_status.get(s)) or "-"))
            print(f"  카메라 {len(cams)}대" + (": " + ", ".join(f"{k} {n}" for k, n in cams.most_common(8))
                                           + (" …" if len(cams) > 8 else "") if cams else ""))
            if taken:
                where = Counter(r["work_batch"] for r in taken)
                print(f"  이미 다른 배치에 든 것 {len(taken)}장(" + ", ".join(f"작업 {b}: {n}장" for b, n in sorted(where.items()))
                      + ") — " + ("새 배치로 옮깁니다(--move)." if args.move else "건드리지 않습니다(옮기려면 --move)."))
            print(f"  새 배치: 이름 {name or '(없음)'} · {'모아서 한꺼번에' if args.hold else '바로 추가(승인이 곧 데이터셋)'}")
            if not chosen:
                sys.exit("묶을 이미지가 없습니다.")
    finally:
        con.close()

    if not args.apply:
        print("\n보기만 했습니다 — 실제로 바꾸려면 --apply.")
        return 0

    store = Store(cfg.db)            # 열·표가 없으면 여기서 더한다(직전에 사본 — store.migrated_from)
    if store.migrated_from is not None:
        print(f"DB 에 작업 배치(표 work_batches · 열 images.work_batch)를 더했습니다. 더하기 직전의 DB 사본: {store.migrated_from}")
    if args.dissolve is not None:
        ok = store.delete_work_batch(args.dissolve)
        print(f"작업 {args.dissolve} 을 지웠습니다." if ok else f"작업 {args.dissolve} 가 그사이 없어졌습니다.")
        return 0 if ok else 1
    assert "manual" in WORK_KINDS
    batch = store.create_work_batch("manual", name=name, hold=args.hold)
    moved = store.set_work_batch([r["id"] for r in chosen], batch)
    if moved != len(chosen):         # 그사이 지워진 이미지가 있다 — 묶인 것은 그대로 둔다
        print(f"[알림] 고른 {len(chosen)}장 가운데 {moved}장만 묶였습니다(나머지는 그사이 지워졌습니다).")
    if not moved:
        store.delete_work_batch(batch, only_if_empty=True)
        sys.exit("묶인 이미지가 없어 배치를 만들지 않았습니다.")
    print(f"\n작업 {batch} 로 {moved}장을 묶었습니다 — 검수 UI '작업 배치' 에서 엽니다. 되돌리기: --dissolve {batch} --apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
