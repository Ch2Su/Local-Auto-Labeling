"""가져온 보관 폴더가 온전한지 본다 — 읽기만 한다(DB 는 immutable 로 열어 -shm·-wal 도 만들지 않는다).

  python -I verify_archive.py <…/20261110> [<lal_oldcctv8 의 labeling.db>]

보는 것: ① DB 가 보관 기록(ARCHIVE_INFO.txt)의 수와 같은가 ② DB 의 모든 이미지가 이 폴더에 있고 끝까지 열리며 크기·dHash 가
DB 와 같은가 ③ responder·cone 박스가 어디에 얼마나 있는가.
"""
import os
import sqlite3
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

EXPECT = {"이미지": 5931, "승인": 4542, "검수 대기": 1331, "휴지통": 58, "박스": 44786, "승인분 박스": 25447, "태그": 84,
          "이미지 태그": 5775, "가릴 영역": 160, "변경 이력": 12908, "클래스": 10, "train": 3167, "val": 681, "test": 694}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".avif"}
MARK = "\\data\\visible\\"


def dhash(path, size=8):
    """src/autolabel/grouping.py 의 dhash 와 같은 식(옮겨 적음)."""
    try:
        with Image.open(path) as im:
            orient = im.getexif().get(0x0112, 1)
            im.draft("L", (size * 8, size * 8))
            src = ImageOps.exif_transpose(im) if orient in (2, 3, 4, 5, 6, 7, 8) else im
            small = src.convert("L").resize((size + 1, size), Image.BILINEAR)
    except Exception:
        return None
    a = np.asarray(small, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return f"{v:016x}"


def check_one(job):
    iid, path, w, h, want = job
    if not os.path.isfile(path):
        return iid, "missing"
    try:
        with Image.open(path) as im:
            size = im.size
            im.load()                      # 끝까지 디코딩 — 잘린 파일이면 여기서 난다
    except Exception as e:                 # noqa: BLE001
        return iid, f"unreadable:{type(e).__name__}"
    if size != (w, h):
        return iid, f"size:{size[0]}x{size[1]}!={w}x{h}"
    if want and dhash(path) != want:
        return iid, "dhash"
    return iid, "ok"


def kind_of(camera):
    if not camera:
        return "크롭(카메라 없음)"
    return camera.split("-")[0] + "-*"


def main() -> int:
    root = Path(sys.argv[1])
    vis = root / "data" / "visible"
    db = vis / "labeling.db"
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro&immutable=1", uri=True)
    con.row_factory = sqlite3.Row
    one = lambda sql, *a: con.execute(sql, a).fetchone()[0]

    print("== ① DB ==")
    print("크기", db.stat().st_size, "바이트 · journal_mode", one("PRAGMA journal_mode"), "· integrity_check", one("PRAGMA integrity_check"))
    status = dict(con.execute("SELECT status, COUNT(*) FROM images GROUP BY 1").fetchall())
    print("상태별", status)
    got = {"이미지": one("SELECT COUNT(*) FROM images"), "승인": status.get("approved", 0),
           "휴지통": status.get("trashed", 0) + status.get("trash", 0),
           "박스": one("SELECT COUNT(*) FROM boxes"),
           "승인분 박스": one("SELECT COUNT(*) FROM boxes b JOIN images i ON i.id = b.image_id WHERE i.status = 'approved'"),
           "태그": one("SELECT COUNT(*) FROM tags"), "이미지 태그": one("SELECT COUNT(*) FROM image_tags"),
           "가릴 영역": one("SELECT COUNT(*) FROM masks"), "변경 이력": one("SELECT COUNT(*) FROM box_history"),
           "클래스": one("SELECT COUNT(*) FROM classes")}
    got["검수 대기"] = got["이미지"] - got["승인"] - got["휴지통"]
    for s in ("train", "val", "test"):
        got[s] = one("SELECT COUNT(*) FROM images WHERE status = 'approved' AND split = ?", s)
    bad = 0
    for k, want in EXPECT.items():
        ok = got.get(k) == want
        bad += not ok
        print(f"  {'같음' if ok else '다름'}  {k}: {got.get(k)} (기록 {want})")
    print("기록과 다른 항목:", bad)
    names = [r["name"] for r in con.execute("SELECT name FROM classes ORDER BY idx")]
    print("클래스", names)
    print("id 범위", tuple(con.execute("SELECT MIN(id), MAX(id) FROM images").fetchone()))

    print("== ② 파일 ==")
    rows = con.execute("SELECT id, path, width, height, dhash, status, camera, split, domain FROM images").fetchall()
    jobs, mapped, unmapped = [], {}, 0
    for r in rows:
        p = r["path"]
        k = p.lower().find(MARK)
        if k < 0:
            unmapped += 1
            continue
        local = str(vis) + "\\" + p[k + len(MARK):]
        mapped[r["id"]] = local
        jobs.append((r["id"], local, r["width"], r["height"], r["dhash"]))
    pre = Counter(r["path"][: r["path"].lower().find(MARK) + len(MARK)] for r in rows if r["path"].lower().find(MARK) >= 0)
    print("DB 경로 앞부분:", dict(pre), "· 맞출 수 없는 경로", unmapped)
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        res = dict(ex.map(check_one, jobs, chunksize=32))
    print(f"확인한 이미지 {len(res)}장 ({time.time() - t0:.0f}초)")
    verdict = Counter(v.split(":")[0] for v in res.values())
    print("결과", dict(verdict))
    has_hash = sum(1 for j in jobs if j[4])
    print("dHash 가 DB 에 있어 견준 이미지", has_hash, "장")
    by_id = {r["id"]: r for r in rows}
    for name in ("missing", "unreadable", "size", "dhash"):
        ids = [i for i, v in res.items() if v.split(":")[0] == name]
        if ids:
            c = Counter((by_id[i]["status"], kind_of(by_id[i]["camera"])) for i in ids)
            print(f"  [{name}] {len(ids)}장 — 상태·종류별 {dict(c)} · 예: id {sorted(ids)[:8]} {res[sorted(ids)[0]]}")
    # 디스크에는 있는데 DB 가 모르는 이미지 파일
    known = {p.lower() for p in mapped.values()}
    disk = Counter()
    extra = Counter()
    for top in ("legacy", "labeled", "unlabeled"):
        for base, _, files in os.walk(vis / top):
            for f in files:
                if os.path.splitext(f)[1].lower() in IMG_EXTS:
                    disk[top] += 1
                    if os.path.join(base, f).lower() not in known:
                        extra[top] += 1
    print("이미지 파일 수", dict(disk), "합", sum(disk.values()), "· DB 에 없는 파일", dict(extra))
    vids = [p for p in (vis / "videos").rglob("*") if p.is_file()]
    print("영상", len(vids), "개", f"{sum(p.stat().st_size for p in vids) / 1e6:.0f} MB")
    total = sum(os.path.getsize(os.path.join(b, f)) for b, _, fs in os.walk(root) for f in fs)
    print(f"폴더 전체 {total / 1e9:.2f} GB ({total / 2**30:.2f} GiB)")

    print("== ③ responder·cone ==")
    for cname in ("responder", "cone"):
        idx = names.index(cname)
        q = con.execute("SELECT b.image_id, b.w, b.h, b.origin, b.conf, i.status, i.camera, i.width, i.height, i.split, i.domain "
                        "FROM boxes b JOIN images i ON i.id = b.image_id WHERE b.cls = ?", (idx,)).fetchall()
        print(f"[{cname}] 박스 {len(q)}개 · 이미지 {len({r['image_id'] for r in q})}장")
        for label, key in (("상태", lambda r: r["status"]), ("종류", lambda r: kind_of(r["camera"])),
                           ("박스 출처", lambda r: r["origin"] or "(없음)"), ("도메인", lambda r: r["domain"] or "(없음)"),
                           ("분할", lambda r: r["split"] or "(미배정)")):
            bc = Counter(key(r) for r in q)
            ic = Counter()
            for i, k in {(r["image_id"], key(r)) for r in q}:
                ic[k] += 1
            print(f"   {label}별 박스 {dict(bc.most_common())} · 이미지 {dict(ic.most_common())}")
        for k in sorted({kind_of(r["camera"]) for r in q}):
            sides = sorted(min(r["w"] * r["width"], r["h"] * r["height"]) for r in q if kind_of(r["camera"]) == k)
            longs = sorted(max(r["width"], r["height"]) for r in q if kind_of(r["camera"]) == k)
            print(f"   {k}: 짧은 변 중앙값 {statistics.median(sides):.0f}px (최소 {sides[0]:.0f}·최대 {sides[-1]:.0f}) · "
                  f"그 이미지 긴 변 중앙값 {statistics.median(longs):.0f}px")
        cams = Counter(r["camera"] for r in q if r["camera"])
        print("   카메라별 박스", dict(sorted(cams.items())))
    rc = {r[0] for r in con.execute("SELECT DISTINCT image_id FROM boxes WHERE cls IN (?, ?)", (names.index("responder"), names.index("cone")))}
    st = Counter((by_id[i]["status"], kind_of(by_id[i]["camera"])) for i in rc)
    print("두 클래스 박스가 든 이미지", len(rc), "장 — 상태·종류별", dict(st.most_common()))
    print("  그 파일의 확인 결과", dict(Counter(res.get(i, "경로 못 맞춤").split(":")[0] for i in rc)))
    owl = con.execute("SELECT batch, COUNT(*), COUNT(DISTINCT image_id) FROM box_history WHERE batch LIKE 'OWL%' GROUP BY 1 ORDER BY 1").fetchall()
    print("OWLv2 묶음(이름, 이력 행, 이미지):", [tuple(r) for r in owl])
    cams = con.execute("SELECT camera, COUNT(*), SUM(status = 'approved'), SUM(status = 'pending') FROM images WHERE camera LIKE 'hw-%' GROUP BY 1 ORDER BY 1").fetchall()
    print("hw 카메라(키, 전체, 승인, 검수 대기):", [tuple(r) for r in cams], "합", sum(r[1] for r in cams))

    # lal_oldcctv8 을 만든 그 DB 인가 — 같은 조건으로 고르면 같은 id 여야 한다
    sel = [r[0] for r in con.execute("SELECT i.id FROM images i WHERE i.status = 'approved' AND i.domain = 'cctv' AND NOT EXISTS "
                                     "(SELECT 1 FROM boxes b WHERE b.image_id = i.id AND b.cls IN (?, ?)) ORDER BY i.id",
                                     (names.index("responder"), names.index("cone")))]
    print("같은 조건(승인·cctv·두 클래스 없음)으로 고른 이미지", len(sel), "장")
    if len(sys.argv) > 2:
        sub = sqlite3.connect(f"file:{Path(sys.argv[2]).as_posix()}?mode=ro&immutable=1", uri=True)
        ids = [r[0] for r in sub.execute("SELECT id FROM images ORDER BY id")]
        print("  lal_oldcctv8 의 id 와", "같다" if ids == sel else f"다르다(그쪽 {len(ids)}장, 공통 {len(set(ids) & set(sel))})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
