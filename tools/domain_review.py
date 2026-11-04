"""도메인(CCTV·비CCTV·드론)과 태그를 데이터셋 전체에서 점검한다 (DB 읽기 전용, 2026-11-04).

사용자가 검수 UI 의 Q·W·E·T 로 6천여 장을 한 장씩 나눴다(11-03~04). 손으로 한 것이라 같은 장면·같은
태그 안에서 어긋나거나, 비슷한 사진끼리 다르게 들어간 것이 섞일 수 있다. 의심 이미지를 이유와 함께
고르고, 검수 UI 'id 로 열기' 에 붙여 넣을 id 목록과 대조 시트를 만든다. **DB 에는 아무것도 쓰지 않는다** —
고치는 것은 사람이 검수 UI 에서 한다.

    python tools/domain_review.py              # → data/visible/domain_review/
    python tools/domain_review.py --no-sheets  # 표만 (임베딩은 legacy_groups 의 캐시를 같이 쓴다)
    python tools/domain_review.py --mark-seen  # 다 본 뒤 — 지금 걸린 것을 seen.csv 에 적어 다음부터 새로 걸린 것만 낸다
    python tools/domain_review.py --mark-tag "<태그 이름>"   # '일부가 빠진 태그' 표에서 그 태그 줄을 다 본 뒤

도메인 검사 (한 이미지에 여럿 걸릴 수 있다):
  group  같은 장면 묶음(grouping.group_images — 지금 규칙으로 다시 묶은 것)의 대부분과 도메인이 다르다
  tag    같은 태그의 대부분(--tag-major 이상)과 도메인이 다르다
  dup    dHash 거리 --dup-dist 이하(묶음 기준 2 보다 느슨)인 다른 묶음 이미지와 도메인이 다르다 — 대비가 낮은
         화면(밤 도로·안개)끼리는 다른 사진이어도 가깝게 나온다
  knn    비슷한 이미지 k 개(사전학습 yolo26s 임베딩, 같은 장면 묶음은 뺀다)의 대부분이 다른 도메인이다
  exif   EXIF 제조사가 DJI 인데 드론이 아니다 / 폰·카메라 제조사인데 CCTV 다
**의심일 뿐이다** — 임베딩은 장면 내용을 볼 뿐 촬영 방식을 알아보지는 않는다(높은 곳에서 찍은 폰 사진은
CCTV 처럼 보인다). 시트로 보고 사람이 정한다.

태그 검사: 태그마다 도메인·분할·장면 묶음 수, 이미지 집합이 같은 태그·포함 관계·일부만 겹치는 태그,
장면 묶음(3장 이상)의 절반 이상에 붙었는데 나머지에 빠진 태그('일부가 빠진 태그' — 빠진 이미지에 붙일지),
다른 태그·파일 이름과 한두 글자 다른 태그 이름, 아직 태그 없는 큰 장면 묶음. 분할(train·val·test)에 걸친
태그와, 잎 태그(다른 태그를 품지 않은 것 — 카메라·장소·작품)를 카메라로 보고 train 과 같은 태그를 쓰는 test
이미지(같은 카메라 누수)도 낸다 — 다시 나눌 때 한쪽에 모아야 한다.
**장면 묶음은 legacy 에서 믿을 만하지 않다** — 30초 안에 찍은 스크린샷은 다른 라이브캠이어도 한 묶음이 되고,
크기가 같은 숫자 이름(로보플로 256×256 크롭)은 CCTV 크롭과 창밖 폰 사진을 한 묶음으로 만든다(11-04 확인).
group·'일부가 빠진 태그' 검사는 그래서 헛것이 많다.

이미 본 것 (11-04): 사람이 본 판정을 점검 때마다 다시 묻지 않으려고, `--mark-seen` 이 지금 걸린 (이미지, 그때
도메인, 검사) 를 `seen.csv` 에 적는다. 다음부터 같은 이미지가 **같은 도메인으로 같은 검사에** 걸리면 report.md 의
id 목록·시트에서 빼고 수만 센다(`--all` 이면 다 낸다. flags.csv 는 늘 전부이고 `new` 칸이 새로 걸린 검사다).
도메인을 바꾸면 다시 나온다. 이유 글(묶음 장수·이웃 비율)은 보지 않는다 — 다른 이미지를 고칠 때마다 바뀐다.
'일부가 빠진 태그' 표는 따로 적는다 — 시트를 다 본 것과 이 표를 본 것은 다른 일이다. 한 태그의 줄을 다 봤으면
`--mark-tag <태그 이름>`(여러 번 줄 수 있다)이 그 태그의 빠진 이미지를 (이미지, `tagmiss:<태그 id>`) 로 적고,
다음부터 그 줄에서 뺀다. 도메인은 보지 않고, 태그는 id 라 이름을 바꿔도 그대로다. 나중에 그 태그를 붙이면 빠진
이미지가 아니니 저절로 빠진다.

나오는 것 (`--out`, git 밖):
  report.md    요약 — 표와 'id 로 열기' 에 붙여 넣을 id 목록
  flags.csv    의심 이미지마다 검사·새로 걸린 검사·이유·이웃 투표·태그·묶음·분할·경로
  tags.csv     태그마다 장수·도메인·분할·묶음 수·대표 이름 계열
  sheets/      검사별 대조 시트(캡션: id·도메인→볼 만한 도메인·분할 / 이유 / 태그)
  seen.csv     `--mark-seen`·`--mark-tag` 로 표시한 이미 본 의심(id·도메인·검사·날짜 — 빠진 태그는 검사 칸이
               `tagmiss:<태그 id>` 이고 도메인 칸은 비운다). 이 파일만 다음 실행이 읽는다. 지우면 다시 전부
               나온다(`--seen` 으로 다른 파일)
"""

from __future__ import annotations

import argparse
import csv
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # tools/ -> 저장소 루트

from autolabel.config import Config, pretrained_path, resolve
from autolabel.grouping import group_images
from legacy_groups import embeddings, family, font   # 같은 폴더의 도구 — 임베딩 캐시도 같이 쓴다

DOMS = ("cctv", "noncctv", "drone")
DN = {"cctv": "CCTV", "noncctv": "비CCTV", "drone": "드론", None: "미분류"}
CHECKS = {
    "group": "같은 장면 묶음과 다름",
    "tag": "같은 태그와 다름",
    "dup": "거의 같은 사진과 다름",
    "knn": "비슷한 이미지와 다름",
    "exif": "EXIF 제조사와 안 맞음",
}
# 폰·카메라 제조사(EXIF Make, 소문자 앞부분). DJI 는 따로 본다.
PHOTO_MAKES = ("apple", "samsung", "google", "xiaomi", "huawei", "lge", "lg electronics", "oneplus", "oppo",
               "vivo", "motorola", "sony", "canon", "nikon", "fujifilm", "panasonic", "olympus", "pentax",
               "ricoh", "leica", "gopro")
THUMB = 200
SHEET_COLS = 6


def load(cfg: Config) -> tuple[list[dict], dict[int, str], list[str]]:
    con = sqlite3.connect(f"file:{cfg.db.as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(
        "SELECT id, name, path, source, split, grp, dhash, width, height, camera, domain"
        " FROM images WHERE status != 'trashed' ORDER BY id")]
    tag_name = {r["id"]: r["name"] for r in con.execute("SELECT id, name FROM tags")}
    tags: dict[int, set[int]] = defaultdict(set)
    for r in con.execute("SELECT image_id, tag_id FROM image_tags"):
        tags[r["image_id"]].add(r["tag_id"])
    boxes: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for r in con.execute("SELECT image_id, cls, h FROM boxes"):
        boxes[r["image_id"]].append((r["cls"], r["h"]))
    classes = [r["name"] for r in con.execute("SELECT idx, name FROM classes ORDER BY idx")]
    con.close()
    # DB 의 grp 는 마지막 분할 때 것이다(옛 g000001 처럼 섞인 것이 남아 있다) — 지금 규칙으로 다시 묶는다
    fresh = group_images(rows)
    for r in rows:
        r["db_grp"], r["grp"] = r["grp"], fresh[r["id"]]
        r["tags"] = tags.get(r["id"], set())
        r["boxes"] = boxes.get(r["id"], [])
        r["family"] = family(r["path"])
    return rows, tag_name, classes


def old_domain(r: dict) -> str:
    """test_baseline.py 가 지금까지 쓴 규칙(경로·출처로 가른 것)."""
    if "stabilized_" in r["path"].lower():
        return "drone"
    return "cctv" if r["source"] == "seed" else "noncctv"


def read_exif(r: dict) -> None:
    r["make"], r["model"], r["gps"] = "", "", False
    try:
        with Image.open(r["path"]) as im:
            ex = im.getexif()
    except OSError:
        return
    r["make"] = str(ex.get(271) or "").strip(" \x00")
    r["model"] = str(ex.get(272) or "").strip(" \x00")
    r["gps"] = 34853 in ex


def exif_hint(r: dict) -> str | None:
    """EXIF 제조사로 본 도메인. 모르면 None."""
    make = r["make"].lower()
    if make.startswith("dji"):
        return "drone"
    if make.startswith(PHOTO_MAKES):
        return "noncctv"
    return None


def knn_vote(rows: list[dict], emb: np.ndarray, k: int) -> np.ndarray:
    """(n, 3) — 같은 장면 묶음을 뺀 가장 비슷한 k 장의 도메인 비율(유사도 가중). 미분류 이웃은 세지 않는다."""
    lab = np.array([DOMS.index(r["domain"]) if r["domain"] in DOMS else -1 for r in rows])
    gid = {g: i for i, g in enumerate(sorted({r["grp"] for r in rows}))}
    grp = np.array([gid[r["grp"]] for r in rows])
    out = np.zeros((len(rows), len(DOMS)))
    for a in range(0, len(rows), 512):
        s = emb[a:a + 512] @ emb.T
        s[(grp[a:a + 512, None] == grp[None, :]) | (lab[None, :] < 0)] = -np.inf
        top = np.argpartition(-s, k, axis=1)[:, :k]
        w = np.take_along_axis(s, top, axis=1).clip(0)
        for j in range(len(DOMS)):
            out[a:a + len(top), j] = (w * (lab[top] == j)).sum(1)
    return out / out.sum(1, keepdims=True).clip(1e-8)


def dup_pairs(rows: list[dict], max_dist: int) -> list[tuple[int, int, int]]:
    """(행 번호 a, b, dHash 거리) — 거리 max_dist 이하."""
    have = [(i, int(r["dhash"], 16)) for i, r in enumerate(rows) if r.get("dhash")]
    idx = np.array([i for i, _ in have])
    h = np.array([v for _, v in have], dtype=np.uint64)
    out = []
    for a in range(len(h) - 1):
        d = np.bitwise_count(h[a] ^ h[a + 1:])
        for b in np.nonzero(d <= max_dist)[0]:
            out.append((int(idx[a]), int(idx[a + 1 + b]), int(d[b])))
    return out


def lev(a: str, b: str) -> int:
    """편집 거리."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def norm(s: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", s.lower())


def fam_words(fam: str) -> str:
    """이름 계열에서 낱말만 — 번호(#)와 로보플로가 붙인 `_png`·`_jpg` 를 뗀다."""
    return norm(re.sub(r"_(png|jpe?g)\b", "", fam.replace("#", "")))


def flag(r: dict, check: str, sugg: str | None, why: str) -> None:
    r.setdefault("flags", {})[check] = (sugg, why)


def sheet(items: list[dict], title: str, path: Path, cap) -> None:
    """썸네일 대조 시트. cap(it) -> 캡션 세 줄."""
    f_title, f_cap = font(19), font(12)
    n_rows = (len(items) + SHEET_COLS - 1) // SHEET_COLS
    cell_h = THUMB + 46
    canvas = Image.new("RGB", (SHEET_COLS * THUMB, 34 + n_rows * cell_h), "white")
    d = ImageDraw.Draw(canvas)
    d.text((8, 6), title, fill="black", font=f_title)
    for k, it in enumerate(items):
        x, y = (k % SHEET_COLS) * THUMB, 34 + (k // SHEET_COLS) * cell_h
        try:
            with Image.open(it["path"]) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((THUMB - 6, THUMB - 6))
                canvas.paste(im, (x + (THUMB - im.width) // 2, y + (THUMB - im.height) // 2))
        except OSError:
            d.text((x + 8, y + 8), "열 수 없음", fill="red", font=f_cap)
        for n, (text, color) in enumerate(cap(it)):
            d.text((x + 3, y + THUMB + 1 + 14 * n), text[:30], fill=color, font=f_cap)
    canvas.save(path, quality=85)


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:   # 엑셀이 한글을 읽게 BOM
        w = csv.DictWriter(fh, list(rows[0]), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_seen(path: Path) -> dict[tuple[int, str], tuple[str, str]]:
    """seen.csv → {(id, 검사): (표시할 때 도메인 — 미분류는 "", 표시한 날)}. 파일이 없으면 빈 것."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return {(int(r["id"]), r["check"]): (r["domain"], r["seen_at"]) for r in csv.DictReader(fh) if r.get("id")}


def tagmiss(tag_id: int) -> str:
    """'일부가 빠진 태그' 표의 빠진 이미지를 seen.csv 에 적을 때의 검사 이름 — 태그는 id 로(이름은 바뀔 수 있고,
    tags.id 는 AUTOINCREMENT 라 지운 태그의 번호를 새 태그가 받지 않는다)."""
    return f"tagmiss:{tag_id}"


def table(head: list[str], body: list[list]) -> list[str]:
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + \
           ["| " + " | ".join(str(c) for c in row) + " |" for row in body]


def ids_line(rows: list[dict]) -> str:
    return ", ".join(str(r["id"]) for r in rows)


def main() -> None:
    p = argparse.ArgumentParser(description="도메인·태그 점검 (DB 읽기 전용).")
    p.add_argument("--config", default=None)
    p.add_argument("--weights", default="yolo26s.pt", help="임베딩에 쓸 가중치 (weights/ 에서 찾는다)")
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--k", type=int, default=15, help="이웃 수")
    p.add_argument("--knn-own", type=float, default=0.2, help="이웃 투표에서 자기 도메인이 이 이하이고")
    p.add_argument("--knn-other", type=float, default=0.6, help="다른 한 도메인이 이 이상이면 knn 의심")
    p.add_argument("--tag-major", type=float, default=0.8, help="태그의 이 비율 이상이 한 도메인이면 나머지를 의심")
    # 6 으로 처음 돌렸더니 18장이 모두 밝기 흐름만 비슷한 다른 사진이었다(밤 도로·안개 낀 화면, 11-04)
    p.add_argument("--dup-dist", type=int, default=4, help="dHash 거리 (묶음 기준은 2)")
    p.add_argument("--broad-tags", default="Korea",
                   help="하위 태그는 없지만 나라처럼 넓은 태그(쉼표로) — 같은 카메라 누수 집계에서 뺀다")
    p.add_argument("--per-sheet", type=int, default=36)
    p.add_argument("--out", default="data/visible/domain_review")
    p.add_argument("--emb-cache", default="data/visible/legacy_groups", help="임베딩 캐시 폴더 (legacy_groups 와 같이)")
    p.add_argument("--no-sheets", action="store_true")
    p.add_argument("--seen", default=None, help="이미 본 의심 목록 (기본 <out>/seen.csv)")
    p.add_argument("--mark-seen", action="store_true", help="지금 걸린 것을 이미 본 것으로 적는다 — 다음 실행부터 빠진다")
    p.add_argument("--mark-tag", action="append", default=[], metavar="태그",
                   help="'일부가 빠진 태그' 표에서 이 태그(이름 그대로)의 빠진 이미지를 이미 본 것으로 적는다 — 여러 번 "
                        "줄 수 있다. --mark-seen 은 이 표를 적지 않는다")
    p.add_argument("--all", action="store_true", help="이미 본 것도 report.md 목록·시트·'일부가 빠진 태그' 표에 낸다")
    args = p.parse_args()

    cfg = Config.load(args.config)
    out = resolve(args.out)
    (out / "sheets").mkdir(parents=True, exist_ok=True)
    rows, tag_name, classes = load(cfg)
    tag_of = {n: t for t, n in tag_name.items()}
    unknown = [n for n in args.mark_tag if n not in tag_of]
    if unknown:
        p.error("--mark-tag: 없는 태그 " + ", ".join(f"'{n}'" for n in unknown) + " — 이름을 그대로(대소문자·띄어쓰기까지)")
    mark_tags = {tag_of[n] for n in args.mark_tag}
    missing = [r for r in rows if not Path(r["path"]).exists()]
    if missing:
        print(f"  파일 없음 {len(missing)}장 — 뺀다 (예: {missing[0]['path']})")
        rows = [r for r in rows if Path(r["path"]).exists()]
    print(f"{len(rows)}장 · 장면 묶음 {len({r['grp'] for r in rows})}개 · 태그 {len(tag_name)}개")

    for r in rows:
        read_exif(r)
    weights = Path(args.weights) if Path(args.weights).exists() else Path(pretrained_path(args.weights))
    cache = resolve(args.emb_cache)
    cache.mkdir(parents=True, exist_ok=True)
    emb = embeddings(rows, weights, args.imgsz, cache / f"emb_{weights.stem}_{args.imgsz}.npz")
    vote = knn_vote(rows, emb, args.k)
    for r, v in zip(rows, vote):
        r["vote"] = {d: float(x) for d, x in zip(DOMS, v)}

    by_grp: dict[str, list[dict]] = defaultdict(list)
    by_tag: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_grp[r["grp"]].append(r)
        for t in r["tags"]:
            by_tag[t].append(r)
    tag_str = lambda r: " ".join(f"#{tag_name[t]}" for t in sorted(r["tags"], key=lambda t: tag_name[t]))

    # ── 도메인 검사 ──
    for g, its in by_grp.items():
        c = Counter(r["domain"] for r in its)
        if len(its) < 2 or len(c) < 2:
            continue
        (maj, m), = c.most_common(1)
        tie = list(c.values()).count(m) > 1
        for r in its:
            if tie or r["domain"] != maj:
                flag(r, "group", None if tie else maj,
                     f"묶음 {g}({len(its)}장): " + "·".join(f"{DN[d]} {n}" for d, n in c.most_common()))
    mixed_tags = []
    for t, its in by_tag.items():
        c = Counter(r["domain"] for r in its)
        (maj, m), = c.most_common(1)
        if len(c) < 2:
            continue
        if m / len(its) >= args.tag_major:
            for r in its:
                if r["domain"] != maj:
                    flag(r, "tag", maj, f"태그 '{tag_name[t]}' {len(its)}장 중 {DN[maj]} {m}")
        else:
            mixed_tags.append((t, c))
    pairs = dup_pairs(rows, args.dup_dist)
    for a, b, dist in pairs:
        ra, rb = rows[a], rows[b]
        if ra["domain"] != rb["domain"] and ra["grp"] != rb["grp"]:
            flag(ra, "dup", rb["domain"], f"dHash {dist}: {rb['id']}({DN[rb['domain']]})")
            flag(rb, "dup", ra["domain"], f"dHash {dist}: {ra['id']}({DN[ra['domain']]})")
    for r in rows:
        own = r["vote"].get(r["domain"], 0.0)
        best = max(DOMS, key=lambda d: r["vote"][d])
        if r["domain"] in DOMS and own <= args.knn_own and best != r["domain"] and r["vote"][best] >= args.knn_other:
            flag(r, "knn", best, f"이웃 {DN[best]} {r['vote'][best]:.0%} · 자기 {own:.0%}")
        hint = exif_hint(r)
        if hint == "drone" and r["domain"] != "drone":
            flag(r, "exif", "drone", f"EXIF {r['make']} {r['model']}")
        elif hint == "noncctv" and r["domain"] == "cctv":
            flag(r, "exif", "noncctv", f"EXIF {r['make']} {r['model']}" + (" · GPS" if r["gps"] else ""))
    flagged = [r for r in rows if r.get("flags")]
    # 여러 검사에 걸린 것, 이웃이 강하게 다르게 본 것부터
    flagged.sort(key=lambda r: (-len(r["flags"]), r["vote"].get(r["domain"], 0.0), r["id"]))

    # 이미 본 것 — 같은 이미지·같은 도메인·같은 검사. 이 실행에서 표시하는 것은 다음 실행부터 빠진다(적는 것은
    # '일부가 빠진 태그' 까지 센 뒤 아래에서 한꺼번에)
    seen_path = resolve(args.seen) if args.seen else out / "seen.csv"
    seen = read_seen(seen_path)
    for r in flagged:
        r["new"] = [c for c in r["flags"] if seen.get((r["id"], c), ("?",))[0] != (r["domain"] or "")]
    listed = lambda r, c: c in r["flags"] and (args.all or c in r["new"])   # report.md 목록·시트에 낼 것

    # 이웃 투표가 사람 판정과 맞는 비율 — 임베딩이 도메인을 얼마나 가르는지(같은 장면 묶음은 뺐다)
    hit = Counter((r["domain"], max(DOMS, key=lambda d: r["vote"][d])) for r in rows if r["domain"] in DOMS)

    # ── 태그 검사 ──
    sets = {t: {r["id"] for r in its} for t, its in by_tag.items()}
    same, subset, partial = [], [], []
    tl = sorted(sets)
    for i, a in enumerate(tl):
        for b in tl[i + 1:]:
            n = len(sets[a] & sets[b])
            if not n:
                continue
            if sets[a] == sets[b]:
                same.append((a, b))
            elif sets[a] <= sets[b] or sets[b] <= sets[a]:
                small, big = (a, b) if len(sets[a]) <= len(sets[b]) else (b, a)
                subset.append((small, big))
            else:
                partial.append((a, b, n))
    names = {t: norm(tag_name[t]) for t in tag_name}
    nodigit = {t: re.sub(r"\d", "", names[t]) for t in tag_name}
    tl_all = sorted(tag_name)
    similar = [(a, b) for i, a in enumerate(tl_all) for b in tl_all[i + 1:]     # 번호만 다른 것(Drone_1·_2)은 뺀다
               if names[a] and names[b] and nodigit[a] != nodigit[b] and 0 < lev(names[a], names[b]) <= 2]
    fam_off = []
    for t, its in by_tag.items():
        (fam, fn), = Counter(r["family"] for r in its).most_common(1)
        f = fam_words(fam)
        if len(names[t]) >= 6 and f and names[t] != f and names[t] not in f and lev(names[t], f) <= 2:
            fam_off.append((t, fam, fn))
    # 일부가 빠진 태그: 3장 이상 묶음에서 절반 이상에 붙었는데 전부는 아닌 태그. 빠진 이미지 중 seen.csv 에
    # (이미지, tagmiss:<태그 id>) 로 적힌 것은 사람이 보고 안 붙이기로 한 것이다 — 도메인은 보지 않는다
    partial_in_grp = []      # (묶음, 태그, 붙은 수, 묶음 장수, 빠진 이미지, 그중 새로 걸린 것)
    for g, its in by_grp.items():
        if len(its) < 3:
            continue
        for t, c in Counter(t for r in its for t in r["tags"]).items():
            if c < len(its) and c / len(its) >= 0.5:
                miss = [r for r in its if t not in r["tags"]]
                partial_in_grp.append((g, t, c, len(its), miss,
                                       [r for r in miss if (r["id"], tagmiss(t)) not in seen]))
    partial_in_grp.sort(key=lambda x: (-len(x[4]), x[0]))
    n_miss = sum(len(x[4]) for x in partial_in_grp)
    n_miss_new = sum(len(x[5]) for x in partial_in_grp)
    miss_new_txt = f"새로 걸린 것 {sum(1 for x in partial_in_grp if x[5])}줄 · {n_miss_new}장"

    # 표시 — --mark-seen 은 지금 걸린 도메인 의심, --mark-tag 는 이 표에서 그 태그의 빠진 이미지. 따로 둔다 —
    # 시트를 다 본 것과 이 표를 본 것은 다른 일이다
    marked = marked_tm = 0
    if args.mark_seen or mark_tags:
        today = f"{datetime.now():%Y-%m-%d}"
        seen_out = dict(seen)
        if args.mark_seen:
            for r in flagged:
                for c in r["new"]:
                    seen_out[(r["id"], c)] = (r["domain"] or "", today)
                    marked += 1
        for g, t, c, n, miss, miss_new in partial_in_grp:
            if t in mark_tags:
                for r in miss_new:
                    seen_out[(r["id"], tagmiss(t))] = ("", today)
                    marked_tm += 1
        if marked or marked_tm:
            seen_path.parent.mkdir(parents=True, exist_ok=True)
            write_csv(seen_path, [{"id": i, "domain": d, "check": c, "seen_at": at}
                                  for (i, c), (d, at) in sorted(seen_out.items())])
    untagged_grp = sorted(((g, its) for g, its in by_grp.items() if not any(r["tags"] for r in its)),
                          key=lambda x: -len(x[1]))
    tag_rows = []
    for t, its in sorted(by_tag.items(), key=lambda x: -len(x[1])):
        dc, sc = Counter(r["domain"] for r in its), Counter(r["split"] for r in its)
        tag_rows.append({
            "tag_id": t, "name": tag_name[t], "images": len(its), "groups": len({r["grp"] for r in its}),
            "cctv": dc["cctv"], "noncctv": dc["noncctv"], "drone": dc["drone"], "unclassified": dc[None],
            "train": sc["train"], "val": sc["val"], "test": sc["test"], "unassigned": sc[None],
            "families": " | ".join(f"{f}({n})" for f, n in Counter(r["family"] for r in its).most_common(3)),
        })
    empty_tags = sorted(t for t in tag_name if t not in by_tag)

    # ── 써 내기 ──
    write_csv(out / "flags.csv", [{
        "id": r["id"], "domain": r["domain"] or "",
        "checks": " ".join(r["flags"]),
        "new": " ".join(r["new"]),
        "suggest": " ".join(sorted({s for s, _ in r["flags"].values() if s})),
        "why": " / ".join(w for _, w in r["flags"].values()),
        "vote_cctv": f"{r['vote']['cctv']:.2f}", "vote_noncctv": f"{r['vote']['noncctv']:.2f}",
        "vote_drone": f"{r['vote']['drone']:.2f}",
        "tags": tag_str(r), "grp": r["grp"], "grp_n": len(by_grp[r["grp"]]), "split": r["split"] or "",
        "size": f"{r['width']}x{r['height']}", "exif": f"{r['make']} {r['model']}".strip(),
        "name": r["name"], "path": r["path"],
    } for r in flagged] or [{"id": ""}])
    if tag_rows:
        write_csv(out / "tags.csv", tag_rows)

    if not args.no_sheets:
        sd = out / "sheets"
        for old in sd.glob("*.jpg"):
            old.unlink()

        def cap(check):
            def f(r):
                sugg, why = r["flags"][check]
                return [(f"{r['id']} {DN[r['domain']]}→{DN.get(sugg, '?') if sugg else '?'} {r['split'] or '-'}", "black"),
                        (why, "#a33"), (tag_str(r) or "(태그 없음)", "#555")]
            return f
        only_new = "새로 걸린 것 " if seen and not args.all else ""
        for check in CHECKS:
            its = [r for r in flagged if listed(r, check)]
            for k in range(0, len(its), args.per_sheet):
                sheet(its[k:k + args.per_sheet],
                      f"{check} — {CHECKS[check]} · {only_new}{k + 1}–{min(k + args.per_sheet, len(its))} / {len(its)}장",
                      sd / f"{check}_{k // args.per_sheet + 1:02d}.jpg", cap(check))

    dc = Counter(r["domain"] for r in rows)
    lines = [f"# 도메인·태그 점검 — {datetime.now():%Y-%m-%d %H:%M} (DB 읽기 전용, tools/domain_review.py)", "",
             f"{len(rows):,}장 · 장면 묶음 {len(by_grp):,}개 · 태그 {len(tag_name)}개(붙은 이미지 "
             f"{sum(1 for r in rows if r['tags']):,}장)", "", "## 1. 도메인 현황", ""]
    sp = ("train", "val", "test", None)
    lines += table(["도메인", "장수", *[s or "미배정" for s in sp], "test 비율"],
                   [[DN[d], dc[d], *[sum(1 for r in rows if r["domain"] == d and r["split"] == s) for s in sp],
                     f"{sum(1 for r in rows if r['domain'] == d and r['split'] == 'test') / max(dc[d], 1):.1%}"]
                    for d in (*DOMS, None) if dc[d]])
    lines += ["", "클래스별 박스 수(도메인별):", ""]
    lines += table(["도메인", *classes],
                   [[DN[d], *[sum(1 for r in rows if r["domain"] == d for c, _ in r["boxes"] if c == i)
                              for i in range(len(classes))]] for d in DOMS if dc[d]])
    lines += ["", "test_baseline.py 의 옛 규칙(드론=경로에 stabilized_, CCTV=seed, 비CCTV=unlabeled) → 지금 판정:", ""]
    oc = Counter((old_domain(r), r["domain"]) for r in rows)
    lines += table(["옛 규칙", *[DN[d] for d in (*DOMS, None)]],
                   [[DN[o], *[oc[(o, d)] for d in (*DOMS, None)]] for o in DOMS])
    lines += ["", f"이웃 투표(k={args.k}, 같은 장면 묶음 뺌)가 사람 판정과 같은 비율 — 임베딩이 도메인을 얼마나 가르나:", ""]
    lines += table(["사람 판정", "장수", *[f"이웃 {DN[d]}" for d in DOMS], "맞음"],
                   [[DN[d], dc[d], *[hit[(d, e)] for e in DOMS], f"{hit[(d, d)] / max(dc[d], 1):.0%}"]
                    for d in DOMS if dc[d]])
    makes = Counter((r["make"] or "(없음)", r["domain"]) for r in rows)
    lines += ["", "EXIF 제조사(Make)별 도메인 — 없음 외 위 10개:", ""]
    top = Counter({m: sum(n for (mm, _), n in makes.items() if mm == m) for m, _ in makes}).most_common(11)
    lines += table(["Make", *[DN[d] for d in DOMS]], [[m, *[makes[(m, d)] for d in DOMS]] for m, _ in top])

    n_new = sum(1 for r in flagged if r["new"])
    lines += ["", "## 2. 도메인 의심 이미지", "",
              f"모두 {len(flagged)}장(검사 둘 이상 {sum(1 for r in flagged if len(r['flags']) > 1)}장). "
              "id 목록은 검수 UI 'id 로 열기' 에 그대로 붙여 넣는다.", ""]
    if seen:
        lines += [f"**새로 걸린 것 {n_new}장**, 이미 본 것 {len(flagged) - n_new}장 — `{seen_path.name}`(마지막 표시 "
                  f"{max(at for _, at in seen.values())})에 같은 도메인·같은 검사로 적힌 것. "
                  + ("아래 목록·시트는 이미 본 것까지 전부다(`--all`)." if args.all
                     else "아래 목록·시트는 새로 걸린 것만이다(전부는 `--all`, flags.csv 는 늘 전부 — `new` 칸)."), ""]
    if args.mark_seen or mark_tags:
        lines += [f"이번 실행에서 이미 본 것으로 적었다 — 도메인 의심 {marked}건(이미지·검사)"
                  + (f" · 일부가 빠진 태그 {marked_tm}건(이미지·태그: {', '.join(args.mark_tag)})" if mark_tags else "")
                  + ". 다음 실행부터 빠진다.", ""]
    elif not seen:
        lines += ["다 본 뒤 `--mark-seen` 으로 다시 돌리면 다음부터 새로 걸린 것만 나온다.", ""]
    for check, title in CHECKS.items():
        all_c = [r for r in flagged if check in r["flags"]]
        its = [r for r in all_c if listed(r, check)]
        n_new_c = sum(1 for r in all_c if check in r["new"])
        lines += [f"### {check} — {title}: {len(all_c)}장" + (f" (새로 걸린 것 {n_new_c}장)" if seen else ""), ""]
        if not its:
            continue
        sc = Counter((r["domain"], r["flags"][check][0]) for r in its)
        lines += ["  " + " · ".join(f"{DN[a]}→{DN.get(b, '?') if b else '반반'} {n}" for (a, b), n in sc.most_common()), ""]
        if check in ("group", "tag"):     # 어느 묶음·태그에서 나왔나 — 나라 태그처럼 원래 섞이는 것을 가려 보게
            src = Counter(r["flags"][check][1].split(":")[0] if check == "group"
                          else r["flags"][check][1].split(" 중 ")[0] for r in its)
            lines += ["  나온 곳: " + " · ".join(f"{s} {n}" for s, n in src.most_common(12)), ""]
        lines += [f"    {ids_line(its)}", ""]
    if mixed_tags:
        lines += ["### 도메인이 고르게 섞인 태그 (한 도메인이 "
                  f"{args.tag_major:.0%} 미만 — 나라·출처 태그면 자연스럽다)", ""]
        lines += table(["태그", "장수", *[DN[d] for d in (*DOMS, None)]],
                       [[tag_name[t], sum(c.values()), *[c[d] for d in (*DOMS, None)]]
                        for t, c in sorted(mixed_tags, key=lambda x: -sum(x[1].values()))])
        lines += [""]

    lines += ["## 3. 태그", "", "태그별 장수·도메인·분할은 tags.csv.", ""]
    if same:
        lines += ["### 이미지 집합이 같은 태그 (하나로 충분할 수 있다)", ""] + \
                 [f"- '{tag_name[a]}' = '{tag_name[b]}' ({len(sets[a])}장)" for a, b in same] + [""]
    if partial:
        lines += ["### 일부만 겹치는 태그 (포함 관계도 아니다 — 위계가 아닌 두 태그가 같은 이미지에 붙었나)", ""] + \
                 [f"- '{tag_name[a]}'({len(sets[a])}) ∩ '{tag_name[b]}'({len(sets[b])}) = {n}장"
                  for a, b, n in sorted(partial, key=lambda x: -x[2])] + [""]
    if subset:
        lines += [f"### 포함 관계 {len(subset)}쌍 (작은 ⊂ 큰 — 나라 ⊃ 장소 같은 위계면 괜찮다)", ""] + \
                 [f"- '{tag_name[s]}'({len(sets[s])}) ⊂ '{tag_name[b]}'({len(sets[b])})"
                  for s, b in sorted(subset, key=lambda x: (tag_name[x[1]], tag_name[x[0]]))] + [""]
    if similar or fam_off:
        lines += ["### 이름이 한두 글자 다른 것 (오타일 수 있다)", ""] + \
                 [f"- 태그 '{tag_name[a]}' ↔ 태그 '{tag_name[b]}'" for a, b in similar] + \
                 [f"- 태그 '{tag_name[t]}' ↔ 파일 이름 계열 '{fam}'({n}장)" for t, fam, n in fam_off] + [""]
    one = sorted((t for t, its in by_tag.items() if len(its) == 1), key=lambda t: tag_name[t])
    if one or empty_tags:
        lines += [f"### 한 장뿐인 태그 {len(one)}개 · 붙은 이미지가 없는 태그 {len(empty_tags)}개", "",
                  "  " + ", ".join(f"'{tag_name[t]}'" for t in one + empty_tags), ""]
    if partial_in_grp:
        shown = sorted(((g, t, c, n, miss if args.all else miss_new, len(miss) - len(miss_new))
                        for g, t, c, n, miss, miss_new in partial_in_grp if args.all or miss_new),
                       key=lambda x: (-len(x[4]), x[0]))
        lines += [f"### 장면 묶음(3장 이상) 안에서 절반 이상 붙었는데 일부가 빠진 태그 (빠진 이미지에 붙일지): "
                  f"{len(partial_in_grp)}줄 · 빠진 이미지 {n_miss}장"
                  + (f" ({miss_new_txt})" if n_miss != n_miss_new else ""), ""]
        note = ("아래는 이미 본 것까지 전부다(`--all`). " if args.all else
                f"`{seen_path.name}` 에 이미 본 것으로 적힌 {n_miss - n_miss_new}장(이미지·태그)은 뺐다(전부는 `--all`). "
                if n_miss != n_miss_new else "")
        lines += [note + "한 태그의 줄을 다 봤으면 `--mark-tag <태그>` 로 적어 둔다 — 다음부터 그 빠진 이미지는 나오지 않는다.", ""]
        if shown:
            lines += table(["묶음", "태그", "붙음/장수", "빠진 이미지 id"],
                           [[g, tag_name[t], f"{c}/{n}", ids_line(ms[:12]) + (" …" if len(ms) > 12 else "")
                             + (f" (본 것 {k}장 뺌)" if k and not args.all else "")]
                            for g, t, c, n, ms, k in shown[:40]])
            if len(shown) > 40:
                lines += ["", f"… {len(shown) - 40}줄 더 — 빠진 장수가 많은 순으로 40줄만 실었다"]
        lines += [""]
    lines += [f"### 태그가 하나도 없는 장면 묶음 — 큰 것 20개 (태그 없는 이미지 모두 "
              f"{sum(1 for r in rows if not r['tags']):,}장: " +
              " · ".join(f"{DN[d]} {sum(1 for r in rows if not r['tags'] and r['domain'] == d)}" for d in DOMS) + ")", ""]
    lines += table(["묶음", "장수", "도메인", "이름 계열", "예시 id"],
                   [[g, len(its), "·".join(f"{DN[d]} {n}" for d, n in Counter(r["domain"] for r in its).most_common()),
                     Counter(r["family"] for r in its).most_common(1)[0][0], ids_line(its[:5])]
                    for g, its in untagged_grp[:20]])
    multi = [tr for tr in tag_rows if sum(1 for s in ("train", "val", "test") if tr[s]) > 1]
    # 같은 카메라 누수: 다른 태그를 품지 않은 잎 태그(카메라·장소·작품)를 카메라로 보고, train 에도 있는 태그가 붙은 test 이미지
    broad = {t for t in tag_name if tag_name[t] in {s.strip() for s in args.broad_tags.split(",")}}
    parents = {b for a in sets for b in sets if a != b and sets[a] < sets[b]}
    by_id = {r["id"]: r for r in rows}
    leak: dict[int, list[int]] = defaultdict(list)
    for t in sets.keys() - parents - broad:
        if {by_id[i]["split"] for i in sets[t]} >= {"train", "test"}:
            for i in sets[t]:
                if by_id[i]["split"] == "test":
                    leak[i].append(t)
    test = [r for r in rows if r["split"] == "test"]
    lc = Counter(by_id[i]["domain"] for i in leak)
    lines += ["", f"### 같은 카메라 누수 — test {len(test):,}장 중 {len(leak):,}장이 train 과 같은 잎 태그를 함께 쓴다",
              "", f"잎 태그 = 다른 태그를 품지 않은 태그(카메라·장소·작품), '{args.broad_tags}' 는 넓어서 뺐다. 도메인별 "
              + " · ".join(f"{DN[d]} {lc[d]}/{sum(1 for r in test if r['domain'] == d)}" for d in DOMS) + ". 많은 태그: "
              + " · ".join(f"{tag_name[t]} {n}" for t, n in Counter(t for ts in leak.values() for t in ts).most_common(10)), ""]
    lines += [f"### 분할에 걸친 태그 {len(multi)}개 (태그가 카메라·출처라면 다시 나눌 때 한쪽에 모아야 한다)", ""]
    lines += table(["태그", "장수", "train", "val", "test", "미배정", "장면 묶음"],
                   [[tr["name"], tr["images"], tr["train"], tr["val"], tr["test"], tr["unassigned"], tr["groups"]]
                    for tr in multi])
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"의심 {len(flagged)}장 — " + " · ".join(
        f"{c} {sum(1 for r in flagged if c in r['flags'])}" for c in CHECKS))
    if seen:
        print(f"  새로 걸린 것 {n_new}장 — " + " · ".join(
            f"{c} {sum(1 for r in flagged if c in r['new'])}" for c in CHECKS) + f"  (이미 본 것: {seen_path})")
    print(f"일부가 빠진 태그 {len(partial_in_grp)}줄 · 빠진 이미지 {n_miss}장"
          + (f" — {miss_new_txt}" if n_miss != n_miss_new else ""))
    if args.mark_seen or mark_tags:
        print(f"  이미 본 것으로 적음 — 도메인 의심 {marked}건 · 빠진 태그 {marked_tm}건 → {seen_path}")
    elif not seen:
        print("  다 봤으면 --mark-seen 으로 다시 돌릴 것 — 다음부터 새로 걸린 것만 나온다")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
