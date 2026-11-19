"""보관 DB 의 responder·cone 박스를 원본 해상도에서 잘라 대조 시트로 붙인다 — 보관 폴더는 읽기만(DB immutable).

칸 하나 = 박스 하나(둘레를 조금 넣어 자르고 박스 선을 그린다). 칸 아래 글: 이미지 id · 출처(o = OWLv2 후보, h = 사람) 점수 · 카메라/태그.
  python -I make_sheets.py <…/20261110> <내보낼 폴더>
"""
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

MARK = "\\data\\visible\\"
GENERIC = {"Korea", "Japan", "United States of America", "United Kingdom", "Indonesia", "Czech Republic", "Ireland",
           "Argentine Republic", "Jamaica", "K-PM", "K-Scooter"}
FONT = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 12)
BIG = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 18)
COLOR = {"responder": (255, 40, 40), "cone": (0, 235, 255)}


def tile(img, box, size, grow, min_side, color):
    """박스 둘레를 넣어 정사각으로 자르고 size 로 맞춘 칸(박스 선 포함)."""
    W, H = img.size
    cx, cy, bw, bh = box["cx"] * W, box["cy"] * H, box["w"] * W, box["h"] * H
    side = max(max(bw, bh) * grow, min_side)
    x1, y1 = cx - side / 2, cy - side / 2
    crop = Image.new("RGB", (int(round(side)), int(round(side))), (60, 60, 60))
    sx1, sy1 = max(0, int(round(x1))), max(0, int(round(y1)))
    sx2, sy2 = min(W, int(round(x1 + side))), min(H, int(round(y1 + side)))
    if sx2 > sx1 and sy2 > sy1:
        crop.paste(img.crop((sx1, sy1, sx2, sy2)), (sx1 - int(round(x1)), sy1 - int(round(y1))))
    k = size / crop.size[0]
    out = crop.resize((size, size), Image.LANCZOS if k < 1 else Image.BICUBIC)
    d = ImageDraw.Draw(out)
    bx1, by1 = (cx - bw / 2 - x1) * k, (cy - bh / 2 - y1) * k
    bx2, by2 = (cx + bw / 2 - x1) * k, (cy + bh / 2 - y1) * k
    d.rectangle((bx1 - 1, by1 - 1, bx2 + 1, by2 + 1), outline=color, width=1)
    return out


def sheet(tiles, caps, size, cols, title, dst):
    cap_h = 15
    rows = (len(tiles) + cols - 1) // cols
    head = 30
    im = Image.new("RGB", (cols * (size + 2) + 2, head + rows * (size + cap_h + 2) + 2), (24, 24, 24))
    d = ImageDraw.Draw(im)
    d.text((6, 5), title, fill=(255, 255, 255), font=BIG)
    for i, (t, c) in enumerate(zip(tiles, caps)):
        x, y = 2 + (i % cols) * (size + 2), head + (i // cols) * (size + cap_h + 2)
        im.paste(t, (x, y))
        d.text((x + 2, y + size + 1), c, fill=(230, 230, 230), font=FONT)
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, "JPEG", quality=88)
    print(dst.name, im.size, len(tiles), "칸", f"{dst.stat().st_size / 1e6:.1f}MB")


def main() -> int:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    vis = root / "data" / "visible"
    con = sqlite3.connect(f"file:{(vis / 'labeling.db').as_posix()}?mode=ro&immutable=1", uri=True)
    con.row_factory = sqlite3.Row
    names = [r["name"] for r in con.execute("SELECT name FROM classes ORDER BY idx")]
    tags = defaultdict(list)
    for r in con.execute("SELECT it.image_id, t.name FROM image_tags it JOIN tags t ON t.id = it.tag_id"):
        tags[r[0]].append(r[1])
    rows = con.execute("SELECT b.id AS bid, b.image_id, b.cls, b.cx, b.cy, b.w, b.h, b.conf, b.origin, i.path, i.camera, i.status "
                       "FROM boxes b JOIN images i ON i.id = b.image_id WHERE b.cls IN (8, 9) AND i.status IN ('approved', 'pending') "
                       "ORDER BY b.image_id, b.id").fetchall()
    by_img = defaultdict(list)
    for r in rows:
        by_img[r["image_id"]].append(r)

    def place(iid):
        t = [x for x in sorted(tags[iid]) if x not in GENERIC] or sorted(tags[iid])
        return (t[0] if t else "-")[:12]

    groups = defaultdict(list)          # (클래스, 종류) -> [(정렬 키, 칸, 글)]
    for iid, bs in by_img.items():
        p = bs[0]["path"]
        src = Path(str(vis) + "\\" + p[p.lower().find(MARK) + len(MARK):])
        with Image.open(src) as im:
            img = ImageOps.exif_transpose(im).convert("RGB")
        cam = bs[0]["camera"]
        kind = "crop" if not cam else ("virat" if cam.startswith("virat") else "hw")
        for b in bs:
            cname = names[b["cls"]]
            if cname == "responder":
                t = tile(img, b, 176, 1.5, 96, COLOR[cname])
            else:
                t = tile(img, b, 112, 4.0, 72, COLOR[cname])
            who = ("h" if b["origin"] == "human" else "o") + (f"{b['conf']:.2f}"[1:] if b["conf"] is not None else "")
            mark = "A" if b["status"] == "approved" else ""
            where = cam if cam else place(iid)
            cap = f"{iid}{mark} {who} {where}"
            groups[(cname, kind)].append(((where, iid, b["bid"]), t, cap[: (24 if cname == "responder" else 15)]))
    for k in groups:
        groups[k].sort(key=lambda x: x[0])
    note = " | cell = one box (drawn), caption: image id (A = approved) / o = OWLv2 candidate + score, h = human / camera or tag"

    g = groups[("responder", "hw")]
    sheet([x[1] for x in g], [x[2] for x in g], 176, 14, f"responder - highway frames: {len(g)} boxes" + note, out / "responder_hw.jpg")
    g = groups[("responder", "crop")]
    sheet([x[1] for x in g], [x[2] for x in g], 176, 14, f"responder - old crops: {len(g)} boxes" + note, out / "responder_crop.jpg")
    g = groups[("cone", "hw")] + groups[("cone", "virat")]
    sheet([x[1] for x in g], [x[2] for x in g], 112, 20, f"cone - highway frames + VIRAT: {len(g)} boxes" + note, out / "cone_hw_virat.jpg")
    g = groups[("cone", "crop")]
    per = 320
    for n in range(0, len(g), per):
        part = g[n:n + per]
        sheet([x[1] for x in part], [x[2] for x in part], 112, 20,
              f"cone - old crops {n + 1}-{n + len(part)} of {len(g)} boxes" + note, out / f"cone_crop_{n // per + 1}.jpg")
    print({f"{k[0]}/{k[1]}": len(v) for k, v in groups.items()})
    rc = Counter(place(r["image_id"]) for r in rows if names[r["cls"]] == "responder" and not r["camera"])
    print("크롭의 responder 박스 — 태그별:", dict(rc.most_common()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
