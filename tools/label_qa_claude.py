"""`label_qa.py` 의 missing 후보를 Claude 에게 보여 진짜 빠진 라벨인지 거른다 (파일럿).

missing 후보는 '모델은 확신하는데 라벨이 없는 곳'이다. 눈으로 보니 약 1/3 만 진짜였고,
나머지는 가방 더미·벽화·등대 같은 오탐이었다. 사람이 366건을 다 열어 보기 전에 Claude 로
한 번 거르는 것이 목적이다. DB 는 읽기 전용으로만 연다.

    python tools/label_qa_claude.py --dry-run            # 크롭·대조군만 만들고 비용 추정 (전송 없음)
    python tools/label_qa_claude.py --limit 20           # 20건만 실제 호출해 형식·단가 확인
    python tools/label_qa_claude.py                      # 나머지 전부 (이미 끝난 건 건너뜀)

**정답을 모르니 대조군을 섞는다.** 후보와 같은 형식으로
- 양성 대조: 이미 라벨이 있는 박스에서 그 초록 박스만 숨긴 것 → 'missing' 이라고 해야 맞다
- 음성 대조: 라벨과 겹치지 않는 임의의 배경 영역 → 'missing' 이라고 하면 틀린 것
을 만든다. 크기는 후보의 짧은 변 분포를 따른다. Claude 는 어느 것이 대조군인지 모른다.
대조군 성적이 곧 이 필터를 믿어도 되는지에 대한 근거다. (음성 대조가 우연히 라벨 없는
객체에 떨어질 수 있어서, 대조군 오답은 시트로 눈으로 확인할 것.)

**보내는 것:** 후보 주변 크롭(깨끗한 것 + 박스 그린 것, 512px)과 전체 장면 축소본(긴 변
512px) 세 장. 벽화·포스터처럼 '사람 그림'인지는 장면을 봐야 알 수 있어서 축소본을 넣었다.

**API 키는 출력하지 않는다.** 환경변수 → Windows 사용자 환경변수(레지스트리) →
`~/.anthropic_api_key` 파일 순으로 찾는다. 사용자 환경변수를 레지스트리에서 직접 읽는
이유는, 설정 창에서 추가한 값이 이미 떠 있는 프로세스에는 안 보이기 때문이다.

**`ANTHROPIC_BASE_URL` 을 따르지 않는다.** Claude Code 안에서 실행하면 이 변수가 앱 쪽
주소로 잡혀 있어, SDK 기본 동작대로면 API 키가 그쪽으로 간다. `--base-url` 로만 바꾼다.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import random
import sqlite3
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # tools/ -> 저장소 루트

from autolabel.config import Config, resolve
from autolabel.yolo_io import display_size

# USD / 1M 토큰 (입력, 출력). 2026-09-18 platform.claude.com/docs/en/about-claude/pricing
PRICES = {
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-opus-5": (5.0, 25.0),
}
VERDICTS = ["missing", "already_labeled", "depiction", "not_object", "unsure"]
CROP = 512
SIDE_BUCKETS = [(0, 32, "<32px"), (32, 64, "32-64px"), (64, 10**9, ">=64px")]

SYSTEM_TMPL = """You audit bounding-box labels of an object-detection dataset. Most images are CCTV or street scenes; some are photos and video frames.
Label classes: {classes}.
Every real instance of these classes should have its own box. Riders get their own person box separate from the vehicle."""

# 이름만으로 뜻이 갈리지 않는 클래스의 설명. 차량 셋은 CLAUDE.md '차량 클래스 판정 순서' 를 따른다.
CLASS_NOTES = {
    "scooter": "stand-up e-scooter / kick scooter",
    "pickup": "open cargo bed separate from the cab, up to 1-ton class",
    "truck": "larger open bed, or a closed box body on a cab-chassis",
    "car": "passenger cars, SUVs, minivans and one-box vans",   # 11-02 에 van 을 car 로 합쳤다
}

# 클래스 목록은 DB 에서 읽어 `set_classes()` 가 채운다. 예전에는 7개가 박혀 있어서
# 2026-09-20 에 생긴 pickup·van 후보를 모델이 car·truck 중에서 고르게 돼 있었다.
SYSTEM = SYSTEM_TMPL.format(classes="")
CLASSES: list[str] = []

INSTRUCTION = """A detector predicted a {cls} (confidence {conf:.2f}) inside the red box, but the dataset has no label there.
Image 1: crop around the red box, without drawings.
Image 2: the same crop; green boxes are existing labels, the red box is the region to check.
Image 3: the whole image, downscaled, with the red box.

What does the red box contain?
- missing: a real person or vehicle from the class list fills roughly the red box and has no green box of its own. Partly visible objects (cut by the image edge or occluded) count if you can clearly identify them.
- already_labeled: that object already has its own green box (maybe drawn a bit differently).
- depiction: a picture of a person or vehicle, not a real one: painting, mural, poster, statue, mannequin, screen, reflection, toy.
- not_object: none of the classes - background, bags, furniture, signs, poles, animals, etc.
- unsure: too small, dark or blurry to tell.
object_class: the class of the object for missing / already_labeled / depiction, otherwise null.
note: at most 8 words."""

def schema_for(classes: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": VERDICTS},
            "object_class": {"anyOf": [
                {"type": "string", "enum": list(classes)},
                {"type": "null"}]},
            "note": {"type": "string"},
        },
        "required": ["verdict", "object_class", "note"],
        "additionalProperties": False,
    }


SCHEMA = schema_for([])


def set_classes(classes: list[str]) -> None:
    """DB 의 클래스 목록으로 시스템 프롬프트와 응답 스키마를 맞춘다."""
    global SYSTEM, SCHEMA, CLASSES
    CLASSES = list(classes)
    SYSTEM = SYSTEM_TMPL.format(classes=", ".join(
        f"{c} ({CLASS_NOTES[c]})" if c in CLASS_NOTES else c for c in CLASSES))
    SCHEMA = schema_for(CLASSES)


def parse_args():
    p = argparse.ArgumentParser(description="label_qa missing 후보를 Claude 로 거른다 (파일럿).")
    p.add_argument("--config", default=None)
    p.add_argument("--qa-dir", default=None, help="label_qa.py 결과 폴더 (기본: runs/label_qa 의 가장 최근)")
    p.add_argument("--model", default="claude-sonnet-5", choices=sorted(PRICES))
    p.add_argument("--controls", type=int, default=40, help="양성·음성 대조군 각각의 수 (기본 40)")
    p.add_argument("--limit", type=int, default=None, help="이번에 호출할 최대 건수 (섞인 순서 기준)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="크롭·대조군만 만들고 비용 추정 (전송 없음)")
    p.add_argument("--base-url", default="https://api.anthropic.com")
    return p.parse_args()


def load_api_key() -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key.strip()
    if sys.platform == "win32":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                return str(winreg.QueryValueEx(k, "ANTHROPIC_API_KEY")[0]).strip()
        except OSError:
            pass
    f = Path.home() / ".anthropic_api_key"
    if f.exists():
        return f.read_text(encoding="utf-8").strip()
    return None


def side_bucket(side: float) -> str:
    return next(name for lo, hi, name in SIDE_BUCKETS if lo <= side < hi)


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def build_items(cfg: Config, qa_dir: Path, n_controls: int, seed: int):
    """후보 + 대조군. 박스는 원본 픽셀 xyxy. hide_gt 는 초록으로 그리지 않을 라벨 번호."""
    con = sqlite3.connect(f"file:{cfg.db.as_posix()}?mode=ro", uri=True)
    names = dict(con.execute("SELECT idx, name FROM classes"))
    set_classes([names[i] for i in sorted(names)])
    images = {}
    for iid, path, w, h, split in con.execute(
            "SELECT id, path, width, height, split FROM images WHERE status='approved'"):
        # 픽셀 좌표는 EXIF 회전을 적용한 화면 기준이다 — label_qa 의 후보 박스가 그렇다.
        # DB 의 width/height 는 회전 전 크기라 회전 사진(정사각 아닌 4장)에서 어긋난다.
        if Path(path).exists():
            w, h = display_size(Path(path))
        images[iid] = {"path": path, "W": w, "H": h, "split": split}
    gt = defaultdict(list)
    for iid, c, cx, cy, w, h in con.execute(
            "SELECT b.image_id, b.cls, b.cx, b.cy, b.w, b.h FROM boxes b "
            "JOIN images i ON i.id = b.image_id WHERE i.status='approved' ORDER BY b.id"):
        im = images[iid]
        gt[iid].append((names[c], [(cx - w / 2) * im["W"], (cy - h / 2) * im["H"],
                                   (cx + w / 2) * im["W"], (cy + h / 2) * im["H"]]))
    con.close()

    cands = [r for r in csv.DictReader(open(qa_dir / "candidates.csv", encoding="utf-8-sig"))
             if r["kind"] == "missing"]
    if not cands and n_controls:    # 대조군 크기·신뢰도를 후보에서 빌리므로 만들 수 없다 (11-05 — rng.choice 가 IndexError)
        print(f"  [알림] {qa_dir / 'candidates.csv'} 에 missing 후보가 없습니다(--dup-only 결과?) — 대조군도 만들지 않습니다.")
        n_controls = 0
    items = []
    for n, r in enumerate(cands):
        items.append({"key": f"cand{n:04d}", "group": "candidate", "image_id": int(r["image_id"]),
                      "box": [float(v) for v in r["pred_box"].split()], "cls": r["pred_cls"],
                      "conf": float(r["conf"]), "hide_gt": None, "split": r["split"]})

    rng = random.Random(seed)
    cand_ids = {it["image_id"] for it in items}
    pool = [i for i in images if i not in cand_ids and Path(images[i]["path"]).exists()]
    sides = [float(r["side_px"]) for r in cands]

    # 양성: 후보와 짧은 변이 비슷한 기존 라벨을 숨겨서 보여 준다
    all_gt = [(iid, j, c, b, min(b[2] - b[0], b[3] - b[1]))
              for iid in pool for j, (c, b) in enumerate(gt[iid])]
    used = set()
    made = tries = 0
    while made < n_controls and tries < 50 * n_controls:
        tries += 1
        target = rng.choice(sides)
        near = [g for g in all_gt if abs(g[4] - target) <= max(4, 0.2 * target) and g[0] not in used]
        if not near:    # 그 크기의 라벨이 없거나 다 썼다 — 다른 후보 크기로 다시 뽑는다 (11-05, 전에는 IndexError)
            continue
        iid, j, c, b, _ = rng.choice(near)
        used.add(iid)
        donor = rng.choice(cands)
        items.append({"key": f"pos{made:03d}", "group": "control_pos", "image_id": iid, "box": b, "cls": c,
                      "conf": float(donor["conf"]), "hide_gt": j, "split": images[iid]["split"] or "미배정"})
        made += 1
    if made < n_controls:
        print(f"  [알림] 양성 대조를 {n_controls}개 중 {made}개만 만들었습니다 — 후보와 크기가 비슷한 라벨이 모자랍니다.")

    # 음성: 후보 하나의 박스 크기를 빌려 라벨과 안 겹치는 곳에 놓는다
    n = tries = 0
    while n < n_controls and pool and tries < 200 * n_controls:     # 놓을 곳이 없으면 끝없이 돌지 않게
        tries += 1
        donor = rng.choice(cands)
        bx = [float(v) for v in donor["pred_box"].split()]
        bw, bh = bx[2] - bx[0], bx[3] - bx[1]
        iid = rng.choice(pool)
        im = images[iid]
        if iid in used or bw >= im["W"] or bh >= im["H"]:
            continue
        for _ in range(100):
            x, y = rng.uniform(0, im["W"] - bw), rng.uniform(0, im["H"] - bh)
            b = [x, y, x + bw, y + bh]
            if all(iou(b, g) < 0.02 for _, g in gt[iid]):
                used.add(iid)
                items.append({"key": f"neg{n:03d}", "group": "control_neg", "image_id": iid, "box": b,
                              "cls": donor["pred_cls"], "conf": float(donor["conf"]), "hide_gt": None,
                              "split": im["split"] or "미배정"})
                n += 1
                break
    if n < n_controls:
        print(f"  [알림] 음성 대조를 {n_controls}개 중 {n}개만 만들었습니다 — 라벨과 안 겹치게 놓을 곳이 모자랍니다.")

    rng.shuffle(items)
    for it in items:
        b = it["box"]
        it["side_px"] = round(min(b[2] - b[0], b[3] - b[1]), 1)
    return items, images, gt


def render(item, images, gt):
    """(깨끗한 크롭, 그린 크롭, 전체 축소본) PIL 이미지."""
    from PIL import Image, ImageDraw, ImageOps

    im = images[item["image_id"]]
    img = ImageOps.exif_transpose(Image.open(im["path"])).convert("RGB")
    W, H = img.size
    b = item["box"]
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    side = min(max(4 * max(b[2] - b[0], b[3] - b[1]), 192), max(W, H))
    x0 = min(max(cx - side / 2, 0), max(W - side, 0))
    y0 = min(max(cy - side / 2, 0), max(H - side, 0))
    s = CROP / side
    clean = img.crop((x0, y0, x0 + side, y0 + side)).resize((CROP, CROP), Image.LANCZOS)

    drawn = clean.copy()
    d = ImageDraw.Draw(drawn)
    for j, (_, g) in enumerate(gt[item["image_id"]]):
        if j != item["hide_gt"]:
            d.rectangle([(g[0] - x0) * s, (g[1] - y0) * s, (g[2] - x0) * s, (g[3] - y0) * s],
                        outline=(0, 230, 0), width=2)
    d.rectangle([(b[0] - x0) * s - 2, (b[1] - y0) * s - 2, (b[2] - x0) * s + 2, (b[3] - y0) * s + 2],
                outline=(255, 0, 0), width=3)

    t = CROP / max(W, H)
    whole = img.resize((max(1, round(W * t)), max(1, round(H * t))), Image.LANCZOS)
    ImageDraw.Draw(whole).rectangle([b[0] * t - 3, b[1] * t - 3, b[2] * t + 3, b[3] * t + 3],
                                    outline=(255, 0, 0), width=3)
    return clean, drawn, whole


def visual_tokens(w: int, h: int) -> int:
    return -(-w // 28) * -(-h // 28)     # 문서의 ⌈w/28⌉×⌈h/28⌉ (512px 는 한도 안이라 축소 없음)


def b64(pil) -> str:
    buf = io.BytesIO()
    pil.save(buf, "JPEG", quality=90)
    return base64.standard_b64encode(buf.getvalue()).decode()


def call(client, model, item, images, gt):
    clean, drawn, whole = render(item, images, gt)
    content = []
    for pil in (clean, drawn, whole):
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64(pil)}})
    content.append({"type": "text", "text": INSTRUCTION.format(cls=item["cls"], conf=item["conf"])})
    resp = client.messages.create(
        model=model, max_tokens=300, system=SYSTEM,
        messages=[{"role": "user", "content": content}],
        thinking={"type": "disabled"},
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
    )
    text = "".join(getattr(blk, "text", "") for blk in resp.content)
    u = resp.usage
    return json.loads(text), {"input": u.input_tokens, "output": u.output_tokens}


def render_sheet(rows, crops_dir: Path, path: Path, title: str):
    from PIL import Image, ImageDraw

    if not rows:
        return
    tile, cols = 256, 6
    sheet = Image.new("RGB", (cols * tile, ((len(rows) + cols - 1) // cols) * (tile + 28) + 24), "white")
    d = ImageDraw.Draw(sheet)
    d.text((6, 6), title, fill="black")
    for k, r in enumerate(rows):
        x, y = (k % cols) * tile, 24 + (k // cols) * (tile + 28)
        sheet.paste(Image.open(crops_dir / f"{r['key']}.jpg").resize((tile, tile)), (x, y))
        d.text((x + 3, y + tile + 1), f"#{k + 1} id{r['image_id']} {r['cls']} {r['conf']:.2f} {r['side_px']:.0f}px", fill="black")
        d.text((x + 3, y + tile + 13), f"{r['verdict']}/{r['object_class'] or '-'}: {r['note'][:34]}", fill="black")
    sheet.save(path)


def summarize(items, results, model, out: Path):
    by_key = {it["key"]: it for it in items}
    rows = [{**by_key[k], **v} for k, v in results.items() if k in by_key]
    lines = []

    def pr(s=""):
        print(s)
        lines.append(s)

    pr(f"모델 {model} / 완료 {len(rows)} / 전체 {len(items)}")
    for group, want in (("control_pos", "missing"), ("control_neg", None)):
        g = [r for r in rows if r["group"] == group]
        if not g:
            continue
        c = Counter(r["verdict"] for r in g)
        if want:
            pr(f"\n양성 대조 {len(g)}건 — 'missing' 이라고 맞힌 비율 {c['missing'] / len(g):.0%}   {dict(c)}")
        else:
            pr(f"\n음성 대조 {len(g)}건 — 'missing' 이라고 잘못 말한 비율 {c['missing'] / len(g):.0%}   {dict(c)}")
        for _, _, name in SIDE_BUCKETS:
            gb = [r for r in g if side_bucket(r["side_px"]) == name]
            if gb:
                pr(f"  {name:>8}: {sum(r['verdict'] == 'missing' for r in gb)}/{len(gb)} missing")

    cand = [r for r in rows if r["group"] == "candidate"]
    if cand:
        pr(f"\n후보 {len(cand)}건 판정: {dict(Counter(r['verdict'] for r in cand).most_common())}")
        for _, _, name in SIDE_BUCKETS:
            gb = [r for r in cand if side_bucket(r["side_px"]) == name]
            if gb:
                pr(f"  {name:>8}: " + ", ".join(f"{k} {v}" for k, v in Counter(r['verdict'] for r in gb).most_common()))
        pr("  모델 예측 클래스별 'missing' 판정:")
        for cls, n in Counter(r["cls"] for r in cand).most_common():
            m = sum(1 for r in cand if r["cls"] == cls and r["verdict"] == "missing")
            pr(f"    {cls:<10} {m}/{n}")

    ip, op = PRICES[model]
    tin = sum(r["usage"]["input"] for r in rows)
    tout = sum(r["usage"]["output"] for r in rows)
    pr(f"\n실제 토큰 입력 {tin:,} / 출력 {tout:,} → ${tin * ip / 1e6 + tout * op / 1e6:.2f}"
       f" (건당 ${(tin * ip / 1e6 + tout * op / 1e6) / max(len(rows), 1):.4f})")
    (out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    crops = out / "crops"
    for v in VERDICTS:
        sel = [r for r in cand if r["verdict"] == v]
        sel.sort(key=lambda r: -r["conf"])
        render_sheet(sel[:60], crops, out / f"sheet_{v}.png", f"candidates judged {v}: {len(sel)} (top 60 by conf)")
    wrong = [r for r in rows if (r["group"] == "control_pos" and r["verdict"] != "missing")
             or (r["group"] == "control_neg" and r["verdict"] == "missing")]
    render_sheet(wrong, crops, out / "sheet_control_errors.png",
                 f"control errors: {len(wrong)} (pos not called missing / neg called missing)")


def main():
    args = parse_args()
    cfg = Config.load(args.config)
    if args.qa_dir:
        qa_dir = resolve(args.qa_dir)
    else:
        qa_dir = max((p for p in resolve("runs/label_qa").iterdir() if (p / "candidates.csv").exists()),
                     key=lambda p: p.stat().st_mtime)
    out = qa_dir / f"claude_{args.model}"
    (out / "crops").mkdir(parents=True, exist_ok=True)

    items_file = out / "items.json"
    if items_file.exists():     # 다시 돌려도 같은 대조군을 쓴다
        items = json.loads(items_file.read_text(encoding="utf-8"))
        _, images, gt = build_items(cfg, qa_dir, 0, args.seed)
    else:
        items, images, gt = build_items(cfg, qa_dir, args.controls, args.seed)
        if items:               # 빈 목록을 남기면 다음 실행도 그것을 읽는다
            items_file.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    # 지금 DB 에 승인 이미지가 없거나 파일이 없는 항목은 그릴 수 없다 — 그 뒤 지우거나 내린 이미지다 (11-05: 09-18 items.json
    # 446건 중 123건이 그래서 render 가 KeyError 로 죽었다). items.json 은 그대로 두고 이번 실행에서만 뺀다.
    def drawable(it) -> bool:
        im = images.get(it["image_id"])
        return im is not None and Path(im["path"]).exists()

    gone = Counter(it["group"] for it in items if not drawable(it))
    if gone:
        items = [it for it in items if drawable(it)]
        print(f"[알림] 지금 DB 에 승인 이미지(또는 그 파일)가 없는 항목 {sum(gone.values())}건을 뺍니다: {dict(gone)}")
    if not items:
        sys.exit(f"거를 항목이 없습니다 ({qa_dir}) — missing 후보가 없거나(--dup-only 결과?) 모두 지금 DB 에 없는 이미지입니다.")
    print(f"결과 폴더 {out}")
    print(f"항목 {len(items)}: " + ", ".join(f"{k} {v}" for k, v in Counter(it['group'] for it in items).items()))

    for it in items:            # 시트용 그린 크롭은 전송 여부와 상관없이 만든다
        f = out / "crops" / f"{it['key']}.jpg"
        if not f.exists():
            render(it, images, gt)[1].save(f, quality=88)

    results_file = out / "results.jsonl"
    results = {}
    if results_file.exists():
        for line in open(results_file, encoding="utf-8"):
            r = json.loads(line)
            results[r["key"]] = r
    todo = [it for it in items if it["key"] not in results]

    if args.dry_run:
        clean, _, whole = render(todo[0] if todo else items[0], images, gt)
        per_img = 2 * visual_tokens(*clean.size)
        whole_tok = sum(visual_tokens(*render(it, images, gt)[2].size) for it in items[:30]) / min(30, len(items))
        text_tok = (len(SYSTEM) + len(INSTRUCTION)) / 3.5 + 20
        est_in, est_out = per_img + whole_tok + text_tok, 60
        print(f"\n드라이런: 남은 {len(todo)}건, 건당 입력 약 {est_in:,.0f} 토큰(이미지 {per_img + whole_tok:,.0f}) + 출력 약 {est_out}")
        for m, (ip, op) in PRICES.items():
            cost = len(todo) * (est_in * ip + est_out * op) / 1e6
            print(f"  {m:<28} 약 ${cost:.2f}")
        print("추정치다. structured output 이 붙이는 시스템 토큰은 빠져 있다 — --limit 20 으로 실측할 것.")
        return

    key = load_api_key()
    if not key:
        sys.exit("API 키를 찾지 못했습니다: ANTHROPIC_API_KEY 환경변수(사용자 범위 포함) 또는 ~/.anthropic_api_key")
    import anthropic
    client = anthropic.Anthropic(api_key=key, base_url=args.base_url, max_retries=6)

    if args.limit is not None:
        todo = todo[:args.limit]
    print(f"이번 호출 {len(todo)}건 ({args.model}, 동시 {args.workers})", flush=True)
    lock = threading.Lock()
    errors = 0
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex, open(results_file, "a", encoding="utf-8") as f:
        futs = {ex.submit(call, client, args.model, it, images, gt): it for it in todo}
        for n, fut in enumerate(as_completed(futs), 1):
            it = futs[fut]
            try:
                ans, usage = fut.result()
            except Exception as exc:        # 끝난 것만 기록한다 — 다시 돌리면 실패분만 재시도
                errors += 1
                print(f"  실패 {it['key']}: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
                if errors >= 5 and errors == n:
                    print("처음부터 연달아 실패해 멈춥니다.", flush=True)
                    ex.shutdown(cancel_futures=True)
                    break
                continue
            rec = {"key": it["key"], **ans, "usage": usage}
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                results[it["key"]] = rec
            if n % 25 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
    print(f"실패 {errors}건\n")
    summarize(items, results, args.model, out)


if __name__ == "__main__":
    main()
