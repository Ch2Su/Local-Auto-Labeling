"""지금 DB 의 라벨로 test 점수를 다시 재고, **결과를 기록에 쌓는다** (모델끼리 비교하는 기준선).

라벨을 고치면 옛 기준선과 비교가 끊긴다. 그때 같은 가중치로 다시 재서 기준선을 갱신하려고
만들었다. `labeler.py train` 의 `build_dataset` 은 DB 에 쓰지만(미배정 이미지를 배정한다)
이 스크립트는 **DB 를 읽기 전용으로만 열고**, test 분할만 따로 복사해 재고 지운다.

    python tools/test_baseline.py --domains                         # 정해진 도메인을 한 번에 (권장)
    python tools/test_baseline.py                                   # models/best.pt 를 지금 라벨로 (전체만)
    python tools/test_baseline.py --weights models/best_visible2.pt # 옛 모델 (점수는 부풀려져 있다 — CLAUDE.md)
    python tools/test_baseline.py --history                         # 쌓인 기록만 표로 본다 (재지 않음)
    python tools/test_baseline.py --data data/visible/dataset/data.yaml   # 이미 만들어 둔 데이터셋으로
    python tools/test_baseline.py --keep                            # 만든 test 폴더를 남긴다

**기록** (2026-09-23): 잴 때마다 `runs/test_baseline/history.jsonl` 에 한 줄씩 쌓는다 — 가중치
이름·SHA-256·해상도·도메인·장수·박스 수와 전체·**클래스별** mAP50-95/mAP50/P/R. 새로 잰 것은
같은 도메인의 직전 기록과 클래스별로 나란히 보여 준다. **test 장수·박스 수가 직전과 다르면
라벨·분할이 바뀐 것이라 점수 차이에 그 몫이 섞인다고 알린다.** 클래스가 7개에서 9개로 늘면서
평균만으로는 모델을 견줄 수 없게 돼서(CLAUDE.md '현재 데이터 상태') 클래스별로 남긴다.

**도메인** (`--domains`, 또는 `--domain 이름` 하나만):
  all 전체 / cctv CCTV(드론·비CCTV·o365 를 뺀 나머지, 11-03 에 방향이 CCTV 검출로 바뀌어 추가) / drone 드론(경로에 stabilized_) / noncctv 비CCTV(source='unlabeled', 나중에 넣은 이미지)
  / nodrone 드론 제외. 비CCTV 는 예전에 `--like "data\\labeled"` 로 쟀는데, 09-21 에 폴더를
  data/visible/ 로 옮긴 뒤로 그 조건은 **0장**에 걸린다 — 그래서 경로가 아니라 source 로 가른다.
  **2026-10-03 부터 all·nodrone 은 Objects365(source='o365')를 뺀다** — 그 사진이 test 로 들어와도
  09-27 기준선과 같은 범위를 잰다. Objects365 만은 o365, 전부 합친 것은 all_o365.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # tools/ -> 저장소 루트

from autolabel.config import Config, resolve
from autolabel.yolo_io import Box, write_yolo_txt

HISTORY = "runs/test_baseline/history.jsonl"

# 이름: (보여 줄 이름, 이미지 한 장을 받을지 가르는 함수). 판정은 DB 행(path, source)으로 한다.
# 'all'·'nodrone' 은 Objects365(source='o365', 10-03 부터)를 뺀다 — 09-27 기준선과 비교가 이어지게.
# 이름표는 그대로 둔다(--history 가 이름표로 묶는다).
DOMAINS = {
    "all": ("전체", lambda r: r["source"] != "o365"),
    "cctv": ("CCTV", lambda r: r["source"] not in ("unlabeled", "o365") and "stabilized_" not in r["path"].lower()),
    "drone": ("드론", lambda r: "stabilized_" in r["path"].lower()),
    "noncctv": ("비CCTV", lambda r: r["source"] == "unlabeled"),
    "nodrone": ("드론 제외", lambda r: "stabilized_" not in r["path"].lower() and r["source"] != "o365"),
    "o365": ("Objects365", lambda r: r["source"] == "o365"),
    "all_o365": ("전체+O365", lambda r: True),
}


def parse_args():
    p = argparse.ArgumentParser(description="지금 라벨로 test 점수를 다시 재고 기록에 쌓는다 (DB 읽기 전용).")
    p.add_argument("--config", default=None)
    p.add_argument("--weights", default=None, help="기본: config 의 운영 가중치(models/best.pt)")
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--data", default=None, help="이미 있는 data.yaml 로 잰다 (새로 만들지 않음)")
    p.add_argument("--imgsz", type=int, default=None, help="기본: model.imgsz (학습 해상도)")
    p.add_argument("--out", default=None, help="만들 폴더 (기본: runs/test_baseline/<날짜>_<가중치>_<split>)")
    p.add_argument("--keep", action="store_true", help="만든 이미지 폴더를 지우지 않는다")
    p.add_argument("--domains", action="store_true", help="정해진 도메인(전체·CCTV·드론·비CCTV·드론 제외·Objects365·전체+O365)을 한 번에 잰다")
    p.add_argument("--domain", choices=sorted(DOMAINS), default=None, help="도메인 하나만")
    p.add_argument("--like", default=None,
                   help="경로에 이 문자열이 든 이미지만 (임시 조건. 정해진 도메인은 --domain)")
    p.add_argument("--not-like", default=None, help="경로에 이 문자열이 든 이미지는 뺀다")
    p.add_argument("--history", action="store_true", help="쌓인 기록만 표로 보여 주고 끝낸다")
    p.add_argument("--no-record", action="store_true", help="기록에 쌓지 않는다")
    p.add_argument("--note", default="", help="기록에 함께 남길 메모 (예: '9클래스 첫 학습')")
    return p.parse_args()


def build_split(cfg: Config, split: str, out: Path,
                like: str | None = None, not_like: str | None = None) -> list[dict]:
    """DB(읽기 전용)에서 그 분할만 복사해 Ultralytics 가 읽을 수 있는 폴더로 만든다.

    복사한 이미지마다 {id, path, source, img(복사본), counts(클래스별 박스 수)} 를 돌려준다.
    도메인별 목록은 이걸 걸러 txt 로 만든다 — 이미지를 도메인마다 다시 복사하지 않는다.
    """
    con = sqlite3.connect(f"file:{cfg.db.as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    sql = "SELECT id, path, source FROM images WHERE status='approved' AND split=?"
    args: list = [split]
    if like:
        sql += " AND path LIKE ?"
        args.append(f"%{like}%")
    if not_like:
        sql += " AND path NOT LIKE ?"
        args.append(f"%{not_like}%")
    rows = con.execute(sql + " ORDER BY id", args).fetchall()
    boxes: dict[int, list[Box]] = {}
    for r in con.execute(
        "SELECT b.image_id, b.cls, b.cx, b.cy, b.w, b.h, b.conf, b.origin FROM boxes b"
        " JOIN images i ON i.id = b.image_id WHERE i.status='approved' AND i.split=?", (split,)
    ):
        boxes.setdefault(r["image_id"], []).append(
            Box(r["cls"], r["cx"], r["cy"], r["w"], r["h"], r["conf"], r["origin"]))
    con.close()

    img_dir, lab_dir = out / "images" / split, out / "labels" / split
    if out.exists():
        shutil.rmtree(out)
    img_dir.mkdir(parents=True)
    lab_dir.mkdir(parents=True)

    items = []
    for r in rows:
        src = Path(r["path"])
        if not src.exists():
            print(f"  파일 없음, 건너뜀: {src}")
            continue
        stem = f"{src.stem}__{r['id']}"     # 다른 폴더의 동명 파일 충돌 방지 (여기선 id 로 충분)
        dst = img_dir / f"{stem}{src.suffix}"
        shutil.copy2(src, dst)
        bs = boxes.get(r["id"], [])
        write_yolo_txt(lab_dir / f"{stem}.txt", bs)
        items.append({"id": r["id"], "path": r["path"], "source": r["source"], "img": dst,
                      "counts": Counter(b.cls for b in bs)})
    return items


def class_names(cfg: Config) -> list[str]:
    con = sqlite3.connect(f"file:{cfg.db.as_posix()}?mode=ro", uri=True)
    try:
        return [r[0] for r in con.execute("SELECT name FROM classes ORDER BY idx")]
    finally:
        con.close()


def write_yaml(path: Path, root: Path, val: str, names: list[str]) -> Path:
    """학습은 하지 않으므로 train 도 val 과 같게 둔다. val 은 폴더나 이미지 목록 txt."""
    listing = "\n".join(f"  {i}: {n}" for i, n in enumerate(names))
    path.write_text(f"path: {root}\ntrain: {val}\nval: {val}\ntest: {val}\nnames:\n{listing}\n",
                    encoding="utf-8")
    return path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize(res, names: list[str], counts: Counter | None) -> dict:
    """Ultralytics 검증 결과를 기록 한 줄로."""
    b = res.box
    per = {}
    for k, c in enumerate(res.ap_class_index):
        p, r, ap50, ap = b.class_result(k)
        per[names[c] if c < len(names) else f"cls_{c}"] = {
            "map": round(float(ap), 4), "map50": round(float(ap50), 4),
            "p": round(float(p), 4), "r": round(float(r), 4),
            "n": int(counts[c]) if counts is not None else None,
        }
    return {"map": round(float(b.map), 4), "map50": round(float(b.map50), 4),
            "precision": round(float(b.mp), 4), "recall": round(float(b.mr), 4), "classes": per}


def load_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def print_result(rec: dict, prev: dict | None, names: list[str]) -> None:
    """새로 잰 것을 직전 기록(같은 도메인·분할)과 클래스별로 나란히."""
    title = f"[{rec['domain_label']}] {rec['images']}장 / 박스 {rec['boxes']}개"
    print(f"\n{title}   mAP50-95 {rec['map']:.4f}   mAP50 {rec['map50']:.4f}"
          f"   P {rec['precision']:.4f}   R {rec['recall']:.4f}")
    if prev:
        same_set = (prev.get("images"), prev.get("boxes")) == (rec["images"], rec["boxes"])
        print(f"  직전 기록: {prev['at'][:16]} {prev['weights_name']} (sha {prev['sha256'][:8]})"
              f" imgsz {prev['imgsz']} — mAP50-95 {prev['map']:.4f} ({rec['map'] - prev['map']:+.4f})")
        if not same_set:
            print(f"  [주의] test 구성이 다릅니다 ({prev.get('images')}장/{prev.get('boxes')}개 ->"
                  f" {rec['images']}장/{rec['boxes']}개). 라벨·분할이 바뀐 몫이 차이에 섞여 있습니다.")
    print(f"  {'클래스':<11}{'박스':>6}{'mAP50-95':>10}{'mAP50':>8}{'P':>7}{'R':>7}" + ("   직전 대비" if prev else ""))
    for n in names:
        c = rec["classes"].get(n)
        if c is None:
            continue
        line = f"  {n:<11}{(c['n'] if c['n'] is not None else '-'):>6}{c['map']:>10.4f}{c['map50']:>8.4f}{c['p']:>7.3f}{c['r']:>7.3f}"
        if prev:
            pc = prev["classes"].get(n)
            line += f"   {c['map'] - pc['map']:+.4f}" if pc else "   (직전엔 없음)"
        print(line)


def print_history(path: Path) -> None:
    recs = load_history(path)
    if not recs:
        print(f"기록이 없습니다: {path}")
        return
    names = []
    for r in recs:
        for n in r["classes"]:
            if n not in names:
                names.append(n)
    by_dom: dict[str, list[dict]] = {}
    for r in recs:
        by_dom.setdefault(f"{r['domain_label']} ({r['split']})", []).append(r)
    short = lambda n: n[:6]
    for dom, rows in by_dom.items():
        print(f"\n== {dom} ==")
        print(f"{'시각':<17}{'가중치':<22}{'imgsz':>6}{'장':>6}{'박스':>7}{'mAP':>8}"
              + "".join(f"{short(n):>8}" for n in names) + "  메모")
        for r in rows:
            label = f"{r['weights_name']}·{r['sha256'][:6]}"
            print(f"{r['at'][:16]:<17}{label[:21]:<22}{r['imgsz']:>6}{r['images']:>6}{r['boxes']:>7}"
                  f"{r['map']:>8.4f}"
                  + "".join(f"{r['classes'][n]['map']:>8.4f}" if n in r["classes"] else f"{'-':>8}"
                            for n in names)
                  + f"  {r.get('note', '')}")


def main():
    args = parse_args()
    cfg = Config.load(args.config)
    hist_path = resolve(HISTORY)
    if args.history:
        print_history(hist_path)
        return

    weights = str(resolve(args.weights)) if args.weights else cfg.active_weights()
    imgsz = args.imgsz or cfg.imgsz
    out = resolve(args.out) if args.out else resolve(
        f"runs/test_baseline/{date.today():%Y%m%d}_{Path(weights).stem}_{args.split}")
    names = class_names(cfg)

    # 무엇을 잴지: [(도메인 키, 보여 줄 이름, 이미지 목록 또는 None=폴더 전체, 데이터 yaml)]
    plan = []
    if args.data:
        data = resolve(args.data)
        print(f"데이터셋 {data} (이미 있는 것)")
        plan.append((f"data:{data.name}", f"데이터셋 {data.name}", None, data))
    else:
        items = build_split(cfg, args.split, out, args.like, args.not_like)
        n_box = sum(sum(it["counts"].values()) for it in items)
        print(f"{args.split} {len(items)}장 / 박스 {n_box}개 -> {out}")
        if args.like or args.not_like:
            keys = ["all"]          # 임시 조건은 그 조건 하나만 잰다 (아래서 이름을 custom 으로)
        elif args.domains or args.domain:
            keys = list(DOMAINS) if args.domains else [args.domain]
        else:
            keys = ["all"]
        for key in keys:
            label, keep = DOMAINS[key]
            if args.like or args.not_like:
                key, label = "custom", f"조건 like={args.like} not_like={args.not_like}"
                keep = lambda r: True       # noqa: E731  임시 조건은 출처를 가리지 않는다
            sub = [it for it in items if keep(it)]
            if not sub:
                print(f"  [{label}] 해당 이미지가 0장이라 건너뜁니다.")
                continue
            listing = out / f"{args.split}_{key}.txt"
            listing.write_text("\n".join(str(it["img"]) for it in sub) + "\n", encoding="utf-8")
            plan.append((key, label, sub, write_yaml(out / f"data_{key}.yaml", out, listing.name, names)))

    if not plan:
        print("잴 것이 없습니다.")
        return

    from ultralytics import YOLO

    model = YOLO(weights)
    digest = sha256(Path(weights))
    print(f"가중치 {weights} (sha256 {digest[:12]}…) / imgsz {imgsz} / split {args.split}", flush=True)
    history = load_history(hist_path)
    new_recs = []
    for key, label, sub, data in plan:
        res = model.val(data=str(data), split="val", imgsz=imgsz, device=cfg.device,
                        workers=0, plots=False, verbose=False,
                        project=str(out.parent), name=f"{out.name}_{key}_val", exist_ok=True)
        counts = sum((it["counts"] for it in sub), Counter()) if sub is not None else None
        rec = {
            "at": datetime.now().isoformat(timespec="seconds"),
            "weights": str(Path(weights)), "weights_name": Path(weights).name, "sha256": digest,
            "imgsz": imgsz, "split": args.split, "domain": key, "domain_label": label,
            "images": len(sub) if sub is not None else None,
            "boxes": sum(counts.values()) if counts is not None else None,
            **summarize(res, names, counts),
            "note": args.note,
        }
        prev = next((r for r in reversed(history)
                     if r["domain"] == key and r["split"] == args.split), None)
        print_result(rec, prev, names)
        new_recs.append(rec)

    if not args.no_record:
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(hist_path, "a", encoding="utf-8") as f:
            for rec in new_recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\n기록에 {len(new_recs)}줄을 더했습니다: {hist_path}  (표로 보기: --history)")

    if not args.keep and not args.data:
        shutil.rmtree(out / "images", ignore_errors=True)
        print(f"복사본은 지웠다 (라벨·목록·data yaml 은 {out} 에 남는다). 남기려면 --keep")


if __name__ == "__main__":      # Windows spawn (CLAUDE.md 함정 7)
    main()
