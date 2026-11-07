"""시드 COCO json 에서 파일이 없는 이미지 항목을 뺀다 (2026-11-08).

scan 은 `data.seed_coco` 아래의 COCO json 을 모두 읽어 파일이 있는 이미지만 승인으로 넣고, 파일이 없는 항목은
'JSON 에 있으나 파일이 없는 이미지 N장' 으로 알린 뒤 건너뛴다(`coco_io.read_coco`). 휴지통 비우기
(`tidy.purge_trashed`)는 원본 파일까지 지우므로, 시드 이미지를 비울 때마다 그 수가 늘어난다. 이 도구는 그 이미지
항목과 그 이미지의 annotations 를 json 에서 빼 경고를 없앤다. **DB 는 건드리지 않는다.** 이미지 찾기는 scan 과
같은 함수(`find_annotation_files`·`_resolve_image`)를 쓴다.

**원본 json 은 먼저 떠 둔다.** `<seed_coco 의 부모>/seed_json_backup/<시각>/` 에 seed_coco 기준 상대 경로 그대로
두고, 뺀 목록 `removed.tsv` 도 함께 둔다. seed_coco 아래에 두면 scan 이 사본까지 시드로 읽으므로 반드시 그 밖이다.

**쓰지 않고 멈추는 경우** — 하나라도 걸리면 어느 json 도 쓰지 않는다.
- 한 json 에서 이미지의 절반 넘게 파일이 없다. 폴더를 옮겼거나 경로가 깨진 것일 수 있다(scan 이 DB 의 절반 넘게
  '파일 없음' 이면 멈추는 것과 같은 생각). 정말 지운 것이면 `--force`.
- 이미지 id 가 겹친다. annotations 는 id 로 이미지를 가리키므로, 겹치면 남길 이미지의 박스까지 빠질 수 있다.
- 다시 저장했을 때 원래 바이트가 그대로 나오는 json 모양을 못 찾았다. 쓰면 손대지 않은 부분까지 모양이 바뀐다.

쓴 뒤에는 json 을 다시 읽어, 파일이 있는 이미지와 그 박스가 전과 같고 빠진 항목이 0 인지 본다. 다르면 떠 둔 원본으로
모두 되돌리고 실패로 끝낸다.

**되살릴 때.** 지운 사진을 다시 넣으려면 파일을 제자리에 두고 떠 둔 json 으로 되돌린 뒤 scan 한다. 그 사진의 라벨은
검수로 고친 최신본이 든 DB 백업 쪽이 json 보다 낫다.

    python tools/prune_seed_json.py            # 무엇을 뺄지 보여만 준다 (아무것도 쓰지 않음)
    python tools/prune_seed_json.py --apply    # 원본을 떠 두고 뺀다
"""

from __future__ import annotations

import argparse
import codecs
import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # tools/ -> 저장소 루트

from autolabel.coco_io import _resolve_image, find_annotation_files, read_coco  # noqa: E402
from autolabel.config import Config  # noqa: E402

# 다시 저장할 때 시험해 볼 모양 — 원래 바이트를 그대로 내는 것을 쓴다 (Roboflow 내보내기는 맨 앞 것이다)
_STYLES = [{"ensure_ascii": ea, **kw}
           for kw in ({"separators": (",", ":")}, {}, {"indent": 2}, {"indent": 4}, {"indent": 1}, {"indent": "\t"})
           for ea in (False, True)]


def find_style(data, raw: bytes) -> tuple[dict, bool, str] | None:
    """data 를 다시 저장해 raw 와 바이트까지 같아지는 (json.dumps 인자, BOM, 끝 줄바꿈). 없으면 None."""
    bom = raw.startswith(codecs.BOM_UTF8)
    body = raw[len(codecs.BOM_UTF8):] if bom else raw
    for style in _STYLES:
        text = json.dumps(data, **style)
        for tail in ("", "\n", "\r\n"):
            if (text + tail).encode("utf-8") == body:
                return style, bom, tail
    return None


def dump(data, style: tuple[dict, bool, str]) -> bytes:
    kw, bom, tail = style
    out = (json.dumps(data, **kw) + tail).encode("utf-8")
    return codecs.BOM_UTF8 + out if bom else out


def records_key(root: Path, index: dict[str, int]) -> tuple[list, list[str]]:
    """scan 이 읽는 그대로의 (이미지 경로·크기·박스) 목록과 파일이 없는 항목."""
    recs, rep = read_coco(root, index)
    rows = sorted((str(r.path.resolve()), r.width, r.height,
                   tuple((b.cls, b.cx, b.cy, b.w, b.h) for b in r.boxes)) for r in recs)
    return rows, rep.missing_images


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="설정 파일 (기본: 저장소의 config.yaml)")
    ap.add_argument("--apply", action="store_true", help="실제로 json 을 고친다 (먼저 원본을 떠 둔다)")
    ap.add_argument("--force", action="store_true", help="이미지의 절반 넘게 파일이 없는 json 도 고친다")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    root = cfg.seed_coco
    if root is None or not root.exists():
        sys.exit(f"설정의 data.seed_coco 가 없습니다: {root}")
    files = find_annotation_files(root)
    if not files:
        sys.exit(f"{root} 아래에 COCO json 이 없습니다")
    base = root if root.is_dir() else root.parent
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = base.parent / "seed_json_backup" / stamp
    if root.is_dir() and root.resolve() in backup_dir.resolve().parents:
        sys.exit(f"떠 둘 곳 {backup_dir} 이 {root} 안이라 scan 이 읽는다 — 멈춥니다")

    # 검증용 클래스 번호 — 모든 json 의 카테고리를 다 넣어 어떤 박스도 버려지지 않게 한다
    index: dict[str, int] = {}
    plans, problems = [], []
    for jf in files:
        raw = jf.read_bytes()
        data = json.loads(raw.decode("utf-8-sig"))
        for c in data.get("categories", []):
            index.setdefault(str(c["name"]).strip().lower(), len(index))
        images = data.get("images", [])
        ids = [im["id"] for im in images]
        gone = [im for im in images if _resolve_image(jf.parent, im["file_name"]) is None]
        rel = jf.relative_to(base)
        if not gone:
            print(f"{rel}: 이미지 {len(images)}장 — 빠진 것 없음")
            continue
        gone_ids = {im["id"] for im in gone}
        anns = data.get("annotations", [])
        n_ann = sum(1 for a in anns if a["image_id"] in gone_ids)
        print(f"{rel}: 이미지 {len(images)}장 중 파일 없음 {len(gone)}장({len(gone) / len(images):.0%}) — "
              f"뺄 annotations {n_ann}개 / {len(anns)}  (예: {[im['file_name'] for im in gone[:2]]})")
        if len(set(ids)) != len(ids):
            problems.append(f"{rel}: 이미지 id 가 겹친다 ({[i for i, n in Counter(ids).items() if n > 1][:5]})")
        if len(gone) * 2 > len(images) and not args.force:
            problems.append(f"{rel}: 이미지의 절반 넘게 파일이 없다 — 폴더를 옮겼는지 볼 것(정말 지운 것이면 --force)")
        style = find_style(data, raw)
        if style is None:
            problems.append(f"{rel}: 다시 저장하면 원래 모양이 안 나온다 — 쓰지 않는다")
        new = {k: v for k, v in data.items()}
        new["images"] = [im for im in images if im["id"] not in gone_ids]
        if "annotations" in data:
            new["annotations"] = [a for a in anns if a["image_id"] not in gone_ids]
        plans.append((jf, rel, raw, new, style, gone, gone_ids, anns))

    total = sum(len(p[5]) for p in plans)
    print(f"\n뺄 것: 이미지 {total}장 · annotations "
          f"{sum(sum(1 for a in p[7] if a['image_id'] in p[6]) for p in plans)}개 (json {len(plans)}개)")
    if problems:
        print("\n[멈춤] 아무것도 쓰지 않았다:\n  " + "\n  ".join(problems))
        return 1
    if not plans:
        print("빼야 할 항목이 없다.")
        return 0
    if not args.apply:
        print(f"\n(보기만 했다. 고치려면 --apply — 원본 json 을 {backup_dir} 에 떠 둔다)")
        return 0

    before, missing_before = records_key(root, index)

    # 1) 원본을 떠 두고 같은지 본다
    backup_dir.mkdir(parents=True, exist_ok=False)
    with open(backup_dir / "removed.tsv", "w", encoding="utf-8") as f:
        f.write("json\timage_id\tfile_name\twidth\theight\tannotations\n")
        for jf, rel, raw, new, style, gone, gone_ids, anns in plans:
            per = Counter(a["image_id"] for a in anns if a["image_id"] in gone_ids)
            for im in gone:
                f.write(f"{rel.as_posix()}\t{im['id']}\t{im['file_name']}\t{im.get('width')}\t{im.get('height')}"
                        f"\t{per[im['id']]}\n")
    for jf, rel, raw, *_ in plans:
        dst = backup_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(jf, dst)
        if dst.read_bytes() != raw:
            sys.exit(f"[멈춤] 떠 둔 {dst} 이 원본과 다르다 — json 은 아직 그대로다")
    print(f"원본을 떠 두었다: {backup_dir}")

    # 2) 고친 json 을 옆 임시 파일에 쓰고 바꿔 넣는다 (임시 이름은 .json 으로 끝나지 않아 scan 이 읽지 않는다)
    done, tmps = [], []
    try:
        for jf, rel, raw, new, style, *_ in plans:
            tmp = jf.with_name(f".{jf.name}.tmp")
            tmps.append(tmp)
            with open(tmp, "wb") as f:
                f.write(dump(new, style))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, jf)
            done.append((jf, rel))
        # 3) scan 이 읽는 그대로 다시 읽어 견준다
        after, missing_after = records_key(root, index)
        ok = after == before and not missing_after
        for jf, rel, raw, new, *_ in plans:
            ok = ok and json.loads(jf.read_bytes().decode("utf-8-sig")) == new
    except Exception as exc:   # noqa: BLE001 — 무엇이든 되돌린 뒤 알린다
        ok = False
        print(f"[오류] {exc!r}")
    if not ok:
        for jf, rel in done:
            shutil.copy2(backup_dir / rel, jf)
        for tmp in tmps:
            tmp.unlink(missing_ok=True)
        sys.exit("[실패] 다시 읽은 결과가 전과 달라 떠 둔 원본으로 되돌렸다")
    print(f"고쳤다: 파일이 있는 이미지 {len(after)}장과 박스는 전과 같고, 빠진 항목 {len(missing_before)} → 0.")
    print(f"뺀 목록: {backup_dir / 'removed.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
