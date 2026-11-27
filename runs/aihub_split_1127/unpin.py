"""AI Hub 카메라(ah-NNNN)의 train 고정을 푼다 — 같은 자리로 이어진 짝만 train 으로 남긴다 (2026-11-27 사용자 결정 '고정 풀고 진행').

    python runs/aihub_split_1127/unpin.py <짝 json>            # 보기만
    python runs/aihub_split_1127/unpin.py <짝 json> --apply    # cameras.yaml 의 split 줄만 고친다(쓰기 전 파일은 backups/)

짝 json 은 같은 폴더의 `ah_groups.py` 와 같은 판정(같은 영상 id 이거나 `import_aihub_pm.same_camera`)으로 뽑은 짝 목록이다 —
[{"why", "a": [카메라, id, 상태], "b": [...]}]. 짝에 든 카메라는 train 고정 그대로 둔다(같은 장소가 두 분할로 갈리지 않게 —
둘을 한 키로 모으려면 그림 파일을 옮겨야 해서 고정으로 대신했다).

DB 는 읽기만 한다(mode=ro). 이미지가 이미 분할에 든 카메라가 하나라도 있으면 멈춘다 — split.py 는 정한 분할을 옮기지 않아,
고정을 바꾸면 그 카메라가 두 분할에 걸칠 수 있다(11-27 에는 실제 DB 가 전부 미배정이었다).
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from autolabel.config import Config  # noqa: E402
from autolabel.provenance import load_cameras, set_splits  # noqa: E402
from import_aihub_pm import KEY_RE  # noqa: E402


def main() -> int:
    pairs = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    write = "--apply" in sys.argv[2:]
    cfg = Config.load(None)
    keep = sorted({p[side][0] for p in pairs for side in ("a", "b")})
    cams = load_cameras(cfg.cameras)
    ah = sorted(k for k in cams if KEY_RE.match(k))
    con = sqlite3.connect(f"file:{Path(cfg.db).as_posix()}?mode=ro", uri=True)
    placed = con.execute("SELECT camera, split, COUNT(*) FROM images WHERE split IS NOT NULL AND camera LIKE 'ah-%'"
                         " GROUP BY camera, split").fetchall()
    con.close()
    if placed:
        raise SystemExit(f"멈춘다 — 이미 분할에 든 AI Hub 이미지가 있다: {placed[:10]}")
    missing = [k for k in keep if k not in cams]
    if missing:
        raise SystemExit(f"멈춘다 — 짝의 카메라가 등록 파일에 없다: {missing}")
    want = {k: ("train" if k in keep else None) for k in ah}
    change = {k: v for k, v in want.items() if cams[k].split != v}
    print(f"ah-* {len(ah)}대 — train 으로 남길 짝 {len(keep)}대 {keep}, 고정을 풀 것 "
          f"{sum(1 for v in change.values() if v is None)}대, train 으로 새로 고정할 것 {sum(1 for v in change.values() if v)}대")
    other = {k: cams[k].split for k in ah if cams[k].split not in (None, "train")}
    if other:
        raise SystemExit(f"멈춘다 — train 이 아닌 고정이 있다(사람이 적은 것): {other}")
    if not change:
        print("바꿀 것이 없다")
        return 0
    done = set_splits(cfg.cameras, change, backup_dir=Path(cfg.db).parent / "backups", dry_run=not write)
    print(("고쳤다" if write else "(보기만 했다 — 고치려면 --apply)") + f": {len(done)}대")
    return 0


if __name__ == "__main__":
    sys.exit(main())
