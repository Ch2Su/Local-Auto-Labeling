"""'같은 카메라 자리' 판정의 근거를 다시 낸다 (2026-11-22).

밝기 축소본 상관만으로(도구의 첫 판) 시범 100장을 고른 뒤, 그 4,950쌍에 도구의 특징점 정합(`orb_match`)과 윤곽 방향 무늬
(`Scene.edges`)를 재서 — 상관만으로는 놓친 같은 자리 쌍이 몇인지, 문턱(`ORB_INLIERS`·`ORB_OVERLAP`·`SAME_EDGES`)이 어디에
놓이는지 본다. 위쪽 쌍들은 그림으로 저장한다(같은 자리인지는 사람이 본다 — 그날 본 결과는 docs/provenance.md).
DB·파일을 쓰지 않는다(읽기만 — 프로젝트 DB 는 이미 넣은 AI Hub 영상을 빼려고 연다). GPU 로 운영 가중치를 돌린다.

    python runs/aihub_pm/scene_probe.py <결과 폴더 — 저장소 밖>
"""

from __future__ import annotations

import io
import itertools
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import import_aihub_pm as ah  # noqa: E402
from autolabel.config import Config  # noqa: E402
from autolabel.store import Store  # noqa: E402


def main() -> int:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    cfg = Config.load(None)
    store = Store(cfg.db)
    cands = ah.candidates(ah.read_index(ah.index_path(cfg)), ah.BUCKETS, 80.0)
    lab = ah.Labeler(cfg, store.get_classes(), None, 0)
    frames = ah.Frames(ah.AIHUB_ROOT / "1.Training" / "원천데이터")
    full = ah.same_camera
    ah.same_camera = lambda a, b: ah.scene_corr(a.sig, b.sig) >= ah.SAME_SCENE      # 첫 판: 밝기 상관만
    try:
        picks, _ = ah.choose(ah.BUCKETS, cands, frames, lab, random.Random(20261122), taken=set(), scenes=[],
                             max_boxes=16, log=lambda *_: None)
    finally:
        ah.same_camera = full
        frames.close()
    n = len(picks)
    print(f"밝기 상관만으로 고른 {n}장, 쌍 {n * (n - 1) // 2:,}개")

    import cv2

    cv2.setNumThreads(8)
    rows = []
    for i, j in itertools.combinations(range(n), 2):
        a, b = picks[i].scene, picks[j].scene
        inl, ov = ah.orb_match(a, b)
        rows.append((inl, ov, float(a.edges @ b.edges), ah.scene_corr(a.sig, b.sig), i, j))
    orb = [r for r in rows if r[0] >= ah.ORB_INLIERS and r[1] >= ah.ORB_OVERLAP]
    edges = [r for r in rows if r[2] >= ah.SAME_EDGES]
    both = {(r[4], r[5]) for r in orb} | {(r[4], r[5]) for r in edges}
    print(f"특징점 정합으로 같은 자리: {len(orb)}쌍  (인라이어 {sorted(r[0] for r in orb)}, 겹침 "
          f"{min((r[1] for r in orb), default=0):.2f}–{max((r[1] for r in orb), default=0):.2f})")
    rest = [r for r in rows if not (r[0] >= ah.ORB_INLIERS and r[1] >= ah.ORB_OVERLAP)]
    print(f"  나머지 쌍: 인라이어 최대 {max(r[0] for r in rest)}, 그중 겹침 {ah.ORB_OVERLAP} 이상인 쌍의 인라이어 최대 "
          f"{max((r[0] for r in rest if r[1] >= ah.ORB_OVERLAP), default=0)}")
    print(f"윤곽 무늬 {ah.SAME_EDGES} 이상: {len(edges)}쌍, 둘 중 하나: {len(both)}쌍")
    e = np.array([r[2] for r in rows])
    print(f"  윤곽 무늬 전체: 중앙값 {np.median(e):.2f} · 95% {np.percentile(e, 95):.2f} · 99% {np.percentile(e, 99):.2f} · 최대 {e.max():.2f}")

    show = sorted(orb, reverse=True)[:8] + sorted((r for r in edges if r not in orb), key=lambda r: -r[2])[:8]
    w, h = 400, 225
    sheet = Image.new("RGB", (w * 2, (h + 4) * len(show)), (20, 20, 20))
    for k, (inl, ov, ed, corr, i, j) in enumerate(show):
        for s, t in enumerate((i, j)):
            im = Image.open(io.BytesIO(picks[t].data)).convert("RGB").resize((w, h), Image.BILINEAR)
            dr = ImageDraw.Draw(im)
            dr.rectangle((0, 0, 330, 13), fill=(0, 0, 0))
            dr.text((3, 1), f"{picks[t].video}  orb {inl}/{ov:.2f}  edges {ed:.2f}  corr {corr:.2f}", fill=(255, 255, 0))
            sheet.paste(im, (s * w, k * (h + 4)))
    sheet.save(out / "scene_pairs.jpg", quality=85)
    print(f"쌍 그림: {out / 'scene_pairs.jpg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
