"""영상에서 프레임을 뽑아 출처(카메라·촬영 시각)가 이름에 박힌 이미지로 `unlabeled/<카메라>/` 에 둔다.

**왜 이 도구로 뽑나 (2026-11-03, docs/provenance.md).** scan 은 카메라를 폴더로, 촬영 시각을 파일 이름으로
읽는다. 프레임을 아무 도구로 뽑으면 `frame_0001.jpg` 처럼 시각이 사라진다. 여기서는
`<카메라>_<YYYYMMDD_HHMMSS>_<밀리초>.jpg` 로 써서 scan 이 그대로 `captured_at` 을 채운다.
시각 = 영상 시작 시각 + 그 프레임의 영상 안 시각(OpenCV `CAP_PROP_POS_MSEC` — 가변 프레임레이트에도 맞다).

**시작 시각을 모르면 뽑지 않는다.** `--start` 를 주거나, 영상 파일 이름에 시각이 있어야 한다
(`provenance.py` 가 읽는 꼴 — 예: `cam01_20261103_140000.mp4`). 시각 없는 프레임을 만들면 나중에 채울 길이 없다.

**거의 같은 프레임 건너뛰기는 기본으로 끈다.** 처음에는 dHash 거리 2 이하(grouping 의 '같은 장면' 기준)를
건너뛰었는데, VIRAT 옥상 카메라 영상(`VIRAT_S_000001`)에서 5초 간격 138장 중 125장이 걸렸다(2026-11-03) — 8x8
dHash 는 배경만 보고, 화면 속 작은 사람·차가 바뀐 것은 못 본다. 그래서 `--min-change <퍼센트>` 를 주면 직전
저장 프레임과 비교해 **밝기가 20 넘게 바뀐 픽셀의 비율**(가로 320px 로 줄여서)이 그보다 작을 때만 건너뛴다.
같은 카메라라 분할은 어차피 한 묶음이다(grouping.py) — 중복은 검수 시간만 늘린다.
같은 영상(5초 간격 137쌍)에서 잰 바뀐 픽셀 비율은 중앙값 0.77%, 0.05% 미만(사실상 같은 화면)이 22쌍이었다 —
똑같은 화면만 빼려면 `--min-change 0.05` 정도. 그보다 올리면 사람 한 명이 지나간 장면도 빠질 수 있다(눈으로는 안 봤다).

**덮어쓰지 않는다.** 같은 이름이 있으면 건너뛴다 — 같은 영상을 다시 돌려도 결과가 그대로다.
카메라는 `cameras.yaml` 에 먼저 등록해야 한다(scan 과 같은 검사). 영상은 읽기만 한다.

검수 UI 의 '영상 캡처' 보기(2026-11-10, `src/autolabel/capture.py`)가 같은 방식으로 뽑고, 뽑은 것을 눈으로 골라 넣는다.
이름·시작 시각·변화량 함수(`frame_name`·`parse_start`·`small_gray`·`changed_pct`)는 그 모듈 한 곳에 있다.

    python tools/extract_frames.py 영상.mp4 --camera pg-cross-01 --start "2026-11-03 14:00:00"
    python tools/extract_frames.py 영상.mp4 --camera pg-cross-01 --every 10 --limit 200
    python tools/extract_frames.py 영상.mp4 --camera pg-cross-01 --start "..." --dry-run   # 쓰지 않고 셈만
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # tools/ -> 저장소 루트

from autolabel.capture import changed_pct, frame_name, small_gray  # noqa: E402
from autolabel.capture import parse_start as _parse_start  # noqa: E402
from autolabel.config import Config  # noqa: E402
from autolabel.provenance import _from_name, load_cameras  # noqa: E402


def parse_start(text: str) -> datetime:
    try:
        return _parse_start(text)
    except ValueError:
        raise SystemExit(f"--start '{text}' 를 읽지 못했습니다 — 예: \"2026-11-03 14:00:00\"") from None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video", type=Path)
    ap.add_argument("--camera", required=True, help="cameras.yaml 의 키")
    ap.add_argument("--start", default=None, help="영상 첫 프레임의 현지 시각 (없으면 영상 이름에서 찾는다)")
    ap.add_argument("--every", type=float, default=5.0, help="몇 초마다 한 장 (기본 5)")
    ap.add_argument("--min-change", type=float, default=0.0,
                    help="직전 저장 프레임과 바뀐 픽셀이 이 퍼센트보다 적으면 건너뛴다 (기본 0 = 끔, 예: 0.5)")
    ap.add_argument("--limit", type=int, default=0, help="최대 저장 장수 (0 = 끝까지)")
    ap.add_argument("--quality", type=int, default=95, help="JPEG 품질")
    ap.add_argument("--dry-run", action="store_true", help="파일을 쓰지 않고 몇 장이 나올지만 센다")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    import cv2

    cfg = Config.load(args.config)
    try:
        cams = load_cameras(cfg.cameras)
    except ValueError as exc:
        raise SystemExit(str(exc))
    if args.camera not in cams:
        raise SystemExit(f"카메라 '{args.camera}' 가 {cfg.cameras} 에 없습니다 — 먼저 등록할 것 (docs/provenance.md)")
    if args.every <= 0:
        raise SystemExit("--every 는 0 보다 커야 합니다")

    start = parse_start(args.start) if args.start else _from_name(args.video)
    if start is None:
        raise SystemExit("시작 시각을 모릅니다 — --start \"YYYY-MM-DD HH:MM:SS\" 를 주거나 영상 이름에 시각을 넣을 것")
    start = start.replace(tzinfo=None)           # 이름에 쓰는 것은 카메라 현지 시각이다

    if not args.video.exists():
        raise SystemExit(f"영상이 없습니다: {args.video}")
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"영상을 열지 못했습니다: {args.video} (경로에 한글이 있으면 OpenCV 가 못 여는 경우가 있다)")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur = n_frames / fps if fps else 0.0
    out_root = cfg.unlabeled / args.camera
    print(f"영상 {args.video.name}: {fps:.2f} fps, {n_frames} 프레임 (약 {dur / 60:.1f}분)")
    print(f"카메라 {args.camera} ({cams[args.camera].place}), 시작 {start:%Y-%m-%d %H:%M:%S}, {args.every:g}초 간격"
          f" -> {out_root}" + ("  [dry-run]" if args.dry_run else ""))

    t0 = time.time()
    next_t = 0.0
    idx = -1
    last_small = None
    saved = similar = exists = 0
    first = last = None
    while True:
        if not cap.grab():
            break
        idx += 1
        pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if pos <= 0 and idx > 0 and fps:
            pos = idx / fps                       # 백엔드가 시각을 안 주면 프레임 번호로
        if pos + 1e-6 < next_t:
            continue
        next_t = (int((pos + 1e-6) // args.every) + 1) * args.every   # 0·5·10초… 격자에 맞춘다
        ok, frame = cap.retrieve()
        if not ok:
            continue
        small = small_gray(frame)
        if args.min_change > 0 and last_small is not None and changed_pct(small, last_small) < args.min_change:
            similar += 1
            continue
        t = start + timedelta(seconds=pos)
        dst = out_root / f"{t:%Y%m%d}" / frame_name(args.camera, t)
        if dst.exists():
            exists += 1
            last_small = small
            continue
        if not args.dry_run:
            okj, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            if not okj:
                print(f"  인코딩 실패: {dst.name}", file=sys.stderr)
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            with open(dst, "xb") as f:            # 있으면 FileExistsError — 덮어쓰지 않는다
                f.write(buf.tobytes())
        last_small = small
        saved += 1
        first = first or t
        last = t
        if args.limit and saved >= args.limit:
            break
    cap.release()

    span = f", {first:%H:%M:%S} ~ {last:%H:%M:%S}" if first else ""
    print(f"{'셀 수 있는' if args.dry_run else '저장'} {saved}장{span} · 비슷해서 건너뜀 {similar}장 · "
          f"이미 있어 건너뜀 {exists}장 · {time.time() - t0:.0f}초")
    if saved and not args.dry_run:
        print("다음: labeler.py scan (카메라·촬영 시각이 DB 에 들어간다)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
