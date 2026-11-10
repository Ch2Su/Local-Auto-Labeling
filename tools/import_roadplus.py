"""ROAD PLUS(고속도로 교통정보) CCTV 를 OBS 로 녹화한 파일에서 **전체 화면으로 재생된 구간만** 골라 프레임을 뽑는다 (2026-11-09).

녹화 하나에는 사이트 화면(지도·작은 재생 창)·전체 화면 재생·빈 화면(영상이 끝났거나 다음 것을 부르는 중 — 어두운 회색)이 섞여
있다. 사이트는 CCTV 를 30초 안팎 파일로 틀어 주고(재생기 '0:17 / 0:25'), 다시 열면 새 파일을 준다. 이 도구는:
- **전체 화면 판별** — 초당 4장씩 훑어, 양옆 끝 띠(폭의 4.7%, 2560 이면 120px)가 검고(평균 밝기 < 6) 가운데가 고르지 않은
  (표준편차 > 20) 프레임을 '영상' 으로 본다. 사이트 화면은 왼쪽 끝이 밝고, 빈 화면(어두운 회색)은 띠가 검지 않아 사이트 화면으로 셈한다.
- **구간** — 0.5초 넘게 끊기지 않은 '영상' 이 `--min-seg`(3초) 이상이면 한 구간. 앞 `--skip-head`(4초 — 전체 화면 안내 'Esc'·
  재생 막대가 겹친다. 11-09 첫 녹화를 2초로 뽑았더니 구간 첫 장 위쪽에 안내 글상자가 남았다)와 뒤 `--skip-tail`(1초)을 뺀다.
  마우스를 움직여 재생 막대가 겹친 프레임은 남을 수 있다 — 검수 때 볼 것.
- **같은 영상을 다시 튼 구간** — 구간의 1초 간격 썸네일(216×144 흑백)마다 앞 구간들 썸네일(초당 4장) 중 가장 가까운 것과의 평균
  밝기 차를 재서, 중앙값 < 0.8 이고 80% 값 < 1.2 이면(거의 같은 화면이 내내 이어지면) 건너뛴다(`--keep-repeats` 면 넣는다).
  처음엔 64×36 썸네일·차 3 미만으로 셌는데, 11-09 hw-05 녹화(차가 드문 거의 멈춘 장면)에서 새 영상 두 구간을 '다시 튼 것'
  으로 잘못 뺐다 — 같은 재생 시점끼리의 차가 중앙값 2.0~2.7 로 구간 안 0.5초 이웃(0.9)보다 컸다. 다시 튼 영상이면 같은 원본이라
  차가 압축 잡음만큼(1 아래)이어야 한다 — 진짜로 다시 튼 녹화로는 아직 재 보지 않았다.
- **자르기·줄이기** — 그 구간에서 검지 않은 행·열의 범위가 영상 영역이다(원본 SD 를 3배로 키운 화면 — 11-07 녹화는 2160×1440 @ x=200).
  높이 720 으로 줄인다(SD 원본이라 잃는 것이 없다 — 11-07 사고 영상도 이렇게 다듬었다).
- **프레임** — 구간 안에서 `--every`(기본 2초)마다 `unlabeled/<카메라>/<YYYYMMDD>/<카메라>_<YYYYMMDD_HHMMSS>_<밀리초>.jpg`
  (JPEG 95, extract_frames.py 와 같은 이름 — scan 이 카메라·촬영 시각을 읽는다). 시각 = 녹화 시작 + 녹화 안 시각, 녹화 시작은
  `--start` 가 없으면 **파일 수정 시각 − 길이**(OBS 는 녹화를 끝낼 때 파일을 닫는다 — 복사한 파일이면 틀리니 `--start` 를 줄 것).
  사이트가 주는 CCTV 는 몇 분 안의 영상이라 촬영 시각도 그 언저리다(정확하지 않다).
- 카메라는 `cameras.yaml` 에 먼저 등록해야 한다(kind: screen — docs/provenance.md). 같은 이름이 있으면 건너뛴다(다시 돌려도 같다).
  녹화 파일은 읽기만 한다.
- **원본을 지우기 전에 영상으로 남기기 (`--save-clips`, 11-09).** 녹화는 2560×1440 60fps 라 크다(30초에 100 MB 남짓). 구간마다
  영상 영역만 잘라 높이 720 · H.264 CRF 18 · 1초마다 키프레임 · 60fps 그대로 · 소리 없이 `<폴더>/<카메라>/<카메라>_<구간 시작 시각>.mp4`
  로 쓴다(11-07 사고 영상을 다듬은 방식 — DetectView 로 재생할 수 있고, extract_frames.py 가 이름에서 시각을 읽는다). 앞뒤는 프레임과
  같은 만큼 뺀다. ffmpeg 이 PATH 에 없으면 `--ffmpeg` 로 준다. 이미 프레임을 뽑은 녹화면 `--no-frames` — **기본값을 바꾼 뒤 다시
  돌리면 프레임 시각이 달라져 새 프레임이 생긴다**(11-09 첫 녹화 5개는 앞 2초를 뺀 프레임이다).

    .venv\\Scripts\\python.exe tools\\import_roadplus.py "<녹화>.mkv" --camera hw-04 --no-frames --save-clips data\\visible\\videos --ffmpeg <ffmpeg.exe>

    .venv\\Scripts\\python.exe tools\\import_roadplus.py "<녹화>.mkv" --camera hw-04 --dry-run   # 구간·장수만
    .venv\\Scripts\\python.exe tools\\import_roadplus.py "<녹화>.mkv" --camera hw-04
    # 그다음: labeler.py scan → labeler.py label → (출동 차량·라바콘) tools/import_proposals.py list --camera … → ovd_propose → apply
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # tools/ -> 저장소 루트

from autolabel.capture import frame_name  # noqa: E402,F401 — 이름 꼴은 extract_frames·영상 캡처와 한 곳에서
from autolabel.config import Config  # noqa: E402
from autolabel.provenance import load_cameras  # noqa: E402

SCAN_FPS = 4          # 판별할 때 초당 몇 장
BAND = 0.047          # 양옆 띠 폭(화면 폭 비율)
BLACK = 6.0           # 띠 평균 밝기가 이보다 작으면 검다
TEXTURE = 20.0        # 가운데 표준편차가 이보다 크면 영상이 있다
GAP = 0.5             # 이만큼(초)까지 끊겨도 한 구간
THUMB = (216, 144)    # 다시 튼 구간을 가릴 썸네일 크기(3:2)
REPEAT_MEDIAN = 0.8   # 가장 가까운 앞 썸네일과의 평균 밝기 차 — 중앙값이 이보다 작고
REPEAT_P80 = 1.2      # 80% 값도 이보다 작으면 다시 튼 구간


def classify(gray, np) -> str:
    h, w = gray.shape
    b = max(4, int(w * BAND))
    if gray[:, :b].mean() < BLACK and gray[:, -b:].mean() < BLACK:
        return "video" if gray[h // 8: h * 7 // 8, w // 6: w * 5 // 6].std() > TEXTURE else "blank"
    return "ui"


def scan(cap, cv2, np, fps: float, n: int):
    """초당 SCAN_FPS 장 — (시각, 판별, 썸네일, 열 평균, 행 평균)."""
    step = max(1, round(fps / SCAN_FPS))
    out = []
    i = 0
    while i < n:
        if not cap.grab():
            break
        if i % step == 0:
            ok, f = cap.retrieve()
            if ok:
                g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                kind = classify(g, np)
                thumb = cv2.resize(g, THUMB, interpolation=cv2.INTER_AREA)   # uint8 — 녹화 4분이면 약 30 MB
                out.append((i / fps, kind, thumb, g.mean(axis=0) if kind == "video" else None,
                            g.mean(axis=1) if kind == "video" else None))
        i += 1
    return out


def segments(samples, min_seg: float, head: float, tail: float):
    segs, cur = [], None
    for t, kind, *_ in samples:
        if kind == "video":
            if cur and t - cur[1] <= GAP + 1.0 / SCAN_FPS:
                cur[1] = t
            else:
                if cur:
                    segs.append(cur)
                cur = [t, t]
        # 다른 판별은 구간을 끊지만 GAP 안의 한두 장은 위의 이어 붙이기가 메운다
    if cur:
        segs.append(cur)
    out = []
    for a, b in segs:
        if b - a >= min_seg:
            a2, b2 = a + head, b - tail
            if b2 - a2 >= 1.0:
                out.append((a, b, a2, b2))
    return out


def crop_box(samples, a: float, b: float, np):
    cols = np.median(np.stack([s[3] for s in samples if a <= s[0] <= b and s[3] is not None]), axis=0)
    rows = np.median(np.stack([s[4] for s in samples if a <= s[0] <= b and s[4] is not None]), axis=0)
    xs, ys = np.where(cols > 10)[0], np.where(rows > 10)[0]
    return int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1


def find_ffmpeg(given: str | None) -> str | None:
    import shutil
    if given:
        return given
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def save_clips(a, plan, start: datetime) -> int:
    """구간마다 영상 영역만 잘라 높이 720·H.264 CRF 18·1초마다 키프레임·60fps 그대로·소리 없이 — 11-07 사고 영상을 다듬은 방식."""
    import subprocess
    exe = find_ffmpeg(a.ffmpeg)
    if exe is None:
        print("[멈춤] ffmpeg 를 찾지 못했다 — --ffmpeg 로 줄 것(예: uv 의 imageio-ffmpeg 가 받은 실행 파일)")
        return 2
    out_dir = a.save_clips / a.camera
    out_dir.mkdir(parents=True, exist_ok=True)
    wrote = skipped = 0
    for (x0, y0, x1, y1), _times, sa, sb in plan:
        dst = out_dir / f"{a.camera}_{start + timedelta(seconds=sa):%Y%m%d_%H%M%S}.mp4"
        if dst.exists():
            skipped += 1
            continue
        cmd = [exe, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{sa:.3f}", "-i", str(a.recording),
               "-t", f"{sb - sa:.3f}", "-vf", f"crop={x1 - x0}:{y1 - y0}:{x0}:{y0},scale=-2:720:flags=area",
               "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-g", "60", "-pix_fmt", "yuv420p", "-an", str(dst)]
        subprocess.run(cmd, check=True)
        wrote += 1
    print(f"  영상 {wrote}개 · 이미 있어 건너뜀 {skipped} → {out_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="ROAD PLUS 녹화 → 전체 화면 구간의 프레임 (머리말 참고)")
    ap.add_argument("recording", type=Path)
    ap.add_argument("--camera", required=True, help="cameras.yaml 의 키")
    ap.add_argument("--every", type=float, default=2.0, help="구간 안에서 몇 초마다 한 장 (기본 2)")
    ap.add_argument("--start", default=None, help="녹화 시작 현지 시각 'YYYY-MM-DD HH:MM:SS' (없으면 파일 수정 시각 − 길이)")
    ap.add_argument("--min-seg", type=float, default=3.0)
    ap.add_argument("--skip-head", type=float, default=4.0)
    ap.add_argument("--skip-tail", type=float, default=1.0)
    ap.add_argument("--keep-repeats", action="store_true", help="다시 튼 구간도 넣는다")
    ap.add_argument("--dry-run", action="store_true", help="구간과 장수만 보인다(파일을 쓰지 않는다)")
    ap.add_argument("--save-clips", type=Path, default=None,
                    help="구간마다 다듬은 영상을 <이 폴더>/<카메라>/<카메라>_<YYYYMMDD_HHMMSS>.mp4 로 (ffmpeg 필요 — 머리말)")
    ap.add_argument("--no-frames", action="store_true", help="프레임은 쓰지 않는다(이미 프레임을 뽑은 녹화에서 영상만 남길 때)")
    ap.add_argument("--ffmpeg", default=None, help="ffmpeg 실행 파일 (없으면 PATH → imageio_ffmpeg 순으로 찾는다)")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")

    import cv2
    import numpy as np

    cfg = Config.load(a.config)
    cams = load_cameras(cfg.cameras)
    if a.camera not in cams:
        print(f"[멈춤] 카메라 '{a.camera}' 가 {cfg.cameras} 에 없습니다 — 먼저 등록할 것")
        return 2
    cap = cv2.VideoCapture(str(a.recording), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"[멈춤] 열지 못함: {a.recording}")
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not fps or n <= 0:
        print(f"[멈춤] 길이를 모른다(fps {fps}, 프레임 {n}) — 녹화가 덜 끝난 파일일 수 있다")
        return 2
    dur = n / fps
    start = (datetime.strptime(a.start, "%Y-%m-%d %H:%M:%S") if a.start
             else datetime.fromtimestamp(a.recording.stat().st_mtime) - timedelta(seconds=dur))
    print(f"{a.recording.name}: {int(cap.get(3))}×{int(cap.get(4))} {fps:.2f}fps {dur:.1f}초 · 녹화 시작 {start:%Y-%m-%d %H:%M:%S}"
          f"{'' if a.start else ' (파일 수정 시각 − 길이)'}", flush=True)
    samples = scan(cap, cv2, np, fps, n)
    kinds = {k: sum(1 for s in samples if s[1] == k) / SCAN_FPS for k in ("video", "ui", "blank")}
    print(f"  판별(초): 전체 화면 {kinds['video']:.0f} · 사이트 화면 {kinds['ui']:.0f} · 빈 화면 {kinds['blank']:.0f}", flush=True)

    plan, seen = [], []
    for raw_a, raw_b, sa, sb in segments(samples, a.min_seg, a.skip_head, a.skip_tail):
        thumbs = [s[2].astype(np.int16) for s in samples if sa <= s[0] <= sb][:: SCAN_FPS]
        near = ""
        repeat = False
        if seen and thumbs:
            best = np.array([min(float(np.abs(t - u).mean()) for u in seen) for t in thumbs])
            med, p80 = float(np.median(best)), float(np.percentile(best, 80))
            repeat = med < REPEAT_MEDIAN and p80 < REPEAT_P80
            near = f" · 앞 구간과 차 중앙 {med:.2f}·80% {p80:.2f}"
        box = crop_box(samples, raw_a, raw_b, np)
        times = [sa + k * a.every for k in range(int((sb - sa) / a.every) + 1)]
        print(f"  구간 {raw_a:6.1f}~{raw_b:6.1f}초 → 쓸 곳 {sa:6.1f}~{sb:6.1f}초 · 영역 {box[2] - box[0]}×{box[3] - box[1]} @ ({box[0]},{box[1]})"
              f" · {len(times)}장{near}{' · 다시 튼 구간(건너뜀)' if repeat and not a.keep_repeats else ''}", flush=True)
        seen.extend(s[2].astype(np.int16) for s in samples if raw_a <= s[0] <= raw_b)
        if not repeat or a.keep_repeats:
            plan.append((box, times, sa, sb))
    total = sum(len(p[1]) for p in plan)
    if a.dry_run:
        print(f"  넣을 프레임 {total}장 · 영상 {len(plan)}개 (보기만 했다)")
        return 0
    if a.save_clips:
        if save_clips(a, plan, start) != 0:
            return 2
    if a.no_frames:
        return 0
    out_dir = cfg.unlabeled / a.camera
    wrote = skipped = 0
    for (x0, y0, x1, y1), times, _sa, _sb in plan:
        for t in times:
            when = start + timedelta(seconds=t)
            dst = out_dir / f"{when:%Y%m%d}" / frame_name(a.camera, when)
            if dst.exists():
                skipped += 1
                continue
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, f = cap.read()
            if not ok:
                continue
            f = f[y0:y1, x0:x1]
            hh = 720
            ww = round(f.shape[1] * hh / f.shape[0])
            f = cv2.resize(f, (ww, hh), interpolation=cv2.INTER_AREA)
            dst.parent.mkdir(parents=True, exist_ok=True)
            ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if ok:
                dst.write_bytes(buf.tobytes())   # imwrite 는 한글 경로를 못 쓸 수 있다
                wrote += 1
    print(f"  썼다 {wrote}장 · 이미 있어 건너뜀 {skipped} → {out_dir}")
    print("  다음: labeler.py scan → labeler.py label → tools/import_proposals.py list --camera "
          f"{a.camera} … (머리말)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
