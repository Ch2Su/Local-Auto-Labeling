"""정해진 시각에 웹 UI 로 시작한 학습을 정상 중단한다.

    python tools/scheduled_stop.py 2026-09-10T06:30

이 세션이나 터미널이 닫혀도 계속 돌도록 분리해서 띄운다.

**웹 UI 의 [중단] 버튼과 같은 경로를 쓴다** (`POST /api/jobs/{id}/stop`). 그래서
Windows 학습이든 'WSL 학습' 버튼으로 시작한 학습이든 똑같이 멈춘다 — WSL 쪽은 서버가
그 요청을 받아 WSL 의 STOP 파일을 만든다. 다음 배치에서 멈추고 검증·저장한 뒤
resume.pt 를 남긴다.

운영 가중치는 따로 백업하지 않는다. 학습이 끝나면(중단 포함) `promote_weights()` 가
기존 `models/best.pt` 를 `best.prev.pt` 로 남긴 뒤 교체한다.

**프로젝트 탭 서버(기본)에서는 프로젝트마다 API 주소가 따로다** (`/p/visible/api/...`).
그래서 `/api/projects` 로 목록을 받아 학습이 도는 프로젝트를 찾는다. `--config` 로 하나만
띄운 서버면 그 목록이 없으므로 루트(`/api/...`)를 본다. 예전에는 루트만 봐서, 탭 서버로
바뀐 뒤(2026-09-21)에는 '중단할 작업이 없다' 며 아무것도 멈추지 못했다(2026-09-23 수정).

CLI 로 돌리는 학습이면 이 스크립트 대신 정지 파일을 쓰면 된다 (`runs/STOP`).
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:7860"
ROOT = Path(__file__).resolve().parents[1]      # tools/ -> 저장소 루트
LOG = ROOT / "runs" / "scheduled_stop.log"


def log(msg: str) -> None:
    line = f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


TRAIN_KINDS = {"train", "resume", "train_wsl", "resume_wsl"}


def api(path: str, method: str = "GET"):
    req = urllib.request.Request(BASE + path, method=method)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def app_prefixes() -> list[str]:
    """앱 주소 앞부분들. 탭 서버면 ['/p/visible', ...], 단일 서버면 ['']."""
    try:
        return [p["base"].rstrip("/") for p in api("/api/projects")["projects"]]
    except urllib.error.HTTPError:          # 단일 설정 서버에는 /api/projects 가 없다
        return [""]
    except (urllib.error.URLError, TimeoutError, OSError):
        return []                           # 서버가 없다


def current_job() -> tuple[str, dict] | None:
    """(앱 주소 앞부분, 작업). 도는 학습이 있으면 그것, 없으면 가장 최근 작업 하나."""
    latest = None
    for prefix in app_prefixes():
        try:
            job = api(f"{prefix}/api/state").get("job")
        except (urllib.error.URLError, TimeoutError, OSError):
            continue
        if not job:
            continue
        if job["status"] == "running" and job["kind"] in TRAIN_KINDS:
            return prefix, job
        latest = latest or (prefix, job)
    return latest


def job_status(prefix: str, job_id: int) -> dict | None:
    try:
        return api(f"{prefix}/api/jobs/{job_id}")
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    target = dt.datetime.fromisoformat(sys.argv[1])
    log(f"예약 중단 대기 시작 — 목표 {target:%Y-%m-%d %H:%M}")

    while True:
        now = dt.datetime.now()
        if now >= target:
            break
        remain = (target - now).total_seconds()
        # 한 시간마다 살아 있음을 남긴다. 마지막 구간은 촘촘히 본다.
        time.sleep(min(remain, 1800))
        if remain > 1800:
            found = current_job()
            st = f"{found[0] or '/'} {found[1]['kind']} {found[1]['status']}" if found else "없음"
            log(f"대기 중 — 남은 {remain/3600:.1f}시간, 학습 상태 {st}")

    found = current_job()
    if found is None:
        log("중단할 작업이 없습니다 (서버가 없거나 학습이 이미 끝났습니다).")
        return 1
    prefix, job = found
    if job["status"] != "running" or job["kind"] not in TRAIN_KINDS:
        log(f"도는 학습이 없습니다 (가장 최근 작업: {prefix or '/'} {job['kind']} {job['status']}). "
            "아무것도 하지 않습니다.")
        return 0

    log(f"작업 {job['id']} ({prefix or '/'} {job['kind']}) 중단 요청 — 현재 진행: {job['progress'][:80]}")
    api(f"{prefix}/api/jobs/{job['id']}/stop", method="POST")

    # 진행 중이던 epoch 과 최종 검증이 끝날 때까지 기다린다.
    deadline = time.time() + 3600
    while time.time() < deadline:
        time.sleep(30)
        job = job_status(prefix, job["id"]) or job
        if job["status"] != "running":
            break
    if job and job["status"] != "running":
        log(f"중단 완료 — 상태 {job['status']}")
        if job.get("result"):
            log(f"결과: {json.dumps(job['result'], ensure_ascii=False)[:400]}")
        if job.get("error"):
            log(f"오류: {job['error']}")
    else:
        log("1시간을 기다렸으나 아직 종료되지 않았습니다. 수동으로 확인하세요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
