"""국가교통정보센터(ITS) 돌발 지점 근처 CCTV 의 '최근 30초' 영상을 받는다 (2026-11-24 사용자 결정 '돌발 지점 근처').

responder·cone 장면이 모자라서 — 돌발상황정보(사고·고장·공사 좌표)와 CCTV 목록(카메라 좌표)을 거리로 맞춰, 가까운 카메라의
동영상(cctvType 5 — 가장 최근 약 30초 mp4)만 받는다. 조사·수치는 docs/provenance.md '국가교통정보센터(ITS) CCTV Open API'.

    python tools/its_incident_clips.py                 미리 보기 — 무엇을 받을지만 찍는다(영상은 안 받는다)
    python tools/its_incident_clips.py --apply         받는다 → data/visible/its/clips/<날짜>/its-<번호>_<시각>.mp4

알아 둘 것
- **API 횟수가 든다** — 돌발 1회 + 목록(고속도로·국도 1회씩). 목록은 `--cache` 폴더에 두고 20시간 안이면 다시 쓴다(영상 주소가
  24시간만 유효하다). 월 한도가 개발용 100건이다 — 미리 보기도 돌발 1회는 쓴다(`--events-file` 로 받아 둔 것을 주면 0회).
- 키는 환경 변수 `ITS_API_KEY`(없으면 Windows 사용자 환경 변수). 값을 찍지 않고, 요청 주소도 찍지 않는다.
- **거리는 좌표끼리 잰 것이다** — 그 카메라에 현장이 보이는지는 받아서 봐야 안다(카메라는 돌아가고 반대 방향을 볼 수 있다).
- DB·`cameras.yaml`·`unlabeled/` 는 건드리지 않는다. 받은 영상은 검수 화면 '영상 캡처' 탭이나 `tools/auto_capture.py` 로 넣는다 —
  이름의 `its-<번호>` 는 등록된 카메라가 아니니 넣을 때 카메라를 고르거나 먼저 등록한다(번호가 날마다 같은지는 아직 안 봤다).
- 받은 것 옆의 `manifest.jsonl` 에 실제 지명·돌발 글이 들어간다 — `data/` 밖(git)으로 옮기지 말 것. 그림을 밖(Runpod 등)으로
  올리기 전에는 사용자에게 물을 것(이용 조건에 그 얘기가 없다).
- 이름의 시각은 '받기 시작한 시각 − 영상 길이' 다(영상에 시각 정보가 없다 — 몇 초 어긋날 수 있다).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://openapi.its.go.kr:9443/"
UA = "Mozilla/5.0 (compatible; cctv-clips)"
BOX_KR = (124.0, 132.0, 33.0, 39.0)            # minX maxX minY maxY — 전국
LIST_TTL = 20 * 3600                           # 목록을 다시 쓰는 한도(초) — 영상 주소는 24시간 유효
ROADS = {"고속도로": "ex", "국도": "its"}       # 돌발의 도로 유형 → 목록의 type (그 밖의 도로는 카메라가 없다)
# 받는 순서 — 드문 것부터. 재난·기상(강풍·적설 알림)과 기타(행사·정체)는 현장이 없어 뺀다.
EVENT_ORDER = ("교통사고", "기타돌발", "공사")
# 현장이 그 자리에 없는 돌발 — 정체 알림, 움직이며 하는 작업(제설·제설제 살포·이동 도색·이동 세척). 11-24 첫 미리 보기에서
# 고른 20개 중 11개가 이것이었다. 글(message)과 세부 유형(eventDetailType)에서 찾는다.
SKIP = r"정체|이동|살포|제설"
MAX_CLIP = 40 * 1024 * 1024                    # 30초 영상은 3–4 MB(720×480)·15 MB 안팎(1080p)이었다

_KEY = ""


def api_key() -> str:
    k = os.environ.get("ITS_API_KEY", "")
    if not k and sys.platform == "win32":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as h:
                k = winreg.QueryValueEx(h, "ITS_API_KEY")[0]
        except OSError:
            k = ""
    k = (k or "").strip()
    if not k:
        raise SystemExit("ITS_API_KEY 가 없습니다 — 사용자 환경 변수에 인증키를 넣으세요.")
    return k


def scrub(text) -> str:
    """오류 글에서 키를 지운다(urllib 의 예외가 주소를 담을 수 있다)."""
    text = str(text)
    if _KEY:
        for s in {_KEY, urllib.parse.quote(_KEY, safe="")}:
            text = text.replace(s, "***")
    return text


def call(path: str, **params) -> bytes:
    """API 한 번. 키가 든 주소는 어디에도 남기지 않는다."""
    global _KEY
    _KEY = _KEY or api_key()
    box = dict(zip(("minX", "maxX", "minY", "maxY"), BOX_KR))
    q = urllib.parse.urlencode({"apiKey": _KEY, **params, **box, "getType": "json"})
    req = urllib.request.Request(API + path + "?" + q, headers={"User-Agent": UA, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        raise SystemExit(f"API {path} 실패: HTTP {e.code} {scrub(e.read()[:200].decode('utf-8', 'replace'))}")
    except Exception as e:
        raise SystemExit(f"API {path} 실패: {type(e).__name__}: {scrub(e)}")


# ---------------------------------------------------------------- 고르기 (순수 함수 — tests/test_its_incident_clips.py)
_CAM_ID = re.compile(r"[A-Za-z0-9_-]{1,40}")


def cam_id(url: str) -> str:
    """영상 주소의 첫 칸 — 목록끼리 같은 카메라를 가리키는 번호(11-24 실측).

    **파일 이름에 들어가는 값이라 영숫자·`_`·`-` 만 받는다 (2026-11-25, Gemini 리뷰)** — 밖(API 응답)에서 온 글자다. 그 밖의
    글자(`..`·`\\`·`:` 등)가 든 칸이나 http(s) 가 아닌 주소는 번호 없음("")으로 본다: 그 카메라는 목록에서 빠진다(parse_cams)."""
    parts = urllib.parse.urlsplit(str(url or ""))
    if parts.scheme not in ("http", "https"):
        return ""
    segs = [s for s in parts.path.split("/") if s]
    return segs[0] if segs and _CAM_ID.fullmatch(segs[0]) else ""


def meters(x1: float, y1: float, x2: float, y2: float) -> float:
    """경위도 두 점의 거리(m) — 수 km 안에서만 쓰는 평면 어림."""
    return math.hypot((x2 - x1) * 111_320 * math.cos(math.radians((y1 + y2) / 2)), (y2 - y1) * 110_540)


def parse_cams(obj: dict, road: str) -> list[dict]:
    """cctvInfo 응답 → [{id, road, name, x, y, url}]. 번호·좌표·주소가 없는 줄은 버린다."""
    data = (obj.get("response") or {}).get("data") or []
    if isinstance(data, dict):
        data = [data]
    out = []
    for it in data:
        url = str(it.get("cctvurl") or "").strip()
        try:
            x, y = float(it.get("coordx")), float(it.get("coordy"))
        except (TypeError, ValueError):
            continue
        if url and cam_id(url):
            out.append({"id": cam_id(url), "road": road, "name": str(it.get("cctvname") or ""), "x": x, "y": y, "url": url})
    return out


def parse_events(obj: dict) -> list[dict]:
    """eventInfo 응답 → 좌표가 있는 돌발만."""
    items = (obj.get("body") or {}).get("items") or []
    if isinstance(items, dict):
        items = [items]
    out = []
    for it in items:
        try:
            x, y = float(it.get("coordX")), float(it.get("coordY"))
        except (TypeError, ValueError):
            continue
        out.append({"road": str(it.get("type") or ""), "type": str(it.get("eventType") or ""),
                    "detail": str(it.get("eventDetailType") or ""), "start": str(it.get("startDate") or ""),
                    "end": str(it.get("endDate") or ""), "lanes": str(it.get("lanesBlocked") or ""),
                    "message": str(it.get("message") or "").strip(), "road_name": str(it.get("roadName") or ""),
                    "x": x, "y": y})
    return out


def choose(events: list[dict], cams: list[dict], *, radius: float = 500.0, per_event: int = 1, limit: int = 20,
           types: tuple[str, ...] = EVENT_ORDER, now: str = "", skip: str = SKIP) -> list[dict]:
    """돌발마다 반경 안의 가까운 카메라를 per_event 대까지 — 같은 카메라는 한 번만, 모두 limit 대까지.

    돌발은 types 의 순서대로, 같은 유형 안에서는 최근에 시작한 것부터 본다. 돌발과 도로 유형이 같은 카메라만 본다
    (고속도로 돌발에 옆 국도 카메라를 붙이지 않는다). 끝난 돌발(endDate 가 now 보다 이름)과, 글·세부 유형이
    skip(정규식 — 빈 글이면 안 거른다)에 걸리는 돌발은 뺀다."""
    import re
    bad = re.compile(skip) if skip else None
    grid: dict[tuple[int, int], list[dict]] = {}
    for c in cams:
        grid.setdefault((int(c["x"] * 50), int(c["y"] * 50)), []).append(c)       # 0.02도 칸(약 2km)
    reach = max(1, int(radius / 1700) + 1)
    rank = {t: i for i, t in enumerate(types)}
    todo = [e for e in events if e["type"] in rank and e["road"] in ROADS and not (now and e["end"] and e["end"] < now)
            and not (bad and bad.search(e["message"] + " " + e["detail"]))]
    todo.sort(key=lambda e: e["start"], reverse=True)       # 최근 것부터(안정 정렬 — 아래에서 유형 순으로 다시)
    todo.sort(key=lambda e: rank[e["type"]])
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for e in todo:
        gx, gy = int(e["x"] * 50), int(e["y"] * 50)
        near = []
        for dx in range(-reach, reach + 1):
            for dy in range(-reach, reach + 1):
                for c in grid.get((gx + dx, gy + dy), ()):
                    if c["road"] != ROADS[e["road"]]:
                        continue
                    d = meters(e["x"], e["y"], c["x"], c["y"])
                    if d <= radius:
                        near.append((d, c))
        near.sort(key=lambda t: (t[0], t[1]["id"]))
        took = 0
        for d, c in near:
            if took >= per_event or len(out) >= limit:
                break
            key = (c["road"], c["id"])
            if key in seen:
                continue
            seen.add(key)
            took += 1
            out.append({"cam": c, "event": e, "dist": round(d)})
        if len(out) >= limit:
            break
    return out


def clip_name(cam: str, start: datetime) -> str:
    """`its-<번호>_<YYYYMMDD_HHMMSS>.mp4` — 영상 캡처·auto_capture 가 이름에서 시각을 읽는 꼴."""
    return f"its-{cam}_{start:%Y%m%d_%H%M%S}.mp4"


# ---------------------------------------------------------------- 받기
def load_list(road: str, cache: Path, refresh: bool) -> tuple[dict, bool]:
    """목록(cctvType 5). 캐시가 LIST_TTL 안이면 다시 쓴다 → (응답, API 를 불렀나)."""
    p = cache / f"list_{road}_5.json"
    if not refresh and p.is_file() and time.time() - p.stat().st_mtime < LIST_TTL:
        return json.loads(p.read_bytes().decode("utf-8", "replace")), False
    body = call("cctvInfo", type=road, cctvType="5")
    cache.mkdir(parents=True, exist_ok=True)
    p.write_bytes(body)
    return json.loads(body.decode("utf-8", "replace")), True


def duration_of(path: Path) -> float | None:
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(path))
    try:
        fps, n = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        return n / fps if fps and fps > 0 and n > 0 else None
    finally:
        cap.release()


def download(url: str, dst: Path) -> int:
    """mp4 하나를 받는다(.part 로 쓰고 이름을 바꾼다). mp4 가 아니면 지우고 예외.
    주소는 http(s) 만 — urllib 은 `file:`·`ftp:` 도 여는데, 주소는 API 응답에서 온 글자다(2026-11-25)."""
    if urllib.parse.urlsplit(str(url or "")).scheme not in ("http", "https"):
        raise ValueError("http(s) 주소가 아닙니다")
    tmp = dst.with_suffix(".part")
    n = 0
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
            while True:
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                n += len(chunk)
                if n > MAX_CLIP:
                    raise ValueError(f"{MAX_CLIP >> 20} MB 를 넘습니다")
                f.write(chunk)
        with open(tmp, "rb") as f:
            if f.read(12)[4:8] != b"ftyp":
                raise ValueError("mp4 가 아닙니다")
        tmp.replace(dst)
        return n
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true", help="영상을 받는다(없으면 미리 보기)")
    ap.add_argument("--radius", type=float, default=500.0, help="돌발 지점에서 카메라까지(m, 기본 500)")
    ap.add_argument("--per-event", type=int, default=1, help="돌발 하나에 카메라 몇 대까지(기본 1)")
    ap.add_argument("--max", type=int, default=20, help="한 번에 받는 영상 수(기본 20)")
    ap.add_argument("--types", default=",".join(EVENT_ORDER), help="돌발 유형, 받는 순서대로(기본 교통사고,기타돌발,공사)")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "visible" / "its" / "clips",
                    help="받는 폴더(그 아래 날짜 폴더) — git 밖이어야 한다")
    ap.add_argument("--cache", type=Path, default=ROOT / "data" / "visible" / "its" / "cache", help="목록을 두는 폴더")
    ap.add_argument("--refresh", action="store_true", help="목록을 다시 받는다(API 2회)")
    ap.add_argument("--events-file", type=Path, help="받아 둔 돌발 응답(json) — 주면 돌발 API 를 부르지 않는다")
    ap.add_argument("--skip", default=SKIP, help=f"글·세부 유형이 이 정규식에 걸리는 돌발은 뺀다(기본 '{SKIP}', 빈 글이면 안 뺀다)")
    ap.add_argument("--pause", type=float, default=1.0, help="영상 사이 쉬는 시간(초)")
    a = ap.parse_args()

    calls = 0
    if a.events_file:
        ev_obj = json.loads(a.events_file.read_bytes().decode("utf-8", "replace"))
    else:
        body = call("eventInfo", type="all", eventType="all")
        calls += 1
        a.cache.mkdir(parents=True, exist_ok=True)
        (a.cache / "events_last.json").write_bytes(body)
        ev_obj = json.loads(body.decode("utf-8", "replace"))
    events = parse_events(ev_obj)
    cams: list[dict] = []
    for road in ("ex", "its"):
        obj, called = load_list(road, a.cache, a.refresh)
        calls += called
        cams += parse_cams(obj, road)
    now = datetime.now()
    types = tuple(t.strip() for t in a.types.split(",") if t.strip())
    picks = choose(events, cams, radius=a.radius, per_event=a.per_event, limit=a.max, types=types,
                   now=f"{now:%Y%m%d%H%M%S}", skip=a.skip)
    print(f"돌발 {len(events)}건 · 카메라 {len(cams)}대 · API {calls}회 → 고른 영상 {len(picks)}개"
          f" (반경 {a.radius:.0f}m, 유형 {'·'.join(types)})")
    for i, p in enumerate(picks, 1):
        e, c = p["event"], p["cam"]
        print(f"{i:3d}. {e['road']} {e['type']:<5} {p['dist']:4d}m  its-{c['id']:<6} {e['message'][:40] or e['detail']}")
    if not a.apply:
        print("미리 보기입니다 — 받으려면 --apply.")
        return 0
    if not picks:
        return 0
    day = a.out / f"{now:%Y%m%d}"
    day.mkdir(parents=True, exist_ok=True)
    got = failed = total = 0
    with open(day / "manifest.jsonl", "a", encoding="utf-8") as man:
        for i, p in enumerate(picks):
            if i:
                time.sleep(a.pause)
            c, e = p["cam"], p["event"]
            t0 = datetime.now()
            tmp = day / f"its-{c['id']}_{t0:%Y%m%d_%H%M%S}_dl.mp4"
            try:
                size = download(c["url"], tmp)
            except Exception as ex:
                failed += 1
                print(f"  실패 its-{c['id']}: {type(ex).__name__}: {str(ex)[:80]}")
                continue
            dur = duration_of(tmp)
            start = t0 - timedelta(seconds=dur or 30.0)
            dst = day / clip_name(c["id"], start)
            if dst.exists():
                tmp.unlink()
                continue
            tmp.replace(dst)
            got += 1
            total += size
            man.write(json.dumps({"file": dst.name, "camera": f"its-{c['id']}", "its_id": c["id"], "road": c["road"],
                                  "place": c["name"], "x": c["x"], "y": c["y"], "fetched_at": f"{t0:%Y-%m-%d %H:%M:%S}",
                                  "duration": round(dur, 1) if dur else None, "bytes": size, "dist_m": p["dist"],
                                  "event": e}, ensure_ascii=False) + "\n")
            man.flush()
    print(f"받음 {got}개 · {total / 1e6:.1f} MB · 실패 {failed} → {day}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
