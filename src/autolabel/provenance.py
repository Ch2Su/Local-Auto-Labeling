"""새 수집 이미지의 출처(카메라·촬영 시각)를 읽는다 (2026-11-03, docs/provenance.md).

**카메라는 폴더로 정한다.** `unlabeled/<카메라키>/…` 의 첫 폴더 이름이 키이고, 그 카메라의 속성(장소 이름·
설치 형태·시간대·권리자)은 `cameras.yaml` 한 곳에만 둔다 — DB 에는 키만 남긴다(두 곳에 두면 어긋난다).
등록 파일은 실제 설치 장소를 담으므로 `data/` 아래(git 에 담지 않는 곳)에 둔다. 저장소는 public 이다.

**등록 안 된 키가 하나라도 있으면 scan 을 통째로 멈춘다.** 폴더 이름 오타가 새 카메라로 조용히 들어가면, 같은
카메라가 두 묶음이 되어 train 과 test 로 갈릴 수 있다(분할 단위가 카메라다).

**촬영 시각은 파일 이름 → EXIF 순으로 읽고, 못 찾으면 비워 둔다.** 파일 수정 시각은 복사·다운로드 때 바뀌므로
쓰지 않는다. 시간대가 없는 값에는 카메라의 `tz` 를 붙인다. 이 PC 의 Python 에는 IANA 시간대 자료(tzdata)가
없어서 `Asia/Seoul` 같은 이름은 풀리지 않는다 — `+09:00` 꼴로 적는다(한국은 서머타임이 없어 고정 오프셋으로 충분).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path

import yaml

from .grouping import base_name

KINDS = ("cctv", "phone", "screen", "dataset")
MOUNTS = ("pole", "rooftop", "wall", "indoor", "drone", "handheld", "unknown")
SPLITS = ("train", "val", "test")
DEFAULT_TZ = "+09:00"

_KEY = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_OFFSET = re.compile(r"^([+-])(\d{2}):?(\d{2})$")
# 20260911_132342 · Screenshot_20261103-141500 · pg-cross-01_20261103_140005_120(밀리초, extract_frames 형식)
_COMPACT = re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})[_-](\d{2})(\d{2})(\d{2})(?:[_-](\d{3}))?(?!\d)")
# 스크린샷 2026-06-08 165037 · 2026-06-08 16-50-37 · 2026-06-08_16.50.37
_DASHED = re.compile(r"(?<!\d)(20\d{2})-(\d{2})-(\d{2})[ _T](\d{2})[-.:]?(\d{2})[-.:]?(\d{2})(?!\d)")

_EXIF_IFD = 0x8769
_DATETIME_ORIGINAL = 0x9003
_OFFSET_TIME_ORIGINAL = 0x9011


@dataclass(frozen=True)
class Camera:
    key: str
    kind: str
    place: str
    tz: tzinfo
    mount: str = "unknown"
    split: str | None = None


def parse_tz(text: str) -> tzinfo:
    """'+09:00' 꼴 고정 오프셋. 이름(Asia/Seoul)은 tzdata 가 있을 때만 받는다."""
    m = _OFFSET.match(str(text).strip())
    if m:
        sign = 1 if m.group(1) == "+" else -1
        return timezone(sign * timedelta(hours=int(m.group(2)), minutes=int(m.group(3))))
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(str(text))
    except Exception as exc:
        raise ValueError(f"시간대 '{text}' 를 풀 수 없습니다 — '+09:00' 꼴로 적을 것") from exc


def load_cameras(path: Path) -> dict[str, Camera]:
    """cameras.yaml 을 읽어 {키: Camera}. 잘못된 항목이 있으면 전부 모아 ValueError."""
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    items = raw.get("cameras") or {}
    if not isinstance(items, dict):
        raise ValueError(f"{path}: 'cameras:' 아래는 키: 속성 사전이어야 합니다")
    out: dict[str, Camera] = {}
    errors: list[str] = []
    for key, attrs in items.items():
        key = str(key)
        attrs = attrs or {}
        if not _KEY.match(key):
            errors.append(f"{key}: 키는 영문 소문자·숫자·'-' 만 (폴더 이름이 된다)")
            continue
        kind = attrs.get("kind")
        if kind not in KINDS:
            errors.append(f"{key}: kind 는 {'|'.join(KINDS)} 중 하나 (지금 {kind!r})")
        mount = attrs.get("mount") or "unknown"
        if mount not in MOUNTS:
            errors.append(f"{key}: mount 는 {'|'.join(MOUNTS)} 중 하나 (지금 {mount!r})")
        split = attrs.get("split") or None
        if split is not None and split not in SPLITS:
            errors.append(f"{key}: split 은 비우거나 {'|'.join(SPLITS)} (지금 {split!r})")
        place = str(attrs.get("place") or "").strip()
        if not place:
            errors.append(f"{key}: place(장소 이름)를 적을 것")
        try:
            tz = parse_tz(attrs.get("tz") or DEFAULT_TZ)
        except ValueError as exc:
            errors.append(f"{key}: {exc}")
            continue
        out[key] = Camera(key=key, kind=str(kind), place=place, tz=tz, mount=str(mount), split=split)
    if errors:
        raise ValueError(f"{path} 에 고칠 곳이 있습니다:\n  " + "\n  ".join(errors))
    return out


def is_camera_key(key: str | None) -> bool:
    return bool(key) and bool(_KEY.match(key))


def camera_key(img: Path, unlabeled_root: Path) -> str | None:
    """`unlabeled/<키>/…` 의 키. 폴더 바로 아래 파일이면 None(출처 없음)."""
    parts = img.relative_to(unlabeled_root).parts
    return parts[0] if len(parts) > 1 else None


def _from_name(path: Path) -> datetime | None:
    name = base_name(str(path))
    for pat in (_COMPACT, _DASHED):
        for m in pat.finditer(name):
            y, mo, d, h, mi, s = (int(x) for x in m.groups()[:6])
            ms = int(m.group(7)) if pat is _COMPACT and m.group(7) else 0
            try:
                return datetime(y, mo, d, h, mi, s, ms * 1000)
            except ValueError:
                continue          # 13월처럼 날짜가 아닌 숫자 — 다음 후보
    return None


def _from_exif(path: Path) -> datetime | None:
    from PIL import Image

    try:
        with Image.open(path) as im:
            ifd = im.getexif().get_ifd(_EXIF_IFD)
    except Exception:
        return None
    text = ifd.get(_DATETIME_ORIGINAL)
    if not text:
        return None
    try:
        dt = datetime.strptime(str(text).strip("\x00 ").strip(), "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None
    off = ifd.get(_OFFSET_TIME_ORIGINAL)
    if off:
        try:
            dt = dt.replace(tzinfo=parse_tz(str(off).strip("\x00 ")))
        except ValueError:
            pass
    return dt


def capture_time(path: Path, tz: tzinfo) -> str | None:
    """촬영 시각 ISO 8601(시간대 포함). 파일 이름 → EXIF DateTimeOriginal 순, 못 찾으면 None."""
    dt = _from_name(path) or _from_exif(path)
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    return dt.isoformat(timespec="milliseconds" if dt.microsecond else "seconds")
