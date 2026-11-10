"""새 수집 이미지의 출처(카메라·촬영 시각)를 읽는다 (2026-11-03, docs/provenance.md).

**카메라는 폴더로 정한다.** `unlabeled/<카메라키>/…` 의 첫 폴더 이름이 키이고, 그 카메라의 속성(장소 이름·
설치 형태·시간대·권리자)은 `cameras.yaml` 한 곳에만 둔다 — DB 에는 키만 남긴다(두 곳에 두면 어긋난다).
등록 파일은 실제 설치 장소를 담으므로 `data/` 아래(git 에 담지 않는 곳)에 둔다. 저장소는 public 이다.

**등록 안 된 키가 하나라도 있으면 scan 을 통째로 멈춘다.** 폴더 이름 오타가 새 카메라로 조용히 들어가면, 같은
카메라가 두 묶음이 되어 train 과 test 로 갈릴 수 있다(분할 단위가 카메라다).

**촬영 시각은 파일 이름 → EXIF 순으로 읽고, 못 찾으면 비워 둔다.** 파일 수정 시각은 복사·다운로드 때 바뀌므로
쓰지 않는다. 시간대가 없는 값에는 카메라의 `tz` 를 붙인다. 이 PC 의 Python 에는 IANA 시간대 자료(tzdata)가
없어서 `America/New_York` 같은 이름은 풀리지 않는다 — `+09:00` 꼴로 적는다(한국은 서머타임이 없어 고정 오프셋으로 충분).
뜻이 하나뿐인 약어(`KST`·`JST`·`UTC`·`GMT`·`Z`)와 `UTC+9`·`GMT-5`·`+9` 꼴, 서머타임 없는 `Asia/Seoul`·`Asia/Tokyo` 는
`normalize_tz` 가 `+09:00` 꼴로 바꿔 받는다(2026-11-10 사용자 요청). 뜻이 여럿인 약어(`CST`·`IST` 등)는 오프셋으로 적게 한다.
"""

from __future__ import annotations

import os
import re
import secrets
import shutil
import threading
import time
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
_OFFSET = re.compile(r"^([+-])(\d{1,2})(?::?(\d{2}))?$")      # +09:00 · +0900 · +09 · +9 · +9:30
_PREFIXED = re.compile(r"^(?:UTC|GMT)\s*([+-]\s*[\d:]+)$", re.IGNORECASE)   # UTC+9 · GMT-05:00 (사람이 쓰는 뜻 그대로)
# 뜻이 하나뿐인 약어·서머타임 없는 이름 → 오프셋. 'Etc/GMT+9' 는 부호가 거꾸로라(UTC-9) 받지 않는다(tzdata 도 없다).
_TZ_ALIASES = {"KST": "+09:00", "JST": "+09:00", "UTC": "+00:00", "GMT": "+00:00", "Z": "+00:00",
               "ASIA/SEOUL": "+09:00", "ASIA/TOKYO": "+09:00", "ETC/UTC": "+00:00"}
# 하나로 정할 수 없는 약어 — (이유, 적을 오프셋 예). 여러 나라가 같은 약어를 쓰거나, 여름에 다른 약어(EDT·PDT)로 바뀌는데
# 사람들이 한 해 내내 그 약어로 부르는 것
_TZ_AMBIGUOUS = {
    "CST": ("뜻이 여럿입니다", "+08:00 중국 · -06:00 미국 중부"),
    "IST": ("뜻이 여럿입니다", "+05:30 인도 · +02:00 이스라엘 · +01:00 아일랜드 여름"),
    "BST": ("뜻이 여럿입니다", "+01:00 영국 여름 · +06:00 방글라데시"),
    "AST": ("뜻이 여럿입니다", "+03:00 아라비아 · -04:00 대서양"),
    "EST": ("여름에는 EDT 로 바뀌어 하나로 정할 수 없습니다", "-05:00 겨울 · -04:00 여름"),
    "PST": ("여름에는 PDT 로 바뀌어 하나로 정할 수 없습니다", "-08:00 겨울 · -07:00 여름"),
}
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


def normalize_tz(text: str) -> str | None:
    """시간대 글을 '+09:00' 꼴로 (2026-11-10). 약어(KST·UTC·GMT…)·'UTC+9'·'+9' 꼴을 받는다.
    고정 오프셋으로 바꿀 수 없는 글(tzdata 의 'America/New_York' 같은 이름)은 None — parse_tz 가 이름으로 다시 본다.
    뜻이 여럿인 약어(CST·IST…)나 범위 밖(±14시간 넘게·60분 이상)은 ValueError."""
    s = str(text or "").strip()
    u = s.upper()
    if u in _TZ_ALIASES:
        return _TZ_ALIASES[u]
    if u in _TZ_AMBIGUOUS:
        why, examples = _TZ_AMBIGUOUS[u]
        raise ValueError(f"시간대 약어 '{s}' 는 {why} — 그 영상에 맞는 오프셋으로 적을 것({examples})")
    m = _PREFIXED.match(s)
    if m:
        s = m.group(1).replace(" ", "")
    m = _OFFSET.match(s)
    if not m:
        return None
    h, mi = int(m.group(2)), int(m.group(3) or 0)
    if h > 14 or mi >= 60 or (h == 14 and mi):
        raise ValueError(f"시간대 '{text}' — UTC 와의 차이는 -14:00 ~ +14:00 사이여야 합니다")
    return f"{m.group(1)}{h:02d}:{mi:02d}"


def parse_tz(text: str) -> tzinfo:
    """시간대 글 → tzinfo. '+09:00' 꼴·약어(KST·UTC·GMT…)·'UTC+9' 는 고정 오프셋으로(normalize_tz), 그 밖의 이름은
    tzdata 가 있을 때만 받는다(이 PC 에는 없다)."""
    off = normalize_tz(text)
    if off is not None:
        sign = 1 if off[0] == "+" else -1
        return timezone(sign * timedelta(hours=int(off[1:3]), minutes=int(off[4:6])))
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(str(text))
    except Exception as exc:
        raise ValueError(f"시간대 '{text}' 를 풀 수 없습니다 — '+09:00' 꼴이나 KST·UTC·GMT·'UTC+9' 로 적을 것") from exc


def tz_text(tz: tzinfo) -> str:
    """화면에 보일 시간대 — 고정 오프셋은 '+09:00', 이름이면 그 이름."""
    off = tz.utcoffset(None) if isinstance(tz, timezone) else None
    if off is None:
        return str(getattr(tz, "key", tz))
    total = int(off.total_seconds() // 60)
    sign = "+" if total >= 0 else "-"
    return f"{sign}{abs(total) // 60:02d}:{abs(total) % 60:02d}"


def load_cameras(path: Path) -> dict[str, Camera]:
    """cameras.yaml 을 읽어 {키: Camera}. 잘못된 항목이 있으면 전부 모아 ValueError.

    파일이 YAML 이 아니거나 모양이 틀린 것(맨 위·항목이 사전이 아님)도 ValueError 다 (2026-11-05) — 예전에는
    yaml.YAMLError·AttributeError 가 그대로 나가, ValueError 만 받는 쪽(검수 서버의 카메라 목록, 분할)을 뚫고 500 이 됐다.
    """
    if not path.exists():
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: YAML 로 읽지 못했습니다 — {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: 맨 위는 'cameras:' 로 시작하는 사전이어야 합니다")
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
        if not isinstance(attrs, dict):
            errors.append(f"{key}: 속성은 'kind: cctv' 처럼 키: 값 사전으로 적을 것 (지금 {type(attrs).__name__})")
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


class CameraExists(ValueError):
    """이미 등록된 키."""


class CameraMissing(ValueError):
    """등록 파일에 없는 키."""


# 등록 파일에 적는 순서 (docs/provenance.md '등록 파일' 의 예와 같다). split 은 비어도 적는다(지금 파일과 같은 꼴)
_FIELD_ORDER = ("kind", "place", "mount", "view", "resolution", "tz", "rights", "split", "note")
_EDIT_LOCK = threading.Lock()          # 등록 파일을 고치는 add_camera·remove_cameras 는 한 번에 하나
_TOP_CAMERAS = re.compile(r"^cameras\s*:\s*(#.*)?$")


def add_camera(path: Path, key: str, attrs: dict, *, backup_dir: Path | None = None,
               comment: str | None = None) -> Camera:
    """cameras.yaml 의 `cameras:` 묶음 끝에 카메라 하나를 더한다 (2026-11-10, 검수 UI '영상 캡처' 의 새 카메라).

    사람이 손으로 쓰는 파일이라(묶음마다 설명 주석·순서·`crop` 같은 다른 속성) 통째로 다시 쓰지 않고 **글자로 덧붙인다**.
    쓰기 전에 붙인 결과를 다시 읽어 '기존 카메라는 속성까지 그대로이고 새 키 하나만 늘었다' 가 아니면 아무것도 쓰지 않는다.
    등록 파일이 이미 깨져 있으면(load_cameras 의 ValueError) 손대지 않는다. 쓰기 전 파일은 backup_dir 에
    `cameras_<시각>.yaml` 로 남기고, 바꿔 끼우기는 한 번에 한다(scan 이 반쯤 쓴 파일을 읽지 않게).
    값 규칙은 load_cameras 와 같다(키 꼴·kind·mount·split·place·tz). 수정·삭제는 하지 않는다 — 파일을 손으로 고친다.
    """
    key = str(key or "").strip()
    if not _KEY.match(key):
        raise ValueError(f"키 '{key}' — 영문 소문자·숫자·'-' 만, 첫 글자는 영문·숫자(폴더 이름이 된다)")
    fields: dict = {}
    for name in _FIELD_ORDER:
        v = attrs.get(name)
        if v is not None and not isinstance(v, str):
            raise ValueError(f"{name}: 글자로 적을 것")
        v = (v or "").strip()
        if any(ord(ch) < 32 for ch in v):
            raise ValueError(f"{name}: 한 줄로 적을 것(줄바꿈·탭 없이)")
        fields[name] = v or None
    if fields["kind"] not in KINDS:
        raise ValueError(f"kind 는 {'|'.join(KINDS)} 중 하나 (지금 {fields['kind']!r})")
    fields["mount"] = fields["mount"] or "unknown"
    if fields["mount"] not in MOUNTS:
        raise ValueError(f"mount 는 {'|'.join(MOUNTS)} 중 하나 (지금 {fields['mount']!r})")
    if fields["split"] is not None and fields["split"] not in SPLITS:
        raise ValueError(f"split 은 비우거나 {'|'.join(SPLITS)} (지금 {fields['split']!r})")
    if not fields["place"]:
        raise ValueError("place(장소 이름)를 적을 것")
    if fields["tz"]:
        fields["tz"] = normalize_tz(fields["tz"]) or fields["tz"]   # 'KST'·'UTC+9' → '+09:00' 꼴로 적는다
    parse_tz(fields["tz"] or DEFAULT_TZ)               # 못 풀면 ValueError
    if fields["tz"] == DEFAULT_TZ:
        fields["tz"] = None                            # 기본값은 적지 않는다(지금 파일의 카메라들처럼)
    written = {k: v for k, v in fields.items() if v is not None or k == "split"}

    with _EDIT_LOCK:
        # read_text 는 CRLF 를 LF 로 바꿔 읽는다 — 바이트로 읽어야 다시 쓸 때 기존 줄끝이 그대로 남는다
        old = path.read_bytes().decode("utf-8") if path.exists() else ""
        before = load_cameras(path)                    # 깨진 파일이면 ValueError — 손대지 않는다
        if key in before:
            raise CameraExists(f"카메라 '{key}' 는 이미 등록돼 있습니다")
        raw_before = (yaml.safe_load(old) or {}).get("cameras") or {}
        new = _append_camera_text(old, key, written, comment)
        tmp = path.with_name(f"{path.name}.{secrets.token_hex(3)}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(new, encoding="utf-8", newline="")
            try:
                after = load_cameras(tmp)
                raw_after = (yaml.safe_load(new) or {}).get("cameras") or {}
            except (ValueError, yaml.YAMLError) as exc:
                raise ValueError(f"덧붙인 결과를 읽지 못해 아무것도 쓰지 않았습니다 — {exc}") from exc
            same_old = (set(after) == set(before) | {key}
                        and all(after[k] == before[k] for k in before)
                        and all(raw_after.get(k) == v for k, v in raw_before.items()))
            if not same_old or raw_after.get(key) != written:
                raise ValueError("덧붙인 결과를 다시 읽으니 기존 카메라나 새 카메라가 기대와 달라 아무것도 쓰지 않았습니다 — "
                                 f"{path.name} 의 모양(들여쓰기·'cameras:' 줄)을 손으로 확인할 것")
            _backup_and_swap(path, tmp, backup_dir)
        finally:
            tmp.unlink(missing_ok=True)
        return after[key]


def remove_cameras(path: Path, keys: list[str], *, backup_dir: Path | None = None) -> list[str]:
    """cameras.yaml 에서 카메라 등록을 뺀다 (2026-11-10, 검수 UI '영상 캡처' 의 카메라 지우기).

    add_camera 와 같은 방식이다 — 다시 쓰지 않고 **그 카메라의 줄만** 뺀다: 키 줄과 그 아래 더 깊이 들여쓴 줄(속성·안쪽 주석),
    그리고 키 바로 위에 같은 들여쓰기로 붙은 주석(add_camera 의 '… 에서 등록' 같은). 맨 왼쪽의 묶음 설명 주석과 다음 키 앞의
    주석·빈 줄은 남긴다. 뺀 결과를 다시 읽어 '그 키들만 없어지고 나머지는 속성까지 그대로' 가 아니면 아무것도 쓰지 않고,
    쓰기 전 파일은 backup_dir 에 남긴다. 하나라도 없는 키면 CameraMissing(아무것도 안 뺀다).
    그 카메라의 이미지가 DB·폴더에 남았는지는 부르는 쪽이 본다(이 모듈은 DB 를 모른다 — server 의 /api/cameras/remove).
    """
    keys = list(dict.fromkeys(str(k or "").strip() for k in keys))
    if not keys:
        raise ValueError("지울 카메라를 고르지 않았습니다")
    with _EDIT_LOCK:
        if not path.exists():
            raise CameraMissing(f"등록 파일이 없습니다: {path}")
        old = path.read_bytes().decode("utf-8")        # 줄끝·BOM 을 그대로 두려고 바이트로 (add_camera 와 같다)
        before = load_cameras(path)                    # 깨진 파일이면 ValueError — 손대지 않는다
        missing = [k for k in keys if k not in before]
        if missing:
            raise CameraMissing(f"등록돼 있지 않은 카메라: {', '.join(missing)}")
        raw_before = (yaml.safe_load(old) or {}).get("cameras") or {}
        new = _remove_camera_text(old, keys)
        tmp = path.with_name(f"{path.name}.{secrets.token_hex(3)}.tmp")
        try:
            tmp.write_text(new, encoding="utf-8", newline="")
            try:
                after = load_cameras(tmp)
                raw_after = (yaml.safe_load(new) or {}).get("cameras") or {}
            except (ValueError, yaml.YAMLError) as exc:
                raise ValueError(f"뺀 결과를 읽지 못해 아무것도 쓰지 않았습니다 — {exc}") from exc
            gone = set(keys)
            ok = (set(after) == set(before) - gone
                  and all(after[k] == before[k] for k in after)
                  and set(raw_after) == set(raw_before) - gone
                  and all(raw_after[k] == raw_before[k] for k in raw_after))
            if not ok:
                raise ValueError("뺀 결과를 다시 읽으니 다른 카메라가 달라져 아무것도 쓰지 않았습니다 — "
                                 f"{path.name} 의 모양(들여쓰기)을 손으로 확인할 것")
            _backup_and_swap(path, tmp, backup_dir)
        finally:
            tmp.unlink(missing_ok=True)
    return keys


def _backup_and_swap(path: Path, tmp: Path, backup_dir: Path | None) -> None:
    """쓰기 전 파일을 backup_dir/cameras_<시각>.yaml 로 남기고 tmp 로 한 번에 바꿔 끼운다."""
    if backup_dir is not None and path.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)
        dst = backup_dir / f"cameras_{datetime.now():%Y%m%d_%H%M%S}.yaml"
        while dst.exists():
            dst = dst.with_name(f"{dst.stem}_{secrets.token_hex(2)}.yaml")
        shutil.copy2(path, dst)
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:                        # Windows: 다른 쪽이 잠깐 열어 둔 때
            if attempt == 9:
                raise
            time.sleep(0.05)


# 'cameras:' 바로 아래 한 단계의 키 줄 — `hw-01:` · `'yes':` · `"017":` · `a-1: {kind: cctv, …}`
_KEY_LINE = re.compile(r"""^[ ]*(?:'(?P<sq>[^']*)'|"(?P<dq>[^"]*)"|(?P<pl>[^\s:#'"][^:#]*?))[ ]*:(?:[ ].*)?$""")


def _indent(s: str) -> int:
    return len(s) - len(s.lstrip(" "))


class _Layout:
    """등록 파일을 줄 단위로 본 것 — 줄(줄끝 포함)·판별용 글자·주석/빈 줄 여부·'cameras:' 줄·묶음 끝·카메라 키 들여쓰기."""

    def __init__(self, old: str, verb: str):
        self.nl = "\r\n" if "\r\n" in old else "\n"
        self.lines = old.splitlines(keepends=True)
        self.plain = [ln.rstrip("\r\n") for ln in self.lines]
        if self.plain:
            self.plain[0] = self.plain[0].lstrip("﻿")
        self.is_note = [not s.strip() or s.lstrip().startswith("#") for s in self.plain]
        self.head = next((i for i, s in enumerate(self.plain) if _TOP_CAMERAS.match(s)), None)
        if self.head is None:
            raise ValueError(f"등록 파일에 'cameras:' 줄이 없어 {verb} 수 없습니다 — 손으로 고칠 것")
        self.child = None
        self.end = len(self.lines)
        for j in range(self.head + 1, len(self.lines)):
            if self.is_note[j]:
                continue
            if _indent(self.plain[j]) == 0:
                self.end = j
                break
            if self.child is None:
                self.child = _indent(self.plain[j])

    def entries(self):
        """(키 줄 번호, 키, 그 아래 더 깊은 줄(안쪽 주석 포함)의 끝) — cameras 묶음 안의 카메라마다."""
        for i in range(self.head + 1, self.end):
            if self.is_note[i] or _indent(self.plain[i]) != self.child:
                continue
            m = _KEY_LINE.match(self.plain[i])
            if not m:
                continue
            k = m.group("sq") if m.group("sq") is not None else m.group("dq") if m.group("dq") is not None \
                else (m.group("pl") or "").strip()
            j = i + 1
            while j < self.end and (self.is_note[j] or _indent(self.plain[j]) > self.child):
                j += 1
            yield i, k, j


def _remove_camera_text(old: str, keys: list[str]) -> str:
    lay = _Layout(old, "뺄")
    plain, is_note, head, child = lay.plain, lay.is_note, lay.head, lay.child
    found: dict[str, tuple[int, int]] = {}
    for i, k, j in lay.entries():
        if k not in keys:
            continue
        b = j                                          # 끝의 빈 줄·얕은 주석은 다음 키(또는 묶음 설명)의 것
        while b > i + 1 and is_note[b - 1] and (not plain[b - 1].strip() or _indent(plain[b - 1]) <= child):
            b -= 1
        a = i                                          # 바로 위에 같은 들여쓰기로 붙은 주석은 이 키의 것
        while a > head + 1 and is_note[a - 1] and plain[a - 1].strip() and _indent(plain[a - 1]) == child:
            a -= 1
        found[k] = (a, b)
    if set(found) != set(keys):
        raise ValueError(f"등록 파일에서 줄을 찾지 못한 카메라: {', '.join(sorted(set(keys) - set(found)))} — 손으로 고칠 것")
    lines = lay.lines
    for a, b in sorted(found.values(), reverse=True):
        del lines[a:b]
    return "".join(lines)


def camera_fields(path: Path) -> dict[str, dict]:
    """카메라마다 화면에서 고칠 수 있는 칸(_FIELD_ORDER)의 지금 값 — 글자(없으면 None). '고치기' 칸을 채운다."""
    if not path.exists():
        return {}
    try:
        raw = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("cameras") or {}
    except (yaml.YAMLError, AttributeError):
        return {}
    return {str(k): {f: (None if a.get(f) is None else str(a.get(f))) for f in _FIELD_ORDER}
            for k, a in raw.items() if isinstance(a, dict)}


def _same_value(field: str, old, new: str | None) -> bool:
    """고치기 칸의 값이 지금 값과 같은 뜻인가 — 기본값(mount unknown·tz +09:00)·빈 값·약어(KST)를 같게 본다."""
    if field == "mount":
        return (str(old) if old is not None else "unknown") == (new or "unknown")
    if field == "tz":
        try:
            return tz_text(parse_tz(str(old) if old is not None else DEFAULT_TZ)) == tz_text(parse_tz(new or DEFAULT_TZ))
        except ValueError:
            return False
    return (str(old).strip() if old is not None and str(old).strip() else None) == new


def edit_camera(path: Path, key: str, values: dict, *, new_key: str | None = None,
                backup_dir: Path | None = None) -> tuple[Camera, list[str]]:
    """등록된 카메라의 칸을 고친다 (2026-11-10, 검수 UI '영상 캡처' 의 카메라 고치기).

    values 에 든 칸(_FIELD_ORDER)만 보고, 지금 값과 뜻이 같은 칸은 건드리지 않는다. 다른 속성(crop 등)·주석·순서는 그대로 —
    바뀐 칸의 줄만 글자로 고친다: 한 줄짜리 값은 그 줄을 바꾸고(줄 끝 주석은 남긴다), 없던 칸은 _FIELD_ORDER 의 자리에 넣고,
    비운 칸은 그 줄을 뺀다(split 은 'split:' 으로 비워 둔다). new_key 는 키 줄의 이름만 바꾼다 — 폴더·파일 이름·DB 는 모르니
    그 카메라를 쓰는 이미지가 없는지는 부르는 쪽(server)이 본다. 값 규칙은 add_camera 와 같다(시간대는 normalize_tz 꼴로).
    다시 읽어 이 카메라만 기대대로 바뀌었을 때만 쓰고(백업은 add 와 같다), 바뀐 것이 없으면 쓰지 않는다.
    돌려주는 것: (고친 뒤의 Camera, 바뀐 칸 이름들 — 키를 바꿨으면 'key' 포함).
    """
    key = str(key or "").strip()
    new_key = str(new_key).strip() if new_key else None
    if new_key == key:
        new_key = None
    if new_key is not None and not _KEY.match(new_key):
        raise ValueError(f"키 '{new_key}' — 영문 소문자·숫자·'-' 만, 첫 글자는 영문·숫자(폴더 이름이 된다)")
    with _EDIT_LOCK:
        if not path.exists():
            raise CameraMissing(f"등록 파일이 없습니다: {path}")
        old = path.read_bytes().decode("utf-8")        # 줄끝·BOM 을 그대로 두려고 바이트로
        before = load_cameras(path)                    # 깨진 파일이면 ValueError — 손대지 않는다
        if key not in before:
            raise CameraMissing(f"등록돼 있지 않은 카메라: {key}")
        if new_key is not None and new_key in before:
            raise CameraExists(f"카메라 '{new_key}' 는 이미 등록돼 있습니다")
        raw_before = (yaml.safe_load(old) or {}).get("cameras") or {}
        cur = dict(raw_before.get(key) or {})
        edits: dict[str, str | None] = {}
        for f in _FIELD_ORDER:
            if f not in values:
                continue
            v = values[f]
            if v is not None and not isinstance(v, str):
                raise ValueError(f"{f}: 글자로 적을 것")
            v = (v or "").strip()
            if any(ord(ch) < 32 for ch in v):
                raise ValueError(f"{f}: 한 줄로 적을 것(줄바꿈·탭 없이)")
            v = v or None
            if f == "tz" and v:
                v = normalize_tz(v) or v
                parse_tz(v)                            # 못 풀면 ValueError
            if _same_value(f, cur.get(f), v):
                continue
            if f == "tz" and v == DEFAULT_TZ:
                v = None                               # 기본값은 적지 않는다
            edits[f] = v
        merged = {**cur, **{f: v for f, v in edits.items() if v is not None or f == "split"}}
        for f, v in edits.items():
            if v is None and f != "split":
                merged.pop(f, None)
        if merged.get("kind") not in KINDS:
            raise ValueError(f"kind 는 {'|'.join(KINDS)} 중 하나 (지금 {merged.get('kind')!r})")
        if (merged.get("mount") or "unknown") not in MOUNTS:
            raise ValueError(f"mount 는 {'|'.join(MOUNTS)} 중 하나 (지금 {merged.get('mount')!r})")
        if merged.get("split") not in (None, *SPLITS):
            raise ValueError(f"split 은 비우거나 {'|'.join(SPLITS)} (지금 {merged.get('split')!r})")
        if not str(merged.get("place") or "").strip():
            raise ValueError("place(장소 이름)를 적을 것")
        if not edits and new_key is None:
            return before[key], []
        new = _edit_camera_text(old, key, edits, new_key)
        final = new_key or key
        tmp = path.with_name(f"{path.name}.{secrets.token_hex(3)}.tmp")
        try:
            tmp.write_text(new, encoding="utf-8", newline="")
            try:
                after = load_cameras(tmp)
                raw_after = (yaml.safe_load(new) or {}).get("cameras") or {}
            except (ValueError, yaml.YAMLError) as exc:
                raise ValueError(f"고친 결과를 읽지 못해 아무것도 쓰지 않았습니다 — {exc}") from exc
            others = set(before) - {key}
            ok = (set(after) == others | {final}
                  and all(after[k] == before[k] for k in others)
                  and set(raw_after) == (set(raw_before) - {key}) | {final}
                  and all(raw_after[k] == raw_before[k] for k in raw_before if k != key)
                  and raw_after.get(final) == merged)
            if not ok:
                raise ValueError("고친 결과를 다시 읽으니 기대와 달라 아무것도 쓰지 않았습니다 — 이 카메라의 줄 모양(여러 줄 값·"
                                 f"들여쓰기)을 {path.name} 에서 손으로 확인할 것")
            _backup_and_swap(path, tmp, backup_dir)
        finally:
            tmp.unlink(missing_ok=True)
        return after[final], list(edits) + (["key"] if new_key else [])


def _line_comment(text: str) -> str:
    """'    place: x  # 메모' 의 줄 끝 주석 — 떼어도 값이 같게 읽히는 첫 ' #' 부터. 없으면 ''(따옴표 안의 # 은 주석이 아니다)."""
    s = text.strip()
    try:
        whole = yaml.safe_load(s)
    except yaml.YAMLError:
        return ""
    for p in range(1, len(s)):
        if s[p] == "#" and s[p - 1] in " \t":
            try:
                if yaml.safe_load(s[:p]) == whole:
                    return s[p:]
            except yaml.YAMLError:
                continue
    return ""


def _edit_camera_text(old: str, key: str, edits: dict, new_key: str | None) -> str:
    lay = _Layout(old, "고칠")
    plain, is_note, nl = lay.plain, lay.is_note, lay.nl
    hit = next(((i, j) for i, k, j in lay.entries() if k == key), None)
    if hit is None:
        raise ValueError(f"등록 파일에서 '{key}' 의 줄을 찾지 못했습니다 — 손으로 고칠 것")
    i, j = hit
    # 이 카메라의 줄을 조각으로: 속성(그 줄 + 더 깊은 이어진 줄) · 그 밖(주석·빈 줄)
    attr_ind = next((_indent(plain[k]) for k in range(i + 1, j) if not is_note[k]), (lay.child or 2) * 2)
    chunks: list[list] = []                            # [이름 또는 None, 줄 목록]
    k = i + 1
    while k < j:
        if is_note[k] or _indent(plain[k]) != attr_ind:
            chunks.append([None, [lay.lines[k]]])
            k += 1
            continue
        m = re.match(r"""^\s*([^\s:#'"][^:#]*?)\s*:(?:\s|$)""", plain[k])
        e = k + 1
        while e < j and not is_note[e] and _indent(plain[e]) > attr_ind:
            e += 1
        chunks.append([m.group(1) if m else None, lay.lines[k:e]])
        k = e

    def line(name: str, value, comment: str = "") -> str:
        text = f"{name}:" if value is None else yaml.safe_dump({name: value}, allow_unicode=True, width=10**9).rstrip("\n")
        return " " * attr_ind + text + (f"  {comment}" if comment else "") + nl

    for name, value in edits.items():
        pos = next((n for n, c in enumerate(chunks) if c[0] == name), None)
        if pos is not None:
            if value is None and name != "split":
                del chunks[pos]                        # 비운 칸은 줄을 뺀다
            else:
                first = chunks[pos][1][0].rstrip("\r\n")
                chunks[pos][1] = [line(name, value, _line_comment(first) if len(chunks[pos][1]) == 1 else "")]
            continue
        if value is None and name != "split":
            continue                                   # 없던 칸을 비우기 — 할 일 없음
        later = _FIELD_ORDER[_FIELD_ORDER.index(name) + 1:]
        at = next((n for n, c in enumerate(chunks) if c[0] in later), None)
        if at is None:                                 # 뒤 칸이 없으면 마지막 속성 바로 뒤(끝의 주석보다 앞)
            at = max((n + 1 for n, c in enumerate(chunks) if c[0] is not None), default=0)
        chunks.insert(at, [name, [line(name, value)]])
    head_line = lay.lines[i]
    if new_key is not None:
        m = re.match(r"""^(\s*)('[^']*'|"[^"]*"|[^\s:#'"][^:#]*?)(\s*:.*)$""", plain[i])
        if not m:
            raise ValueError(f"'{key}' 의 키 줄을 읽지 못했습니다 — 손으로 고칠 것")
        text = new_key if yaml.safe_load(new_key) == new_key else f"'{new_key}'"
        head_line = m.group(1) + text + m.group(3) + head_line[len(head_line.rstrip("\r\n")):]
    body = [ln for c in chunks for ln in c[1]]
    lay.lines[i:j] = [head_line] + body
    return "".join(lay.lines)


def _append_camera_text(old: str, key: str, fields: dict, comment: str | None) -> str:
    """`cameras:` 묶음의 끝(다음 맨 위 키가 있으면 그 앞, 그 키에 붙은 주석·빈 줄보다 앞)에 항목 글자를 넣는다.
    들여쓰기는 첫 카메라의 것을 따른다(없으면 2·4칸). 줄끝(CRLF·LF)도 파일을 따른다."""
    nl = "\r\n" if "\r\n" in old else "\n"
    lines = old.splitlines(keepends=True)
    plain = [ln.rstrip("\r\n") for ln in lines]
    if plain:
        plain[0] = plain[0].lstrip("﻿")           # 메모장이 붙인 BOM — 글자는 그대로 두고 판별만 뺀다
    is_note = [not s.strip() or s.lstrip().startswith("#") for s in plain]
    head = next((i for i, s in enumerate(plain) if _TOP_CAMERAS.match(s)), None)
    if head is None:
        if not all(is_note):
            raise ValueError("등록 파일에 'cameras:' 줄이 없거나 'cameras: {}' 처럼 한 줄로 적혀 있어 덧붙일 수 없습니다 — "
                             "손으로 고칠 것")
        lines.append(("" if not lines or lines[-1].endswith("\n") else nl) + "cameras:" + nl)
        plain.append("cameras:")
        is_note.append(False)
        head = len(lines) - 1
    child = attr = None
    end = len(lines)
    for j in range(head + 1, len(lines)):
        if is_note[j]:
            continue
        ind = len(plain[j]) - len(plain[j].lstrip(" "))
        if ind == 0:
            end = j                                    # 다음 맨 위 키(또는 문서 끝 표시)
            break
        if child is None:
            child = ind
        elif attr is None and ind > child:
            attr = ind
    child = child or 2
    attr = attr or child * 2
    at = end
    if end < len(lines):
        while at > head + 1 and is_note[at - 1]:
            at -= 1
    elif lines and not lines[-1].endswith("\n"):
        lines[-1] += nl
    key_text = key if yaml.safe_load(key) == key else f"'{key}'"     # 'yes'·'123' 같은 키도 글자로 읽히게
    block = []
    if comment:
        block.append(" " * child + "# " + " ".join(comment.split()))
    block.append(" " * child + key_text + ":")
    for name, v in fields.items():
        text = f"{name}:" if v is None else yaml.safe_dump({name: v}, allow_unicode=True, width=10**9).rstrip("\n")
        block.append(" " * attr + text)
    lines[at:at] = [b + nl for b in block]
    return "".join(lines)


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
