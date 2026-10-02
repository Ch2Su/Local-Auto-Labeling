"""같은 장면에서 나온 이미지끼리 묶는다 (train/val/test 누수 방지용).

**왜 필요한가.** 같은 CCTV 영상의 인접 프레임이 train 과 val 로 갈리면, 모델이 외운
배경을 그대로 다시 보는 셈이라 점수가 부풀려진다. 2026-09-08 측정에서 val 1,214장 중
743장(61%)이 그런 경우였고, 누수/청정을 갈라 재니 mAP50-95 가 0.72 대 0.62 였다.

**무엇으로 묶나 (2026-09-16 에 실제 데이터로 정했다).** 셋을 합집합으로 쓴다.

1. **연번 계열** — `000001, 000002 …` / `cctv01_000098` 처럼 끝이 번호인 이름은 앞부분이
   같으면 **계열째** 한 묶음이다. 처음에는 번호가 3 이내로 이어질 때만 묶었는데
   `ala_2_001` 과 `ala_2_016` 이 train·val 로 갈렸고, 열어 보니 같은 카메라의 같은
   장면이었다(2026-09-16). 가장 큰 묶음(고정 카메라 한 대)이 이렇게 생긴다.
   **이름 전체가 숫자면 자릿수와 이미지 크기까지 같아야 한 계열이다** (2026-09-24). 앞부분이
   비어 있으면 출처를 가를 단서가 없어서, 예전에는 숫자 이름이 전부 한 묶음(g000001, 1,433장)
   이었다 — 열어 보니 고정 CCTV 640×640 6자리 795장에 12자리 COCO 식 사진·256px 3자리
   이미지·폰 사진(2992px)이 섞여 있었고, 그 사진들은 묶음째 train 에 갇혔다.
2. **30초 이내 연사** — `스크린샷 2026-06-08 165037`, `20260911_132342` 같은 시각 기반
   이름. 5분으로 묶어 봤더니 **다른 카메라 화면을 찍은 사진들까지 한 덩어리**가 됐다
   (134장짜리 묶음을 열어 보니 파주·판교·건대가 섞여 있었다). 그래서 30초로 좁혔다.
3. **dHash 거리 ≤2** — 이름이 달라도 화면이 사실상 같은 것. 8×8 dHash 로 6,466장을
   전수 비교하면 23%가 여기 걸린다. 이름 규칙이 없는 데이터(로보플로 재내보내기 등)를
   잡아 주는 안전망이다.

**한계.** 같은 카메라를 다른 날·다른 조명에서 찍은 것은 dHash 거리가 벌어져 따로 묶인다.
완전히 못 막는다는 뜻이고, 남은 정도는 `split.leak_report()` 로 재서 본다.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

BURST_SEC = 30      # 시각 기반 이름이 이 이내면 같은 연사로 본다
DHASH_MAX = 2        # 8x8 dHash(64비트) 해밍 거리

# 로보플로가 붙이는 꼬리표: `_jpg.rf.<해시>` / `.rf.<해시>`
_RF = re.compile(r"[._](?:jpg|jpeg|png)?\.rf\.[0-9a-z]+", re.I)
_SHOT = re.compile(r"^스크린샷 (\d{4}-\d{2}-\d{2}) (\d{6})$")
_PHONE = re.compile(r"^(\d{8})_(\d{6})$")
# 끝이 번호로 끝나는 이름은 한 출처의 연번으로 본다: `000123`, `cctv01_000098`,
# `ala_2_016`, `T100`. 번호는 2자리 이상만 (한 자리는 우연이 잦다).
_SERIAL = re.compile(r"^(.*?)_?(\d{2,})$")


class DSU:
    """묶음 합치기 (union-find)."""

    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def base_name(path: str) -> str:
    """로보플로 꼬리표를 떼어낸 원래 이름."""
    return Path(_RF.sub("", Path(path).name)).stem


def name_series(path: str, size: tuple[int, int] | None = None) -> tuple[str, int] | None:
    """(계열 이름, 순서값). 계열을 못 찾으면 None. 순서값은 초 또는 프레임 번호.

    size 는 DB 의 (width, height). 이름 전체가 숫자일 때만 계열 이름에 들어간다.
    """
    b = base_name(path)
    m = _SHOT.match(b)
    if m:
        h = m.group(2)
        return f"shot {m.group(1)}", int(h[:2]) * 3600 + int(h[2:4]) * 60 + int(h[4:])
    m = _PHONE.match(b)
    if m:
        h = m.group(2)
        return f"phone {m.group(1)}", int(h[:2]) * 3600 + int(h[2:4]) * 60 + int(h[4:])
    m = _SERIAL.match(b)
    if m:
        if m.group(1):
            return f"serial {m.group(1)}", int(m.group(2))
        # 이름이 숫자뿐이면 출처를 가를 앞부분이 없다. 자릿수와 크기로 가른다 (머리말 1번).
        dims = f" {size[0]}x{size[1]}" if size else ""
        return f"serial #{len(m.group(2))}{dims}", int(m.group(2))
    return None


def dhash(path: Path, size: int = 8) -> str | None:
    """8x8 dHash 를 16자리 16진수로. 읽지 못하면 None.

    JPEG 은 `draft()` 로 디코딩 단계에서 줄여 받는다 (6,466장에 약 2분).
    """
    from PIL import Image

    try:
        with Image.open(path) as im:
            im.draft("L", (size * 8, size * 8))
            small = im.convert("L").resize((size + 1, size), Image.BILINEAR)
    except Exception:
        return None
    a = np.asarray(small, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return f"{v:016x}"


def _bit_matrix(hashes: list[str]) -> np.ndarray:
    arr = np.array([int(h, 16) for h in hashes], dtype=np.uint64)
    return np.unpackbits(arr.view(np.uint8).reshape(-1, 8)[:, ::-1], axis=1)


def close_pairs(ids: list[int], hashes: list[str], max_dist: int = DHASH_MAX,
                block: int = 512) -> list[tuple[int, int]]:
    """dHash 거리가 max_dist 이하인 쌍. 6,466장 전수 비교가 몇 초면 끝난다."""
    if not ids:
        return []
    bits = _bit_matrix(hashes)
    out: list[tuple[int, int]] = []
    for s in range(0, len(ids), block):
        e = min(s + block, len(ids))
        d = (bits[s:e, None, :] != bits[None, :, :]).sum(axis=2)
        for a, b in np.argwhere(d <= max_dist):
            if s + a < b:
                out.append((ids[s + a], ids[b]))
    return out


def group_images(items: list[dict], *, max_dist: int = DHASH_MAX) -> dict[int, str]:
    """{이미지 id: 묶음 키}. items 는 {"id", "path", "dhash"} (+ 있으면 "width", "height") dict 목록.

    묶음 키는 그 묶음에서 가장 작은 이미지 id 로 만든다(`g000123`) — 새 이미지가
    들어와도 기존 묶음의 키가 바뀌지 않는다.
    """
    dsu = DSU()
    for it in items:
        dsu.find(it["id"])

    # 1) 연번 계열은 **통째로** 한 묶음 — 같은 출처(카메라·영상)로 본다.
    #    번호 간격으로 자르던 때는 `ala_2_001` 과 `ala_2_016` 이 train·val 로 갈렸고,
    #    열어 보니 같은 카메라의 같은 장면이었다 (2026-09-16).
    # 2) 시각 기반 이름은 30초 이내 연사만 묶는다. 하루치를 통째로 묶으면 다른 카메라
    #    화면을 찍은 사진까지 한 덩어리가 된다(134장짜리 묶음에서 확인).
    series: dict[str, list[tuple[int, int]]] = {}
    for it in items:
        size = (it["width"], it["height"]) if it.get("width") and it.get("height") else None
        s = name_series(it["path"], size)
        if s:
            series.setdefault(s[0], []).append((s[1], it["id"]))
    for key, vals in series.items():
        vals.sort()
        if not key.startswith(("shot ", "phone ")):
            for (_, i0), (_, i1) in zip(vals, vals[1:]):
                dsu.union(i0, i1)
            continue
        for (v0, i0), (v1, i1) in zip(vals, vals[1:]):
            if v1 - v0 <= BURST_SEC:
                dsu.union(i0, i1)

    # 3) 내용이 거의 같은 것
    have = [(it["id"], it["dhash"]) for it in items if it.get("dhash")]
    for a, b in close_pairs([i for i, _ in have], [h for _, h in have], max_dist):
        dsu.union(a, b)

    members: dict[int, list[int]] = {}
    for it in items:
        members.setdefault(dsu.find(it["id"]), []).append(it["id"])
    out: dict[int, str] = {}
    for ids in members.values():
        key = f"g{min(ids):06d}"
        for i in ids:
            out[i] = key
    return out
