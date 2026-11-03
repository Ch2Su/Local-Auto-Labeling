"""train / val / test 배정 — 묶음 단위로, 계층화해서, DB 에 고정한다.

세 가지를 한꺼번에 푼다.

1. **DB 에 고정** (`images.split`). 예전에는 파일 경로 해시로 매번 계산해서, `tidy` 가
   파일을 옮기면 배정이 뒤집혔다. 이제 경로가 바뀌어도 그대로다.
2. **계층화.** 해시·무작위 분할은 클래스별 비중이 20% ±0.7~1.9%p 로 흔들린다(무작위
   200회 실측). 계층화하면 ±0.5%p 안에 든다. 희소 클래스(scooter·bus)의 점수가 덜 흔들린다.
3. **묶음 단위 분할.** 같은 영상·같은 연사·내용이 같은 이미지는 통째로 한쪽에만 넣는다
   (`grouping.py`). 인접 프레임이 train 과 val 로 갈리면 점수가 부풀려지기 때문이다.
   출처를 기록한 이미지는 **카메라째** 한 묶음이고(2026-11-03, docs/provenance.md), `cameras.yaml` 에
   `split:` 을 적은 카메라는 그 분할로 고정한다. 그래서 test 점수가 '처음 보는 카메라' 성능을 잰다.

**test 는 학습에도, best.pt 고르기에도 쓰지 않는다.** val 은 학습 중 best 를 고르는 데
쓰이므로 val 점수에는 '여러 epoch 중 가장 좋은 것을 골랐다'는 이득이 섞인다. test 는
그게 없어서 모델 간 비교에 쓸 수 있다. `data.yaml` 의 `test:` 로 나가고,
`YOLO(w).val(split="test")` 로 잰다.

**배정은 한 번 정하면 움직이지 않는다.** 새로 승인된 이미지만 계층화로 채운다. 전체를
다시 나누려면 `labeler.py split --reset` 을 사람이 명시적으로 돌려야 한다 — 그러면
이전 학습들과 점수 비교가 끊긴다(그 모델들이 새 val/test 이미지를 학습에 썼을 수 있다).
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .grouping import close_pairs, dhash, group_images
from .provenance import load_cameras
from .store import Store

SPLITS = ("train", "val", "test")


@dataclass
class SplitReport:
    total: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    groups: int = 0
    biggest_group: int = 0
    hashed: int = 0          # 이번에 dHash 를 새로 계산한 장수
    assigned: int = 0        # 이번에 새로 배정한 장수 (배정이 없던 이미지)
    moved: dict[str, int] = field(default_factory=dict)   # 이미 배정돼 있다가 옮겨진 것 {"val->train": n}
    # 기존 배정이 서로 다른 이미지가 한 묶음으로 이어진 것 {묶음: {split: 장수}} — 옮기지 않고 알리기만 한다
    mixed: dict[str, dict[str, int]] = field(default_factory=dict)
    # 카메라별 배정 {카메라: {split: 장수}} — 한 카메라가 두 분할에 걸치면 경고한다
    cameras: dict[str, dict[str, int]] = field(default_factory=dict)
    # cameras.yaml 의 split: 고정과 다른 분할에 이미 있어 그대로 둔 이미지 {묶음: {split: 장수}}
    pin_kept: dict[str, dict[str, int]] = field(default_factory=dict)
    reset: bool = False
    ratios: dict[str, float] = field(default_factory=dict)
    shares: dict[str, dict[str, float]] = field(default_factory=dict)  # split -> 클래스 -> 비중

    def worst(self) -> float:
        out = 0.0
        for s, per in self.shares.items():
            for v in per.values():
                out = max(out, abs(v - self.ratios.get(s, 0)))
        return out

    def summary(self) -> str:
        n = max(self.total, 1)
        head = "  ".join(f"{s} {self.counts.get(s, 0)}장 ({self.counts.get(s, 0)/n:.1%})"
                         for s in SPLITS if self.counts.get(s))
        lines = [f"분할: {head}",
                 f"  묶음 {self.groups}개 (가장 큰 묶음 {self.biggest_group}장)"]
        if self.hashed:
            lines.append(f"  dHash 새로 계산 {self.hashed}장")
        if self.reset:
            lines.append("  **전체 재배정** — 이전 학습과 점수 비교는 끊깁니다")
        else:
            if self.assigned:
                lines.append(f"  새로 배정 {self.assigned}장")
            if self.moved:      # reset 이 아니면 일어나지 않아야 한다 (2026-10-25 부터 기존 배정을 고정)
                detail = ", ".join(f"{k} {v}장" for k, v in sorted(self.moved.items()))
                lines.append(
                    f"  [경고] 이미 배정돼 있던 이미지가 옮겨졌습니다: {detail}. train 에서 val·test 로 "
                    "간 것이 있으면 그 이미지를 학습한 이전 모델의 val·test 점수는 부풀려집니다.")
            if self.mixed:
                detail = "; ".join(f"{g} " + "·".join(f"{s} {n}" for s, n in sorted(c.items()))
                                   for g, c in sorted(self.mixed.items())[:5])
                more = f" 외 {len(self.mixed) - 5}개" if len(self.mixed) > 5 else ""
                lines.append(
                    f"  [알림] 서로 다른 분할에 있던 이미지가 한 묶음으로 이어졌습니다 {len(self.mixed)}개"
                    f"({detail}{more}). 기존 배정은 그대로 두었습니다 — 그 장면은 두 분할에 걸칩니다.")
            if self.pin_kept:
                detail = "; ".join(f"{g} " + "·".join(f"{s} {n}" for s, n in sorted(c.items()))
                                   for g, c in sorted(self.pin_kept.items())[:5])
                lines.append(
                    f"  [알림] cameras.yaml 의 split 고정과 다른 분할에 이미 있던 이미지는 옮기지 않았습니다"
                    f" ({detail}). 새 이미지만 고정한 분할로 갑니다.")
        if self.cameras:
            def fmt(c: dict[str, int]) -> str:
                return "·".join(f"{s} {c[s]}" for s in SPLITS if c.get(s))
            lines.append("  카메라별: " + ", ".join(f"{k} {fmt(c)}" for k, c in sorted(self.cameras.items())))
            split_cams = [k for k, c in self.cameras.items() if sum(1 for v in c.values() if v) > 1]
            if split_cams:
                lines.append(f"  [경고] 두 분할에 걸친 카메라: {', '.join(sorted(split_cams))} — 같은 배경이 "
                             "train 과 평가셋에 함께 있습니다(기존 배정을 고정해서 생긴 것일 수 있음).")
        for s in SPLITS:
            per = self.shares.get(s)
            if per:
                lines.append(f"  {s} 클래스 비중: "
                             + ", ".join(f"{k} {v:.1%}" for k, v in sorted(per.items())))
        if self.shares:
            lines.append(f"  목표에서 벗어난 폭: {self.worst():.1%}p")
        return "\n".join(lines)


# ------------------------------------------------------------------ 내부 계산

def _ratios(cfg: Config) -> dict[str, float]:
    val = float(cfg.get("train.val_ratio", 0.2))
    test = float(cfg.get("train.test_ratio", 0.0))
    if val + test >= 1.0:
        raise RuntimeError("config.yaml 의 train.val_ratio + test_ratio 가 1 이상입니다.")
    out = {"train": 1.0 - val - test, "val": val}
    if test > 0:
        out["test"] = test
    return out


def _camera_pins(cfg: Config, groups: dict[str, list[dict]],
                 ratios: dict[str, float]) -> dict[str, str]:
    """{묶음: cameras.yaml 의 split}. 카메라가 있는 묶음이 없으면 등록 파일을 읽지 않는다.

    한 묶음에 고정 분할이 다른 카메라가 함께 있으면(dHash 로 이어짐) 어느 쪽도 지킬 수 없어 멈춘다 —
    사람이 정할 일이다. 등록 파일에서 빠진 카메라는 고정 없이 계층화를 탄다.
    """
    used = {it["camera"] for ms in groups.values() for it in ms if it.get("camera")}
    if not used:
        return {}
    try:
        reg = load_cameras(cfg.cameras)
    except ValueError as exc:
        raise RuntimeError(f"카메라 등록 파일을 읽지 못해 분할을 멈췄습니다 (아무것도 배정하지 않음).\n{exc}") from exc
    out: dict[str, str] = {}
    for g, ms in groups.items():
        want: dict[str, set[str]] = {}
        for it in ms:
            cam = reg.get(it.get("camera") or "")
            if cam and cam.split:
                want.setdefault(cam.split, set()).add(cam.key)
        if not want:
            continue
        if len(want) > 1:
            detail = ", ".join(f"{s}: {'·'.join(sorted(k))}" for s, k in sorted(want.items()))
            raise RuntimeError(
                f"묶음 {g} 에 split 고정이 서로 다른 카메라가 함께 있습니다({detail}) — 내용이 거의 같은 이미지로 "
                "이어진 것입니다. cameras.yaml 의 split 을 맞추거나 한쪽 고정을 지운 뒤 다시 돌리세요.")
        s = next(iter(want))
        if s not in ratios:
            raise RuntimeError(f"카메라 {'·'.join(sorted(want[s]))} 의 split: {s} 은 지금 비율 설정에 없습니다"
                               " (train.test_ratio 가 0 이면 test 를 쓸 수 없다).")
        out[g] = s
    return out


def _ensure_dhash(store: Store, items: list[dict]) -> int:
    """dHash 가 없는 이미지만 계산해 DB 에 넣는다. 원본이 없으면 건너뛴다."""
    todo = [it for it in items if not it.get("dhash")]
    done = []
    for it in todo:
        p = Path(it["path"])
        if not p.exists():
            continue
        h = dhash(p)
        if h:
            it["dhash"] = h
            done.append((it["id"], h))
    store.set_dhashes(done)
    return len(done)


def _stratify_groups(groups: dict[str, list[dict]], ratios: dict[str, float],
                     fixed: dict[str, str], jitter: int = 0) -> dict[str, str]:
    """묶음을 split 에 배분한다 (묶음 단위 계층화).

    **큰 묶음부터** 넣는다 — 이 데이터에는 천 장이 넘는 묶음(고정 CCTV 한 대)이 있어서,
    작은 것부터 채우면 마지막에 큰 것을 놓을 자리가 없다. 각 묶음은 '넣고 났을 때
    전체가 가장 덜 틀어지는' split 으로 간다. 틀어짐은 클래스별 비중과 장수 비중을
    같이 본다 — 클래스만 보면 장수가 쏠리고, 장수만 보면 희소 클래스가 쏠린다.
    (먼저 '모자란 쪽'을 고르는 방식으로 짰다가 70/15/15 가 65/14/21 로 나와서 바꿨다.)
    """
    totals: Counter = Counter()
    n_all = 0
    for members in groups.values():
        for it in members:
            totals.update(it["counts"])
            n_all += 1

    got = {s: Counter() for s in ratios}
    got_n = {s: 0 for s in ratios}
    assign: dict[str, str] = {}
    for g, s in fixed.items():           # 이미 배정된 묶음은 그대로 두고 채워 넣는다
        if g in groups and s in ratios:
            assign[g] = s
            for it in groups[g]:
                got[s].update(it["counts"])
                got_n[s] += 1

    def counts_of(g: str) -> Counter:
        c: Counter = Counter()
        for it in groups[g]:
            c.update(it["counts"])
        return c

    def delta(s: str, c: Counter, size: int) -> float:
        """이 묶음을 s 에 넣으면 '목표에서 벗어난 정도'가 얼마나 늘어나나."""
        r = ratios[s]
        out = 0.0
        for k, n in c.items():
            if not totals[k]:
                continue
            before = abs(got[s][k] / totals[k] - r)
            after = abs((got[s][k] + n) / totals[k] - r)
            out += after - before
        before_n = abs(got_n[s] / n_all - r)
        after_n = abs((got_n[s] + size) / n_all - r)
        return out + (after_n - before_n) * len(totals)   # 장수도 클래스들만큼 무겁게

    rng = random.Random(jitter)
    todo = sorted((g for g in groups if g not in assign),
                  key=lambda g: (-len(groups[g]) * (1 if not jitter else rng.uniform(0.7, 1.3)),
                                 min((totals[k] for k in counts_of(g)), default=10**9)))
    for g in todo:
        c = counts_of(g)
        size = len(groups[g])
        best = min(ratios, key=lambda s: (delta(s, c, size), -ratios[s]))
        assign[g] = best
        got[best].update(c)
        got_n[best] += size
    return assign


def _deviation(groups: dict[str, list[dict]], assign: dict[str, str],
               ratios: dict[str, float]) -> float:
    """배정 전체가 목표에서 벗어난 정도 (클래스 비중 + 장수). 작을수록 좋다."""
    totals: Counter = Counter()
    per: dict[str, Counter] = {s: Counter() for s in ratios}
    n_all = 0
    n_per = {s: 0 for s in ratios}
    for g, members in groups.items():
        s = assign.get(g)
        for it in members:
            totals.update(it["counts"])
            n_all += 1
            if s:
                per[s].update(it["counts"])
                n_per[s] += 1
    out = 0.0
    for s, r in ratios.items():
        out += sum(abs(per[s][k] / totals[k] - r) for k in totals if totals[k])
        out += abs(n_per[s] / max(n_all, 1) - r) * len(totals)
    return out


def _shares(items: list[dict], assign: dict[int, str],
            classes: list[str]) -> dict[str, dict[str, float]]:
    """이미지별 배정({이미지 id: split})으로 센다 — 한 묶음이 두 분할에 걸칠 수 있어서."""
    tot: Counter = Counter()
    per: dict[str, Counter] = {s: Counter() for s in SPLITS}
    for it in items:
        s = assign.get(it["id"])
        for k, n in it["counts"].items():
            tot[k] += n
            if s:
                per[s][k] += n
    name = lambda k: classes[k] if k < len(classes) else f"cls_{k}"
    return {s: {name(k): per[s][k] / tot[k] for k in sorted(tot) if tot[k]}
            for s in SPLITS if per[s]}


# -------------------------------------------------------------------- 공개 API

def ensure_splits(cfg: Config, store: Store, *, reset: bool = False,
                  regroup: bool = False) -> tuple[dict[int, str], SplitReport]:
    """승인 이미지 전부에 train/val/test 배정을 보장하고 {이미지 id: split} 을 돌려준다.

    reset=True 면 전부 다시 나눈다 (이전 학습과 비교가 끊긴다).
    regroup=True 면 묶음도 다시 계산한다 (새 이미지가 기존 묶음에 붙을 수 있으므로,
    dHash 를 새로 계산한 이미지가 있으면 자동으로 켜진다).
    """
    ratios = _ratios(cfg)
    items = store.approved_class_counts()
    report = SplitReport(total=len(items), ratios=ratios, reset=reset)
    if not items:
        return {}, report

    report.hashed = _ensure_dhash(store, items)
    need_group = reset or regroup or report.hashed > 0 or any(not it["grp"] for it in items)
    keys = group_images(items) if need_group else {it["id"]: it["grp"] for it in items}
    groups: dict[str, list[dict]] = {}
    for it in items:
        groups.setdefault(keys[it["id"]], []).append(it)
    # 카메라 고정은 DB 에 무엇이든 쓰기 전에 확인한다 — 등록 파일 오류·고정 충돌이면 여기서 멈춘다.
    pins = _camera_pins(cfg, groups, ratios)
    if need_group:
        store.set_groups([(it["id"], keys[it["id"]]) for it in items
                          if it["grp"] != keys[it["id"]]])
        for it in items:
            it["grp"] = keys[it["id"]]
    report.groups = len(groups)
    report.biggest_group = max(len(v) for v in groups.values())

    # 이미 배정된 이미지가 **하나라도** 있는 묶음은 그 split 을 유지한다(다수결) — 새로 들어온
    # 이미지는 그쪽을 따라간다. (전부 배정된 묶음만 고정했더니, 새 이미지가 낀 묶음이 통째로
    # 재배정돼 기존 이미지 7장이 따라 움직였다 — 2026-09-16 시험에서 잡았다.)
    # **기존 이미지 자체는 묶음의 다수결과 상관없이 제자리다** (2026-10-25). 새 이미지가 dHash 로
    # train 묶음과 test 묶음을 이어 붙이면, 예전에는 소수 쪽 기존 이미지가 다수결로 옮겨졌다 —
    # test 기준선이 바뀌거나 학습한 이미지가 test 로 들어갈 수 있었다. 이제는 옮기지 않고
    # 그 묶음을 `report.mixed` 로 알린다(그 장면은 두 분할에 걸친다).
    fixed: dict[str, str] = {}
    pinned: dict[int, str] = {}
    if not reset:
        for g, members in groups.items():
            have = Counter(it["split"] for it in members if it["split"] in ratios)
            if have:
                fixed[g] = have.most_common(1)[0][0]
                if len(have) > 1:
                    report.mixed[g] = dict(have)
        pinned = {it["id"]: it["split"] for it in items if it["split"] in ratios}
    # cameras.yaml 의 split: 은 다수결보다 앞선다 — 새 이미지는 그리로 간다. 이미 다른 분할에 있던 이미지는
    # 위의 pinned 규칙대로 제자리에 두고 알리기만 한다.
    for g, s in pins.items():
        fixed[g] = s
        other = Counter(it["split"] for it in groups[g]
                        if not reset and it["split"] in ratios and it["split"] != s)
        if other:
            report.pin_kept[g] = dict(other)

    # 큰 묶음이 어디로 가느냐에 따라 결과가 꽤 달라진다. 순서를 몇 번 흔들어 보고
    # 가장 덜 틀어진 배정을 고른다 (시드 고정이라 항상 같은 결과가 나온다).
    assign_g = min(
        (_stratify_groups(groups, ratios, fixed, jitter=j) for j in range(12)),
        key=lambda a: _deviation(groups, a, ratios),
    )
    assign = {it["id"]: pinned.get(it["id"], assign_g[g])
              for g, members in groups.items() for it in members}

    changed = [(it["id"], assign[it["id"]]) for it in items
               if it["split"] != assign[it["id"]]]
    # reset 이 아니면 위의 pinned 때문에 0 이어야 한다. 그래도 조용히 지나가면 안 되는 일이라 센다.
    moved = Counter(f"{it['split']}->{assign[it['id']]}" for it in items
                    if it["split"] in ratios and it["split"] != assign[it["id"]])
    report.moved = {} if reset else dict(moved)
    report.assigned = len(changed) - (0 if reset else sum(moved.values()))
    if changed:
        store.set_splits(changed)

    report.counts = Counter(assign.values())
    report.shares = _shares(items, assign, store.get_classes())
    for it in items:
        if it.get("camera"):
            report.cameras.setdefault(it["camera"], {})
            c = report.cameras[it["camera"]]
            c[assign[it["id"]]] = c.get(assign[it["id"]], 0) + 1
    return assign, report


def leak_report(cfg: Config, store: Store, *, max_dist: int = 6) -> str:
    """val·test 이미지가 train 에 '비슷한 이미지'를 몇 장이나 두고 있는지.

    묶음 기준(dHash ≤2)보다 느슨한 거리로 재서, 묶음으로 못 막은 잔여 누수를 본다.
    """
    items = [it for it in store.approved_class_counts() if it.get("dhash")]
    by_id = {it["id"]: it for it in items}
    ids = [it["id"] for it in items]
    hashes = [it["dhash"] for it in items]
    pairs = close_pairs(ids, hashes, max_dist)
    near: dict[int, set[str]] = {}
    for a, b in pairs:
        near.setdefault(a, set()).add(by_id[b]["split"])
        near.setdefault(b, set()).add(by_id[a]["split"])
    lines = [f"[잔여 누수] dHash 거리 ≤{max_dist} 로 본 '비슷한 이미지' (묶음 기준은 ≤2)"]
    for s in ("val", "test"):
        members = [i for i in ids if by_id[i]["split"] == s]
        if not members:
            continue
        hit = sum(1 for i in members if "train" in near.get(i, ()))
        lines.append(f"  {s} {len(members)}장 중 train 쪽에 비슷한 것이 있는 이미지: "
                     f"{hit}장 ({hit / len(members):.1%})")
    return "\n".join(lines)
