"""승인된 라벨로 YOLO 데이터셋을 만들고 학습한다.

데이터셋은 매번 config 의 data.dataset(가시광은 data/visible/dataset) 아래에 새로
빌드된다. 이미지는 복사한다 — 하드링크는 원본까지 덮어써질 수 있어 기본으로 끈다
(`_link_or_copy`, CLAUDE.md 함정 3).

train/val/test 배정은 DB(`images.split`)에 고정돼 있다(`split.py`). 검수가 쌓여 데이터가
늘어나도 기존 이미지는 제자리에 남아 성능 비교가 의미를 가진다. 같은 DB 로 다시 빌드하면
결과가 똑같다는 뜻이기도 하다(`ensure_resume_dataset`).
"""

from __future__ import annotations

import hashlib
import re
import sys
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import yaml

from .config import Config, resolve
from .masks import split_boxes, write_masked
from .split import SplitReport, ensure_splits
from .store import Store
from .yolo_io import write_yolo_txt


@dataclass
class BuildReport:
    train: int = 0
    val: int = 0
    test: int = 0
    boxes: int = 0
    per_class: dict[str, int] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    data_yaml: Path | None = None
    split: SplitReport | None = None
    masked: int = 0             # 가릴 영역을 칠해 쓴 이미지 (masks.py)
    mask_dropped: int = 0       # 그 영역에 반 넘게 가려져 학습 라벨에서 뺀 박스
    masked_test: int = 0        # 영역이 있지만 test 라 칠하지 않은 이미지 (평가는 실제 화면 그대로)

    def summary(self) -> str:
        lines = [
            f"학습셋 {self.train}장 / 검증셋 {self.val}장"
            + (f" / 테스트셋 {self.test}장 (학습에 쓰지 않음)" if self.test else "")
            + f" (박스 {self.boxes}개)",
            "클래스별 박스 수: "
            + (
                ", ".join(f"{k}={v}" for k, v in sorted(self.per_class.items()))
                or "(없음)"
            ),
        ]
        if self.missing:
            lines.append(f"  원본이 사라진 이미지 {len(self.missing)}장은 건너뜀")
        if self.masked:
            lines.append(f"가릴 영역: {self.masked}장을 회색(114)으로 칠해 썼고, 반 넘게 가려진 박스 "
                         f"{self.mask_dropped}개는 학습 라벨에서 뺐습니다 (DB 박스는 그대로)")
        if self.masked_test:
            lines.append(f"  test 이미지 {self.masked_test}장의 가릴 영역은 칠하지 않았습니다 (평가는 실제 화면 그대로)")
        if self.split is not None:
            lines.append(self.split.summary())
        return "\n".join(lines)


def _link_or_copy(src: Path, dst: Path, *, hardlink: bool = False) -> None:
    """학습 데이터셋으로 이미지를 가져온다.

    hardlink=True 는 용량을 아끼지만 원본과 같은 실체를 가리킨다. Ultralytics 는
    스캔 중 깨진 JPEG 을 발견하면 그 자리에 다시 저장하므로(corrupt JPEG restored
    and saved), 하드링크를 쓰면 원본 이미지까지 덮어써진다. 기본은 복사.
    """
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if hardlink:
        try:
            os.link(src, dst)  # 같은 NTFS 볼륨이면 즉시, 추가 용량 0
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def build_dataset(cfg: Config, store: Store) -> BuildReport:
    root = cfg.dataset
    if root.exists():
        shutil.rmtree(root)

    classes = store.get_classes()
    if not classes:
        raise RuntimeError("클래스 목록이 비어 있습니다. 먼저 'scan'을 실행하세요.")

    hardlink = bool(cfg.get("train.hardlink_images", False))
    report = BuildReport()

    records = store.approved_with_boxes()
    if not records:
        raise RuntimeError(
            "승인된 라벨이 하나도 없습니다. 시드 데이터를 넣고 'scan'을 실행하거나 "
            "검수 UI에서 이미지를 승인하세요."
        )

    # train/val 배정은 DB 에 고정돼 있다. 새로 승인된 것만 계층화로 채운다(split.py).
    assign, split_report = ensure_splits(cfg, store)
    report.split = split_report
    masks = store.masks_by_image()          # 가릴 영역이 있는 이미지만 (masks.py)

    for row, boxes in records:
        src = Path(row["path"])
        if not src.exists():
            report.missing.append(str(src))
            continue

        # 서로 다른 폴더의 동명 파일이 충돌하지 않도록 경로 해시를 덧붙인다.
        digest = hashlib.sha1(str(src).encode("utf-8")).hexdigest()[:8]
        stem = f"{src.stem}__{digest}"
        split = assign[row["id"]]

        rects = masks.get(row["id"])
        if rects and split == "test":       # 평가는 실제 화면 그대로 — 칠하지도, 박스를 빼지도 않는다
            report.masked_test += 1
            rects = None
        if rects:
            # 가릴 영역을 회색으로 칠한 JPEG(원본은 그대로)과, 반 넘게 가려진 박스를 뺀 라벨
            boxes, dropped = split_boxes(boxes, rects)
            try:
                write_masked(src, root / "images" / split / f"{stem}.jpg", rects)
            except (OSError, SyntaxError, ValueError) as exc:
                raise RuntimeError(f"가릴 영역을 칠하지 못했습니다 (id {row['id']}): {src}: {exc}") from exc
            report.masked += 1
            report.mask_dropped += len(dropped)
        else:
            _link_or_copy(src, root / "images" / split / f"{stem}{src.suffix}", hardlink=hardlink)
        write_yolo_txt(root / "labels" / split / f"{stem}.txt", boxes)

        if split == "val":
            report.val += 1
        elif split == "test":
            report.test += 1
        else:
            report.train += 1
        report.boxes += len(boxes)
        for b in boxes:
            name = classes[b.cls] if 0 <= b.cls < len(classes) else f"cls_{b.cls}"
            report.per_class[name] = report.per_class.get(name, 0) + 1

    if report.train == 0:
        raise RuntimeError(
            "학습셋이 비었습니다. 승인된 이미지가 너무 적거나 val_ratio가 큽니다."
        )
    if report.val == 0:
        # 데이터가 아주 적을 때는 학습셋 일부를 검증에도 쓴다.
        # (과적합을 감지하지는 못하지만 학습 자체는 돌아간다.)
        (root / "labels" / "val").mkdir(parents=True, exist_ok=True)
        for p in sorted((root / "images" / "train").iterdir()):
            _link_or_copy(p, root / "images" / "val" / p.name, hardlink=hardlink)
            shutil.copy2(
                root / "labels" / "train" / f"{p.stem}.txt",
                root / "labels" / "val" / f"{p.stem}.txt",
            )
            report.val += 1
            break

    spec = {"path": str(root), "train": "images/train", "val": "images/val"}
    if report.test:
        # 학습은 train/val 만 본다. test 는 `YOLO(w).val(split="test")` 로 따로 잰다.
        spec["test"] = "images/test"
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {**spec, "names": {i: n for i, n in enumerate(classes)}},
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    report.data_yaml = data_yaml
    return report


def check_box_distance_gain(model, dfl: float) -> str | None:
    """`dfl` 게인이 0 이면 박스 거리 손실이 통째로 꺼지므로 경고한다.

    이름이 'dfl' 이라 DFL 전용 옵션처럼 보이지만, Ultralytics 에서 이 값은
    '박스 거리 손실 게인'이다 (default.yaml: "DFL when reg_max > 1, L1 on DFL-free YOLO26").

      - reg_max > 1  (yolo11 등) : loss[2] = DFL
      - reg_max == 1 (yolo26)    : loss[2] = ltrb 거리에 대한 L1

    어느 쪽이든 `loss[2] *= hyp.dfl` 이 적용되므로, dfl=0.0 은 '해당 없는 손실 끄기'가
    아니라 '거리 회귀 보조 감독 제거'가 된다.

    박스 감독이 전부 사라지는 것은 아니다. CIoU(loss[0], gain=box)는 그대로 남으므로
    검출 자체는 학습된다. 실제로 손해를 보는 쪽은 좌표 정밀도다.
    합성 데이터 20 epoch A/B 기준: mAP50 은 0.995 로 동일, mAP50-95 는 0.902 -> 0.953.
    에러 없이 조용히 진행되므로 경고를 띄운다.
    """
    if dfl != 0:
        return None
    try:
        reg_max = int(model.model.model[-1].reg_max)
    except (AttributeError, IndexError, TypeError):
        reg_max = None
    kind = "L1" if reg_max == 1 else "DFL"
    return (
        f"dfl=0.0 이면 박스 거리 손실({kind})이 꺼집니다"
        + (f" (reg_max={reg_max})" if reg_max is not None else "")
        + ". 검출은 CIoU 로 학습되지만 박스 좌표 정밀도(mAP50-95)가 떨어집니다. "
        "config.yaml 의 train.overrides.dfl 을 1.5 로 되돌리세요."
    )


def check_ram_cache(cache, workers: int) -> str | None:
    """Windows 에서 RAM 캐시 + 워커 프로세스 조합을 경고한다.

    Windows 는 DataLoader 워커를 spawn 으로 띄우므로 데이터셋 객체 전체가 피클되어
    각 워커로 전달된다. cache=True('ram') 이면 그 안에 디코딩된 이미지 전체가 들어
    있어(수 GB), 파이프로 보내다 다음처럼 죽는다.

        OSError: [Errno 22] Invalid argument
          ... when serializing ultralytics.data.base.BaseDataset._ImageCache state

    뒤이어 나오는 '_MultiProcessingDataLoaderIter object has no attribute
    _workers_status' 는 뒷정리 과정의 잡음이지 원인이 아니다.

    cache='disk' 는 .npy 경로만 들고 있어 피클이 가볍고 안전하다.
    """
    if sys.platform != "win32":
        return None
    if workers <= 0:
        return None
    if cache is True or (isinstance(cache, str) and cache.lower() == "ram"):
        return (
            "Windows 에서 cache=ram 과 workers>0 을 함께 쓰면 데이터가 커질수록 "
            "OSError: [Errno 22] 로 학습이 죽습니다(RAM 캐시를 워커마다 피클해 보냄). "
            "config.yaml 의 train.overrides.cache 를 disk 로 바꾸거나 workers 를 0 으로 두세요."
        )
    return None


@dataclass
class ResumePoint:
    """이어서 학습할 수 있는 체크포인트."""

    ckpt: Path
    epoch: int          # 0-기반. 이 epoch 까지 마쳤다.
    total_epochs: int
    run_dir: Path
    kind: str           # resume.pt / last.pt / epochN.pt
    lost: int = 0       # 이 지점으로 되돌리면 버려지는 epoch 수
    data: str = ""      # 이 학습의 data.yaml (체크포인트의 train_args 에 적힌 그대로)

    @property
    def done(self) -> int:
        """마친 epoch 수 (사람이 세는 방식, results.csv 의 epoch 열과 같다)."""
        return self.epoch + 1

    def summary(self) -> str:
        lines = [
            f"이어 학습 지점 : {self.ckpt}",
            f"  {self.done}/{self.total_epochs} epoch 까지 완료 → "
            f"{self.done + 1} epoch 부터 재개",
        ]
        if self.lost:
            lines.append(
                f"  주의: 이 체크포인트 이후 {self.lost} epoch 분이 버려집니다 "
                f"({self.kind} 에는 그만큼이 담겨 있지 않습니다)"
            )
        return "\n".join(lines)


def _read_ckpt_meta(path: Path) -> tuple[int, int, str] | None:
    """체크포인트가 '이어 학습 가능한지' 보고 (epoch, total_epochs, data.yaml) 를 준다.

    Ultralytics 는 학습이 끝나면 final_eval() -> strip_optimizer() 로 last.pt 와
    best.pt 를 덮어쓴다. 그 과정에서 optimizer/ema/scaler/best_fitness 가 None 이
    되고 epoch 가 -1 로 바뀌므로(utils/torch_utils.py), 그 파일로는 이어서 학습할
    수 없다. 여기서 그 상태를 걸러낸다.
    """
    from ultralytics.utils.patches import torch_load

    try:
        ckpt = torch_load(path, map_location="cpu")
    except Exception:
        return None
    if not isinstance(ckpt, dict):
        return None
    epoch = int(ckpt.get("epoch", -1))
    if epoch < 0 or ckpt.get("optimizer") is None:
        return None
    targs = ckpt.get("train_args") or {}
    total = int(targs.get("epochs", 0))
    if total and epoch + 1 >= total:
        return None  # 이미 끝까지 간 학습
    return epoch, total, str(targs.get("data") or "")


def uses_dataset(point: ResumePoint, data_yaml: str | Path) -> bool:
    """이 체크포인트가 그 data.yaml 로 학습했는가.

    프로젝트가 여럿이면 `runs/` 를 (WSL 은 `runs_wsl/` 를) 같이 쓴다. 이걸 보지 않으면
    가장 최근 폴더가 다른 프로젝트 것일 때 그 학습을 이어서 돌리고, 끝나면
    `promote_weights()` 가 **이쪽 운영 가중치를 그 결과로 덮어쓴다.** 폴더 이름은
    config 로 바뀌므로 믿지 않고, 체크포인트에 적힌 데이터셋 경로로 가른다.
    Windows 경로(E:\\...)와 WSL 경로(/home/...) 둘 다 받는다.
    """
    norm = lambda s: os.path.normcase(str(s).replace("\\", "/").rstrip("/"))
    return bool(point.data) and norm(point.data) == norm(data_yaml)


def _ckpt_finished(path: Path) -> bool:
    """final_eval() 의 strip_optimizer() 를 거친 체크포인트인가 (= 학습이 끝났다).

    strip 은 epoch 를 -1 로 바꾸고 옵티마이저를 지운다(CLAUDE.md 함정 8). 읽지 못하는
    파일은 '끝났다'고 단정하지 않는다 — 그건 비정상 종료일 수 있다.
    """
    from ultralytics.utils.patches import torch_load

    try:
        ckpt = torch_load(path, map_location="cpu")
    except Exception:
        return False
    return isinstance(ckpt, dict) and int(ckpt.get("epoch", 0)) == -1


def _csv_last_epoch(run_dir: Path) -> int:
    """results.csv 에 기록된 마지막 epoch 번호 (1-기반). 없으면 0."""
    csv = run_dir / "results.csv"
    if not csv.exists():
        return 0
    try:
        rows = [ln for ln in csv.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return int(float(rows[-1].split(",")[0])) if len(rows) > 1 else 0
    except (ValueError, IndexError):
        return 0


def resume_point_in(run_dir: Path) -> ResumePoint | None:
    """학습 폴더 하나에서 이어 학습할 체크포인트를 고른다.

    우선순위는 resume.pt(중단 시 남긴 스냅샷) > last.pt(강제 종료로 살아남은 것)
    > epochN.pt(save_period 로 저장된 주기 체크포인트) 순이며, 그중 epoch 가 가장
    큰 것을 고른다.
    """
    w = run_dir / "weights"
    if not w.is_dir():
        return None
    best: ResumePoint | None = None

    # 최신일 가능성이 높은 두 개를 먼저 본다 (resume.pt 는 60MB 라 여는 비용이 있다).
    for name in ("resume.pt", "last.pt"):
        p = w / name
        if not p.exists():
            continue
        meta = _read_ckpt_meta(p)
        if meta and (best is None or meta[0] > best.epoch):
            best = ResumePoint(p, meta[0], meta[1], run_dir, name, data=meta[2])

    # 둘 다 못 쓰면 주기 체크포인트(epochN.pt)로 내려간다 — 단 **끝난 학습은 제외**.
    #
    # last.pt 가 final_eval 의 strip 을 거쳤다면(epoch == -1) 그 학습은 끝까지 갔거나
    # 조기 종료된 것이다. 사람이 멈췄다면 resume.pt 가 있었을 것이다. 그런데 epochN.pt
    # 는 옵티마이저가 살아 있어 '이어 갈 수 있음'으로 판정되므로, 이 확인이 없으면
    # save_period 로 epoch0.pt 가 늘 남는 탓에 **끝난 학습마다 이어하기 버튼이 뜬다.**
    # (WSL 학습 버튼을 시험하다 드러났다.)
    #
    # 이 폴백은 원래 정지 스냅샷 기능이 생기기 전에 UI 로 멈춘 학습을 살리려던
    # 것이었다. 지금은 사람이 멈추면 resume.pt 가 남으므로 그 용도는 사라졌다.
    if best is None:
        last = w / "last.pt"
        if last.exists() and _ckpt_finished(last):
            return None
        periodic = sorted(
            w.glob("epoch*.pt"),
            key=lambda p: int(re.search(r"\d+", p.stem).group()),
            reverse=True,
        )
        for p in periodic:
            meta = _read_ckpt_meta(p)
            if meta:
                best = ResumePoint(p, meta[0], meta[1], run_dir, p.name, data=meta[2])
                break

    if best is not None:
        best.lost = max(0, _csv_last_epoch(run_dir) - best.done)
    return best


def find_resume_point(cfg: Config) -> ResumePoint | None:
    """이 설정의 데이터셋으로 돈 학습 중 가장 최근 것에서 이어 학습할 체크포인트를 찾는다."""
    runs = resolve("runs")
    if not runs.exists():
        return None

    dirs = sorted(
        (d for d in runs.iterdir() if (d / "weights").is_dir()),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    want = cfg.dataset / "data.yaml"
    for d in dirs:
        point = resume_point_in(d)
        if point is not None and uses_dataset(point, want):
            return point
    return None


def trim_results_csv(run_dir: Path, keep_through: int) -> int:
    """results.csv 를 keep_through epoch 까지로 자른다. 자른 줄 수를 돌려준다.

    주기 체크포인트로 되돌아가면 그 이후 epoch 행이 csv 에 남아 있어, 이어 학습이
    같은 번호를 다시 쓰며 중복 행을 만든다(그래프도 망가진다). 원본은 .bak 으로 남긴다.
    """
    csv = run_dir / "results.csv"
    if not csv.exists():
        return 0
    lines = csv.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        return 0
    head, rows = lines[0], lines[1:]
    kept = [r for r in rows if r.strip() and float(r.split(",")[0]) <= keep_through]
    if len(kept) == len(rows):
        return 0
    shutil.copy2(csv, csv.with_suffix(".csv.bak"))
    csv.write_text("\n".join([head, *kept]) + "\n", encoding="utf-8")
    return len(rows) - len(kept)


def attach_callbacks(model, should_stop: Callable[[], bool] | None) -> dict:
    """중단 요청 처리와 '이어 학습용 스냅샷' 콜백을 붙인다.

    스냅샷이 필요한 이유: 학습이 어떤 식으로든 정상 종료하면 final_eval() 이
    last.pt 에서 옵티마이저 상태를 지우고 epoch 를 -1 로 만든다. 그 전에 온전한
    사본을 resume.pt 로 남겨두어야 나중에 이어서 학습할 수 있다.
    """
    state = {"stopped": False, "snapped": False, "snapshot": None}

    if should_stop is not None:
        def _check(trainer) -> None:
            if should_stop() and not trainer.stop:
                trainer.stop = True
                state["stopped"] = True
                print(
                    "\n[중단] 요청에 따라 학습을 끊습니다. "
                    "검증과 가중치 저장은 마칩니다.",
                    flush=True,
                )

        # 배치 단위로 봐야 응답이 빠르다. epoch 끝에서도 한 번 더 확인한다.
        model.add_callback("on_train_batch_end", _check)
        model.add_callback("on_train_epoch_end", _check)

    def _snapshot(trainer) -> None:
        # 이 콜백은 save_model() 직후에 불린다. 즉 last.pt 가 방금 쓰였고 아직
        # final_eval() 은 돌지 않은 시점이다.
        #
        # **사람이 멈춘 때만** 남긴다. trainer.stop 을 보면 안 된다 — Ultralytics 는
        # 조기 종료(patience)에도 `self.stop |= self.stopper(...)` 로 같은 플래그를
        # 세운다. 그걸 보고 스냅샷을 남기면 끝난 학습이 '이어서 학습' 대상으로 뜬다.
        # 실제로 runs/train 이 epoch 259 조기 종료 때 resume.pt 를 남겼다.
        if state["snapped"] or not state["stopped"]:
            return
        if trainer.epoch + 1 >= trainer.epochs:
            return  # 끝까지 간 학습은 이어 붙일 것이 없다
        last = Path(trainer.last)
        if not last.exists():
            return
        dst = last.with_name("resume.pt")
        shutil.copy2(last, dst)
        state["snapped"] = True
        state["snapshot"] = str(dst)
        print(f"[중단] 이어 학습용 스냅샷을 남겼습니다: {dst}", flush=True)

    model.add_callback("on_fit_epoch_end", _snapshot)
    return state


def promote_weights(best: Path, cfg: Config) -> Path | None:
    """학습 결과 best.pt 를 운영 가중치(models/best.pt)로 올린다. 백업 경로를 돌려준다.

    덮어쓰기 전에 기존 운영 가중치를 `best.prev.pt` 로 한 벌 남긴다. 학습을 도중에
    끊어도 이 경로를 타므로(정상 중단 경로), 끊을 때마다 운영 가중치가 조용히
    바뀐다. 실제로 epoch 3 짜리(mAP 0.44)가 0.72 짜리를 덮어쓴 적이 있다.
    이름을 `.pt` 로 끝내는 이유는 Ultralytics 가 확장자를 검사해서
    (`best.pt.bak_...` 같은 이름은 검증에 못 넣는다) 바로 비교할 수 있게 하려는 것.

    Windows 학습(`_finish`)과 WSL 학습(`wsl_train`) 이 같이 쓴다. 규칙이 두 군데서
    달라지지 않도록 여기 하나로 둔다.
    """
    out = cfg.weights
    out.parent.mkdir(parents=True, exist_ok=True)
    if not best.exists():
        return None
    prev = None
    if out.exists() and not _same_file(out, best):
        prev = out.with_name(f"{out.stem}.prev{out.suffix}")
        shutil.copy2(out, prev)
    shutil.copy2(best, out)
    return prev


def clear_stale_snapshot(run_dir: Path, stopped: bool) -> bool:
    """사람이 멈춘 게 아니면(= 끝까지 갔거나 조기 종료) 남은 resume.pt 를 치운다.

    중단 → 이어하기 → 완주 순서면, 처음 중단 때 만든 resume.pt 가 그대로 남는다.
    그 파일은 여전히 '이어 학습 가능'으로 판정되므로(옵티마이저가 살아 있다) UI 에
    '이어서 학습' 이 떠서, 누르면 끝난 학습을 되돌려 epoch 을 버리게 된다.
    """
    if stopped:
        return False
    snap = Path(run_dir) / "weights" / "resume.pt"
    if snap.exists():
        snap.unlink()
        return True
    return False


def _finish(cfg: Config, results, state: dict, extra: dict) -> dict:
    """최고 가중치를 models/ 로 올리고(promote_weights) 결과 dict 를 만든다."""
    out = cfg.weights
    prev = promote_weights(Path(results.save_dir) / "weights" / "best.pt", cfg)
    clear_stale_snapshot(Path(results.save_dir), state["stopped"])

    metrics = {}
    try:
        metrics = {
            k: float(v) for k, v in results.results_dict.items()
            if isinstance(v, (int, float))
        }
    except Exception:
        pass

    return {
        "run_dir": str(results.save_dir),
        "weights": str(out),
        "previous_weights": str(prev) if prev else None,
        "stopped_early": state["stopped"],
        "resume_snapshot": state["snapshot"],
        "metrics": metrics,
        **extra,
    }


def _same_file(a: Path, b: Path) -> bool:
    """내용이 같은가. 크기부터 보고 같을 때만 해시한다 (가중치는 20MB 안팎)."""
    if a.stat().st_size != b.stat().st_size:
        return False
    h = lambda p: hashlib.md5(p.read_bytes()).hexdigest()
    return h(a) == h(b)


def train_args(cfg: Config, data_yaml: Path | str, *, epochs: int | None = None) -> dict:
    """학습 버튼이 Ultralytics `model.train()` 에 넘기는 인자 전부.

    config.yaml 의 train.* 와 train.overrides 로 만든다. **WSL 학습 버튼도 이 결과를
    그대로 넘긴다** (`wsl_train`). 인자를 한 곳에서 만들어야 config 를 고쳤을 때 두
    버튼이 같은 학습을 한다. 적지 않은 키는 양쪽 다 Ultralytics 기본값이다 —
    예컨대 optimizer 는 auto 다.
    """
    args = dict(
        data=str(data_yaml),
        epochs=int(epochs or cfg.get("train.epochs", 100)),
        batch=int(cfg.get("train.batch", 8)),
        imgsz=cfg.imgsz,
        device=cfg.device,
        patience=int(cfg.get("train.patience", 30)),
        workers=int(cfg.get("train.workers", 4)),
        seed=int(cfg.get("train.seed", 0)),
        project=str(resolve("runs")),
        name="train",
        exist_ok=False,
        plots=True,
    )
    # config.yaml 의 train.overrides 를 그대로 Ultralytics 에 넘긴다.
    overrides = cfg.get("train.overrides") or {}
    if not isinstance(overrides, dict):
        raise RuntimeError("config.yaml 의 train.overrides 는 키:값 형태여야 합니다.")
    args.update(overrides)
    # config.yaml 처럼 `project: runs` 를 적으면 Ultralytics 는 **실행한 폴더**
    # 기준으로 푼다. 이어하기(find_resume_point)는 프로젝트의 runs/ 를 보므로 맞춘다.
    args["project"] = str(resolve(args["project"]))
    return args


def train(
    cfg: Config,
    store: Store,
    *,
    epochs: int | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict:
    """데이터셋을 빌드하고 학습한 뒤, 최고 가중치를 models/best.pt 로 복사한다.

    should_stop 을 주면 배치마다 호출해 True 일 때 학습을 끊는다. 스레드를 죽이는
    것이 아니라 Ultralytics 의 trainer.stop 을 세우는 방식이라, 최종 검증과 가중치
    저장까지 정상적으로 끝난 뒤 결과가 돌아온다.
    """
    from ultralytics import YOLO

    build = build_dataset(cfg, store)
    print(build.summary(), flush=True)

    # 이어서 학습하지 않고 매번 사전학습 가중치에서 다시 미세조정한다.
    # (자기학습 반복에서 모델이 자기 오류를 굳혀버리는 것을 피하기 위함)
    model = YOLO(cfg.base_weights)
    args = train_args(cfg, build.data_yaml, epochs=epochs)

    warning = check_box_distance_gain(model, float(args.get("dfl", 1.5)))
    if warning:
        print(f"\n[경고] {warning}\n", flush=True)

    cache_warning = check_ram_cache(args.get("cache"), int(args.get("workers", 0)))
    if cache_warning:
        print(f"\n[경고] {cache_warning}\n", flush=True)

    state = attach_callbacks(model, should_stop)
    results = model.train(**args)
    return _finish(
        cfg, results, state,
        {"train_images": build.train, "val_images": build.val},
    )


def ensure_resume_dataset(cfg: Config, store: Store, run_dir: Path) -> None:
    """이어 학습에 쓸 data/dataset 을 준비한다. 있으면 그대로 쓰고, 없으면 DB 로 다시 만든다.

    이어 학습은 중단 당시의 데이터셋을 그대로 써야 분할과 val 이 맞는다. 다시 만들어도
    되는 근거는 분할이 DB(`images.split`)에 고정돼 있어 **DB 가 그대로면 결과가 똑같다**는
    점이다. (경로 해시로 나누던 2026-09-11 에도 확인했다 — data/dataset 을 지운 뒤 DB 로
    다시 계산한 라벨 5,961개가 train-3 과 같은 DB 로 만든 WSL 복사본과 한 글자도 다르지
    않았다.) 그 사이 새로 승인된 이미지는 다시 만들 때 배정돼 들어간다.

    그래서 '학습을 시작한 뒤 DB 에서 바뀐 이미지 수'를 같이 보여준다. 0 이면 원래와 같다.
    학습 폴더가 생긴 시각을 기준으로 삼는다 — 데이터셋을 만든 직후에 생긴다.
    (WSL 학습 버튼도 data/dataset 을 새로 만든다. 같은 DB 면 결과도 같다.)
    """
    data_yaml = cfg.dataset / "data.yaml"
    if data_yaml.exists():
        # 데이터셋이 그 사이에 바뀌었는지 알려준다 (막지는 않는다).
        # test 도 세야 한다 — 빼면 승인 수와 늘 test 장수만큼 어긋나 잘못 경고한다.
        built = sum(
            1 for split in ("train", "val", "test")
            for _ in (cfg.dataset / "labels" / split).glob("*.txt")
        )
        approved = store.counts().get("approved", 0)
        if built != approved:
            print(
                f"\n[경고] 데이터셋에는 {built}장이 들어 있는데 현재 승인된 이미지는 "
                f"{approved}장입니다 ({approved - built:+d}장 차이). 이어 학습은 데이터셋을 "
                "다시 만들지 않으므로 그 사이에 검수한 라벨은 반영되지 않습니다.\n",
                flush=True,
            )
        return

    st = run_dir.stat()
    started = datetime.fromtimestamp(getattr(st, "st_birthtime", st.st_ctime), timezone.utc)
    # 데이터셋에 들어가는 것은 승인 이미지뿐이다. 그 밖의 변경(새로 들어와 자동 라벨링된
    # 검수 대기 등)은 대개 무관하다 — 2026-09-12 에 검수 대기 167장이 이 경고를 잘못 띄웠다.
    # 다만 '승인에서 내려온' 이미지도 그쪽에 섞여 있을 수 있어 따로 알린다.
    approved = store.count_updated_since(started, status="approved")
    others = store.count_updated_since(started) - approved
    print(f"\n{cfg.dataset} 가 없어 DB 로 다시 만듭니다 "
          f"(학습 시작 {started.astimezone():%Y-%m-%d %H:%M}).", flush=True)
    if approved:
        print(
            f"[경고] 학습을 시작한 뒤 박스나 상태가 바뀐 승인 이미지가 {approved}장입니다. "
            "다시 만든 데이터셋은 원래와 다르며, 그 이미지들이 val 에 들어가면 이전 "
            "epoch 들과 점수를 바로 비교할 수 없습니다. 새로 학습하는 편이 깔끔합니다.\n",
            flush=True,
        )
    else:
        print("  학습을 시작한 뒤 바뀐 승인 이미지가 없으므로 원래 데이터셋과 같습니다.", flush=True)
    if others:
        print(f"  (승인이 아닌 이미지 {others}장도 바뀌었습니다. 새로 들어온 것이면 무관하고, "
              "승인에서 내려온 것이면 데이터셋이 그만큼 줄어듭니다.)", flush=True)
    print(build_dataset(cfg, store).summary(), flush=True)


def resume_train(
    cfg: Config,
    store: Store,
    *,
    batch: int | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict:
    """중단된 학습을 이어서 돌린다.

    data/dataset 이 남아 있으면 그대로 쓴다. 이어 학습은 중단 당시의 데이터셋을 써야
    배치 수·스케줄러·분할이 맞아떨어지기 때문이다. 그래서 그 사이에 검수한 라벨은
    반영되지 않는다 (그러려면 새로 학습해야 한다). 없으면 DB 로 다시 만든다
    (`ensure_resume_dataset`).

    하이퍼파라미터도 체크포인트에 저장된 것을 쓴다. Ultralytics 의 check_resume()
    이 일부 키만 덮어쓰기를 허용하므로(engine/trainer.py), config.yaml 을 고쳐도
    대부분 무시된다. 여기서는 기계 사정에 해당하는 device/workers 만 넘긴다.

    batch 를 주면 그것도 함께 넘긴다. Ultralytics 가 "reduce memory" 용도로 명시적으로
    허용하는 키이고, 손실 누적이 `accumulate = round(nbs / batch)` 로 맞춰지므로
    유효 배치가 크게 흔들리지 않는다. Windows 에서 VRAM 이 모자라 공유 메모리(시스템
    RAM)로 페이징되기 시작하면 에러 없이 몇 배 느려지는데, 그때 쓰는 탈출구다.
    """
    from ultralytics import YOLO

    point = find_resume_point(cfg)
    if point is None:
        raise RuntimeError(
            "이어서 학습할 체크포인트가 없습니다.\n"
            "  - 정상 종료·중단으로 끝난 last.pt 와 best.pt 는 최종 검증 단계에서 "
            "옵티마이저 상태가 제거되어 이어 학습에 쓸 수 없습니다.\n"
            "  - 이 기능을 넣기 전에 중단한 학습이라면 resume.pt 스냅샷이 없습니다. "
            "runs/*/weights/epochN.pt (save_period 로 저장됨) 도 없다면 새로 학습하세요."
        )

    print(point.summary(), flush=True)
    ensure_resume_dataset(cfg, store, point.run_dir)

    trimmed = trim_results_csv(point.run_dir, point.done)
    if trimmed:
        print(
            f"results.csv 에서 {point.done} epoch 이후의 {trimmed}행을 잘라냈습니다 "
            f"(원본은 results.csv.bak). 이어 학습이 같은 번호를 다시 쓰기 때문입니다.",
            flush=True,
        )

    model = YOLO(str(point.ckpt))
    state = attach_callbacks(model, should_stop)
    # resume 에는 파일 경로를 넘겨야 한다. resume=True 로 두면 Ultralytics 가
    # get_latest_run() 으로 가장 최근 last.pt 를 집는데, 그 파일은 strip 된 것일 수 있다.
    args = dict(
        resume=str(point.ckpt),
        device=cfg.device,
        workers=int(cfg.get("train.workers", 4)),
    )
    if batch:
        args["batch"] = int(batch)
        print(f"배치 크기를 {batch} 로 바꿔 이어갑니다 (VRAM 절약).", flush=True)

    results = model.train(**args)
    return _finish(
        cfg, results, state,
        {
            "resumed_from": str(point.ckpt),
            "resumed_at_epoch": point.done,
            "total_epochs": point.total_epochs,
            "discarded_epochs": point.lost,
            "batch": batch,
        },
    )
