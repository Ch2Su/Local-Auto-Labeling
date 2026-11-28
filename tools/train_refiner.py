"""VehicleRefiner 1단계 — 크롭 세트(`tools/build_vehicle_crops.py`)로 카메라째 교차 검증 학습을 돌려 fold 밖 예측(OOF)을 남긴다
(2026-11-28, 사용자 '학습 도구 만들기'. 설계는 docs/vehicle-refiner.md '3. 구조'·'4. 학습과 평가'·'5. 비교 실험').

**하는 일.** `manifest.csv` 의 `fold` 0…k-1 마다 나머지 fold 로 학습하고 그 fold 를 예측한다 — 고르는 데 쓰는 점수는 이 OOF 뿐이다.
**fold 안에서 검증으로 epoch 을 고르지 않는다**(정해 둔 epoch 끝까지 — 고르면 OOF 가 낙관적이 된다). `fold=test` 크롭은 학습에도
OOF 에도 넣지 않는다. `--save-final` 이면 모든 fold 로 한 번 더 학습해 `final.pt` 와 test 크롭의 예측(`test_preds.csv`)을 남긴다
(2단계용). 결과는 `<crops>/runs/<이름>/`(저장소 밖 — 크롭 폴더 옆): `oof.csv`·`config.json`·`log.jsonl`. 지표는
`tools/eval_refiner.py report` 가 센다.

**모델.** 크롭 → (박스를 `--pad` 만큼 넓힌 자리, 학습 때는 박스 흔들기) → `--view 1280` 이면 그림 긴 변 1280 배율로 줄였다가 →
레터박스 `--imgsz`(회색 114) → 백본 → GAP → (`--geom` 이면 [log(w/h), w/W, h/H, cy/H] 를 MLP 4→32→32 로 이어 붙임) → Dropout 0.2 →
Linear 5. 백본(`--backbone`):
- `yolo:<가중치>` — Ultralytics 검출 가중치의 층 0–10(SPPF·C2PSA 까지). B1 `yolo:weights/yolo26m.pt`(COCO) · B2
  `yolo:models/best_y26m_8cls.pt`(운영 m) · B2b 옛 데이터 10클래스 가중치. 입력은 0–1 RGB(Ultralytics 와 같다).
- `tv:resnet18`·`tv:convnext_tiny` — torchvision ImageNet 가중치. **내려받지 않는다** — torch 캐시(`torch.hub.get_dir()/checkpoints`)
  에 파일이 없으면 멈춘다(ConvNeXt 는 약 110MB — 사용자 허락을 받고 따로 받을 것). 입력은 ImageNet 평균·표준편차로 맞춘다.
- `tinycnn` — 사전학습 없는 작은 합성곱(테스트·연기 시험용, 비교에 쓰지 말 것).

**학습.** AdamW(백본 `--lr`, 머리·기하 `--head-lr`), weight decay `--wd`, warmup 2 epoch + cosine, `--epochs`, `--batch`, CUDA 면
bf16 autocast, 교차 엔트로피 + label smoothing `--smooth`. 불균형 `--balance`: `sampler`(클래스 빈도의 제곱근 역수로 뽑기, 기본) ·
`loss`(같은 몫을 손실 가중으로) · `none`. `--freeze` 는 백본을 얼린다(BN 통계도 — 선형 탐침, R6). 증강: 좌우 뒤집기 0.5 · 밝기·대비·
채도 ±0.3 · 색조 ±0.02 · 흑백 0.1 · 흐림 0.2 · JPEG 품질 30–90 0.3 · 박스 중심 ±8%·크기 0.9–1.15배. `--min-long`(기본 48 —
설계의 문턱) 미만 크롭은 학습·예측 둘 다에서 뺀다. 크롭은 처음에 전부 메모리로 읽는다(1,169개 약 60MB).

**지킬 것.** AI Hub 크롭(`local_only=1`)이 든 세트다 — 결과 폴더·가중치를 PC 밖으로 옮기지 말 것(루트 CLAUDE.md). Claude 는 크롭을
열지 않는다. GPU 는 검수 서버의 작업 잠금 밖에서 쓰니, 학습·자동 라벨링이 도는 중이면 다른 GPU(`--device 1`)를 쓸 것.

  .venv\\Scripts\\python.exe tools\\train_refiner.py --crops C:\\datasets\\vehicle_crops_20261128 --name b1 \\
      --backbone yolo:weights/yolo26m.pt --device 1 [--view 1280] [--no-geom] [--freeze] [--save-final]
"""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))   # tools/ -> 저장소 루트
sys.path.insert(0, str(ROOT / "tools"))

from autolabel.config import resolve  # noqa: E402  (autolabel 을 먼저 — YOLO_AUTOINSTALL=false 기본값)
from autolabel.yolo_io import plain_image_open  # noqa: E402
from build_vehicle_crops import MARKER, VEHICLES  # noqa: E402

GRAY = (114, 114, 114)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VIEW_LONG = 1280
TV_WEIGHTS = {"resnet18": "ResNet18_Weights", "convnext_tiny": "ConvNeXt_Tiny_Weights"}


# ---------------------------------------------------------------- 크롭 읽기·기하

def read_manifest(crops: Path, classes=VEHICLES, min_long: float = 0.0) -> list[dict]:
    """manifest.csv 행 가운데 고른 클래스·크기만. 수 칸은 수로 바꾼다."""
    rows = []
    with open(crops / "manifest.csv", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["cls"] not in classes or float(r["long_px"]) < min_long:
                continue
            for k in ("img_w", "img_h", "crop_x1", "crop_y1", "crop_x2", "crop_y2", "image_id", "box_idx"):
                r[k] = int(r[k])
            for k in ("x1", "y1", "x2", "y2", "long_px", "short_px"):
                r[k] = float(r[k])
            r["y"] = list(classes).index(r["cls"])
            rows.append(r)
    return rows


def box_in_crop(r: dict) -> tuple[float, float, float, float]:
    """크롭 그림 안에서의 박스 (x1, y1, x2, y2) — 저장한 크롭은 박스를 넓혀 자른 것이다."""
    return r["x1"] - r["crop_x1"], r["y1"] - r["crop_y1"], r["x2"] - r["crop_x1"], r["y2"] - r["crop_y1"]


def jitter_box(b, rng: random.Random, shift: float = 0.08, scale: tuple[float, float] = (0.9, 1.15)):
    """검출기 박스의 어긋남을 흉내 — 중심을 박스 크기의 ±shift, 크기를 scale 배."""
    x1, y1, x2, y2 = b
    w, h = x2 - x1, y2 - y1
    cx = (x1 + x2) / 2 + rng.uniform(-shift, shift) * w
    cy = (y1 + y2) / 2 + rng.uniform(-shift, shift) * h
    s = rng.uniform(*scale)
    return cx - w * s / 2, cy - h * s / 2, cx + w * s / 2, cy + h * s / 2


def context_rect(b, pad: float, cw: int, ch: int) -> tuple[int, int, int, int]:
    """박스를 사방으로 pad 만큼 넓혀 크롭 그림 안으로 자른 정수 사각형(1px 이상)."""
    x1, y1, x2, y2 = b
    w, h = x2 - x1, y2 - y1
    a = max(0, math.floor(x1 - pad * w))
    t = max(0, math.floor(y1 - pad * h))
    c = min(cw, math.ceil(x2 + pad * w))
    d = min(ch, math.ceil(y2 + pad * h))
    if c <= a:                      # 흔들린 박스가 크롭 밖으로 나간 때 — 가장자리 1px 이라도
        a = min(a, cw - 1)
        c = a + 1
    if d <= t:
        t = min(t, ch - 1)
        d = t + 1
    return a, t, c, d


def geom_features(b, img_w: int, img_h: int, offset: tuple[float, float]) -> list[float]:
    """[log(w/h), w/W, h/H, cy/H] — b 는 크롭 안 좌표, offset 은 크롭의 그림 안 왼쪽 위."""
    x1, y1, x2, y2 = b
    w, h = max(x2 - x1, 1e-3), max(y2 - y1, 1e-3)
    cy = (y1 + y2) / 2 + offset[1]
    return [math.log(w / h), w / img_w, h / img_h, cy / img_h]


def view_factor(r: dict, view: str) -> float:
    """'1280' 이면 검출기가 1280 입력에서 본 배율(그림 긴 변 1280 — 이미 작으면 1)."""
    if view == "native":
        return 1.0
    return min(1.0, VIEW_LONG / max(r["img_w"], r["img_h"]))


def letterbox(im, size: int):
    """가로세로비를 지켜 size×size 에 넣는다(빈칸 회색 114)."""
    from PIL import Image

    w, h = im.size
    s = size / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    im = im.resize((nw, nh), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), GRAY)
    canvas.paste(im, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def make_sample(im, r: dict, *, imgsz: int, pad: float, view: str, train: bool, rng: random.Random):
    """(PIL 그림 imgsz×imgsz, 기하 4개). train 이면 박스 흔들기·색·흐림·JPEG·뒤집기."""
    from PIL import Image, ImageFilter, ImageOps

    b = box_in_crop(r)
    if train:
        b = jitter_box(b, rng)
    a, t, c, d = context_rect(b, pad, *im.size)
    region = im.crop((a, t, c, d))
    f = view_factor(r, view)
    if f < 1.0:
        region = region.resize((max(1, round(region.width * f)), max(1, round(region.height * f))), Image.Resampling.BOX)
    if train:
        region = color_jitter(region, rng)
        if rng.random() < 0.1:
            region = ImageOps.grayscale(region).convert("RGB")
        if rng.random() < 0.2:
            region = region.filter(ImageFilter.GaussianBlur(rng.uniform(0.1, 1.5)))
        if rng.random() < 0.3:
            buf = io.BytesIO()
            region.save(buf, "JPEG", quality=rng.randint(30, 90))
            buf.seek(0)
            with plain_image_open()(buf) as j:
                region = j.convert("RGB")
    out = letterbox(region, imgsz)
    if train and rng.random() < 0.5:
        out = ImageOps.mirror(out)
    return out, geom_features(b, r["img_w"], r["img_h"], (r["crop_x1"], r["crop_y1"]))


def color_jitter(im, rng: random.Random):
    """밝기·대비·채도 0.7–1.3배, 색조 ±0.02 바퀴."""
    from PIL import Image, ImageEnhance

    for enh in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        im = enh(im).enhance(rng.uniform(0.7, 1.3))
    shift = round(rng.uniform(-0.02, 0.02) * 255)
    if shift:
        h, s, v = im.convert("HSV").split()
        h = h.point(lambda p: (p + shift) % 256)
        im = Image.merge("HSV", (h, s, v)).convert("RGB")
    return im


def load_images(crops: Path, rows: list[dict]) -> dict[str, object]:
    out = {}
    for r in rows:
        with plain_image_open()(crops / r["crop"]) as im:
            out[r["crop"]] = im.convert("RGB")
    return out


# ---------------------------------------------------------------- 백본·모델

def tv_cached_file(name: str) -> Path:
    import torch
    import torchvision

    url = getattr(torchvision.models, TV_WEIGHTS[name]).IMAGENET1K_V1.url
    return Path(torch.hub.get_dir()) / "checkpoints" / os.path.basename(url)


def make_backbone(spec: str):
    """(nn.Module: B×3×H×W → B×C×h×w, C, 입력 맞춤 'unit'|'imagenet')."""
    import torch
    from torch import nn

    if spec == "tinycnn":
        net = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.BatchNorm2d(16), nn.SiLU(),
                            nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.SiLU(),
                            nn.Conv2d(32, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.SiLU())
        return net, 32, "unit"
    if spec.startswith("yolo:"):
        from ultralytics import YOLO

        path = resolve(spec[5:])
        if not path.exists():
            raise SystemExit(f"[중단] 가중치가 없습니다: {path}")
        det = YOLO(str(path)).model
        layers = list(det.model[:11])
        if any(getattr(m, "f", -1) != -1 for m in layers):
            raise SystemExit("[중단] 층 0–10 이 차례로 이어지지 않는 구조입니다(yolo26 계열만 확인했다)")
        net = copy.deepcopy(nn.Sequential(*layers)).float()
        for p in net.parameters():
            p.requires_grad_(True)
        with torch.no_grad():
            c = net.eval()(torch.zeros(1, 3, 64, 64)).shape[1]
        return net, int(c), "unit"
    if spec.startswith("tv:"):
        import torchvision

        name = spec[3:]
        if name not in TV_WEIGHTS:
            raise SystemExit(f"[중단] 모르는 torchvision 백본: {name} (쓸 수 있는 것: {', '.join(TV_WEIGHTS)})")
        f = tv_cached_file(name)
        if not f.exists():
            raise SystemExit(f"[중단] {name} 의 ImageNet 가중치가 torch 캐시에 없습니다({f}). 이 도구는 내려받지 않는다 — "
                             "사용자 허락을 받고 따로 받을 것")
        model = getattr(torchvision.models, name)(weights=None)
        model.load_state_dict(torch.load(f, map_location="cpu", weights_only=True))
        if name == "resnet18":
            return nn.Sequential(*list(model.children())[:-2]), 512, "imagenet"
        return model.features, 768, "imagenet"
    raise SystemExit(f"[중단] 백본 꼴을 모릅니다: {spec} (yolo:<가중치> · tv:resnet18 · tv:convnext_tiny · tinycnn)")


def build_model(spec: str, n_cls: int, geom: bool, dropout: float = 0.2):
    import torch
    from torch import nn

    backbone, c, norm = make_backbone(spec)

    class Refiner(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.geom = nn.Sequential(nn.Linear(4, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU()) if geom else None
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(c + (32 if geom else 0), n_cls))
            mean, std = (IMAGENET_MEAN, IMAGENET_STD) if norm == "imagenet" else ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
            self.register_buffer("px_mean", torch.tensor(mean).view(1, 3, 1, 1))
            self.register_buffer("px_std", torch.tensor(std).view(1, 3, 1, 1))
            self.register_buffer("g_mean", torch.zeros(4))
            self.register_buffer("g_std", torch.ones(4))

        def forward(self, x, g):
            f = self.backbone((x - self.px_mean) / self.px_std).mean((2, 3))
            if self.geom is not None:
                f = torch.cat([f, self.geom((g - self.g_mean) / self.g_std)], 1)
            return self.head(f)

    return Refiner()


# ---------------------------------------------------------------- 학습·예측

def class_weights(ys: list[int], n_cls: int) -> list[float]:
    """클래스 빈도의 제곱근 역수, 평균 1 로(없는 클래스는 0)."""
    n = [ys.count(c) for c in range(n_cls)]
    w = [1 / math.sqrt(k) if k else 0.0 for k in n]
    m = sum(w[c] * n[c] for c in range(n_cls)) / max(1, len(ys))
    return [x / m if m else 0.0 for x in w]


def to_batch(samples):
    import numpy as np
    import torch

    x = torch.from_numpy(np.stack([np.asarray(s[0], dtype=np.uint8) for s in samples])).permute(0, 3, 1, 2).float() / 255
    g = torch.tensor([s[1] for s in samples], dtype=torch.float32)
    return x, g


def train_one(rows: list[dict], images: dict, args, seed: int, device, log) -> object:
    """rows 로 모델 하나를 학습해 돌려준다(정해 둔 epoch 끝까지)."""
    import torch
    from torch import nn

    rng = random.Random(seed)
    torch.manual_seed(seed)
    n_cls = len(args.classes)
    model = build_model(args.backbone, n_cls, args.geom).to(device)
    if args.geom:
        g = torch.tensor([geom_features(box_in_crop(r), r["img_w"], r["img_h"], (r["crop_x1"], r["crop_y1"]))
                          for r in rows])
        model.g_mean.copy_(g.mean(0))
        model.g_std.copy_(g.std(0).clamp_min(1e-6))
    if args.freeze:
        for p in model.backbone.parameters():
            p.requires_grad_(False)
    groups = [{"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": args.lr},
              {"params": [p for n, p in model.named_parameters() if not n.startswith("backbone.")], "lr": args.head_lr}]
    opt = torch.optim.AdamW([gr for gr in groups if gr["params"]], weight_decay=args.wd)
    ys = [r["y"] for r in rows]
    cw = class_weights(ys, n_cls)
    loss_w = torch.tensor(cw, device=device) if args.balance == "loss" else None
    crit = nn.CrossEntropyLoss(weight=loss_w, label_smoothing=args.smooth)
    steps_per_epoch = max(1, math.ceil(len(rows) / args.batch))
    total, warm = args.epochs * steps_per_epoch, 2 * steps_per_epoch

    def lr_at(step):
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    amp = device.type == "cuda"
    step = 0
    for epoch in range(args.epochs):
        model.train()
        if args.freeze:
            model.backbone.eval()
        if args.balance == "sampler":
            order = rng.choices(range(len(rows)), weights=[cw[y] for y in ys], k=len(rows))
        else:
            order = list(range(len(rows)))
            rng.shuffle(order)
        t0, run_loss = time.time(), 0.0
        for i in range(0, len(order), args.batch):
            idx = order[i:i + args.batch]
            x, g = to_batch([make_sample(images[rows[j]["crop"]], rows[j], imgsz=args.imgsz, pad=args.pad,
                                         view=args.view, train=True, rng=rng) for j in idx])
            y = torch.tensor([ys[j] for j in idx], device=device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                out = model(x.to(device), g.to(device))
                loss = crit(out.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            run_loss += float(loss) * len(idx)
        log({"epoch": epoch + 1, "loss": round(run_loss / len(order), 4), "sec": round(time.time() - t0, 1)})
    return model


def predict(model, rows: list[dict], images: dict, args, device) -> list[list[float]]:
    """rows 의 logits(학습 증강 없이)."""
    import torch

    model.eval()
    out = []
    amp = device.type == "cuda"
    with torch.no_grad():
        for i in range(0, len(rows), args.batch):
            part = rows[i:i + args.batch]
            x, g = to_batch([make_sample(images[r["crop"]], r, imgsz=args.imgsz, pad=args.pad, view=args.view,
                                         train=False, rng=random.Random(0)) for r in part])
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                z = model(x.to(device), g.to(device)).float()
            out += z.cpu().tolist()
    return out


def write_preds(path: Path, rows: list[dict], logits: list[list[float]], classes) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["crop", "image_id", "group", "fold", "cls", "pred", "long_px", "local_only"] +
                    [f"z_{c}" for c in classes])
        for r, z in zip(rows, logits):
            wr.writerow([r["crop"], r["image_id"], r["group"], r["fold"], r["cls"], classes[max(range(len(z)), key=z.__getitem__)],
                         r["long_px"], r["local_only"]] + [round(v, 5) for v in z])


def run(args) -> Path:
    import torch

    crops = Path(args.crops).resolve()
    info = json.loads((crops / MARKER).read_text(encoding="utf-8"))
    if args.pad > float(info.get("pad", 0)) + 1e-9:
        raise SystemExit(f"[중단] --pad {args.pad} 가 크롭을 만들 때의 넓히기({info.get('pad')})보다 크다 — 크롭에 그만큼의 주변이 "
                         "없다. build_vehicle_crops.py --pad 를 키워 다시 만들 것")
    rows = read_manifest(crops, args.classes, args.min_long)
    cv = [r for r in rows if r["fold"] != "test"]
    test = [r for r in rows if r["fold"] == "test"]
    folds = sorted({int(r["fold"]) for r in cv})
    pick = folds if args.folds == "all" else [int(x) for x in args.folds.split(",")]
    out = crops / "runs" / args.name
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"[중단] {out} 가 이미 있습니다 — 다른 --name 을 주거나 --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.device}" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    cfg = {k: (v if not isinstance(v, Path) else str(v)) for k, v in vars(args).items()}
    cfg.update({"crops_made_at": info.get("made_at"), "started": datetime.now().isoformat(timespec="seconds"),
                "device": str(device), "n_cv": len(cv), "n_test": len(test), "folds_run": pick})
    (out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    images = load_images(crops, rows)
    logf = open(out / "log.jsonl", "a", encoding="utf-8")

    def logger(tag):
        def log(rec):
            rec = {"part": tag, **rec}
            logf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logf.flush()
            if not args.quiet:
                print(json.dumps(rec, ensure_ascii=False), flush=True)
        return log

    oof_rows, oof_z = [], []
    t0 = time.time()
    for f in pick:
        tr = [r for r in cv if int(r["fold"]) != f]
        te = [r for r in cv if int(r["fold"]) == f]
        model = train_one(tr, images, args, args.seed * 100 + f, device, logger(f"fold{f}"))
        oof_rows += te
        oof_z += predict(model, te, images, args, device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_preds(out / "oof.csv", oof_rows, oof_z, args.classes)
    if args.save_final:
        model = train_one(cv, images, args, args.seed * 100 + 99, device, logger("final"))
        torch.save({"state_dict": model.state_dict(), "config": cfg, "classes": list(args.classes)}, out / "final.pt")
        if test:
            write_preds(out / "test_preds.csv", test, predict(model, test, images, args, device), args.classes)
    logf.close()
    cfg["minutes"] = round((time.time() - t0) / 60, 2)
    (out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"끝: {out} (OOF {len(oof_rows)}개, {cfg['minutes']}분) — 지표는 tools/eval_refiner.py report --crops {crops} --runs {args.name}")
    return out


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--crops", required=True, type=Path, help="build_vehicle_crops.py 가 쓴 폴더")
    ap.add_argument("--name", required=True, help="결과 폴더 이름(<crops>/runs/<이름>)")
    ap.add_argument("--backbone", default="yolo:weights/yolo26m.pt")
    ap.add_argument("--classes", nargs="+", default=list(VEHICLES))
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--pad", type=float, default=0.15, help="박스 크기 대비 주변(크롭을 만들 때의 값 이하)")
    ap.add_argument("--view", choices=["native", "1280"], default="native")
    ap.add_argument("--geom", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--balance", choices=["sampler", "loss", "none"], default="sampler")
    ap.add_argument("--min-long", type=float, default=48.0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--head-lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--smooth", type=float, default=0.05)
    ap.add_argument("--folds", default="all", help="all 또는 0,2 처럼")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="0", help="GPU 번호 또는 cpu")
    ap.add_argument("--save-final", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(parser().parse_args())
    return 0


if __name__ == "__main__":
    sys.exit(main())
