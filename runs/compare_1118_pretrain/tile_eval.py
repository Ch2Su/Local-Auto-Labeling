"""타일 모드(운영 '자동 라벨링 (타일)')가 같은 차에 겹쳐 낸 박스(중복)를 얼마나 줄이는지 잰다 — 추론만.

검수 라벨 갈래(FZ11R)의 짝 없는 responder 예측 가운데 절반 넘게가, 같은 차를 이미 car·truck 으로 찾은 자리에 겹친 중복이었다
(responder_sheet.py). 타일 모드는 차량 클래스끼리 IoU 0.6 이상 겹치면 신뢰도 높은 쪽만 남긴다(tiling.merge_cross 의
class_iou — 11-17). 그 병합의 몫과 타일 추론이 예측 자체를 바꾸는 몫을 가르려고, 25 epoch 의 last.pt 를 다섯 방식으로 돌린다:
  P   통째 — 지금까지의 평가와 같다(cmp_common.count: 경로 8장씩 model.predict, conf 0.25 · iou 0.6 · max_det 300).
  PN  P 의 박스에 merge_cross 의 같은 클래스 NMS(iou 0.6)만 — 병합 함수 자체의 몫(YOLO26 은 NMS 없이 내므로 P 와 다를 수 있다).
  PC  P 의 박스에 merge_cross + 차량 클래스끼리 병합(class_iou) — 같은 박스에서 병합만의 몫.
  T0  타일, 차량 클래스 병합 없음(class_iou 0 — 11-17 전의 타일 모드).
  T   타일 — 운영 그대로(tiling.detect 와 같은 순서·기준. 설정은 저장소 config.yaml 의 predict 와 predict.tiles).
T0·T 는 성분(2×2 타일 + 통째) 추론을 한 번 하고 병합만 두 번 한다 — 첫 모델의 앞 몇 장으로 tiling.detect 와 같은지 견준다.
max_det 는 다섯 다 300 이다(평가와 같게 — 운영 predict.max_det 는 100 이라 한 장의 예측 수 최댓값도 센다).

짝짓기는 cmp_common.count 와 같다(conf 순 탐욕 1:1, IoU 0.5, 클래스 무관 — 클래스가 맞는지는 따로 센다). 짝 없는 예측의 갈래는
responder_sheet.py 와 같다: 중복 = 가장 많이 겹친 정답(IoU ≥ 0.5)이 이미 다른(신뢰도 높은) 예측과 짝이 됨 · 일부 = 0.1–0.5 · 없음.
P 의 찾음·클래스까지·헛 박스는 results.jsonl(학습 때 같은 가중치로 한 같은 평가)과 견준다. 출력에 파일 이름·카메라 키는 없다.
  python -I tile_eval.py            # 결과: tile_eval.jsonl (세트·방식마다 한 줄)
"""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cmp_common as cc  # noqa: E402

sys.path.insert(0, str(cc.REPO / "src"))
from autolabel import tiling  # noqa: E402

ARMS = ("FZ11", "FZ11R", "FZ11U")
ARM_NAME = {"FZ11": "기준", "FZ11R": "R(검수 라벨)", "FZ11U": "U(검수 전 라벨)", "operating": "운영 8클래스"}
OPERATING = cc.REPO / "models" / "best_y26m_8cls.pt"   # --arms operating — 지금 자동 라벨링 가중치(responder·cone 모름, seed 하나)
SEEDS = (0, 1, 2)
MODES = ("P", "PN", "PC", "T0", "T")
MODE_NAME = {"P": "통째", "PN": "통째+같은 클래스 NMS", "PC": "통째+차량 병합", "T0": "타일(병합 없음)", "T": "타일(운영)"}
VEH = set(tiling.VEHICLES)                 # car truck bus pickup responder
CAR4 = VEH - {"responder"}
MAXDET = 300


def settings():
    """저장소 config.yaml 의 predict(conf·iou·imgsz)와 predict.tiles 를 tiling.settings 로 읽는다(운영과 같은 해석)."""
    import yaml
    raw = yaml.safe_load((cc.REPO / "config.yaml").read_text(encoding="utf-8"))["predict"]

    class Cfg:
        @staticmethod
        def get(key):
            return raw.get("tiles") if key == "predict.tiles" else None

    return tiling.settings(Cfg()), float(raw["conf"]), float(raw["iou"]), int(raw["imgsz"])


def weights(arm: str, fold: int, seed: int) -> Path:
    if arm == "operating":
        return OPERATING
    return cc.ROOT / "runs" / (f"{arm}_f{fold}_e25" + (f"_s{seed}" if seed else "")) / "weights" / "last.pt"


def predict_plain(model, files, imgsz, conf, iou):
    """cmp_common.count 와 같은 호출(경로 8장씩) — 파일마다 열 8개(x1 y1 x2 y2 conf cls 0 0)와 원본 크기, 걸린 초."""
    out, dt = {}, 0.0
    for i in range(0, len(files), 8):
        chunk = files[i:i + 8]
        t0 = time.perf_counter()
        res = model.predict([str(f) for f in chunk], imgsz=imgsz, conf=conf, iou=iou, max_det=MAXDET, device=0, verbose=False)
        dt += time.perf_counter() - t0
        for f, r in zip(chunk, res):
            d = r.boxes.data.cpu().numpy().astype(np.float32)[:, :6].reshape(-1, 6)
            out[f] = (np.concatenate([d, np.zeros((len(d), 2), np.float32)], 1), r.orig_shape)
    return out, dt


def tile_parts(model, img, s, imgsz):
    """tiling.detect 의 앞부분 그대로 — 성분 검출을 이어 붙이고(타일 먼저) 잘린 타일 박스를 버린 것(열 8개)."""
    h, w = img.shape[:2]
    comps = [f"g{s['grid']}", "full"] if max(w, h) > s["min_side"] else ["full"]
    parts = []
    for c in comps:
        d, _ = tiling.run_component(model, img, c, overlap=s["overlap"], edge_px=s["edge_px"], imgsz=imgsz, device=0)
        parts.append(np.concatenate([d, np.full((len(d), 1), c != "full", np.float32)], 1))
    d = np.concatenate(parts)
    return d[d[:, 6] == 0]


def tally(t: Counter, gts, xy, cf, names, scale: float) -> int:
    """한 장을 짝짓고 t 에 더한다. scale = 입력 크기로 줄이는 배율(정답 크기 칸). 돌려주는 값은 그 장의 예측 수."""
    n = len(cf)
    t["images"] += 1
    t["pred"] += n
    gt_xy = np.array([g[1:] for g in gts], dtype=float).reshape(-1, 4)
    used = np.zeros(len(gts), dtype=bool)
    hit_by = {}                                      # 정답 번호 -> 짝이 된 예측 클래스
    bins = [cc.size_bin(min(g[3] - g[1], g[4] - g[2]) * scale) for g in gts]
    for g, sb in zip(gts, bins):
        t["total"] += 1
        t["total:" + g[0]] += 1
        t["size_total:" + sb] += 1
    unmatched = []
    for j in np.argsort(-cf):
        name = names[j]
        if len(gts):
            v = cc.iou_one(xy[j], gt_xy)
            v[used] = -1
            k = int(v.argmax())
            if v[k] >= cc.MATCH_IOU:
                used[k] = True
                hit_by[k] = name
                gc = gts[k][0]
                t["found"] += 1
                t["found:" + gc] += 1
                t["size_found:" + bins[k]] += 1
                if gc == name:
                    t["klass"] += 1
                    t["klass:" + gc] += 1
                elif gc in CAR4 and name == "responder":
                    t["veh2resp"] += 1
                elif gc == "responder" and name in CAR4:
                    t["resp2veh"] += 1
                continue
        unmatched.append(j)
    for j in unmatched:
        name = names[j]
        t["false"] += 1
        t["false:" + name] += 1
        if len(gts):
            ious = cc.iou_one(xy[j], gt_xy)
            k = int(ious.argmax())
            best = float(ious[k])
        else:
            k, best = None, 0.0
        kind = "dup" if best >= 0.5 else "part" if best >= 0.1 else "none"
        t[kind] += 1
        t[f"{kind}:{name}"] += 1
        if kind != "dup":
            continue
        first, gc = hit_by.get(k), gts[k][0]
        if first is None:
            t["dup_unhit"] += 1                      # 생기면 안 된다(짝 없는데 IoU ≥ 0.5 인 정답이 비어 있음)
        elif first == name:
            t["dup_same"] += 1
        elif first in VEH and name in VEH:
            t["dup_vcross"] += 1
            if name == "responder":
                t["dup_resp_on_resp" if gc == "responder" else "dup_resp_on_veh"] += 1
            elif first == "responder":
                t["dup_veh_after_resp"] += 1
        else:
            t["dup_ocross"] += 1
    # 정답 없이 센 것: 예측끼리 IoU ≥ 0.5 인 쌍 — 차량끼리 다른 클래스 / 같은 클래스
    vi = [j for j in range(n) if names[j] in VEH]
    for a in range(len(vi) - 1):
        rest = vi[a + 1:]
        v = cc.iou_one(xy[vi[a]], xy[rest])
        for b, val in zip(rest, v):
            if val >= 0.5:
                t["pair_same" if names[b] == names[vi[a]] else "pair_vcross"] += 1
    return n


def mean_rng(vals):
    return f"{np.mean(vals):.1f} ({min(vals)}–{max(vals)})" if len(vals) > 1 else f"{vals[0]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--check", type=int, default=3, help="첫 모델에서 tiling.detect 와 견줄 장 수")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--plain-imgsz", type=int, default=0,
                    help="P·PN·PC 의 입력 크기(기본 config 의 predict.imgsz) — 타일은 그대로 조각마다 config 값")
    ap.add_argument("--out", default="tile_eval.jsonl")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    arm_seeds = {a: [0] if a == "operating" else seeds for a in args.arms.split(",")}
    from ultralytics import YOLO
    from ultralytics.utils.patches import imread

    s, conf, iou, imgsz = settings()
    if conf != cc.CONF:
        raise SystemExit(f"config 의 predict.conf {conf} 가 평가의 {cc.CONF} 와 다르다")
    plain_imgsz = args.plain_imgsz or imgsz
    print(f"설정: conf {conf} · iou {iou} · imgsz {imgsz} (통째 {plain_imgsz}) · max_det {MAXDET} · 타일 {s}", flush=True)
    ref = {}
    for line in (cc.ROOT / "results.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            ref[r["name"]] = r.get("last")
    tallies = defaultdict(Counter)                   # (arm, seed, mode) -> Counter
    maxpred = defaultdict(int)
    secs = Counter()
    checked = False
    for arm in args.arms.split(","):
        for seed in arm_seeds[arm]:
            for fold in (0, 1, 2):
                name = f"{arm}_f{fold}_e25" + (f"_s{seed}" if seed else "")
                files = cc.fold_files(fold, "val")
                model = YOLO(str(weights(arm, fold, seed)))
                names = model.names
                vids = tiling.vehicle_ids(names)
                plain, dt = predict_plain(model, files, plain_imgsz, conf, iou)
                secs["P"] += dt
                fold_t = {m: Counter() for m in MODES}
                for n, f in enumerate(files):
                    dP, (h, w) = plain[f]
                    gts = cc.load_gt(f, w, h)
                    t0 = time.perf_counter()
                    img = imread(str(f))
                    d = tile_parts(model, img, s, imgsz)
                    dT0 = tiling.merge_cross(d, iou, s["cross_iou"], MAXDET, conf)
                    dT = tiling.merge_cross(d, iou, s["cross_iou"], MAXDET, conf, group=vids, iou_class=s["class_iou"])
                    secs["T"] += time.perf_counter() - t0
                    if not checked and n < args.check:     # 옮긴 순서가 운영 함수와 같은지
                        ref_d, _ = tiling.detect(model, img, imgsz=imgsz, device=0, conf=conf, iou=iou, max_det=MAXDET, **s)
                        same = ref_d.shape == dT.shape and np.allclose(ref_d, dT, atol=1e-3)
                        print(f"  tiling.detect 대조 {n + 1}: 박스 {len(ref_d)} / {len(dT)} · {'같음' if same else '다름'}", flush=True)
                        if not same:
                            raise SystemExit("tiling.detect 와 다르다 — 옮긴 순서를 볼 것")
                    outs = {"P": dP,
                            "PN": tiling.merge_cross(dP, iou, s["cross_iou"], MAXDET, conf),
                            "PC": tiling.merge_cross(dP, iou, s["cross_iou"], MAXDET, conf, group=vids, iou_class=s["class_iou"]),
                            "T0": dT0, "T": dT}
                    for m, dd in outs.items():
                        k = tally(fold_t[m], gts, dd[:, :4].astype(float), dd[:, 4], [names[int(c)] for c in dd[:, 5]],
                                  imgsz / max(w, h))
                        maxpred[(arm, seed, m)] = max(maxpred[(arm, seed, m)], k)
                secs["images"] += len(files)
                checked = True
                last = ref.get(name)
                if last:
                    want = (sum(last["found"].values()), sum(last["klass"].values()), sum(last["false"].values()))
                    got = (fold_t["P"]["found"], fold_t["P"]["klass"], fold_t["P"]["false"])
                    note = "같음" if want == got else f"다름 — 학습 때 {want} · 지금 {got}"
                else:
                    note = "학습 때 기록 없음"
                p, t = fold_t["P"], fold_t["T"]
                print(f"{name}: 통째 찾음 {p['found']} · 클래스까지 {p['klass']} · 헛 {p['false']} (중복 {p['dup']}) → 타일 "
                      f"{t['found']} · {t['klass']} · 헛 {t['false']} (중복 {t['dup']}) · 통째 재현 {note}", flush=True)
                for m in MODES:
                    tallies[(arm, seed, m)].update(fold_t[m])
    with open(cc.ROOT / args.out, "a", encoding="utf-8") as fp:
        for (arm, seed, m), t in tallies.items():
            fp.write(json.dumps({"arm": arm, "seed": seed, "mode": m, "plain_imgsz": plain_imgsz,
                                 "maxpred": maxpred[(arm, seed, m)], **t},
                                ensure_ascii=False) + "\n")
    n_img = secs["images"] or 1
    print(f"\n추론 시간(장당, 모델 올리기 제외): 통째 {secs['P'] / n_img * 1000:.0f}ms · 타일(성분 추론+병합 둘) "
          f"{secs['T'] / n_img * 1000:.0f}ms")
    print("한 장의 예측 수 최댓값:", {m: max(maxpred[(a, sd, m)] for a in args.arms.split(",") for sd in arm_seeds[a]) for m in MODES})

    def row(arm, m, keys):
        cells = []
        for k in keys:
            vals = [tallies[(arm, sd, m)][k] for sd in arm_seeds[arm]]
            cells.append(mean_rng(vals))
        return f"| {ARM_NAME[arm]} | {MODE_NAME[m]} | " + " | ".join(cells) + " |"

    arms = args.arms.split(",")
    print("\n## 중복 — seed 셋 평균 (가장 낮은 값–높은 값), 3묶음 합")
    keys = ["dup", "dup_vcross", "dup_same", "dup:responder", "dup_resp_on_resp", "dup_resp_on_veh", "pair_vcross"]
    print("| 갈래 | 방식 | 중복 전체 | 차량끼리 다른 클래스 | 같은 클래스 | 중복 responder | ─ 정답 responder | ─ 정답 차량 | 예측끼리 차량 다른 클래스 쌍 |")
    print("|---|---|" + "---|" * len(keys))
    for arm in arms:
        for m in MODES:
            print(row(arm, m, keys))
    print("\n## 찾음·클래스·헛 박스 — seed 셋 평균 (가장 낮은 값–높은 값), 3묶음 합")
    keys = ["found", "klass", "false", "klass:responder", "found:responder", "veh2resp", "resp2veh", "false:responder", "none", "part"]
    print("| 갈래 | 방식 | 찾음 | 클래스까지 | 헛 박스 | responder 클래스까지 | responder 찾음 | 차량→responder | responder→차량 | 짝 없는 responder | 헛: 정답 없음 | 헛: 일부만 |")
    print("|---|---|" + "---|" * len(keys))
    for arm in arms:
        for m in MODES:
            print(row(arm, m, keys))
    tot = tallies[(arms[0], 0, "P")]
    print("\n## 정답 크기(입력 640 으로 줄였을 때 짧은 변)별 찾음 — seed 셋 평균")
    print("| 갈래 | 방식 | " + " | ".join(f"{b}px (정답 {tot['size_total:' + b]})" for b in cc.SIZE_BINS) + " |")
    print("|---|---|" + "---|" * len(cc.SIZE_BINS))
    for arm in arms:
        for m in ("P", "T"):
            print(row(arm, m, [f"size_found:{b}" for b in cc.SIZE_BINS]))
    print(f"\n정답: {tot['total']}개 (responder {tot['total:responder']}) · 이미지 {tot['images']}장")
    bad = sum(t["dup_unhit"] for t in tallies.values())
    if bad:
        print(f"[경고] 짝 없는데 IoU ≥ 0.5 인 정답이 비어 있던 예측 {bad}개 — 짝짓기를 볼 것")
    return 0


if __name__ == "__main__":
    sys.exit(main())
