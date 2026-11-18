"""학습 없이 재는 기준선 — COCO 로만 학습한 yolo26m 과 운영 가중치를 같은 175장(fold 별 평가 몫)에 같은 방식으로 센다."""
import json
import sys

import cmp_common as cc


def main() -> int:
    from ultralytics import YOLO

    models = [("coco_yolo26m", cc.REPO / "weights" / "yolo26m.pt", cc.COCO_OK),
              ("operating_m_8cls", cc.REPO / "models" / "best_y26m_8cls.pt", cc.SELF_OK)]
    out = cc.ROOT / "baseline.jsonl"
    out.unlink(missing_ok=True)
    for name, path, ok in models:
        m = YOLO(str(path))
        for sz in (640, 1280):
            for fold in (0, 1, 2):
                c = cc.count(m, cc.fold_files(fold, "val"), ok, sz, batch=4)
                with open(out, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"name": name, "imgsz": sz, "fold": fold, **c}, ensure_ascii=False) + "\n")
                print(f"{name} imgsz {sz} fold {fold}: {cc.brief(c)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
