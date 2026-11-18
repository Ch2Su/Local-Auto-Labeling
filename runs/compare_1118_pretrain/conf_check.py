"""문턱(conf)을 바꿔도 순서가 같은가 — 미세조정한 A 가 COCO 그대로보다 덜 찾는 것이 conf 0.25 탓인지 본다(추론만)."""
import json
import sys
from collections import Counter

import cmp_common as cc


def main() -> int:
    from ultralytics import YOLO

    rows = []
    specs = [("coco_yolo26m", None, cc.COCO_OK), ("operating_m_8cls", None, cc.SELF_OK)]
    specs += [(f"{arm}_e100_last", arm, cc.SELF_OK) for arm in "ABC"]
    for conf in (0.25, 0.10):
        cc.CONF = conf
        for name, arm, ok in specs:
            tot = Counter(); found = Counter(); klass = Counter(); false = 0
            for fold in (0, 1, 2):
                if arm is None:
                    path = cc.REPO / ("weights/yolo26m.pt" if name.startswith("coco") else "models/best_y26m_8cls.pt")
                else:
                    path = cc.ROOT / "runs" / f"{arm}_f{fold}_e100" / "weights" / "last.pt"
                c = cc.count(YOLO(str(path)), cc.fold_files(fold, "val"), ok, 640, batch=8)
                tot.update(c["total"]); found.update(c["found"]); klass.update(c["klass"]); false += sum(c["false"].values())
            n = sum(tot.values())
            rec = {"name": name, "conf": conf, "imgsz": 640, "total": n, "found": sum(found.values()),
                   "klass": sum(klass.values()), "false": false,
                   "per_class_found": {k: f"{found[k]}/{tot[k]}" for k in cc.NAMES if tot[k]}}
            rows.append(rec)
            print(f"conf {conf} {name}: 찾음 {rec['found']}/{n} ({rec['found'] / n:.1%}) · 클래스까지 {rec['klass']} · 헛 박스 {false} · "
                  + ", ".join(f"{k} {v}" for k, v in rec["per_class_found"].items()), flush=True)
    with open(cc.ROOT / "conf_check.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
