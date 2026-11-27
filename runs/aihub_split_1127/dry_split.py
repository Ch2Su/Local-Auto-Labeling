"""고정을 푼 등록 파일로 첫 분할을 DB 사본에서 미리 돌린다 — 같은 입력으로 두 번 돌려 같은지 보고, 학습 세트를 만들고,
AI Hub 그림의 확인 시트(평가셋으로 갈 것 / train)를 data/visible/aihub_pm/ 에 그린다. 실제 DB 는 mode=ro 로 읽기만."""
import collections, json, re, sqlite3, sys
from datetime import datetime
from pathlib import Path
REPO = Path(r"C:\local_auto_labeling"); W = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
from PIL import Image, ImageDraw, ImageFont, ImageOps
from autolabel.config import Config
from autolabel.store import Store
from autolabel.split import ensure_splits
from autolabel.trainer import build_dataset
from autolabel.yolo_io import plain_image_open

def copy(dst):
    a = sqlite3.connect(f"file:{(REPO / 'data/visible/labeling.db').as_posix()}?mode=ro", uri=True); b = sqlite3.connect(dst)
    a.backup(b); b.close(); a.close()

def cfg_for(tag, db):
    t = (REPO / "config.yaml").read_text(encoding="utf-8")
    t, n1 = re.subn(r"(?m)^  db: .*$", "  db: " + db.as_posix(), t, count=1)
    t, n2 = re.subn(r"(?m)^  dataset: data/visible/dataset$", "  dataset: " + (W / f"ds_{tag}").as_posix(), t, count=1)
    t, n3 = re.subn(r"(?m)^  keep: \d+", "  keep: 0", t, count=1)
    assert (n1, n2, n3) == (1, 1, 1)
    p = W / f"config_{tag}.yaml"; p.write_text(t, encoding="utf-8")
    c = Config.load(p); assert Path(c.db) == db
    return c

db1, db2 = W / "dry1.db", W / "dry2.db"
copy(db1); copy(db2)
c1, c2 = cfg_for("n1280", db1), cfg_for("dry2", db2)
a2, _ = ensure_splits(c2, Store(db2))
rep = build_dataset(c1, Store(db1))
print(rep.summary().split("\n  카메라별")[0])
con = sqlite3.connect(db1); con.row_factory = sqlite3.Row
rows = [dict(r) for r in con.execute("SELECT id, path, camera, split, status FROM images WHERE status='approved' ORDER BY id")]
a1 = {r["id"]: r["split"] for r in rows}
print("두 번 돌린 분할이 같은가:", a1 == {i: s for i, s in a2.items()}, len(a1))
ah = [r for r in rows if (r["camera"] or "").startswith("ah-")]
own = collections.Counter(r["split"] for r in rows if not (r["camera"] or "").startswith("ah-"))
print("AI Hub:", dict(collections.Counter(r["split"] for r in ah)), "직접 모은 것:", dict(own))
json.dump({str(k): v for k, v in a1.items()}, open(W / "dry_split.json", "w"), indent=0)

out = REPO / "data/visible/aihub_pm"; out.mkdir(parents=True, exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
try:
    font = ImageFont.truetype("malgun.ttf", 16)
except OSError:
    font = ImageFont.load_default()
def sheet(items, title, path, cols=5, tw=384, th=216):
    rows_n = (len(items) + cols - 1) // cols
    img = Image.new("RGB", (cols * tw, 36 + rows_n * (th + 24)), (24, 24, 24))
    d = ImageDraw.Draw(img); d.text((8, 8), title, fill="white", font=font)
    for k, r in enumerate(items):
        x, y = (k % cols) * tw, 36 + (k // cols) * (th + 24)
        with plain_image_open()(r["path"]) as im:
            t = ImageOps.exif_transpose(im).convert("RGB"); t.thumbnail((tw - 4, th - 4))
        img.paste(t, (x + 2, y + 2))
        d.text((x + 4, y + th + 2), f"#{r['id']} {r['camera']} [{r['split']}]", fill="yellow" if r["split"] != "train" else "white", font=font)
    img.save(path, quality=88)
    return path
ev = [r for r in ah if r["split"] in ("val", "test")]
tr = [r for r in ah if r["split"] == "train"]
p1 = sheet(ev, f"AI Hub -> val/test {len(ev)}장 (이 장소가 아래 train 시트에도 있으면 알려 주세요)", out / f"split_check_{stamp}_eval.jpg")
p2 = sheet(tr, f"AI Hub -> train {len(tr)}장", out / f"split_check_{stamp}_train.jpg", cols=8, tw=256, th=144)
print("시트:", p1, p2)
