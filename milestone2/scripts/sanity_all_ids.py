import json, re, random, csv
from pathlib import Path
import numpy as np, cv2, albumentations as A, yaml
from ultralytics import YOLO

IMG_ROOT = Path("celeba_images")
IDMAP    = Path("milestone2/data/id_to_class.json")
CFG      = Path("milestone2/data/celeb_id.yaml")
WEIGHTS  = Path("milestone2/runs/yolov8s_m2_clean/weights/best.pt")

OUT_DIR  = Path("milestone2/id_cards"); OUT_DIR.mkdir(parents=True, exist_ok=True)
FAIL_DIR = Path("milestone2/id_cards_failures"); FAIL_DIR.mkdir(parents=True, exist_ok=True)
CSV_OUT  = Path("milestone2/id_card_audit.csv")

def extract_id(name):
    m = re.search(r"(\d+)(?=\D*$)", name)
    return int(m.group(1)) if m else None

def aug():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.GaussianBlur(blur_limit=(3,5), p=0.2),
        A.RandomBrightnessContrast(p=0.3),
    ])

# index images by identity
by_id = {}
for p in IMG_ROOT.rglob("*.jpg"):
    ident = extract_id(p.parent.name)
    if ident is not None:
        by_id.setdefault(ident, []).append(p)

id_to_class = {int(k):v for k,v in json.load(open(IDMAP)).items()}
class_to_id = {v:k for k,v in id_to_class.items()}
names = yaml.safe_load(open(CFG))["names"]
model = YOLO(str(WEIGHTS))

W, H = 1024, 1024
cw, ch = W//2, H//2
AUG = aug()
random.seed(9)

rows=[]
for ident, paths in sorted(by_id.items()):
    if ident not in id_to_class: 
        continue
    if len(paths) < 4:
        rows.append({"identity": ident, "status": "skip(<4 imgs)", "pred_set": ""})
        continue

    picks = random.sample(paths, 4)
    canvas = np.full((H, W, 3), 220, np.uint8)

    # place four tiles
    for i, ip in enumerate(picks):
        r, c = divmod(i, 2)
        x0, y0 = c*cw, r*ch
        img = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB)
        img = AUG(image=img)["image"]
        img = cv2.resize(img, (cw, ch))
        canvas[y0:y0+ch, x0:x0+cw] = img

    # save + predict
    stem = f"id_{ident}"
    img_path = OUT_DIR / f"{stem}.jpg"
    cv2.imwrite(str(img_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    res = model.predict(str(img_path), imgsz=W, conf=0.01, verbose=False)[0]
    draw = cv2.imread(str(img_path))

    def draw_inside(img, x1,y1,x2,y2, label):
        x1,y1,x2,y2 = map(int,[x1,y1,x2,y2])
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        (tw,th),base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,0.8,2)
        pad=3; xb,yb=x1+2,y1+2
        cv2.rectangle(img,(xb,yb),(min(x2-2,xb+tw+2*pad),min(y2-2,yb+th+base+2*pad)),(0,255,0),-1)
        cv2.putText(img,label,(xb+pad,yb+th+pad-1),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2,cv2.LINE_AA)

    pred_ids=set()
    for b in res.boxes:
        cls=int(b.cls.item()); conf=float(b.conf.item())
        x1,y1,x2,y2 = map(float, b.xyxy[0].tolist())
        lbl = names[cls]
        pred_ids.add(lbl)
        draw_inside(draw, x1,y1,x2,y2, f"{lbl} {conf:.2f}")

    out_pred = OUT_DIR / f"{stem}_pred.jpg"
    cv2.imwrite(str(out_pred), draw)

    ok = (len(pred_ids)==1 and str(ident) in pred_ids)
    row = {"identity": ident,
           "status": "OK" if ok else "FAIL",
           "pred_set": ",".join(sorted(pred_ids))}
    rows.append(row)

    if not ok:
        # copy the failing pred image for quick inspection
        cv2.imwrite(str(FAIL_DIR / f"{stem}_pred.jpg"), draw)

# write CSV
with open(CSV_OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["identity","status","pred_set"])
    w.writeheader(); w.writerows(rows)

print("Wrote:", CSV_OUT)
print("Cards:", OUT_DIR)
print("Failures (if any):", FAIL_DIR)
