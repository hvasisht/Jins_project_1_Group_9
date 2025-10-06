import argparse, json, re, random
from pathlib import Path
import cv2, numpy as np, albumentations as A
from ultralytics import YOLO
import yaml

def extract_id(name):
    m = re.search(r"(\d+)(?=\D*$)", name)
    return int(m.group(1)) if m else None

def aug():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.GaussianBlur(blur_limit=(3,5), p=0.2),
        A.RandomBrightnessContrast(p=0.3),
    ])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--identity", required=True, type=int, help="CelebA identity number, e.g. 8265")
    ap.add_argument("--img_root", default="celeba_images")
    ap.add_argument("--id_map",  default="milestone2/data/id_to_class.json")
    ap.add_argument("--cfg",     default="milestone2/data/celeb_id.yaml")
    ap.add_argument("--weights", default="milestone2/runs/yolov8s_m2_clean/weights/best.pt")
    ap.add_argument("--w", type=int, default=1024)
    ap.add_argument("--h", type=int, default=1024)
    args = ap.parse_args()

    id_to_class = {int(k):v for k,v in json.load(open(args.id_map)).items()}
    class_to_id = {v:k for k,v in id_to_class.items()}
    names = yaml.safe_load(open(args.cfg))["names"]

    ident = args.identity
    assert ident in id_to_class, f"Identity {ident} not in id_to_class.json"

    # collect images for this identity
    root = Path(args.img_root)
    img_paths = []
    for p in root.rglob("*.jpg"):
        if extract_id(p.parent.name) == ident:
            img_paths.append(p)
    if len(img_paths) < 4:
        raise RuntimeError(f"Need at least 4 images for identity {ident}, found {len(img_paths)}")

    # build 2x2 with 4 different images of the same identity
    random.seed(7)
    picks = random.sample(img_paths, 4)
    AUG = aug()
    W,H = args.w, args.h
    cw, ch = W//2, H//2
    canvas = np.full((H,W,3), 220, np.uint8)
    labels = []

    def place(idx, img):
        r, c = divmod(idx, 2)
        x0, y0 = c*cw, r*ch
        img = cv2.cvtColor(cv2.imread(str(img)), cv2.COLOR_BGR2RGB)
        img = AUG(image=img)["image"]
        img = cv2.resize(img, (cw, ch))
        canvas[y0:y0+ch, x0:x0+cw] = img
        cx = (x0 + cw/2)/W; cy = (y0 + ch/2)/H; w = cw/W; h = ch/H
        return cx, cy, w, h

    for i, ip in enumerate(picks):
        cx,cy,w,h = place(i, ip)
        labels.append((id_to_class[ident], cx, cy, w, h))

    outdir = Path("milestone2/sanity"); outdir.mkdir(parents=True, exist_ok=True)
    stem = f"sameid_{ident}"
    img_path = outdir / f"{stem}.jpg"
    lbl_path = outdir / f"{stem}.txt"
    cv2.imwrite(str(img_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    with open(lbl_path,"w") as f:
        for cls,cx,cy,w,h in labels:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    # print ground-truth classes → identities
    uniq = sorted(set([lbl[0] for lbl in labels]))
    print("GT class indices in this grid:", uniq)
    print("GT identities in this grid   :", [class_to_id[c] for c in uniq], "(should all be the same)")

    # predict
    model = YOLO(args.weights)
    res = model.predict(str(img_path), imgsz=W, conf=0.01, verbose=False)[0]
    img = cv2.imread(str(img_path))

    def draw_inside(img, x1,y1,x2,y2, label):
        x1,y1,x2,y2 = map(int,[x1,y1,x2,y2])
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        (tw,th),base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,0.8,2)
        pad=3; xb,yb=x1+2,y1+2
        cv2.rectangle(img,(xb,yb),(min(x2-2, xb+tw+2*pad), min(y2-2, yb+th+base+2*pad)),(0,255,0),-1)
        cv2.putText(img,label,(xb+pad,yb+th+pad-1),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2,cv2.LINE_AA)

    print("Predictions:")
    for b in res.boxes:
        cls = int(b.cls.item())
        conf = float(b.conf.item())
        x1,y1,x2,y2 = map(float, b.xyxy[0].tolist())
        ident_str = names[cls]
        print(f"  ID {ident_str}  conf={conf:.2f}  box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")
        draw_inside(img,x1,y1,x2,y2,f"{ident_str}  {conf:.2f}")

    cv2.imwrite(str(outdir/f"{stem}_pred.jpg"), img)
    print("Saved:", outdir/f"{stem}.jpg", "and", outdir/f"{stem}_pred.jpg")

if __name__ == "__main__":
    main()
