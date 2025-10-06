import json, random, argparse, re
from pathlib import Path
import numpy as np, cv2, albumentations as A

def extract_id(name):
    m = re.search(r"(\d+)(?=\D*$)", name)
    return int(m.group(1)) if m else None

def load_index(img_root, id_map_json):
    id_to_class = {int(k):v for k,v in json.load(open(id_map_json)).items()}
    by_id = {}
    for p in Path(img_root).rglob("*"):
        if p.is_file() and p.suffix.lower() in {".jpg",".jpeg",".png"}:
            ident = extract_id(p.parent.name)
            if ident is not None and ident in id_to_class:
                by_id.setdefault(ident, []).append(p)
    ids = [i for i,lst in by_id.items() if len(lst)>0]
    return by_id, id_to_class, ids

def aug():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.GaussianBlur(blur_limit=(3,5), p=0.2),
        A.RandomBrightnessContrast(p=0.3),
    ])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_root", default="celeba_images")
    ap.add_argument("--id_map", default="milestone2/data/id_to_class.json")
    ap.add_argument("--out", default="milestone2/outputs")
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--w", type=int, default=1024)
    ap.add_argument("--h", type=int, default=1024)
    ap.add_argument("--n", type=int, default=400)
    args = ap.parse_args()

    out_img = Path(args.out)/"images"; out_lbl = Path(args.out)/"labels"
    out_img.mkdir(parents=True, exist_ok=True); out_lbl.mkdir(parents=True, exist_ok=True)

    by_id, id_to_class, ids = load_index(args.img_root, args.id_map)
    random.seed(42)
    cell_w, cell_h = args.w//args.cols, args.h//args.rows
    AUG = aug()

    for i in range(args.n):
        canvas = np.full((args.h, args.w, 3), 220, np.uint8)
        labels = []
        choose = random.sample(ids, k=min(args.rows*args.cols, len(ids)))
        k = 0
        for r in range(args.rows):
            for c in range(args.cols):
                ident = choose[k % len(choose)]; k+=1
                path = random.choice(by_id[ident])
                img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
                img = AUG(image=img)["image"]
                img = cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_LINEAR)
                x0, y0 = c*cell_w, r*cell_h
                canvas[y0:y0+cell_h, x0:x0+cell_w] = img
                cx = (x0 + cell_w/2)/args.w; cy = (y0 + cell_h/2)/args.h
                w  = cell_w/args.w;         h  = cell_h/args.h
                labels.append((id_to_class[ident], cx, cy, w, h))
        img_path = out_img/f"grid_{i:06d}.jpg"
        lbl_path = out_lbl/f"grid_{i:06d}.txt"
        cv2.imwrite(str(img_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        with open(lbl_path,"w") as f:
            for cls,cx,cy,w,h in labels:
                f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

if __name__ == "__main__":
    main()
