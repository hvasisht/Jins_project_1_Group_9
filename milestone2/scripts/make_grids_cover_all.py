import argparse, json, re, random, csv
from pathlib import Path
import numpy as np, cv2, albumentations as A

def extract_id(name):
    m = re.search(r"(\d+)(?=\D*$)", name)
    return int(m.group(1)) if m else None

def aug():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.GaussianBlur(blur_limit=(3,5), p=0.2),
        A.RandomBrightnessContrast(p=0.3),
    ])

ap = argparse.ArgumentParser()
ap.add_argument("--img_root", default="celeba_images")
ap.add_argument("--id_map",  default="milestone2/data/id_to_class.json")
ap.add_argument("--out",     default="milestone2/outputs_cover")
ap.add_argument("--rows", type=int, default=2)
ap.add_argument("--cols", type=int, default=2)
ap.add_argument("--w",   type=int, default=1024)
ap.add_argument("--h",   type=int, default=1024)
args = ap.parse_args()

IMG_ROOT = Path(args.img_root)
id_to_class = {int(k):v for k,v in json.load(open(args.id_map)).items()}

# Gather every source file with its identity
exts = {".jpg",".jpeg",".png"}
sources = []
for p in IMG_ROOT.rglob("*"):
    if p.is_file() and p.suffix.lower() in exts:
        ident = extract_id(p.parent.name)
        if ident in id_to_class:
            sources.append((ident, p))

assert sources, f"No images found under {IMG_ROOT}"
random.seed(42)
random.shuffle(sources)

out = Path(args.out)
img_dir = out/"images"; lbl_dir = out/"labels"; meta_csv = out/"sources.csv"
img_dir.mkdir(parents=True, exist_ok=True); lbl_dir.mkdir(parents=True, exist_ok=True)

W,H = args.w, args.h
cw, ch = W//args.cols, H//args.rows
AUG = aug()

rows = []
faces_per_grid = args.rows*args.cols
grids = (len(sources)+faces_per_grid-1)//faces_per_grid

def place(canvas, img, r, c):
    x0, y0 = c*cw, r*ch
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = AUG(image=img)["image"]
    img = cv2.resize(img, (cw, ch))
    canvas[y0:y0+ch, x0:x0+cw] = img
    cx=(x0+cw/2)/W; cy=(y0+ch/2)/H; w=cw/W; h=ch/H
    return cx,cy,w,h

i = 0
for g in range(grids):
    canvas = np.full((H,W,3), 220, np.uint8)
    labels = []
    used = []
    chunk = sources[i:i+faces_per_grid]
    i += faces_per_grid
    # If last chunk is short, pad with random samples (won't affect "each at least once")
    if len(chunk) < faces_per_grid:
        chunk = chunk + random.sample(sources, faces_per_grid - len(chunk))

    k=0
    for r in range(args.rows):
        for c in range(args.cols):
            ident, path = chunk[k]; k+=1
            img = cv2.imread(str(path))
            cx,cy,w,h = place(canvas, img, r, c)
            labels.append((id_to_class[ident], cx, cy, w, h))
            used.append((f"grid_{g:06d}.jpg", r*args.cols+c, ident, path.name))

    # save grid + labels
    stem=f"grid_{g:06d}"
    cv2.imwrite(str(img_dir/f"{stem}.jpg"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    with open(lbl_dir/f"{stem}.txt","w") as f:
        for cls,cx,cy,w,h in labels:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    # add to metadata rows
    for rec in used:
        rows.append(rec)

# write sidecar mapping: grid, tile_idx, identity, source_filename
with open(meta_csv,"w",newline="") as f:
    w=csv.writer(f); w.writerow(["grid","tile","identity","source_file"])
    w.writerows(rows)

print("grids_written:", grids)
print("each source image used at least once.")
print("out:", out.resolve())
print("metadata:", meta_csv.resolve())
