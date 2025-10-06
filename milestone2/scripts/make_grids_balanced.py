
import argparse, json, random, re, sys
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

def load_id_map(p):
    m = json.load(open(p))
    return {int(k):v for k,v in m.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_root", required=True)
    ap.add_argument("--id_map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--w", type=int, default=1024)
    ap.add_argument("--h", type=int, default=1024)
    ap.add_argument("--per_id", type=int, default=5)
    args = ap.parse_args()

    IMG_ROOT = Path(args.img_root)
    id_to_class = load_id_map(args.id_map)

    by_id = {}
    exts = {".jpg", ".jpeg", ".png"}
    for p in IMG_ROOT.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            ident = extract_id(p.parent.name)
            if ident in id_to_class:
                by_id.setdefault(ident, []).append(p)

    ids = sorted(by_id.keys())
    if not ids:
        print("ERROR: Found 0 identities under", IMG_ROOT)
        print("Tip: folder names must look like images_<IDENTITY>")
        sys.exit(1)

    for ident in ids:
        random.shuffle(by_id[ident])

    faces_total = len(ids) * args.per_id
    faces_per_grid = args.rows * args.cols
    n_grids = (faces_total + faces_per_grid - 1) // faces_per_grid

    out = Path(args.out)
    img_dir = out / "images"; lbl_dir = out / "labels"
    img_dir.mkdir(parents=True, exist_ok=True); lbl_dir.mkdir(parents=True, exist_ok=True)

    W,H = args.w, args.h
    rw, rh = args.cols, args.rows
    cw, ch = W // rw, H // rh
    AUG = aug()

    pools = {i:list(v) for i,v in by_id.items()}
    idx_counters = {i:0 for i in ids}
    random.seed(42)

    def sample_one(ident):
        lst = pools[ident]
        k = idx_counters[ident]
        if k >= len(lst):
            random.shuffle(lst); idx_counters[ident] = 0; k = 0
        p = lst[k]; idx_counters[ident] += 1
        return p

    print(f"identities: {len(ids)}")
    print(f"faces_per_identity_target: {args.per_id}")
    print(f"grids_to_write: {n_grids}")

    wrote = 0
    for g in range(n_grids):
        canvas = np.full((H,W,3), 220, np.uint8)
        labels = []
        pick_ids = random.sample(ids, faces_per_grid)
        for i, ident in enumerate(pick_ids):
            ip = sample_one(ident)
            img = cv2.imread(str(ip))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = AUG(image=img)["image"]
            img = cv2.resize(img, (cw, ch))
            r,c = divmod(i, rw)
            x0,y0 = c*cw, r*ch
            canvas[y0:y0+ch, x0:x0+cw] = img
            cx = (x0 + cw/2)/W; cy = (y0 + ch/2)/H; w = cw/W; h = ch/H
            labels.append((id_to_class[ident], cx, cy, w, h))

        stem = f"grid_{g:06d}"
        cv2.imwrite(str(img_dir/f"{stem}.jpg"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        with open(lbl_dir/f"{stem}.txt","w") as f:
            for cls,cx,cy,w,h in labels:
                f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        wrote += faces_per_grid
        if (g+1) % 10 == 0 or g == n_grids-1:
            print(f"wrote {g+1}/{n_grids} grids…")

    print("faces_written:", wrote)
    print("out:", out.resolve())

if __name__ == "__main__":
    main()
