# make_grid_collages.py — FIXED 2x3 grid, every cell filled, YOLO labels
from pathlib import Path
import os, random, json
import cv2
import numpy as np
from tqdm import tqdm

# -------- Config (fixed grid) --------
RNG_SEED          = 42
CANVAS_SIZE       = (1024, 1024)          # (H, W)
GRID_ROWS, GRID_COLS = 2, 3               # fixed 2x3 grid
PEOPLE_PER_IMAGE  = 6                     # always fill all cells
CELL_MARGIN_PCT   = 0.08                  # margin inside each cell
SPLITS            = {"train": 0.8, "val": 0.1, "test": 0.1}
IMAGES_PER_SPLIT  = {"train": 800, "val": 120, "test": 120}
AUG_FLIP_PROB     = 0.50
AUG_BLUR_PROB     = 0.25
AUG_JITTER_PROB   = 0.50

# -------- Paths --------
SCRIPT_DIR = Path(__file__).resolve().parent
M2_DIR     = SCRIPT_DIR.parent
DATA_DIR   = M2_DIR / "data"
FACES_DIR  = DATA_DIR / "faces"
YOLO_IMG   = DATA_DIR / "yolo" / "images"
YOLO_LBL   = DATA_DIR / "yolo" / "labels"
YOLO_IMG.mkdir(parents=True, exist_ok=True)
YOLO_LBL.mkdir(parents=True, exist_ok=True)

ID_LIST_TXT = DATA_DIR / "id_list.txt"

random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

# -------- Helpers --------
def read_id_list(p: Path):
    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]

def load_faces_index(root: Path, ids):
    idx = {}
    for cid in ids:
        d = root / cid
        files = []
        if d.is_dir():
            for f in os.listdir(d):
                if f.lower().endswith((".jpg",".jpeg",".png")):
                    files.append(str(d / f))
        random.shuffle(files)
        idx[cid] = files
    return idx

def split_files_per_id(index, splits):
    per = {s:{} for s in splits}
    for cid, files in index.items():
        n = len(files)
        ntr = int(n * splits["train"])
        nva = int(n * splits["val"])
        per["train"][cid] = files[:ntr]
        per["val"][cid]   = files[ntr:ntr+nva]
        per["test"][cid]  = files[ntr+nva:]
    return per

def brighten_contrast(img, alpha_range=(0.9,1.1), beta_range=(-10,10)):
    alpha = np.random.uniform(*alpha_range)
    beta  = np.random.uniform(*beta_range)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def maybe_blur(img):
    if np.random.rand() < AUG_BLUR_PROB:
        k = random.choice([3,5])
        return cv2.GaussianBlur(img, (k,k), 0)
    return img

def paste_in_cell(canvas, face, cell_xyxy, margin_pct):
    x1, y1, x2, y2 = cell_xyxy
    cw = x2 - x1
    ch = y2 - y1
    mx = int(cw * margin_pct)
    my = int(ch * margin_pct)
    x1i, y1i = x1 + mx, y1 + my
    x2i, y2i = x2 - mx, y2 - my
    iw, ih = max(1, x2i - x1i), max(1, y2i - y1i)

    h, w = face.shape[:2]
    scale = min(iw / max(1, w), ih / max(1, h))
    nw, nh = max(1, int(w*scale)), max(1, int(h*scale))
    face = cv2.resize(face, (nw, nh), interpolation=cv2.INTER_AREA)

    ox = x1i + (iw - nw)//2
    oy = y1i + (ih - nh)//2
    canvas[oy:oy+nh, ox:ox+nw] = face
    return (ox, oy, ox+nw, oy+nh)  # xyxy abs

def to_yolo(b, W, H):
    x1,y1,x2,y2 = b
    xc = (x1+x2)/2.0 / W
    yc = (y1+y2)/2.0 / H
    w  = (x2-x1)/W
    h  = (y2-y1)/H
    return xc, yc, w, h

# -------- Generation --------
def generate_split(split, per_id_files, id2cls, n_images):
    out_img = YOLO_IMG / split
    out_lbl = YOLO_LBL / split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    H, W = CANVAS_SIZE
    cell_w = W // GRID_COLS
    cell_h = H // GRID_ROWS

    pbar = tqdm(total=n_images, desc=f"grid {split}")
    made = 0
    while made < n_images:
        # pick k=6 IDs WITH replacement so we always fill every cell
        valid = [cid for cid, files in per_id_files.items() if len(files) > 0]
        if not valid:
            continue
        chosen_ids = [random.choice(valid) for _ in range(PEOPLE_PER_IMAGE)]

        canvas = np.full((H, W, 3), 220, dtype=np.uint8)
        boxes, labels = [], []

        # fill cells row-major: (0,0),(0,1),(0,2),(1,0),(1,1),(1,2)
        cell_positions = [(r, c) for r in range(GRID_ROWS) for c in range(GRID_COLS)]

        for cid, (r, c) in zip(chosen_ids, cell_positions):
            fpath = random.choice(per_id_files[cid])
            img = cv2.imread(fpath)
            if img is None:
                continue
            if np.random.rand() < AUG_FLIP_PROB:
                img = cv2.flip(img, 1)
            if np.random.rand() < AUG_JITTER_PROB:
                img = brighten_contrast(img)
            img = maybe_blur(img)

            x1 = c * cell_w
            y1 = r * cell_h
            x2 = x1 + cell_w
            y2 = y1 + cell_h
            bbox = paste_in_cell(canvas, img, (x1,y1,x2,y2), CELL_MARGIN_PCT)
            boxes.append(bbox)
            labels.append(cid)

        if not boxes:
            continue

        name = f"{split}_{made:06d}"
        cv2.imwrite(str(out_img / f"{name}.jpg"), canvas)
        with open(out_lbl / f"{name}.txt", "w") as f:
            for b, cid in zip(boxes, labels):
                x,y,w,h = to_yolo(b, W, H)
                f.write(f"{id2cls[cid]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

        made += 1
        pbar.update(1)
    pbar.close()

def main():
    ids = read_id_list(ID_LIST_TXT)
    id2cls = {cid:i for i, cid in enumerate(ids)}
    (DATA_DIR / "class_map.json").write_text(json.dumps(id2cls, indent=2))

    index_all = load_faces_index(FACES_DIR, ids)
    per_split = split_files_per_id(index_all, SPLITS)

    for split in ["train","val","test"]:
        work = {cid:list(files) for cid,files in per_split[split].items()}
        generate_split(split, work, id2cls, IMAGES_PER_SPLIT[split])

    print("\n[done] Fixed-grid composites & labels →", DATA_DIR / "yolo")

if __name__ == "__main__":
    main()
