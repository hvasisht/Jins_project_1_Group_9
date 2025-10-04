# make_collages.py
# Build composite "group" images from milestone2/data/faces/<ID>
# and write YOLO labels to milestone2/data/yolo/{images,labels}/{train,val,test}

from pathlib import Path
import os, random, json
import cv2
import numpy as np
from tqdm import tqdm

# ---------------- Config ----------------
RNG_SEED          = 42
CANVAS_SIZE       = (1024, 1024)            # (H, W)
PEOPLE_PER_IMAGE  = (3, 4)                  # min/max faces per composite
SPLITS            = {"train": 0.8, "val": 0.1, "test": 0.1}
IMAGES_PER_SPLIT  = {"train": 1200, "val": 150, "test": 150}
MIN_IOU_SEP       = 0.00                    # 0 = allow overlap (simpler)
SCALE_RANGE       = (0.45, 0.70)            # pasted face height as % of canvas H
AUG_BLUR_PROB     = 0.30
AUG_JITTER_PROB   = 0.60

# ---------------- Paths ----------------
SCRIPT_DIR = Path(__file__).resolve().parent
M2_DIR     = SCRIPT_DIR.parent
DATA_DIR   = M2_DIR / "data"
FACES_DIR  = DATA_DIR / "faces"
YOLO_IMG   = DATA_DIR / "yolo" / "images"
YOLO_LBL   = DATA_DIR / "yolo" / "labels"
YOLO_IMG.mkdir(parents=True, exist_ok=True)
YOLO_LBL.mkdir(parents=True, exist_ok=True)

ID_LIST_TXT = DATA_DIR / "id_list.txt"  # one ID per line (your 45 IDs)

random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

# ---------------- Helpers ----------------
def read_id_list(path: Path):
    ids = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(line)
    return ids

def load_faces_index(faces_root: Path, ids):
    """Return dict: id -> [absolute face file paths]"""
    idx = {}
    for cid in ids:
        d = faces_root / cid
        if d.is_dir():
            files = [str(d / f) for f in os.listdir(d)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            if files:
                random.shuffle(files)
            idx[cid] = files
        else:
            idx[cid] = []
    return idx

def split_files_per_id(index, splits):
    """Split each ID's face files into train/val/test lists."""
    per_split = {s: {} for s in splits}
    for cid, files in index.items():
        n = len(files)
        n_train = int(n * splits["train"])
        n_val   = int(n * splits["val"])
        train = files[:n_train]
        val   = files[n_train:n_train+n_val]
        test  = files[n_train+n_val:]
        per_split["train"][cid] = train
        per_split["val"][cid]   = val
        per_split["test"][cid]  = test
    return per_split

def iou(a, b):
    # a,b: (x1,y1,x2,y2)
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area_a = (a[2]-a[0])*(a[3]-a[1]); area_b = (b[2]-b[0])*(b[3]-b[1])
    union = area_a + area_b - inter + 1e-9
    return inter / union

def brighten_contrast(img, alpha_range=(0.85, 1.15), beta_range=(-15, 15)):
    alpha = np.random.uniform(*alpha_range)
    beta  = np.random.uniform(*beta_range)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def maybe_blur(img, k_choices=(3, 5)):
    if np.random.rand() < AUG_BLUR_PROB:
        k = int(random.choice(k_choices))
        return cv2.GaussianBlur(img, (k, k), 0)
    return img

def paste_with_bbox(canvas, face_img, top_left_xy):
    x, y = top_left_xy
    h, w, _ = face_img.shape
    canvas[y:y+h, x:x+w] = face_img
    return (x, y, x + w, y + h)

def to_yolo(bbox_xyxy, canvas_w, canvas_h):
    x1, y1, x2, y2 = bbox_xyxy
    xc = (x1 + x2) / 2.0
    yc = (y1 + y2) / 2.0
    w  = x2 - x1
    h  = y2 - y1
    return xc / canvas_w, yc / canvas_h, w / canvas_w, h / canvas_h

# ---------------- Main generation ----------------
def generate_split(split_name, per_id_files, id_to_class, n_images):
    out_img_dir = YOLO_IMG / split_name
    out_lbl_dir = YOLO_LBL / split_name
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    H, W = CANVAS_SIZE
    pbar = tqdm(total=n_images, desc=f"make {split_name}")
    made = 0

    while made < n_images:
        # how many faces this composite should have
        k = random.randint(PEOPLE_PER_IMAGE[0], PEOPLE_PER_IMAGE[1])

        # valid IDs: must have at least 1 file for this split
        valid_ids = [cid for cid, files in per_id_files.items() if len(files) > 0]
        if not valid_ids:
            # nothing usable; try again (or break to avoid infinite loop)
            continue

        k = min(k, len(valid_ids))
        chosen_ids = random.sample(valid_ids, k)

        # blank canvas (neutral gray)
        canvas = np.full((H, W, 3), 200, dtype=np.uint8)
        boxes_xyxy = []
        labels = []

        for cid in chosen_ids:
            src_list = per_id_files[cid]
            if not src_list:
                continue
            # IMPORTANT: pick without removing so classes don't get "used up"
            fpath = random.choice(src_list)
            img = cv2.imread(fpath)
            if img is None:
                continue

            # scale face to target height %
            tgt_h = int(np.random.uniform(*SCALE_RANGE) * H)
            scale = max(1e-6, tgt_h / max(1, img.shape[0]))
            new_w = max(1, int(img.shape[1] * scale))
            new_h = max(1, int(img.shape[0] * scale))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # simple augs
            if np.random.rand() < 0.5:
                img = cv2.flip(img, 1)
            if np.random.rand() < AUG_JITTER_PROB:
                img = brighten_contrast(img)
            img = maybe_blur(img)

            # try to place to respect MIN_IOU_SEP
            placed = False
            for _ in range(30):
                x = random.randint(0, max(0, W - img.shape[1]))
                y = random.randint(0, max(0, H - img.shape[0]))
                cand = (x, y, x + img.shape[1], y + img.shape[0])

                ok = True
                if MIN_IOU_SEP > 0:
                    for b in boxes_xyxy:
                        if iou(cand, b) > MIN_IOU_SEP:
                            ok = False
                            break
                if ok:
                    bbox = paste_with_bbox(canvas, img, (x, y))
                    boxes_xyxy.append(bbox)
                    labels.append(cid)
                    placed = True
                    break

            if not placed:
                # skip this face if can't place after tries
                continue

        if not boxes_xyxy:
            # nothing placed, try again
            continue

        # save composite + YOLO label file
        img_name = f"{split_name}_{made:06d}.jpg"
        lbl_name = f"{split_name}_{made:06d}.txt"
        cv2.imwrite(str(out_img_dir / img_name), canvas)

        with open(out_lbl_dir / lbl_name, "w") as f:
            for bbox, cid in zip(boxes_xyxy, labels):
                x, y, w, h = to_yolo(bbox, W, H)
                cls = id_to_class[cid]
                f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

        made += 1
        pbar.update(1)

    pbar.close()

def main():
    ids = read_id_list(ID_LIST_TXT)
    id_to_class = {cid: i for i, cid in enumerate(ids)}
    (DATA_DIR / "class_map.json").write_text(json.dumps(id_to_class, indent=2))

    index_all = load_faces_index(FACES_DIR, ids)
    per_split = split_files_per_id(index_all, SPLITS)

    for split_name in ["train", "val", "test"]:
        work = {cid: list(files) for cid, files in per_split[split_name].items()}
        generate_split(split_name, work, id_to_class, IMAGES_PER_SPLIT[split_name])

    print("\n[done] Composites & labels written to:", DATA_DIR / "yolo")

if __name__ == "__main__":
    main()
