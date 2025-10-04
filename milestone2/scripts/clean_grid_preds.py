# clean_grid_preds.py  — read *source* test images, write clean, non-overlapping boxes
from pathlib import Path
import cv2
import numpy as np

# ----- Fixed grid layout -----
H, W = 1024, 1024
GRID_ROWS, GRID_COLS = 2, 3
cell_w, cell_h = W // GRID_COLS, H // GRID_ROWS

# ----- I/O -----
M2 = Path(__file__).resolve().parents[1]
SRC_IMG_DIR = M2 / "data" / "yolo" / "images" / "test"         # clean images
RAW_RUN     = M2 / "outputs" / "pred_grid_raw"                  # where YOLO put raw preds
RAW_LBL_DIR = RAW_RUN / "labels"

OUT_RUN     = M2 / "outputs" / "pred_grid_clean_cells"
OUT_IMG_DIR = OUT_RUN
OUT_LBL_DIR = OUT_RUN / "labels"
OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_LBL_DIR.mkdir(parents=True, exist_ok=True)

# optional: class index -> celeb ID text
id_txt = (M2 / "data" / "id_list.txt").read_text().splitlines()
ID_NAMES = [s.strip() for s in id_txt if s.strip()]

def class_to_name(ci:int) -> str:
    try:
        return ID_NAMES[int(ci)]
    except:
        return str(ci)

def norm_to_xyxy(x, y, w, h, W, H):
    xc, yc, ww, hh = x*W, y*H, w*W, h*H
    x1 = int(max(0, xc - ww/2))
    y1 = int(max(0, yc - hh/2))
    x2 = int(min(W-1, xc + ww/2))
    y2 = int(min(H-1, yc + hh/2))
    return x1, y1, x2, y2

def xyxy_to_norm(x1,y1,x2,y2,W,H):
    xc = ((x1+x2)/2)/W
    yc = ((y1+y2)/2)/H
    w  = (x2-x1)/W
    h  = (y2-y1)/H
    return xc, yc, w, h

def which_cell(x1,y1,x2,y2):
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    col = min(GRID_COLS-1, max(0, int(cx // cell_w)))
    row = min(GRID_ROWS-1, max(0, int(cy // cell_h)))
    return row*GRID_COLS + col

def draw_box(img, x1,y1,x2,y2, txt, color):
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 3)
    (tw, th), bl = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    yb = max(0, y1 - 10)
    cv2.rectangle(img, (x1, yb-th-8), (x1+tw+10, yb), color, -1)
    cv2.putText(img, txt, (x1+5, yb-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)

# nice distinct colors per cell
COLORS = [(46,204,113),(52,152,219),(155,89,182),(241,196,15),(230,126,34),(231,76,60)]

images = sorted([p for p in SRC_IMG_DIR.glob("test_*.jpg")])

kept_total = 0
for img_path in images:
    lab_path = RAW_LBL_DIR / (img_path.stem + ".txt")

    img = cv2.imread(str(img_path))
    if img is None:
        continue

    # one best detection per cell
    best = [None]*(GRID_ROWS*GRID_COLS)

    if lab_path.exists():
        for line in lab_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            x,y,w,h = map(float, parts[1:5])
            conf = float(parts[5]) if len(parts) >= 6 else 0.0
            x1,y1,x2,y2 = norm_to_xyxy(x,y,w,h,W,H)
            cell_idx = which_cell(x1,y1,x2,y2)
            prev = best[cell_idx]
            if prev is None or conf > prev[1]:
                best[cell_idx] = (cls, conf, (x1,y1,x2,y2))

    # draw + save
    kept = []
    for i, b in enumerate(best):
        if b is None:
            continue
        cls, conf, (x1,y1,x2,y2) = b
        label = f"{class_to_name(cls)} {conf:.2f}"
        draw_box(img, x1,y1,x2,y2, label, COLORS[i % len(COLORS)])
        kept.append((cls, conf, (x1,y1,x2,y2)))

    # image
    cv2.imwrite(str(OUT_IMG_DIR / img_path.name), img)
    # labels
    with open(OUT_LBL_DIR / (img_path.stem + ".txt"), "w") as f:
        for cls, conf, (x1,y1,x2,y2) in kept:
            xc,yc,w,h = xyxy_to_norm(x1,y1,x2,y2,W,H)
            f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {conf:.4f}\n")

    kept_total += len(kept)

print(f"[done] cleaned {len(images)} images → {OUT_IMG_DIR}")
print(f"        total boxes kept: {kept_total} (≤ {GRID_ROWS*GRID_COLS} per image)")
