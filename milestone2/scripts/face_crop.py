# face_crop.py
# Run from: Jins_project_1/milestone2/scripts
# Creates cropped-face images under milestone2/data/faces/<ID> from raw_drive/<ID>

from pathlib import Path
import os
import cv2
import mediapipe as mp
from tqdm import tqdm

# ---------- Resolve paths relative to THIS file ----------
SCRIPT_DIR = Path(__file__).resolve().parent                 # .../milestone2/scripts
M2_DIR     = SCRIPT_DIR.parent                                # .../milestone2
RAW_DIR    = M2_DIR / "data" / "raw_drive"                    # input: 45 ID folders
FACE_DIR   = M2_DIR / "data" / "faces"                        # output: cropped faces
FACE_DIR.mkdir(parents=True, exist_ok=True)

print(f"[paths] RAW_DIR : {RAW_DIR}")
print(f"[paths] FACE_DIR: {FACE_DIR}")
if not RAW_DIR.exists():
    raise FileNotFoundError(f"RAW_DIR not found: {RAW_DIR}\n"
                            "→ Put your ID folders inside milestone2/data/raw_drive/")

# ---------- Face detector ----------
mp_fd = mp.solutions.face_detection

def crop_with_margin(img_bgr, bbox_rel, margin_px=10):
    """Crop using MediaPipe relative bbox + pixel margin."""
    h, w, _ = img_bgr.shape
    x1 = max(int(bbox_rel.xmin * w) - margin_px, 0)
    y1 = max(int(bbox_rel.ymin * h) - margin_px, 0)
    x2 = min(int((bbox_rel.xmin + bbox_rel.width)  * w) + margin_px, w)
    y2 = min(int((bbox_rel.ymin + bbox_rel.height) * h) + margin_px, h)
    if x2 <= x1 or y2 <= y1:
        return None
    return img_bgr[y1:y2, x1:x2]

def is_image(fname: str) -> bool:
    return fname.lower().endswith((".jpg", ".jpeg", ".png"))

def process_id_folder(detector, celeb_id: str):
    in_dir  = RAW_DIR / celeb_id
    out_dir = FACE_DIR / celeb_id
    if not in_dir.is_dir():
        return 0, 0
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [f for f in os.listdir(in_dir) if is_image(f)]
    saved, skipped = 0, 0

    for fname in tqdm(files, desc=f"[{celeb_id}]"):
        src = in_dir / fname
        img = cv2.imread(str(src))
        if img is None:
            skipped += 1
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = detector.process(rgb)

        if not res.detections:
            skipped += 1
            continue

        det  = res.detections[0]  # first face
        bbox = det.location_data.relative_bounding_box
        crop = crop_with_margin(img, bbox, margin_px=10)
        if crop is None:
            skipped += 1
            continue

        # Always save as .jpg
        out_path = out_dir / (Path(fname).stem + ".jpg")
        cv2.imwrite(str(out_path), crop)
        saved += 1

    return saved, skipped

def main():
    total_saved, total_skipped = 0, 0
    with mp_fd.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
        for celeb_id in sorted(os.listdir(RAW_DIR)):
            saved, skipped = process_id_folder(detector, celeb_id)
            total_saved  += saved
            total_skipped += skipped
            print(f"  -> {celeb_id}: saved {saved}, skipped {skipped}")
    print("\n[done] Total cropped:", total_saved, "| skipped:", total_skipped)
    print(f"[out] Crops in: {FACE_DIR}")

if __name__ == "__main__":
    main()
