from ultralytics import YOLO
import yaml, sys
from pathlib import Path
import cv2

CFG = "milestone2/data/celeb_id.yaml"
WEIGHTS = "milestone2/runs/yolov8n_m2/weights/best.pt"
OUT_DIR = Path("milestone2/preds")
OUT_DIR.mkdir(parents=True, exist_ok=True)

names = yaml.safe_load(open(CFG))["names"]
model = YOLO(WEIGHTS)

def run_on_image(img_path: Path):
    res = model.predict(str(img_path), imgsz=1024, conf=0.01, verbose=False)[0]
    img = cv2.imread(str(img_path))

    def draw_box_with_label_inside(img, x1, y1, x2, y2, label):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        # place the label background INSIDE the box at the top-left
        pad = 3
        x_bg1, y_bg1 = x1 + 2, y1 + 2
        x_bg2 = min(x2 - 2, x_bg1 + tw + 2*pad)
        y_bg2 = min(y2 - 2, y_bg1 + th + baseline + 2*pad)
        cv2.rectangle(img, (x_bg1, y_bg1), (x_bg2, y_bg2), (0,255,0), -1)
        x_txt, y_txt = x_bg1 + pad, y_bg1 + th + pad - 1
        cv2.putText(img, label, (x_txt, y_txt), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)

    for b in res.boxes:
        cls  = int(b.cls.item())
        conf = float(b.conf.item())
        x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
        ident = names[cls]
        draw_box_with_label_inside(img, x1, y1, x2, y2, f"{ident}  {conf:.2f}")

    out_path = OUT_DIR / f"{img_path.stem}_pred.jpg"
    cv2.imwrite(str(out_path), img)


def main():
    if len(sys.argv) < 2:
        # default: run on one of your validation grids
        val_dir = Path("milestone2/dataset/val/images")
        img_path = sorted(val_dir.glob("*.jpg"))[0]
        run_on_image(img_path)
        print(f"Saved: {OUT_DIR/(img_path.stem + '_pred.jpg')}")
        return

    target = Path(sys.argv[1])
    if target.is_dir():
        for p in sorted(target.glob("*.jpg")):
            run_on_image(p)
        print(f"Saved predictions to: {OUT_DIR}/")
    else:
        run_on_image(target)
        print(f"Saved: {OUT_DIR/(target.stem + '_pred.jpg')}")

if __name__ == "__main__":
    main()
