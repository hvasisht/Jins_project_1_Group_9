from pathlib import Path
import argparse, yaml, cv2

def yolo_to_xyxy(xc,yc,w,h,W,H):
    x1=(xc-w/2)*W; y1=(yc-h/2)*H; x2=(xc+w/2)*W; y2=(yc+h/2)*H
    return int(x1),int(y1),int(x2),int(y2)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root", required=True)   # e.g., milestone2/outputs_preview
    ap.add_argument("--cfg",  default="milestone2/data/celeb_id.yaml")
    ap.add_argument("--out",  default="milestone2/labels_viz")
    args=ap.parse_args()

    names = yaml.safe_load(open(args.cfg))["names"]
    root = Path(args.root)
    img_dir = root/"images"
    lbl_dir = root/"labels"
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted(img_dir.glob("*.jpg"))
    for ip in imgs:
        lp = lbl_dir/(ip.stem + ".txt")
        img = cv2.imread(str(ip))
        H,W = img.shape[:2]
        if lp.exists():
            for line in lp.read_text().strip().splitlines():
                c,xc,yc,w,h = map(float, line.split())
                c=int(c); x1,y1,x2,y2 = yolo_to_xyxy(xc,yc,w,h,W,H)
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                label = names[c]
                (tw,th),base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                yt = max(0,y1-5)
                cv2.rectangle(img,(x1,yt-th-base-4),(x1+tw+6,yt+2),(0,255,0),-1)
                cv2.putText(img,label,(x1+3,yt-2),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2,cv2.LINE_AA)
        cv2.imwrite(str(out_dir/f"{ip.stem}_gt.jpg"), img)

if __name__ == "__main__":
    main()
