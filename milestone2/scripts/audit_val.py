from ultralytics import YOLO
from pathlib import Path
import yaml, csv, cv2, math

CFG = "milestone2/data/celeb_id.yaml"
WEIGHTS = "milestone2/runs/yolov8s_m2_clean/weights/best.pt"
VAL_IMG = Path("milestone2/dataset/val/images")
VAL_LBL = Path("milestone2/dataset/val/labels")
OUT_CSV = Path("milestone2/val_audit.csv")
FAIL_DIR = Path("milestone2/val_failures"); FAIL_DIR.mkdir(parents=True, exist_ok=True)

names = yaml.safe_load(open(CFG))["names"]
model = YOLO(WEIGHTS)

def yolo_to_xyxy(xc,yc,w,h,W,H):
    x1=(xc-w/2)*W; y1=(yc-h/2)*H; x2=(xc+w/2)*W; y2=(yc+h/2)*H
    return [x1,y1,x2,y2]

def iou(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1,iy1=max(ax1,bx1),max(ay1,by1)
    ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1)
    inter=iw*ih
    if inter<=0: return 0.0
    areaA=(ax2-ax1)*(ay2-ay1); areaB=(bx2-bx1)*(by2-by1)
    return inter/(areaA+areaB-inter+1e-9)

rows=[]
tot_gt=tot_pred=tot_match=tot_correct=tot_wrong=tot_miss=tot_extra=0

for ip in sorted(VAL_IMG.glob("*.jpg")):
    img=cv2.imread(str(ip)); H,W=img.shape[:2]
    lp=VAL_LBL/(ip.stem+".txt")
    gts=[]
    for ln in lp.read_text().strip().splitlines():
        c,xc,yc,w,h=map(float,ln.split())
        c=int(c); box=yolo_to_xyxy(xc,yc,w,h,W,H)
        gts.append((c,box))
    res=model.predict(str(ip), imgsz=1024, conf=0.01, verbose=False)[0]
    preds=[]
    if res.boxes is not None:
        for b in res.boxes:
            cls=int(b.cls.item()); conf=float(b.conf.item())
            x1,y1,x2,y2=map(float,b.xyxy[0].tolist())
            preds.append((cls,[x1,y1,x2,y2],conf))

    used_pred=[False]*len(preds)
    matched=correct=wrong=miss=0

    for (gc,gb) in gts:
        # best pred for this GT
        best_iou=-1; best_j=-1
        for j,(pc,pb,conf) in enumerate(preds):
            if used_pred[j]: continue
            i=iou(gb,pb)
            if i>best_iou: best_iou, best_j=i, j
        if best_iou>=0.5 and best_j>=0:
            matched+=1; used_pred[best_j]=True
            pc,_,_ = preds[best_j]
            if pc==gc: correct+=1
            else: wrong+=1
        else:
            miss+=1

    extra = sum(1 for u in used_pred if not u)

    rows.append({
        "image": ip.name,
        "gt": len(gts),
        "pred": len(preds),
        "matched": matched,
        "correct_ids": correct,
        "wrong_ids": wrong,
        "misses": miss,
        "extras": extra
    })

    # save failures overlay
    if wrong>0 or miss>0 or extra>0:
        draw=img.copy()
        # draw GT in blue
        for c,b in gts:
            x1,y1,x2,y2=map(int,b); cv2.rectangle(draw,(x1,y1),(x2,y2),(255,0,0),2)
        # draw preds in green (label)
        for (pc,pb,conf) in preds:
            x1,y1,x2,y2=map(int,pb)
            cv2.rectangle(draw,(x1,y1),(x2,y2),(0,255,0),2)
            lab=f"{names[pc]} {conf:.2f}"
            (tw,th),base=cv2.getTextSize(lab,cv2.FONT_HERSHEY_SIMPLEX,0.7,2)
            bx,by=x1+2,y1+2
            cv2.rectangle(draw,(bx,by),(min(x2-2,bx+tw+8),min(y2-2,by+th+base+6)),(0,255,0),-1)
            cv2.putText(draw,lab,(bx+3,by+th+2),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2,cv2.LINE_AA)
        cv2.imwrite(str(FAIL_DIR/ip.name), draw)

    tot_gt     += len(gts)
    tot_pred   += len(preds)
    tot_match  += matched
    tot_correct+= correct
    tot_wrong  += wrong
    tot_miss   += miss
    tot_extra  += extra

# write CSV
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_CSV,"w",newline="") as f:
    w=csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
    # add a blank line + totals
    f.write("\n")
    f.write(f"# totals,gt={tot_gt},pred={tot_pred},matched={tot_match},correct_ids={tot_correct},wrong_ids={tot_wrong},misses={tot_miss},extras={tot_extra}\n")
    # derived rates
    recall = tot_match/tot_gt if tot_gt else math.nan
    id_acc = tot_correct/tot_match if tot_match else math.nan
    f.write(f"# recall={recall:.4f},id_accuracy={id_acc:.4f}\n")

print("Wrote:", OUT_CSV)
print("Failures (if any) saved to:", FAIL_DIR)
