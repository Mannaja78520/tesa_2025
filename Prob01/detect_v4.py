# detect_folder_v3_7_strong_dilate.py
# Drone Detector — Strong-Dilate + Full Horizon + BH-RAW Fusion

import os, csv, cv2, numpy as np
from glob import glob

# ================= CONFIG =================
INPUT_FOLDER   = 'TEST_DATA'
OUTPUT_FOLDER  = 'RESULTS'
BH_FOLDER      = 'RESULTS_BLACKHAT'
IMG_EXTS       = ('*.jpg','*.jpeg','*.png')
DRAW_THICK     = 2
# ==========================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(BH_FOLDER, exist_ok=True)

# ---------- Utilities ----------
def clamp_box(x, y, w, h, W, H):
    x = max(0, x); y = max(0, y)
    w = min(w, W - x); h = min(h, H - y)
    return x, y, w, h

def iou(a,b):
    ax1,ay1,aw,ah=a; ax2,ay2=ax1+aw,ay1+ah
    bx1,by1,bw,bh=b; bx2,by2=bx1+bw,by1+bh
    ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1)
    inter=iw*ih
    return inter/(aw*ah+bw*bh-inter+1e-6)

def merge_boxes(boxes, thr=0.30, loops=3):
    boxes = boxes[:]
    for _ in range(loops):
        if not boxes: break
        used=[False]*len(boxes)
        out=[]
        for i,a in enumerate(boxes):
            if used[i]: continue
            ax1,ay1,aw,ah=a; ax2,ay2=ax1+aw,ay1+ah
            for j,b in enumerate(boxes):
                if i==j or used[j]: continue
                if iou(a,b)>=thr:
                    bx1,by1,bw,bh=b; bx2,by2=bx1+bw,by1+bh
                    ax1,ay1=min(ax1,bx1),min(ay1,by1)
                    ax2,ay2=max(ax2,bx2),max(ay2,by2)
                    used[j]=True
            used[i]=True
            out.append((ax1,ay1,ax2-ax1,ay2-ay1))
        if len(out)==len(boxes): break
        boxes=out
    return boxes

# ---------- Sky mask ----------
def apply_clahe_v(hsv, clip=2.0, tile=(8,8)):
    H,S,V=cv2.split(hsv)
    V=cv2.createCLAHE(clip,tile).apply(V)
    return cv2.merge([H,S,V])

def auto_sky_range(hsv):
    H,S,V=cv2.split(hsv)
    h_m,s_m,v_m=np.median(H),np.median(S),np.median(V)
    if s_m>70 and v_m>140: low,high=(max(80,h_m-28),15,60),(min(140,h_m+28),255,255)
    elif v_m<120 and s_m<80: low,high=(0,0,35),(179,110,230)
    elif v_m>160 and (h_m<30 or h_m>160): low,high=(0,20,65),(55,255,255)
    else: low,high=(68,10,50),(152,255,255)
    return tuple(map(int,low)),tuple(map(int,high))

def make_sky_mask(bgr):
    hsv = apply_clahe_v(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))
    low, high = auto_sky_range(hsv)
    sky = cv2.inRange(hsv, low, high)
    clouds = cv2.inRange(hsv, (0,0,85), (179,100,255))
    sky = cv2.bitwise_or(sky, clouds)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    sky = cv2.morphologyEx(sky, cv2.MORPH_CLOSE, k)
    sky = cv2.morphologyEx(sky, cv2.MORPH_OPEN,  k)
    return sky

# ---------- Black-hat ----------
def multi_scale_blackhat(gray):
    # เน้นขอบเข้ม + ขยาย blob
    sizes = [9,13,17]  # <-- เดิม [7, 11, 15]
    acc = np.zeros_like(gray, np.float32)
    for s in sizes:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s, s))
        acc += cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k).astype(np.float32)
    
    out = np.clip(acc / len(sizes), 0, 255).astype(np.uint8)

    # เพิ่มรอบ Dilate ให้หนาขึ้น
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    for _ in range(5):     # เดิม 3 → ลอง 5
        out = cv2.dilate(out, k2, iterations=1)
    
    return out


# ---------- ROI proposal ----------
def propose_roi(img_u8, sky_mask=None, perc=96.0, min_area=20, pad=16):
    H,W=img_u8.shape
    work=img_u8.copy()
    if sky_mask is not None:
        work=cv2.bitwise_and(work,work,mask=sky_mask)
    work=cv2.GaussianBlur(work,(5,5),0)
    nz=work[work>0]
    t=int(np.percentile(nz, perc)) if nz.size else 255
    _,bin_hard=cv2.threshold(work,t,255,cv2.THRESH_BINARY)
    bin_hard=cv2.morphologyEx(bin_hard,cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
    bin_hard=cv2.dilate(bin_hard, np.ones((5,5),np.uint8), 3)
    cnts,_=cv2.findContours(bin_hard,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    rois=[]
    for c in cnts:
        if cv2.contourArea(c)<min_area: continue
        x,y,w,h=cv2.boundingRect(c)
        x,y,w,h=clamp_box(x-pad,y-pad,w+2*pad,h+2*pad,W,H)
        rois.append((x,y,w,h))
    return merge_boxes(rois,0.30)

# ---------- Detection ----------
def detect_in_rois(src_u8, rois, area_min=12, area_max=15000):
    boxes=[]
    for (rx,ry,rw,rh) in rois:
        sub=src_u8[ry:ry+rh, rx:rx+rw]
        _,b=cv2.threshold(sub,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        b=cv2.morphologyEx(b,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
        cnts,_=cv2.findContours(b,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            a=cv2.contourArea(c)
            if a<area_min or a>area_max: continue
            x,y,w,h=cv2.boundingRect(c)
            X,Y,W,H=rx+x,ry+y,w,h
            boxes.append((X,Y,W,H))
    return boxes

# ---------- Main pipeline ----------
def process_image(path):
    bgr=cv2.imread(path)
    if bgr is None: return None,[],None
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)

    sky_mask=make_sky_mask(bgr)
    gray=cv2.bitwise_and(gray,gray,mask=sky_mask)
    bh_raw=multi_scale_blackhat(gray)

    rois=propose_roi(bh_raw, sky_mask=sky_mask, perc=96.0)
    boxes=detect_in_rois(bh_raw, rois, area_min=15)

    boxes=merge_boxes(boxes,0.35)
    vis=rgb.copy()
    for (x,y,w,h) in boxes:
        cv2.rectangle(vis,(x,y),(x+w,y+h),(255,0,0),DRAW_THICK)
    cv2.putText(vis,f"Detected Drones (N={len(boxes)})",(16,40),
                cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,0,0),2,cv2.LINE_AA)
    return vis,boxes,bh_raw

# ---------- Run folder ----------
paths=[]
for e in IMG_EXTS: paths+=glob(os.path.join(INPUT_FOLDER,e))
paths=sorted(paths)
results=[]
for p in paths:
    fname=os.path.basename(p)
    print("Processing:",fname)
    vis,boxes,bh=process_image(p)
    if vis is None: continue
    cv2.imwrite(os.path.join(OUTPUT_FOLDER,fname),cv2.cvtColor(vis,cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(BH_FOLDER,os.path.splitext(fname)[0]+"_bh.png"),bh)
    results.append((fname,len(boxes)))
    print(f"  -> drones={len(boxes)}")

with open(os.path.join(OUTPUT_FOLDER,"results_log.csv"),"w",newline='') as f:
    w=csv.writer(f); w.writerow(["filename","num_drones"]); w.writerows(results)
print("✅ Done (v3.7 strong-dilate)")
