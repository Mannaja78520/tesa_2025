# detect_folder_v3_5_bhraw.py
# Drone Detector (Folder) — BH-RAW Fusion
# ตรวจจาก black-hat แบบ raw โดยตรง + รวมผลกับเส้นทาง gated (union)

import os, csv, cv2, numpy as np
from glob import glob

# ================= CONFIG =================
INPUT_FOLDER   = 'TEST_DATA'
OUTPUT_FOLDER  = 'RESULTS'
BH_FOLDER      = 'RESULTS_BLACKHAT'
IMG_EXTS       = ('*.jpg','*.jpeg','*.png')
DRAW_THICK     = 2
OSD_TOP_FRAC   = 0.15
OSD_LEFT_FRAC  = 0.65
DEBUG_REJECTS  = False
DEBUG_DIR      = 'DEBUG_REJECTS'
# =========================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(BH_FOLDER, exist_ok=True)
if DEBUG_REJECTS: os.makedirs(DEBUG_DIR, exist_ok=True)

# ---------------- Utils ----------------
def clamp_box(x, y, w, h, W, H):
    x = max(0, x); y = max(0, y)
    w = min(w, W - x); h = min(h, H - y)
    return x, y, w, h

def iou(a,b):
    ax1,ay1,aw,ah=a; ax2,ay2=ax1+aw,ay1+ah
    bx1,by1,bw,bh=b; bx2,by2=bx1+bw,by1+bh
    ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); inter=iw*ih
    return inter/(aw*ah+bw*bh-inter+1e-6)

def merge_boxes(boxes, thr=0.25, loops=3):
    boxes=boxes[:]
    for _ in range(loops):
        if not boxes: break
        used=[False]*len(boxes); out=[]
        for i,a in enumerate(boxes):
            if used[i]: continue
            ax1,ay1,aw,ah=a; ax2,ay2=ax1+aw,ay1+ah
            for j,b in enumerate(boxes):
                if i==j or used[j]: continue
                if iou(a,b)>=thr:
                    bx1,by1,bw,bh=b; bx2,by2=bx1+bw,by1+bh
                    ax1,ay1=min(ax1,bx1),min(ay1,by1)
                    ax2,ay2=max(ax2,bx2),max(ay2,by2); used[j]=True
            used[i]=True; out.append((ax1,ay1,ax2-ax1,ay2-ay1))
        if len(out)==len(boxes): break
        boxes=out
    return boxes

# -------------- Sky mask --------------
def apply_clahe_v(hsv, clip=2.0, tile=(8,8)):
    H,S,V = cv2.split(hsv)
    V = cv2.createCLAHE(clip, tile).apply(V)
    return cv2.merge([H,S,V])

def auto_sky_range(hsv):
    H,S,V=cv2.split(hsv); h_m,s_m,v_m=np.median(H),np.median(S),np.median(V)
    if s_m>70 and v_m>140: low,high=(max(80,h_m-28),15,60),(min(140,h_m+28),255,255)
    elif v_m<120 and s_m<80: low,high=(0,0,35),(179,110,230)
    elif v_m>160 and (h_m<30 or h_m>160): low,high=(0,20,65),(55,255,255)
    else: low,high=(68,10,50),(152,255,255)
    return tuple(map(int,low)),tuple(map(int,high))

def make_sky_mask(bgr):
    h,w=bgr.shape[:2]
    hsv=apply_clahe_v(cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV))
    low,high=auto_sky_range(hsv)
    sky=cv2.inRange(hsv,low,high)
    clouds=cv2.inRange(hsv,(0,0,85),(179,100,255))
    sky=cv2.bitwise_or(sky,clouds)
    sky[int(0.60*h):,:]=0
    sky[0:int(OSD_TOP_FRAC*h),int(OSD_LEFT_FRAC*w):]=0
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    sky=cv2.morphologyEx(sky,cv2.MORPH_CLOSE,k)
    sky=cv2.morphologyEx(sky,cv2.MORPH_OPEN,k)
    return sky

# ---------- Remove poles (vertical) ----------
def remove_poles_auto(gray):
    H,W=gray.shape
    edges=cv2.Canny(cv2.GaussianBlur(gray,(5,5),0),80,200)
    lines=cv2.HoughLinesP(edges,1,np.pi/180,120,minLineLength=int(H*0.25),maxLineGap=20)
    mask=np.ones_like(gray,np.uint8)*255
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0,:]:
            angle=abs(np.degrees(np.arctan2((y2-y1),(x2-x1))))
            if angle>85:
                cv2.rectangle(mask,(max(0,x1-8),0),(min(W,x2+8),H),0,-1)
    mask[0:int(OSD_TOP_FRAC*H),int(OSD_LEFT_FRAC*W):]=0
    return cv2.bitwise_and(gray,gray,mask=mask)

# -------------- Black-hat --------------
def multi_scale_blackhat(gray):
    sizes=[7,11,15]
    acc=np.zeros_like(gray,np.float32)
    for s in sizes:
        k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(s,s))
        acc+=cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,k).astype(np.float32)
    out=np.clip(acc/len(sizes),0,255).astype(np.uint8)
    return cv2.dilate(out,np.ones((3,3),np.uint8),2)

# -------- ROI (generic, reusable) --------
def propose_roi(img_u8, sky_mask=None, perc=98.8, min_area=20, pad=16):
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
    bin_hard=cv2.dilate(bin_hard,np.ones((3,3),np.uint8),2)

    bin_hard[int(0.58*H):,:]=0
    bin_hard[0:int(OSD_TOP_FRAC*H),int(OSD_LEFT_FRAC*W):]=0

    cnts,_=cv2.findContours(bin_hard,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    rois=[]
    for c in cnts:
        a=cv2.contourArea(c)
        if a<min_area: continue
        x,y,w,h=cv2.boundingRect(c)
        x,y,w,h=clamp_box(x-pad,y-pad,w+2*pad,h+2*pad,W,H)
        rois.append((x,y,w,h))
    return merge_boxes(rois,0.30)

# -------- Cloud-ish reject (light) --------
def _entropy(img_u8):
    hist=cv2.calcHist([img_u8],[0],None,[256],[0,256]).flatten()
    p=hist/(img_u8.size+1e-6); p=p[p>0]
    return float(-(p*np.log2(p)).sum())

def is_cloud_like(patch):
    if patch.size==0: return True
    var=float(np.var(patch))
    if var<30: return True
    edges=cv2.Canny(patch,80,200)
    edge_den=np.count_nonzero(edges)/(patch.size+1e-6)
    if edge_den<0.02: return True
    ent=_entropy(patch)
    if ent<3.8: return True
    return False

# -------- Detect inside ROIs --------
def detect_in_rois(src_u8, rois, area_min=12, area_max=12000):
    boxes=[]
    for (rx,ry,rw,rh) in rois:
        sub=src_u8[ry:ry+rh, rx:rx+rw]
        # ใช้ Otsu + refine
        _,b=cv2.threshold(sub,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        b=cv2.morphologyEx(b,cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
        b=cv2.morphologyEx(b,cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
        cnts,_=cv2.findContours(b,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area=cv2.contourArea(c)
            if area<area_min or area>area_max: continue
            x,y,w,h=cv2.boundingRect(c)
            X,Y,W,H=rx+x,ry+y,w,h
            # shape filter อ่อน
            aspect=W/float(H+1e-6)
            if not (0.3<=aspect<=3.5): continue
            if is_cloud_like(src_u8[Y:Y+H, X:X+W]): continue
            boxes.append((X,Y,W,H))
    return boxes

# ------------- Pipeline per-image -------------
def process_image(path):
    bgr=cv2.imread(path)
    if bgr is None: return None,[],None,None
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)

    sky_mask=make_sky_mask(bgr)
    gray=cv2.cvtColor(cv2.bitwise_and(rgb,rgb,mask=sky_mask),cv2.COLOR_RGB2GRAY)
    gray=remove_poles_auto(gray)

    # BH RAW
    bh_raw=multi_scale_blackhat(gray)

    # BH GATED (ตามเดิม)
    edges=cv2.Canny(gray,80,200)
    edges=cv2.dilate(edges,np.ones((3,3),np.uint8),1)
    bh_gate=cv2.bitwise_and(bh_raw,bh_raw,mask=edges)

    # --- เส้นทาง 1: ใช้ BH RAW ตรง ๆ ---
    rois_raw=propose_roi(bh_raw, sky_mask=sky_mask, perc=98.8, min_area=20, pad=14)
    boxes_raw=detect_in_rois(bh_raw, rois_raw, area_min=12)

    # --- เส้นทาง 2: ใช้ BH GATED ---
    rois_gate=propose_roi(bh_gate, sky_mask=sky_mask, perc=99.1, min_area=20, pad=14)
    boxes_gate=detect_in_rois(bh_gate, rois_gate, area_min=12)

    # รวมผล (union + merge)
    final_boxes=merge_boxes(boxes_raw+boxes_gate, thr=0.35)

    # วาดผล
    vis=rgb.copy()
    for (x,y,w,h) in final_boxes:
        cv2.rectangle(vis,(x,y),(x+w,y+h),(255,0,0),DRAW_THICK)
    cv2.putText(vis,f"Detected Drones (N={len(final_boxes)})",(16,40),
                cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,0,0),2,cv2.LINE_AA)
    return vis,final_boxes,bh_raw,bh_gate

# ---------------- Run on folder ----------------
paths=[]
for ext in IMG_EXTS: paths+=glob(os.path.join(INPUT_FOLDER,ext))
paths=sorted(paths)
results=[]
for p in paths:
    fname=os.path.basename(p)
    print("Processing:",fname)
    vis,boxes,bh_raw,bh_gate=process_image(p)
    if vis is None: 
      print("  skip (read error)"); 
      continue
    cv2.imwrite(os.path.join(OUTPUT_FOLDER,fname),cv2.cvtColor(vis,cv2.COLOR_RGB2BGR))
    stem=os.path.splitext(fname)[0]
    cv2.imwrite(os.path.join(BH_FOLDER,f"{stem}_bh_raw.png"),bh_raw)
    cv2.imwrite(os.path.join(BH_FOLDER,f"{stem}_bh_gate.png"),bh_gate)
    cv2.imwrite(os.path.join(BH_FOLDER,f"{stem}_bh_raw_jet.png"),cv2.applyColorMap(bh_raw,cv2.COLORMAP_JET))
    cv2.imwrite(os.path.join(BH_FOLDER,f"{stem}_bh_gate_jet.png"),cv2.applyColorMap(bh_gate,cv2.COLORMAP_JET))
    results.append((fname,len(boxes)))
    print(f"  -> drones={len(boxes)}")

with open(os.path.join(OUTPUT_FOLDER,"results_log.csv"),"w",newline='') as f:
    w=csv.writer(f); w.writerow(["filename","num_drones"]); w.writerows(results)
print("✅ Done.")
