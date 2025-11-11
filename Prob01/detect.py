# Final Drone Detector v3  (folder mode)
# - Auto-sky mask + CLAHE
# - Auto pole removal (Hough + morphology)
# - OSD top-right masking
# - Black-hat enhancer
# - ROI propose (percentile w/ gloomy-sky adaptation)
# - Shape + Texture (cloud) filtering
# - NMS-like merge
# - Save annotated images + results_log.csv

import os
from glob import glob
import csv
import cv2
import numpy as np

# ============== CONFIG =================
INPUT_FOLDER  = 'TEST_DATA'           # โฟลเดอร์ภาพเข้า
OUTPUT_FOLDER = 'RESULTS'             # โฟลเดอร์บันทึกผล
IMG_EXTS = ('*.jpg','*.jpeg','*.png') # นามสกุลภาพ
DRAW_THICK = 2                        # ความหนาเส้นกรอบ
OSD_TOP_FRAC = 0.15                   # สัดส่วนความสูงมุมขวาบนที่มี OSD
OSD_LEFT_FRAC = 0.65                  # สัดส่วนความกว้างจากซ้ายที่เริ่ม OSD
# ======================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------- Utils ----------
def clamp_box(x, y, w, h, W, H):
    x = max(0, x); y = max(0, y)
    w = min(w, W - x); h = min(h, H - y)
    return x, y, w, h

def iou_box(a, b):
    ax1, ay1, aw, ah = a; ax2, ay2 = ax1+aw, ay1+ah
    bx1, by1, bw, bh = b; bx2, by2 = bx1+bw, by1+bh
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    union = aw*ah + bw*bh - inter + 1e-6
    return inter/union

def merge_overlapping_boxes(boxes, iou_thr=0.2, max_loops=3):
    boxes = boxes[:]
    for _ in range(max_loops):
        if not boxes: break
        used = [False]*len(boxes)
        merged = []
        for i,a in enumerate(boxes):
            if used[i]: continue
            ax1,ay1,aw,ah = a
            ax2,ay2 = ax1+aw, ay1+ah
            for j,b in enumerate(boxes):
                if i==j or used[j]: continue
                if iou_box(a,b) >= iou_thr:
                    bx1,by1,bw,bh = b
                    bx2,by2 = bx1+bw, by1+bh
                    ax1, ay1 = min(ax1,bx1), min(ay1,by1)
                    ax2, ay2 = max(ax2,bx2), max(ay2,by2)
                    used[j] = True
            used[i] = True
            merged.append((ax1, ay1, ax2-ax1, ay2-ay1))
        if len(merged)==len(boxes): break
        boxes = merged
    return boxes

# ---------- Sky mask ----------
def apply_clahe_v(hsv, clip=2.0, tile=(8,8)):
    H,S,V = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    Vc = clahe.apply(V)
    return cv2.merge([H,S,Vc])

def auto_sky_range(hsv):
    H,S,V = cv2.split(hsv)
    h_m, s_m, v_m = np.median(H), np.median(S), np.median(V)
    if s_m > 70 and v_m > 140:      # ฟ้าใส
        low, high = (max(80, h_m-28), 15, 60), (min(140, h_m+28), 255, 255)
    elif v_m < 120 and s_m < 80:    # ฟ้าครึ้มเทา
        low, high = (0, 0, 35), (179, 110, 230)
    elif v_m > 160 and (h_m < 30 or h_m > 160):  # ยามเย็น/โทนส้มชมพู
        low, high = (0, 20, 65), (55, 255, 255)
    else:                            # ปกติ
        low, high = (68, 10, 50), (152, 255, 255)
    return tuple(map(int, low)), tuple(map(int, high))

def make_sky_mask(bgr):
    h, w = bgr.shape[:2]
    hsv = apply_clahe_v(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))
    low, high = auto_sky_range(hsv)
    sky = cv2.inRange(hsv, low, high)
    clouds = cv2.inRange(hsv, (0,0,85), (179,100,255))   # เทา/ขาว
    sky = cv2.bitwise_or(sky, clouds)
    # กันพื้น/อัฒจันทร์/OSD
    sky[int(0.60*h):, :] = 0
    sky[0:int(OSD_TOP_FRAC*h), int(OSD_LEFT_FRAC*w):] = 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    sky = cv2.morphologyEx(sky, cv2.MORPH_CLOSE, k)
    sky = cv2.morphologyEx(sky, cv2.MORPH_OPEN,  k)
    return sky

# ---------- Auto pole removal ----------
def remove_poles_auto(gray):
    """ลบเสาไฟแนวตั้งจากภาพเทา (sky-only) โดยอัตโนมัติ"""
    H, W = gray.shape
    work = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(work, 40, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120,
                            minLineLength=int(H*0.25), maxLineGap=20)
    mask = np.ones_like(gray, np.uint8) * 255
    if lines is not None:
        for l in lines:
            x1,y1,x2,y2 = l[0]
            if abs(x2-x1) < 20 and abs(y2-y1) > H*0.25:  # เส้นแนวตั้งจริง
                cv2.rectangle(mask, (max(0,x1-10), 0), (min(W,x2+10), H), 0, -1)
    # ปิด OSD มุมขวาบน
    mask[0:int(OSD_TOP_FRAC*H), int(OSD_LEFT_FRAC*W):] = 0
    return cv2.bitwise_and(gray, gray, mask=mask)

# ---------- Enhancer ----------
def multi_scale_blackhat(gray):
    sizes = [7, 11, 15]
    acc = np.zeros_like(gray, np.float32)
    for s in sizes:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s, s))
        acc += cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k).astype(np.float32)
    out = np.clip(acc/len(sizes), 0, 255).astype(np.uint8)
    return cv2.dilate(out, np.ones((3,3), np.uint8), 2)

# ---------- ROI proposal ----------
def propose_roi_from_blackhat(enh, sky_mask=None, min_area=120, pad=10):
    H, W = enh.shape
    work = enh.copy()
    if sky_mask is not None:
        work = cv2.bitwise_and(work, work, mask=sky_mask)
    work = cv2.GaussianBlur(work, (5,5), 0)

    # gloomy-sky adaptation
    nz = work[work > 0]
    mean_val = float(np.mean(nz)) if nz.size > 0 else 0
    perc = 98.5 if mean_val < 80 else 99.0

    t = int(np.percentile(nz, perc)) if nz.size > 0 else 255
    _, bin_hard = cv2.threshold(work, t, 255, cv2.THRESH_BINARY)

    bin_hard = cv2.morphologyEx(bin_hard, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
    bin_hard = cv2.dilate(bin_hard, np.ones((3,3), np.uint8), 2)

    # ตัดส่วนล่าง/OSD
    bin_hard[int(0.58*H):, :] = 0
    bin_hard[0:int(OSD_TOP_FRAC*H), int(OSD_LEFT_FRAC*W):] = 0

    cnts, _ = cv2.findContours(bin_hard, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area: 
            continue
        x,y,w,h = cv2.boundingRect(c)
        aspect = w/float(h+1e-6)
        if aspect < 0.5 or aspect > 3.0:
            continue
        hull = cv2.convexHull(c)
        solidity = a / (cv2.contourArea(hull)+1e-6)
        if solidity < 0.6:
            continue
        x,y,w,h = clamp_box(x-pad, y-pad, w+2*pad, h+2*pad, W,H)
        rois.append((x,y,w,h))

    return merge_overlapping_boxes(rois, iou_thr=0.30)

# ---------- Cloud / texture filter ----------
def is_cloud_like(region):
    """Return True if patch looks like cloud (low texture & edges)."""
    if region.size == 0:
        return True
    var = float(np.var(region))
    if var < 40:           # เรียบเกินไป -> เมฆ
        return True
    edges = cv2.Canny(region, 40, 120)
    density = np.count_nonzero(edges) / (region.shape[0]*region.shape[1] + 1e-6)
    if density < 0.02:     # ขอบน้อย -> เมฆ
        return True
    return False

# ---------- Main per-image pipeline ----------
def process_image(path):
    bgr = cv2.imread(path)
    if bgr is None:
        return None, []

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Sky-only
    sky_mask = make_sky_mask(bgr)
    sky_only = cv2.bitwise_and(rgb, rgb, mask=sky_mask)
    gray = cv2.cvtColor(sky_only, cv2.COLOR_RGB2GRAY)

    # Remove poles + OSD automatically
    gray = remove_poles_auto(gray)

    # Enhancer
    enh = multi_scale_blackhat(gray)

    # ROI propose
    rois = propose_roi_from_blackhat(enh, sky_mask=sky_mask)

    # Fine detection inside ROIs
    final_boxes = []
    for (rx, ry, rw, rh) in rois:
        sub = enh[ry:ry+rh, rx:rx+rw]
        # Adaptive threshold in ROI (robust on gradients)
        sub_bin = cv2.adaptiveThreshold(sub, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 21, -5)
        sub_bin = cv2.morphologyEx(sub_bin, cv2.MORPH_OPEN,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
        sub_bin = cv2.morphologyEx(sub_bin, cv2.MORPH_CLOSE,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))

        cnts,_ = cv2.findContours(sub_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 80 or area > 5000:      # scale ได้ตามความละเอียดภาพ
                continue
            x,y,w,h = cv2.boundingRect(c)
            X,Y,W,H = rx+x, ry+y, w, h

            # Shape filters
            aspect = W/float(H+1e-6)
            extent = area/float(W*H+1e-6)
            hull = cv2.convexHull(c)
            solidity = area/(cv2.contourArea(hull)+1e-6)
            circ = 4*np.pi*area/((cv2.arcLength(c, True)+1e-6)**2)

            if not (0.4 <= aspect <= 3.0):  continue
            if extent < 0.25:               continue
            if solidity < 0.6:              continue
            if circ > 0.9:                  continue  # กลมมาก ๆ -> เมฆ/นก

            # Cloud-like reject
            roi_gray = enh[Y:Y+H, X:X+W]
            if is_cloud_like(roi_gray):
                continue

            final_boxes.append((X,Y,W,H))

    final_boxes = merge_overlapping_boxes(final_boxes, iou_thr=0.35)

    # Draw
    vis = rgb.copy()
    for (x,y,w,h) in final_boxes:
        cv2.rectangle(vis, (x,y), (x+w,y+h), (255,0,0), DRAW_THICK)
    cv2.putText(vis, f"Detected Drones (N={len(final_boxes)})", (16,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0), 2, cv2.LINE_AA)

    return vis, final_boxes

# ---------- Run on folder ----------
results = []
paths = []
for ext in IMG_EXTS:
    paths.extend(glob(os.path.join(INPUT_FOLDER, ext)))
paths = sorted(paths)

for p in paths:
    fname = os.path.basename(p)
    print(f"Processing: {fname}")
    vis, boxes = process_image(p)
    if vis is None:
        print("  ⚠️ cannot read, skipped.")
        continue
    out_path = os.path.join(OUTPUT_FOLDER, fname)
    cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"  ✅ saved {out_path}  drones={len(boxes)}")
    results.append((fname, len(boxes)))

# Save CSV log
csv_path = os.path.join(OUTPUT_FOLDER, "results_log.csv")
with open(csv_path, "w", newline='') as f:
    w = csv.writer(f)
    w.writerow(["filename","num_drones"])
    for r in results:
        w.writerow(list(r))
print(f"📊 log saved: {csv_path}")
print("✅ All images processed.")
