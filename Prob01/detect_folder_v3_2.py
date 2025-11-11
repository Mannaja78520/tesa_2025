# detect_folder_v3_3.py
# Final Drone Detector v3.3 (Small-drones tuned)
# - Folder mode with logs & blackhat saving
# - Edge-gated black-hat + adaptive ROI (gloomy aware)
# - Cloud filter (relaxed to avoid missing tiny drones)
# - Merge boxes + Save annotated + Save blackhat (raw & gated) + CSV log

import os, csv, cv2, numpy as np
from glob import glob

# ================ CONFIG =================
INPUT_FOLDER   = 'TEST_DATA'
OUTPUT_FOLDER  = 'RESULTS'
BH_FOLDER      = 'RESULTS_BLACKHAT'
IMG_EXTS       = ('*.jpg','*.jpeg','*.png')
DRAW_THICK     = 2
OSD_TOP_FRAC   = 0.15
OSD_LEFT_FRAC  = 0.65
DEBUG_REJECTS  = True                 # เซฟแพตช์ที่ถูกกรองทิ้ง
DEBUG_DIR      = 'DEBUG_REJECTS'
# ========================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(BH_FOLDER, exist_ok=True)
if DEBUG_REJECTS: os.makedirs(DEBUG_DIR, exist_ok=True)

# --------------- Utils ----------------
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

def merge_overlapping_boxes(boxes, iou_thr=0.35, max_loops=3):
    boxes = boxes[:]
    for _ in range(max_loops):
        if not boxes: break
        used = [False]*len(boxes); merged = []
        for i,a in enumerate(boxes):
            if used[i]: continue
            ax1,ay1,aw,ah = a; ax2,ay2 = ax1+aw, ay1+ah
            for j,b in enumerate(boxes):
                if i==j or used[j]: continue
                if iou_box(a,b) >= iou_thr:
                    bx1,by1,bw,bh = b; bx2,by2 = bx1+bw, by1+bh
                    ax1, ay1 = min(ax1,bx1), min(ay1,by1)
                    ax2, ay2 = max(ax2,bx2), max(ay2,by2)
                    used[j] = True
            used[i] = True; merged.append((ax1, ay1, ax2-ax1, ay2-ay1))
        if len(merged)==len(boxes): break
        boxes = merged
    return boxes

# --------------- Sky mask ---------------
def apply_clahe_v(hsv, clip=2.0, tile=(8,8)):
    H,S,V = cv2.split(hsv)
    V = cv2.createCLAHE(clip, tile).apply(V)
    return cv2.merge([H,S,V])

def auto_sky_range(hsv):
    H,S,V = cv2.split(hsv)
    h_m, s_m, v_m = np.median(H), np.median(S), np.median(V)
    if s_m > 70 and v_m > 140:      low, high = (max(80,h_m-28),15,60), (min(140,h_m+28),255,255)
    elif v_m < 120 and s_m < 80:    low, high = (0,0,35), (179,110,230)
    elif v_m > 160 and (h_m < 30 or h_m > 160): low, high = (0,20,65), (55,255,255)
    else:                            low, high = (68,10,50), (152,255,255)
    return tuple(map(int,low)), tuple(map(int,high))

def make_sky_mask(bgr):
    h, w = bgr.shape[:2]
    hsv = apply_clahe_v(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))
    low, high = auto_sky_range(hsv)
    sky = cv2.inRange(hsv, low, high)
    clouds = cv2.inRange(hsv, (0,0,85), (179,100,255))
    sky = cv2.bitwise_or(sky, clouds)
    sky[int(0.60*h):, :] = 0
    sky[0:int(OSD_TOP_FRAC*h), int(OSD_LEFT_FRAC*w):] = 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    sky = cv2.morphologyEx(sky, cv2.MORPH_CLOSE, k)
    sky = cv2.morphologyEx(sky, cv2.MORPH_OPEN,  k)
    return sky

# --------- Auto pole removal ----------
def remove_poles_auto(gray):
    H, W = gray.shape
    work = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(work, 40, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120,
                            minLineLength=int(H*0.25), maxLineGap=20)
    mask = np.ones_like(gray, np.uint8) * 255
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0,:]:
            if abs(x2-x1) < 20 and abs(y2-y1) > H*0.25:
                cv2.rectangle(mask, (max(0,x1-10),0), (min(W,x2+10),H), 0, -1)
    mask[0:int(OSD_TOP_FRAC*H), int(OSD_LEFT_FRAC*W):] = 0
    return cv2.bitwise_and(gray, gray, mask=mask)

# --------------- Enhancer ---------------
def multi_scale_blackhat(gray):
    sizes = [7, 11, 15]
    acc = np.zeros_like(gray, np.float32)
    for s in sizes:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s, s))
        acc += cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k).astype(np.float32)
    out = np.clip(acc/len(sizes), 0, 255).astype(np.uint8)
    return cv2.dilate(out, np.ones((3,3), np.uint8), 2)

# -------- ROI proposal (recall boost) --------
def propose_roi_from_blackhat(enh, sky_mask=None, min_area=25, pad=18):
    H, W = enh.shape
    work = enh.copy()
    if sky_mask is not None:
        work = cv2.bitwise_and(work, work, mask=sky_mask)
    work = cv2.GaussianBlur(work, (5,5), 0)

    nz = work[work > 0]
    mean_val = float(np.mean(nz)) if nz.size else 0
    perc = 99.1 if mean_val < 80 else 98.7   # อ่อนลงเพื่อเก็บโดรนเล็ก

    t = int(np.percentile(nz, perc)) if nz.size else 255
    _, bin_hard = cv2.threshold(work, t, 255, cv2.THRESH_BINARY)

    bin_hard = cv2.morphologyEx(bin_hard, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
    bin_hard = cv2.dilate(bin_hard, np.ones((3,3), np.uint8), 2)

    # ตัดส่วนล่าง & OSD
    bin_hard[int(0.58*H):, :] = 0
    bin_hard[0:int(OSD_TOP_FRAC*H), int(OSD_LEFT_FRAC*W):] = 0

    # Edge gate ซ้ำ
    eg = cv2.Canny(work, 60, 150)
    eg = cv2.dilate(eg, np.ones((3,3), np.uint8), 1)
    bin_hard = cv2.bitwise_and(bin_hard, eg)

    cnts, _ = cv2.findContours(bin_hard, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area: continue
        x,y,w,h = cv2.boundingRect(c)
        aspect = w/float(h+1e-6)
        if not (0.45 <= aspect <= 3.2): continue
        hull = cv2.convexHull(c)
        solidity = a / (cv2.contourArea(hull)+1e-6)
        if solidity < 0.55: continue
        x,y,w,h = clamp_box(x-pad, y-pad, w+2*pad, h+2*pad, W,H)
        rois.append((x,y,w,h))
    return merge_overlapping_boxes(rois, iou_thr=0.30)

# -------- Cloud / texture filter (relaxed) --------
def _entropy(img_u8):
    hist = cv2.calcHist([img_u8],[0],None,[256],[0,256]).flatten()
    p = hist / (img_u8.size + 1e-6); p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def is_cloud_like(region):
    if region.size == 0: return True

    var = float(np.var(region))
    if var < 35: return True             # เดิม 45

    edges = cv2.Canny(region, 60, 150)
    edge_density = np.count_nonzero(edges) / (region.size + 1e-6)
    if edge_density < 0.018: return True # เดิม 0.025

    lap = cv2.Laplacian(region, cv2.CV_32F, ksize=3)
    hf_ratio = (np.abs(lap) > 8).mean()
    if hf_ratio < 0.007: return True     # เดิม 0.010

    ent = _entropy(region)
    if ent < 4.0: return True            # เดิม 4.4

    return False

# ------------- Per-image pipeline -------------
def process_image(path):
    bgr = cv2.imread(path)
    if bgr is None:
        return None, [], None, None

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Sky-only
    sky_mask = make_sky_mask(bgr)
    sky_only = cv2.bitwise_and(rgb, rgb, mask=sky_mask)
    gray = cv2.cvtColor(sky_only, cv2.COLOR_RGB2GRAY)

    # Remove poles + OSD
    gray = remove_poles_auto(gray)

    # Black-hat + Edge gate
    enh_raw = multi_scale_blackhat(gray)               # BH เดิม
    edges = cv2.Canny(gray, 60, 150)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), 1)
    enh = cv2.bitwise_and(enh_raw, enh_raw, mask=edges)   # BH หลัง gate

    # ROI propose
    rois = propose_roi_from_blackhat(enh, sky_mask=sky_mask)

    # Fine detection in ROIs
    final_boxes = []
    for (rx, ry, rw, rh) in rois:
        sub = enh[ry:ry+rh, rx:rx+rw]
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
            if area < 20 or area > 8000:  # ลดขั้นต่ำเพื่อไม่พลาดโดรนเล็ก
                continue
            x,y,w,h = cv2.boundingRect(c)
            X,Y,W,H = rx+x, ry+y, w, h

            # Shape filters
            aspect = W/float(H+1e-6)
            extent = area/float(W*H+1e-6)
            hull = cv2.convexHull(c)
            solidity = area/(cv2.contourArea(hull)+1e-6)
            circ = 4*np.pi*area/((cv2.arcLength(c, True)+1e-6)**2)
            if not (0.35 <= aspect <= 3.3): continue
            if extent < 0.18: continue
            if solidity < 0.50: continue
            if circ > 0.93: continue

            roi_gray = enh[Y:Y+H, X:X+W]
            if is_cloud_like(roi_gray):
                if DEBUG_REJECTS:
                    cv2.imwrite(os.path.join(
                        DEBUG_DIR, f"{os.path.splitext(os.path.basename(path))[0]}_{X}_{Y}_{W}x{H}.png"
                    ), roi_gray)
                continue

            final_boxes.append((X,Y,W,H))

    final_boxes = merge_overlapping_boxes(final_boxes, iou_thr=0.35)

    # Draw
    vis = rgb.copy()
    for (x,y,w,h) in final_boxes:
        cv2.rectangle(vis, (x,y), (x+w,y+h), (255,0,0), DRAW_THICK)
    cv2.putText(vis, f"Detected Drones (N={len(final_boxes)})", (16,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0), 2, cv2.LINE_AA)

    return vis, final_boxes, enh_raw, enh

# ---------------- Run on folder ----------------
results = []
paths = []
for ext in IMG_EXTS: paths.extend(glob(os.path.join(INPUT_FOLDER, ext)))
paths = sorted(paths)

for p in paths:
    fname = os.path.basename(p)
    print(f"Processing: {fname}")
    vis, boxes, bh_raw, bh_gate = process_image(p)
    if vis is None:
        print("  ⚠️ cannot read, skipped."); continue

    # Save annotated
    out_path = os.path.join(OUTPUT_FOLDER, fname)
    cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    # Save black-hat raw & gated (+ colormap)
    stem = os.path.splitext(fname)[0]
    raw_path  = os.path.join(BH_FOLDER, f"{stem}_bh_raw.png")
    gate_path = os.path.join(BH_FOLDER, f"{stem}_bh_gate.png")
    cv2.imwrite(raw_path,  bh_raw)
    cv2.imwrite(gate_path, bh_gate)
    cv2.imwrite(os.path.join(BH_FOLDER, f"{stem}_bh_raw_jet.png"),
                cv2.applyColorMap(bh_raw,  cv2.COLORMAP_JET))
    cv2.imwrite(os.path.join(BH_FOLDER, f"{stem}_bh_gate_jet.png"),
                cv2.applyColorMap(bh_gate, cv2.COLORMAP_JET))

    print(f"  ✅ saved {out_path}  drones={len(boxes)} | BH saved")
    results.append((fname, len(boxes)))

# Save CSV log
csv_path = os.path.join(OUTPUT_FOLDER, "results_log.csv")
with open(csv_path, "w", newline='') as f:
    w = csv.writer(f); w.writerow(["filename","num_drones"])
    for r in results: w.writerow(list(r))
print(f"📊 log saved: {csv_path}")
print(f"🖼️ black-hat images in: {BH_FOLDER}")
if DEBUG_REJECTS: print(f"🧪 rejects saved in: {DEBUG_DIR}")
print("✅ All images processed.")
