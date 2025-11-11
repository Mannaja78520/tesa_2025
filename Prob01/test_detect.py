# !pip install opencv-python
import cv2, numpy as np, matplotlib.pyplot as plt

# ---------- Knobs (ปรับตามภาพของคุณ) ----------
IMG_PATH = 'TEST_DATA/test_0003.jpg'
MIN_AREA   = 600        # พื้นที่ขั้นต่ำของโดรน (px^2)
MAX_AREA   = 2e5        # กันวัตถุใหญ่อย่างเมฆทึบ/อาคาร
AR_MIN, AR_MAX = 0.6, 4.0   # อัตราส่วน w/h ที่ยอมรับ
EXTENT_MIN = 0.15          # area / (w*h) – ความแน่น
SOLIDITY_MIN = 0.4         # area / hull_area – ความทึบ
MERGE_IOU = 0.2            # ค่าต่ำ ๆ เพื่อรวมกล่องใกล้กัน
BLACKHAT_SIZE = 13         # ขนาดวัตถุที่คาด (px) → ปรับให้สอดคล้องกับโดรนในภาพ
CLEAN_KERNEL = 9           # kernel ปิด/เปิดสำหรับทำความสะอาด mask

# ---------- Utils ----------
def nms(boxes, scores=None, iou_thr=0.3):
    if len(boxes) == 0: return []
    boxes = np.array(boxes, dtype=np.float32)
    if scores is None: scores = np.array([1.0]*len(boxes), dtype=np.float32)
    else: scores = np.array(scores, dtype=np.float32)

    x1, y1 = boxes[:,0], boxes[:,1]
    x2, y2 = boxes[:,0]+boxes[:,2], boxes[:,1]+boxes[:,3]
    areas = (x2-x1+1)*(y2-y1+1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2-xx1+1)
        h = np.maximum(0.0, yy2-yy1+1)
        inter = w*h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds+1]
    return boxes[keep].astype(int).tolist()

# ---------- 1) Load & show ----------
bgr = cv2.imread(IMG_PATH)
if bgr is None:
    raise FileNotFoundError(f"Cannot read image at {IMG_PATH}")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
plt.imshow(rgb); plt.axis('off'); plt.title("Original"); plt.show()

# ---------- 2) HSV & Sky/Cloud mask (inRange) ----------
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
H, S, V = cv2.split(hsv)

# ท้องฟ้าฟ้า: Hue ~ [90,135] ใน OpenCV (0-179), ความสว่างพอสมควร
sky_blue = cv2.inRange(hsv, (90,  0,  80), (135, 255, 255))
# เมฆ: สีซีด/ขาว – S ต่ำ, V สูง
clouds   = cv2.inRange(hsv, (0,   0, 180), (179,  40, 255))
sky_mask = cv2.bitwise_or(sky_blue, clouds)

# ทำความสะอาด
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLEAN_KERNEL, CLEAN_KERNEL))
sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_CLOSE, k)
sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_OPEN,  k)
plt.imshow(sky_mask, cmap='gray'); plt.axis('off'); plt.title("Sky Mask"); plt.show()

# ---------- 3) Keep sky only ----------
sky_only = cv2.bitwise_and(rgb, rgb, mask=sky_mask)
plt.imshow(sky_only); plt.axis('off'); plt.title("Sky Only"); plt.show()

# ---------- 4) Black-hat emphasize dark small objects ----------
gray = cv2.cvtColor(sky_only, cv2.COLOR_RGB2GRAY)
# ปรับ kernel ให้สัมพันธ์กับขนาดโดรนในภาพ (ยิ่งโดรนใหญ่ → kernel ใหญ่ขึ้น)
k_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BLACKHAT_SIZE, BLACKHAT_SIZE))
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k_bh)
# ขยายสัญญาณเล็กน้อย
blackhat = cv2.dilate(blackhat, np.ones((3,3), np.uint8), iterations=1)
plt.imshow(blackhat, cmap='gray'); plt.axis('off'); plt.title("Black-hat"); plt.show()

# ---------- 5) Threshold + clean small noise ----------
_, cand = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
plt.imshow(cand, cmap='gray'); plt.axis('off'); plt.title("Candidates"); plt.show()

# ---------- 6) Contours + shape filters ----------
cnts, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
boxes, scores = [], []

for c in cnts:
    area = cv2.contourArea(c)
    if area < MIN_AREA or area > MAX_AREA: 
        continue

    x,y,w,h = cv2.boundingRect(c)
    ar = w / max(h,1)
    extent = area / float(w*h + 1e-6)

    hull = cv2.convexHull(c)
    solidity = area / (cv2.contourArea(hull) + 1e-6)

    if not (AR_MIN <= ar <= AR_MAX): 
        continue
    if extent < EXTENT_MIN: 
        continue
    if solidity < SOLIDITY_MIN: 
        continue

    boxes.append((x,y,w,h))
    # ใช้คะแนนจากค่าความเข้มเฉลี่ยใน blackhat ภายในกล่องเป็น pseudo-score
    roi = blackhat[y:y+h, x:x+w]
    scores.append(float(roi.mean()) if roi.size else 1.0)

# ---------- 7) Merge with NMS ----------
merged = nms(boxes, scores, iou_thr=MERGE_IOU)

# ---------- 8) Visualize ----------
vis = rgb.copy()
for (x,y,w,h) in merged:
    cv2.rectangle(vis, (x,y), (x+w,y+h), (255,0,0), 2)

plt.imshow(vis); plt.axis('off'); plt.title(f"Detections (N={len(merged)})"); plt.show()
print("Detections:", merged)
