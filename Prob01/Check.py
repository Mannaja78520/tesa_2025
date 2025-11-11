# !pip install opencv-python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. โหลดและแสดงภาพ (BGR→RGB)
# โหลดภาพ
bgr = cv2.imread('TEST_DATA/test_0003.jpg')
# แปลงเป็น RGB
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
# แสดงผล
plt.imshow(rgb); plt.axis('off')
plt.title("Original")
plt.show()

# 2.HSV + สร้าง Sky Mask (ท้องฟ้า + เมฆ)
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
H, S, V = cv2.split(hsv)

blue_sky = (H>=90) & (H<=135) & (S>=0) & (V>=80)
clouds   = (S<=35) & (V>=180)
sky_mask = (blue_sky | clouds).astype(np.uint8) * 255
plt.imshow(sky_mask, cmap='gray'); plt.axis('off'); plt.title("Step 1: Sky Mask"); plt.show()

# 3. ทำความสะอาด Mask ด้วย Morphology
kernel = np.ones((5,5), np.uint8)
sky_mask = cv2.erode(sky_mask, kernel, iterations=8)
plt.imshow(sky_mask, cmap='gray'); plt.axis('off'); plt.title("Sky Mask"); plt.show()

# ใช้โครงสร้างวงรีขนาดใหญ่เพื่อ close/open
kernel_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (120,120))
sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_CLOSE, kernel_big)
sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_OPEN,  kernel_big)
plt.imshow(sky_mask, cmap='gray'); plt.axis('off'); plt.title("Step 2: Sky Mask (clean)"); plt.show()

# 4. แยกภูเขา/พื้นดิน และเก็บเฉพาะท้องฟ้า
mountain_mask = cv2.bitwise_not(sky_mask)
plt.imshow(mountain_mask, cmap='gray'); plt.axis('off'); plt.title("Step 3a: Mountains/Ground Mask"); plt.show()
sky_only = cv2.bitwise_and(rgb, rgb, mask=sky_mask)
plt.imshow(sky_only); plt.axis('off'); plt.title("Step 3b: Sky Only"); plt.show()

# 5. Black-hat — เน้นวัตถุสีมืดเล็ก ๆ บนท้องฟ้าที่สว่าง
gray = cv2.cvtColor(sky_only, cv2.COLOR_RGB2GRAY)
kernel_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_bh)
plt.imshow(blackhat, cmap='gray'); plt.axis('off'); plt.title("Step 5: Black-hat"); plt.show()

# 6. ขยายสัญญาณด้วย Dilate
kernel_d = np.ones((5,5), np.uint8)
blackhat_d = cv2.dilate(blackhat, kernel_d, iterations=2)
plt.imshow(blackhat_d, cmap='gray'); plt.axis('off'); plt.title("Step 5: Black-hat (Dilated)"); plt.show()

# 7. Threshold + เปิดรูเล็ก ๆ (Open) → Candidate Mask
_, cand = cv2.threshold(blackhat_d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))

plt.imshow(cand, cmap='gray'); plt.axis('off'); plt.title("Step 7: Candidates"); plt.show()

# 8. Contour Filtering → วาดกรอบครอบโดรน
cnts, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

vis = rgb.copy()
boxes = []
for c in cnts:
    area = cv2.contourArea(c)
    if area < 1000:   # ปรับตามความละเอียดภาพ/ขนาดโดรนที่คาดหวัง
        continue
    x,y,w,h = cv2.boundingRect(c)

    boxes.append((x,y,w,h))
    cv2.rectangle(vis, (x,y), (x+w, y+h), (255,0,0), 2)

plt.imshow(vis); plt.axis('off'); plt.title(f"Step 8: Detections (N={len(boxes)})"); plt.show()
print("Detections:", boxes)
