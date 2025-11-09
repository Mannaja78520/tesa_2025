import cv2
import numpy as np

# ===== Fixed HSV for orange (คงเดิม) =====
LOWER_ORANGE = (5, 120, 120)
UPPER_ORANGE = (25, 255, 255)

def make_odd(n):
    # บังคับให้เป็นเลขคี่ขั้นต่ำ 1
    return max(1, n if n % 2 == 1 else n + 1)

def nothing(x): pass

# ===== Source: 0 = webcam (เปลี่ยนเป็น path ไฟล์ได้) =====
cap = cv2.VideoCapture(0)

# ===== Trackbar Window =====
cv2.namedWindow("Adjust", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Adjust", 1280, 720)

# Trackbars เฉพาะกรอง noise + contour
cv2.createTrackbar("Blur",        "Adjust", 7,   31,  nothing)   # Gaussian ksize
cv2.createTrackbar("Kernel",      "Adjust", 5,   31,  nothing)   # Morph kernel k
cv2.createTrackbar("Open iters",  "Adjust", 5,   10,  nothing)
cv2.createTrackbar("Close iters", "Adjust", 5,   10,  nothing)
cv2.createTrackbar("Min Area",    "Adjust", 300, 5000, nothing)

# แสดงภาพแรกให้ trackbars โผล่ชัวร์
blank = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.putText(blank, "Adjust noise filters & contour params here...",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)
cv2.imshow("Adjust", blank)
cv2.waitKey(1)

while True:
    ok, frame = cap.read()
    if not ok:
        print("✅ วิดีโอจบหรืออ่านกล้องไม่ได้")
        break

    # อ่านค่า trackbars
    blur_ksize = make_odd(cv2.getTrackbarPos("Blur",        "Adjust"))
    kern_size  = make_odd(cv2.getTrackbarPos("Kernel",      "Adjust"))
    open_it    = cv2.getTrackbarPos("Open iters",  "Adjust")
    close_it   = cv2.getTrackbarPos("Close iters", "Adjust")
    min_area   = cv2.getTrackbarPos("Min Area",    "Adjust")

    # ===== Pipeline =====
    # 1) Blur เพื่อลด noise ก่อนแปลงสี
    blurred = cv2.GaussianBlur(frame, (blur_ksize, blur_ksize), 0)

    # 2) HSV mask (คงช่วงสีส้มเดิมไว้)
    hsv  = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)

    # 3) Morphology (ใช้ kernel และ iterations จาก trackbars)
    kernel = np.ones((kern_size, kern_size), np.uint8)
    if open_it  > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=open_it)
    if close_it > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_it)

    # 4) หา contours และกรองตามพื้นที่ขั้นต่ำ
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = frame.copy()
    kept = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < max(1, min_area):
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        kept += 1

    # 5) แสดงผลรวม 3 ช่องในหน้าต่างเดียว (ให้ trackbars แสดงแน่นอน)
    f1 = cv2.resize(frame, (426, 240))
    f2 = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (426, 240))
    f3 = cv2.resize(output, (426, 240))
    combined = np.hstack([f1, f2, f3])

    # ใส่ overlay ค่าพารามิเตอร์/จำนวนคอนทัวร์
    cv2.putText(combined, f"Blur={blur_ksize}  Kernel={kern_size}  Open={open_it}  Close={close_it}  MinArea={min_area}  Contours={kept}",
                (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.imshow("Adjust", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
