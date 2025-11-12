import cv2
import os

# ---------- CONFIG ----------  
img_path = "P2_DATA_TEST/TEST_RESULTS_SAHI/img_0190.jpg"
# ---------- END CONFIG ----------

# อ่านภาพ
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"ไม่พบไฟล์รูปภาพ: {img_path}")

# สร้าง callback function สำหรับ mouse
def show_xy(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        # สร้างสำเนาภาพชั่วคราว
        temp = img.copy()
        # วาดจุดเล็ก ๆ ที่ตำแหน่งเมาส์
        cv2.circle(temp, (x, y), 1, (0, 255, 0), -1)
        # เขียนข้อความแสดงพิกัด
        cv2.putText(temp, f"({x}, {y})", (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("Mouse Position", temp)

# เปิดหน้าต่างแสดงภาพ
cv2.namedWindow("Mouse Position")
cv2.setMouseCallback("Mouse Position", show_xy)

cv2.imshow("Mouse Position", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
