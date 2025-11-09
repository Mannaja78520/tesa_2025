import cv2

# === Source Config ===
# ถ้าอยากใช้เว็บแคม → ตั้ง source = 0 # เปลี่ยนเป็น 1/2 ถ้ามีหลายกล้อง
# ถ้าอยากอ่านจากไฟล์ → ใส่ path เช่น "videos/ball.mp4"
source = 0  # เปลี่ยนเป็น "videos/ball.mp4" ถ้ามีไฟล์วิดีโอ
cap = cv2.VideoCapture(source)


if not cap.isOpened():
    raise RuntimeError("เปิดเว็บแคมไม่ได้ ลองเปลี่ยน CAM_INDEX")

while True:
    ok, frame = cap.read()
    if not ok: break
    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # กด q เพื่อออก
        break

cap.release()
cv2.destroyAllWindows()
