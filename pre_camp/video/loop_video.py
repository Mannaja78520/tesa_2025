import cv2

# === Source Config ===
source = "basketball_hoop.mp4"  # หรือ 0 ถ้าใช้เว็บแคม

cap = cv2.VideoCapture(source)

if not cap.isOpened():
    raise RuntimeError("เปิดเว็บแคมไม่ได้ ลองเปลี่ยน CAM_INDEX")

while True:
    ok, frame = cap.read()

    # ถ้าวิดีโอจบ → กลับไปเฟรมแรก
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    cv2.imshow("Webcam", frame)

    # กด q เพื่อออก
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()