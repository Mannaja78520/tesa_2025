from ultralytics import YOLO
import cv2

model = YOLO("drone.pt")
cap = cv2.VideoCapture("P1_DATASET/VIDEOS/P1_VIDEO_1.mp4")

DISPLAY_SCALE = 0.3
FRAME_SKIP = 1
frame_count = 0
last_boxes = []

while True:
    ok, frame = cap.read()
    if not ok:
        break

    H, W = frame.shape[:2]
    frame_count += 1

    if frame_count % FRAME_SKIP == 0:
        sky_ratio = 1.0
        sky_h = int(H * sky_ratio)
        sky = frame[0:sky_h, :]
        y_offset = 0

        # 🔹 1) ไม่ต้องซูมแล้ว
        ZOOM = 1.0
        sky_zoom = sky

        # 🔹 2) ใช้ imgsz ใหญ่ขึ้น + conf ต่ำลงหน่อย
        results = model(
            sky_zoom,
            device='cpu',
            conf=0.25,     # ยอมให้มั่นใจน้อยลง
            imgsz=1920     # ให้ YOLO เห็นภาพละเอียดขึ้นจริง ๆ
        )[0]

        last_boxes = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            # 🔹 3) ไม่ต้องหาร ZOOM แล้ว
            last_boxes.append((
                x1, y1, x2, y2,
                int(box.cls[0].item()),
                float(box.conf[0].item())
            ))

    for (x1, y1, x2, y2, cls_id, conf) in last_boxes:
        x1_full = int(x1)
        y1_full = int(y1)
        x2_full = int(x2)
        y2_full = int(y2)

        cx = (x1_full + x2_full) // 2
        cy = (y1_full + y2_full) // 2

        label = f"{model.names[cls_id]} {conf:.2f}"
        cv2.rectangle(frame, (x1_full, y1_full), (x2_full, y2_full), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(frame, label, (x1_full, y1_full - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    disp = cv2.resize(frame, (int(W * DISPLAY_SCALE), int(H * DISPLAY_SCALE)))
    cv2.imshow("YOLO Drone (low-res display)", disp)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
