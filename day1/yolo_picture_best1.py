from ultralytics import YOLO
import cv2
import os

# ---------- CONFIG ----------
model = YOLO("drone.pt")          # โมเดลที่ train แล้ว (มี class drone, bird)
test_dir = "P1_DATASET/TEST_DATA" # โฟลเดอร์รูป test_0001.jpg ...
conf_thresh = 0.35                # ความมั่นใจขั้นต่ำ
imgsz = 5120                      # ขนาด input ของ YOLO (ลดได้ถ้าช้า)
DRONE_CLASS_NAME = "drone"        # ชื่อคลาสโดรนใน model.names
MAX_DRONES = 2                    # จำนวนโดรนสูงสุดต่อภาพ
# -----------------------------

print("class map:", model.names)  # ดู mapping class id -> name สักรอบ

# เอาเฉพาะไฟล์รูป
image_files = sorted([
    f for f in os.listdir(test_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

for i, filename in enumerate(image_files, 1):
    img_path = os.path.join(test_dir, filename)
    img = cv2.imread(img_path)
    if img is None:
        print(f"ข้าม {filename} (อ่านรูปไม่สำเร็จ)")
        continue

    H, W = img.shape[:2]
    img_area = H * W

    # ----- รัน YOLO -----
    results = model(img, device="cpu", conf=conf_thresh, imgsz=imgsz)[0]

    drone_candidates = []

    for box in results.boxes:
        # ดึงพิกัดกล่อง
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # class / confidence
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        class_name = model.names[cls_id]

        # 1) เอาเฉพาะคลาสโดรน
        if class_name != DRONE_CLASS_NAME:
            continue

        # ขนาดกล่อง
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            continue

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # 2) ตัดของที่อยู่ต่ำกว่า "แนวต้นไม้" (กันคน/ต้นไม้)
        ground_line = int(H * 0.65)   # ปรับได้ 0.60–0.70 ตามคลิป
        if cy > ground_line:
            continue

        # 3) กรองขนาดกล่อง (กันกล่องใหญ่เวอร์ / เล็กเวอร์)
        box_area = w * h
        min_ratio = 0      # สัดส่วนเล็กสุดของกล่องเทียบทั้งภาพ
        max_ratio = 1      # สัดส่วนใหญ่สุดของกล่องเทียบทั้งภาพ
        if not (min_ratio * img_area <= box_area <= max_ratio * img_area):
            continue

        # 4) กรองรูปทรง: เอาเฉพาะสี่เหลี่ยมจัตุรัสหรือผืนผ้าแนวนอน
        aspect = w / float(h)
        # ถ้า aspect < 0.8 แสดงว่า "แนวตั้ง" เกินไป → ตัดทิ้ง
        if aspect < 0.6:
            continue

        # ผ่านทุกเงื่อนไข -> เก็บไว้เป็น candidate
        drone_candidates.append((conf, x1, y1, x2, y2, cx, cy))

    # ----- จำกัดจำนวนโดรนไม่เกิน MAX_DRONES -----
    drone_candidates.sort(key=lambda d: d[0], reverse=True)
    drone_candidates = drone_candidates[:MAX_DRONES]

    # ----- วาดกรอบเฉพาะโดรนที่ผ่านฟิลเตอร์ -----
    for conf, x1, y1, x2, y2, cx, cy in drone_candidates:
        label = f"{DRONE_CLASS_NAME} {conf:.2f}"
        # กรอบเขียว
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # *ไม่มี* จุดแดงแล้ว
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print(f"[{i}/{len(image_files)}] {filename} -> {label} center=({cx},{cy})")

    # ----- แสดงภาพแล้วรอให้กดปุ่ม -----
    img_disp = cv2.resize(img, (960, 540))
    cv2.imshow("YOLO Test Viewer", img_disp)
    print("➡️  Space/Enter = ไปภาพต่อไป, q = ออก")

    key = cv2.waitKey(0) & 0xFF
    if key in [ord("q"), 27]:  # q หรือ ESC เพื่อออก
        break

cv2.destroyAllWindows()
