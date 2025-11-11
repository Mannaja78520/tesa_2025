from ultralytics import YOLO
import cv2
import os

# ---------- CONFIG ----------
model = YOLO("drone.pt")
test_dir = "P1_DATASET/TEST_DATA"
save_dir = "P1_DATASET/TEST_RESULTS"
conf_thresh = 0.35
imgsz = 7680
DRONE_CLASS_NAME = "drone"
MAX_DRONES = 2
# -----------------------------

print("class map:", model.names)
os.makedirs(save_dir, exist_ok=True)

# ---------- เลือกโหมด ----------
print("\n=== เลือกโหมดการทำงาน ===")
print("1: เซฟผลลัพธ์อัตโนมัติทั้งหมด (Save All)")
print("2: ดูทีละภาพ ไม่เซฟ (View Only)")
mode = input("เลือกโหมด (1/2): ").strip()

save_all = (mode == "1")
if save_all:
    print("💾 [โหมดเซฟอัตโนมัติ] จะบันทึกภาพทั้งหมดใน:", save_dir)
else:
    print("👁️ [โหมดดูอย่างเดียว] จะไม่บันทึกไฟล์\n")

# ---------- เตรียมรายการรูป ----------
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
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        class_name = model.names[cls_id]

        # --- เอาเฉพาะโดรน ---
        if class_name != DRONE_CLASS_NAME:
            continue

        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            continue

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # --- กรองขนาดกล่อง ---
        box_area = w * h
        min_ratio = 0.0
        max_ratio = 1.0
        if not (min_ratio * img_area <= box_area <= max_ratio * img_area):
            continue

        # --- เอาเฉพาะแนวนอน/จัตุรัส ---
        aspect = w / float(h)
        if aspect < 0.7:
            continue

        drone_candidates.append((conf, x1, y1, x2, y2, cx, cy))

    # --- จำกัดไม่เกิน 2 ลำ ---
    drone_candidates.sort(key=lambda d: d[0], reverse=True)
    drone_candidates = drone_candidates[:MAX_DRONES]

    # --- วาดเฉพาะกล่องโดรน ---
    for conf, x1, y1, x2, y2, cx, cy in drone_candidates:
        label = f"{DRONE_CLASS_NAME} {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print(f"[{i}/{len(image_files)}] {filename} -> {label} center=({cx},{cy})")

    # --- ถ้าโหมด Save All ให้เซฟเลย ---
    if save_all:
        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, img)
        print(f"💾 Saved: {save_path}")

    # --- โหมดดูอย่างเดียว แสดงภาพและรอกดปุ่ม ---
    if not save_all:
        img_disp = cv2.resize(img, (960, 540))
        cv2.imshow("YOLO Test Viewer", img_disp)
        print("➡️  Space/Enter = ถัดไป, q = ออก")
        key = cv2.waitKey(0) & 0xFF
        if key in [ord("q"), 27]:
            break

cv2.destroyAllWindows()
