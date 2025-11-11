from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import cv2
import os
import csv

# ---------- CONFIG ----------
DRONE_MODEL_PATH = "drone_lastest.pt"
TEST_DIR = "P1_DATASET/TEST_DATA"
SAVE_DIR = "P1_DATASET/TEST_RESULTS_SAHI"
CSV_PATH = "output.csv"   # ไฟล์ CSV เก็บผลลัพธ์

DRONE_CLASS_NAME = "drone"
MAX_DRONES = 2

CONF_THRESH = 0.6
CONF_UNDER_LINE_THRESH = 0.71

SLICE_W = 640
SLICE_H = 640
OVERLAP = 0.35
ZOOM = 2.0
AUTO_DELAY_MS = 500

GROUND_RATIO = 0.65
BIG_OBJ_RATIO = 0.0
MIN_RATIO = 0.0
MAX_RATIO = 1.0
# -----------------------------

os.makedirs(SAVE_DIR, exist_ok=True)

# ----- สร้าง SAHI detection model -----
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=DRONE_MODEL_PATH,
    confidence_threshold=CONF_THRESH,
    device="cpu",
)
print("✅ SAHI + YOLO model ready")

# ----- เตรียม CSV -----
csvfile = open(CSV_PATH, "w", newline="")
writer = csv.writer(csvfile)
writer.writerow(["image_name", "center_x", "center_y", "width", "height"])
print(f"📁 สร้างไฟล์ CSV ใหม่: {CSV_PATH}")

# ----- เลือกโหมด -----
print("\n=== เลือกโหมดการทำงาน ===")
print("1: เซฟผลลัพธ์ทุกภาพ + โชว์ + เลื่อนอัตโนมัติ (Save All)")
print("2: ดูทีละภาพ ไม่เซฟ (View Only)")
mode = input("เลือกโหมด (1/2): ").strip()
save_all = (mode == "1")

if save_all:
    print(f"💾 โหมด 1: เซฟทุกภาพลงใน {SAVE_DIR}\n")
else:
    print("👁️ โหมด 2: ดูอย่างเดียว ไม่เซฟไฟล์\n")

# ----- เตรียมรายการรูป -----
image_files = sorted(
    f for f in os.listdir(TEST_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
)

for i, filename in enumerate(image_files, 1):
    img_path = os.path.join(TEST_DIR, filename)
    img = cv2.imread(img_path)
    if img is None:
        print(f"ข้าม {filename} (อ่านรูปไม่สำเร็จ)")
        continue

    H, W = img.shape[:2]
    img_area = H * W
    ground_line = int(H * GROUND_RATIO)

    # ===== 1) ขยายภาพก่อน detect =====
    img_zoom = cv2.resize(
        img, None, fx=ZOOM, fy=ZOOM,
        interpolation=cv2.INTER_LINEAR
    )

    # ===== 2) SAHI + YOLO slicing inference =====
    result = get_sliced_prediction(
        image=img_zoom,
        detection_model=detection_model,
        slice_height=int(SLICE_H * ZOOM),
        slice_width=int(SLICE_W * ZOOM),
        overlap_height_ratio=OVERLAP,
        overlap_width_ratio=OVERLAP,
    )

    drone_candidates = []

    for obj in result.object_prediction_list:
        class_name = obj.category.name
        score = float(obj.score.value)
        if class_name != DRONE_CLASS_NAME:
            continue

        # พิกัดจากภาพซูม -> map กลับภาพจริง
        zx1, zy1, zx2, zy2 = obj.bbox.to_xyxy()
        x1 = int(zx1 / ZOOM)
        y1 = int(zy1 / ZOOM)
        x2 = int(zx2 / ZOOM)
        y2 = int(zy2 / ZOOM)

        x1 = max(0, min(W - 1, x1))
        x2 = max(0, min(W - 1, x2))
        y1 = max(0, min(H - 1, y1))
        y2 = max(0, min(H - 1, y2))

        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            continue

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # ---------- ฟิลเตอร์ต่าง ๆ ----------
        box_area = w * h
        if not (MIN_RATIO * img_area <= box_area <= MAX_RATIO * img_area):
            continue

        if (cy > ground_line) and (box_area > BIG_OBJ_RATIO * img_area):
            if score < CONF_UNDER_LINE_THRESH:
                continue

        aspect1 = w / float(h)
        if aspect1 < 0.8:
            continue
        aspect2 = h / float(w)
        if aspect2 < 0.65:
            continue

        # เก็บทั้ง bbox และ center/size ไว้ใช้ทีเดียว
        drone_candidates.append((score, x1, y1, x2, y2, cx, cy, w, h))

    # ----- จำกัดไม่เกิน 2 ลำ -----
    drone_candidates.sort(key=lambda d: d[0], reverse=True)
    drone_candidates = drone_candidates[:MAX_DRONES]

    # ----- เขียนลง CSV + วาดกรอบ -----
    for score, x1, y1, x2, y2, cx, cy, w, h in drone_candidates:
        # เขียนลง CSV (ไม่มี score)
        writer.writerow([filename, cx, cy, w, h])

        # วาดกรอบโชว์
        label = f"{DRONE_CLASS_NAME} {score:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print(f"[{i}/{len(image_files)}] {filename} -> center=({cx},{cy}), w={w}, h={h}")

    # ----- Save / View -----
    if save_all:
        save_path = os.path.join(SAVE_DIR, filename)
        cv2.imwrite(save_path, img)
        print(f"💾 Saved: {save_path}")

    cv2.line(img, (0, ground_line), (W - 1, ground_line), (0, 0, 255), 2)
    img_disp = cv2.resize(img, (720, 480))
    cv2.imshow("Detect_Image", img_disp)

    if save_all:
        key = cv2.waitKey(AUTO_DELAY_MS) & 0xFF
        if key in [ord("q"), 27]:
            break
    else:
        print("➡️  Space/Enter = ถัดไป, q = ออก")
        key = cv2.waitKey(0) & 0xFF
        if key in [ord("q"), 27]:
            break

csvfile.close()
cv2.destroyAllWindows()
