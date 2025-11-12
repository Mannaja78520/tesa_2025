from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import cv2, os, csv

# ---------- CONFIG ----------
MODEL_1_PATH = "drone_lastest.pt"
MODEL_2_PATH = "drone.pt"   # <<-- โมเดลตัวที่ 2
TEST_DIR = "../P2_DATA_TEST/P2_DATA_TEST"
SAVE_DIR = "../P2_DATA_TEST/TEST_RESULTS_SAHI"
CSV_PATH = "../P2_DATA_TEST/output.csv"

DRONE_CLASS_NAME = "drone"
MAX_DRONES = 2

MODEL1_CONF_THRESH = 0.25
MODEL2_CONF_THRESH = 0.25
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

print("⏳ Loading model #1 ...")
model1 = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=MODEL_1_PATH,
    confidence_threshold=MODEL1_CONF_THRESH,
    device="cpu",
)
print("✅ Model #1 ready")

print("⏳ Loading model #2 ...")
model2 = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=MODEL_2_PATH,
    confidence_threshold=MODEL2_CONF_THRESH,
    device="cpu",
)
print("✅ Model #2 ready")

# ----- CSV -----
csvfile = open(CSV_PATH, "w", newline="")
writer = csv.writer(csvfile)
writer.writerow(["image_file", "center_x", "center_y", "width", "height"])
print(f"📁 Created CSV: {CSV_PATH}")

# ----- เลือกโหมด -----
print("\n=== เลือกโหมดการทำงาน ===")
print("1: เซฟผลลัพธ์ทุกภาพ + auto next (Save All)")
print("2: ดูทีละภาพ ไม่เซฟ (View Only)")
mode = input("เลือกโหมด (1/2): ").strip()
save_all = (mode == "1")

# ----- รายการรูป -----
image_files = sorted(
    f for f in os.listdir(TEST_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
)

def iou(boxA, boxB):
    # box = (x1, y1, x2, y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def detect_with_model(detection_model, img_zoom, H, W, img_area, ground_line):
    """
    รัน SAHI + YOLO บนภาพที่ซูมแล้ว (img_zoom)
    แล้ว map พิกัดกลับมาที่ภาพจริง (W,H)
    คืน list: (score, x1, y1, x2, y2, cx, cy, w, h)
    """
    result = get_sliced_prediction(
        image=img_zoom,
        detection_model=detection_model,
        slice_height=int(SLICE_H * ZOOM),
        slice_width=int(SLICE_W * ZOOM),
        overlap_height_ratio=OVERLAP,
        overlap_width_ratio=OVERLAP,
    )

    candidates = []
    for obj in result.object_prediction_list:
        class_name = obj.category.name
        score = float(obj.score.value)
        if class_name != DRONE_CLASS_NAME:
            continue

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

        candidates.append((score, x1, y1, x2, y2, cx, cy, w, h))

    return candidates


for i, filename in enumerate(image_files, 1):
    img_path = os.path.join(TEST_DIR, filename)
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Skip {filename} (อ่านรูปไม่สำเร็จ)")
        continue

    H, W = img.shape[:2]
    img_area = H * W
    ground_line = int(H * GROUND_RATIO)

    # ===== 1) ซูมภาพ =====
    img_zoom = cv2.resize(
        img, None, fx=ZOOM, fy=ZOOM,
        interpolation=cv2.INTER_LINEAR
    )

    # ===== 2) ใช้ model1 ก่อน =====
    drones1 = detect_with_model(model1, img_zoom, H, W, img_area, ground_line)
    print(f"[{i}/{len(image_files)}] {filename} -> model1 เจอ {len(drones1)} ลำ")

    drones = drones1[:]  # copy

    # ===== 3) ถ้ายังเจอไม่ครบ / หรือน้อย → ให้ model2 ช่วยหาในจุดที่ยังไม่เจอ =====
    if len(drones1) < MAX_DRONES:
        # สร้างภาพ zoom ใหม่ แล้วทาสีดำทับบริเวณโดรนที่ model1 เจอแล้ว
        img_zoom_masked = img_zoom.copy()
        PAD = 10
        for _, x1, y1, x2, y2, cx, cy, w, h in drones1:
            x1_pad = max(0, x1 - PAD)
            y1_pad = max(0, y1 - PAD)
            x2_pad = min(W - 1, x2 + PAD)
            y2_pad = min(H - 1, y2 + PAD)
            
            # map กลับไปเป็นพิกัดบนภาพที่ซูมแล้ว
            zx1 = int(x1_pad * ZOOM)
            zy1 = int(y1_pad * ZOOM)
            zx2 = int(x2_pad * ZOOM)
            zy2 = int(y2_pad * ZOOM)
            cv2.rectangle(img_zoom_masked, (zx1, zy1), (zx2, zy2), (0, 0, 0), -1)

        # รัน model2 บนภาพที่ mask แล้ว (จะไม่เห็นจุดที่ model1 เจอ)
        drones2 = detect_with_model(model2, img_zoom_masked, H, W, img_area, ground_line)
        print(f"   ➕ model2 ช่วยหาเพิ่ม: เจอ {len(drones2)} ลำ")

        filtered_drones2 = []
        for d2 in drones2:
            _, x1_2, y1_2, x2_2, y2_2, _, _, _, _ = d2
            box2 = (x1_2, y1_2, x2_2, y2_2)
            max_iou = 0.0
            for d1 in drones1:
                _, x1_1, y1_1, x2_1, y2_1, _, _, _, _ = d1
                box1 = (x1_1, y1_1, x2_1, y2_1)
                max_iou = max(max_iou, iou(box1, box2))
            if max_iou < 0.3:   # ถ้าซ้อนกันน้อยกว่า 30% ค่อยถือว่าเป็น drone ใหม่
                filtered_drones2.append(d2)

        print(f"   ✅ หลัง IoU filter: ใช้จาก model2 แค่ {len(filtered_drones2)} ลำ")
        drones += filtered_drones2
        # รวมผล model1 + model2
        # drones += drones2

    # ===== 4) จำกัดไม่เกิน MAX_DRONES =====
    drones.sort(key=lambda d: d[0], reverse=True)
    drones = drones[:MAX_DRONES]

    # ===== 5) เขียน CSV + วาดกรอบ =====
    for score, x1, y1, x2, y2, cx, cy, w, h in drones:
        writer.writerow([filename, cx, cy, w, h])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"drone {score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print(f"   -> keep: center=({cx},{cy}), w={w}, h={h}")

    # ===== 6) แสดง / เซฟ =====
    if save_all:
        save_path = os.path.join(SAVE_DIR, filename)
        cv2.imwrite(save_path, img)
    cv2.line(img, (0, ground_line), (W - 1, ground_line), (0, 0, 255), 2)
    img_disp = cv2.resize(img, (720, 480))
    cv2.imshow("Detect_Image", img_disp)

    key = cv2.waitKey(AUTO_DELAY_MS if save_all else 0) & 0xFF
    if key in [ord("q"), 27]:
        break

csvfile.close()
cv2.destroyAllWindows()
print(f"✅ Done. CSV saved at {CSV_PATH}")
