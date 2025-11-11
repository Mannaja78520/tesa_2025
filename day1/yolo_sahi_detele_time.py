from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import cv2
import os

# ---------- CONFIG ----------
DRONE_MODEL_PATH = "drone.pt"
TEST_DIR = "P1_DATASET/TEST_DATA"
SAVE_DIR = "P1_DATASET/TEST_RESULTS_SAHI"

DRONE_CLASS_NAME = "drone"
MAX_DRONES = 2

CONF_THRESH = 0.275
CONF_UNDER_LINE_THRESH = 0.70

SLICE_W = 320
SLICE_H = 320
OVERLAP = 0.2

ZOOM = 5.0
AUTO_DELAY_MS = 500

GROUND_RATIO = 0.65
BIG_OBJ_RATIO = 0.0

MIN_RATIO = 0.0
MAX_RATIO = 1.0
# -----------------------------


def mask_datetime_by_contour(img):
    """
    ลบเฉพาะบริเวณวันที่เวลา (มุมขวาบน) ด้วยการหา contour
    """
    H, W = img.shape[:2]

    # ----- 1) กำหนด ROI มุมขวาบน (ปรับเปอร์เซ็นต์ได้) -----
    roi_x1 = int(W * 0.55)
    roi_y1 = 0
    roi_x2 = W
    roi_y2 = int(H * 0.20)     # เอาแค่ 20% บนสุดพอ

    roi = img[roi_y1:roi_y2, roi_x1:roi_x2]
    if roi.size == 0:
        return img

    # ----- 2) หา mask ของตัวหนังสือสีขาว -----
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # เบลอหน่อยให้ threshold เนียน
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # ตัวเลข/ตัวหนังสือสีอ่อน -> ใช้ binary inverse
    # ปรับ 200 ตามความสว่างได้
    _, th = cv2.threshold(gray_blur, 200, 255, cv2.THRESH_BINARY)

    # ปิดช่องว่างเล็ก ๆ เพื่อให้แต่ละตัวเชื่อมกัน
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    th = cv2.dilate(th, kernel, iterations=1)
    th = cv2.erode(th, kernel, iterations=1)

    # ----- 3) หา contour -----
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        # กรองขนาดกล่อง (กัน noise เล็ก ๆ/ใหญ่เกิน)
        if area < 50:      # เล็กเกินไป (จุด noise)
            continue
        if area > 0.1 * (roi.shape[0] * roi.shape[1]):  # ใหญ่เกิน (ทั้ง ROI) ตัดทิ้ง
            continue

        # padding รอบ ๆ ตัวเลขเล็กน้อย
        pad = 2
        x1 = max(0, roi_x1 + x - pad)
        y1 = max(0, roi_y1 + y - pad)
        x2 = min(W - 1, roi_x1 + x + w + pad)
        y2 = min(H - 1, roi_y1 + y + h + pad)

        # ลบเฉพาะตัวหนังสือ (เติมดำ)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)

    return img


# ---------- สร้าง model ----------
os.makedirs(SAVE_DIR, exist_ok=True)

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=DRONE_MODEL_PATH,
    confidence_threshold=CONF_THRESH,
    device="cpu",
)

print("✅ SAHI + YOLO model ready")

print("\n=== เลือกโหมดการทำงาน ===")
print("1: เซฟผลลัพธ์ทุกภาพ + โชว์ + เลื่อนอัตโนมัติ (Save All)")
print("2: ดูทีละภาพ ไม่เซฟ (View Only)")
mode = input("เลือกโหมด (1/2): ").strip()
save_all = (mode == "1")

if save_all:
    print(f"💾 โหมด 1: เซฟทุกภาพลงใน {SAVE_DIR}\n")
else:
    print("👁️ โหมด 2: ดูอย่างเดียว ไม่เซฟไฟล์\n")

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

    # ===== 1) ทำสำเนาสำหรับ detection แล้วลบวันที่เวลาด้วย contour =====
    img_for_det = img.copy()
    img_for_det = mask_datetime_by_contour(img_for_det)

    # ===== 2) ขยายภาพก่อน detect =====
    img_zoom = cv2.resize(
        img_for_det, None, fx=ZOOM, fy=ZOOM,
        interpolation=cv2.INTER_LINEAR
    )

    # ===== 3) SAHI + YOLO slicing inference =====
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
            if (score < CONF_UNDER_LINE_THRESH):
                continue

        aspect1 = w / float(h)
        if aspect1 < 0.8:
            continue
        aspect2 = h / float(w)
        if aspect2 < 0.65:
            continue

        drone_candidates.append((score, x1, y1, x2, y2, cx, cy))

    drone_candidates.sort(key=lambda d: d[0], reverse=True)
    drone_candidates = drone_candidates[:MAX_DRONES]

    for score, x1, y1, x2, y2, cx, cy in drone_candidates:
        label = f"{DRONE_CLASS_NAME} {score:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print(f"[{i}/{len(image_files)}] {filename} -> {label} center=({cx},{cy})")

    if save_all:
        save_path = os.path.join(SAVE_DIR, filename)
        cv2.imwrite(save_path, img)
        print(f"💾 Saved: {save_path}")

    # debug: วาด ground line ให้เห็น
    cv2.line(img, (0, ground_line), (W-1, ground_line), (0, 0, 255), 2)

    img_disp = cv2.resize(img, (720, 480))
    cv2.imshow("Detect_Image", img_for_det)
    # cv2.imshow("Detect_Image", img_disp)

    if save_all:
        key = cv2.waitKey(AUTO_DELAY_MS) & 0xFF
        if key in [ord("q"), 27]:
            break
    else:
        print("➡️  Space/Enter = ถัดไป, q = ออก")
        key = cv2.waitKey(0) & 0xFF
        if key in [ord("q"), 27]:
            break

cv2.destroyAllWindows()
print("✅ SAHI + YOLO (datetime contour mask) เสร็จเรียบร้อย")
