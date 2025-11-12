# finetune_drone_only.py  (เวอร์ชันแก้ path สำหรับ Roboflow: train/valid/test)
from ultralytics import YOLO
from pathlib import Path
import yaml, datetime, os, sys

# ======= CONFIG (แก้ให้ตรงเครื่องคุณ) =======
OLD_MODEL_WEIGHTS = "drone.pt"                 # โมเดลเก่า (2 คลาส)
NEW_DATA_ROOT     = "Drone_detect_dataset"     # โฟลเดอร์ที่มี train/, valid/, test/
IMG_SIZE = 768
EPOCHS   = 60
BATCH    = 16
LR0      = 0.003
PATIENCE = 20
FREEZE   = 0
PROJECT  = "runs_det"
RUN_NAME_PREFIX = "finetune_drone_only"

# ถ้า GPU ไม่รองรับ ให้บังคับใช้ CPU ชั่วคราว (ตัดสินใจจาก env หรือ flag)
# ตั้งค่านี้เป็น 'cpu' ถ้าเจอปัญหา CUDA (1050 Ti + torch ใหม่)
FORCE_DEVICE = os.environ.get("YOLO_DEVICE", "").strip()  # "", "cpu", "0", "0,1"
if FORCE_DEVICE == "":
    # ลองตรวจสอบสภาพแวดล้อมคร่าว ๆ — ถ้าเป็น 1050Ti + torch รุ่นใหม่ ให้ default เป็น cpu
    FORCE_DEVICE = "cpu"

import torch
torch.set_num_threads(10)          # ลอง 8–12 ตามความเหมาะสม (i7-8750H มี 12 threads)
torch.set_num_interop_threads(2)


# ======= ตรวจโครงสร้าง dataset =======
root = Path(NEW_DATA_ROOT).resolve()
paths_to_check = [
    root / "train" / "images",
    root / "train" / "labels",
    root / "valid" / "images",   # Roboflow ใช้ 'valid' (ไม่ใช่ 'val')
    root / "valid" / "labels",
]
missing = [p for p in paths_to_check if not p.exists()]
if missing:
    print("❌ Dataset paths not found:")
    for p in missing:
        print("   -", p)
    print("\nโปรดตรวจว่าคุณแตกไฟล์ถูกที่หรือยัง ถ้าเป็นรูปแบบอื่น ให้แก้ path ด้านล่างนี้ให้ถูกต้อง:")
    print("train → 'train/images', 'train/labels'")
    print("val   → 'valid/images', 'valid/labels' (Roboflow ใช้คำว่า valid)")
    sys.exit(1)

# ======= สร้าง data.yaml แบบ 1 คลาส (drone) เข้ากับ Roboflow layout =======
data_dict = {
    "path": str(root),           # root ของ dataset
    "train": "train/images",     # Roboflow
    "val":   "valid/images",     # Roboflow ใช้ 'valid'
    "test": "test/images",
    "names": ["drone"]           # เหลือคลาสเดียว
}
cfg_dir = Path("data_cfg_auto"); cfg_dir.mkdir(parents=True, exist_ok=True)
yaml_path = cfg_dir / "drone_only.yaml"
with open(yaml_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(data_dict, f, allow_unicode=True, sort_keys=False)

# ======= ชื่อรันไม่ให้ทับ =======
stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"{RUN_NAME_PREFIX}_{stamp}"

# ======= เทรน (finetune จากน้ำหนักเก่า) =======
model = YOLO(OLD_MODEL_WEIGHTS)
result = model.train(
    data=str(yaml_path),
    imgsz=IMG_SIZE,
    epochs=EPOCHS,
    batch=8,             # แนะนำเริ่ม 4–8 บน CPU
    workers=8,           # DataLoader workers (ลอง 6–10)
    lr0=LR0,
    patience=PATIENCE,
    project=PROJECT,
    name=run_name,
    freeze=FREEZE if FREEZE > 0 else None,
    device=FORCE_DEVICE,   # "cpu"
    amp=False,             # CPU ไม่ใช้ AMP
    deterministic=True
)

best = Path(result.save_dir) / "weights" / "best.pt"
print("BEST:", best)

# ======= วัดผลบน val ใหม่ =======
metrics = YOLO(str(best)).val(data=str(yaml_path), imgsz=IMG_SIZE, device=FORCE_DEVICE)
try:
    print(f"mAP50-95: {metrics.box.map:.4f}  mAP50: {metrics.box.map50:.4f}  Precision: {metrics.box.mp:.4f}  Recall: {metrics.box.mr:.4f}")
except Exception:
    pass

# ======= Export ถ้าต้องใช้ต่อ =======
for fmt in ["onnx", "torchscript"]:
    print(f"Export -> {fmt}")
    YOLO(str(best)).export(format=fmt, device=FORCE_DEVICE)
