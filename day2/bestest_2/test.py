from ultralytics import YOLO

# โหลดโมเดล .pt
model = YOLO("drone_lastest.pt")

# ดูค่าขนาดภาพที่ใช้เทรน
print("Image size (imgsz):", model.args.get("imgsz", "unknown"))

# ถ้าอยากดูพารามิเตอร์อื่น ๆ ทั้งหมด
print("\n== Model arguments ==")
for k, v in model.args.items():
    print(f"{k}: {v}")
