import cv2, base64, json, time
from datetime import datetime, timezone
import paho.mqtt.client as mqtt

BROKER = "192.168.50.242"
CAM_ID = "cam_01"
TOPIC_DATA = f"od/cam/{CAM_ID}"   # topic ข้อมูลตรวจจับ

def now_iso():
    return datetime.now(timezone.utc).isoformat()

# 1) เตรียม MQTT client
cli = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"pub-{CAM_ID}")
cli.connect(BROKER, 1883, keepalive=30)

# 2) โหลดรูป และบีบอัดเป็น JPEG -> base64
frame = cv2.imread("img_0004.jpg")
ok, buf = cv2.imencode(".jpg", frame)
if not ok:
    raise RuntimeError("imencode failed")

img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

# 3) สร้าง payload JSON (ยกตัวอย่าง metadata)
payload = {
    "cam_id": CAM_ID,
    "timestamp": now_iso(),
    "objects": [
        {"label": "target", "confidence": 0.93, "bbox": [100,120,80,60],
         "lat": 18.796143, "lon": 98.979263, "height_m": 120.3}
    ],
    "image_b64": img_b64,   # <<— รูป (base64)
    "image_format": "jpg",
    "quality": 75
}

# 4) publish
cli.publish(TOPIC_DATA, json.dumps(payload), qos=1, retain=False)
cli.disconnect()
print("✅ published")
