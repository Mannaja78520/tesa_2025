import cv2, base64, json, time, socketio
from datetime import datetime, timezone
import paho.mqtt.client as mqtt

# ---------- Config ----------
BROKER = "192.168.50.242"           # MQTT broker (เครื่อง server)
CAM_ID = "cam_01"
TOPIC_META = f"od/cam/{CAM_ID}/meta"

SOCKET_SERVER = "http://192.168.50.242:3000"  # Socket.IO server ของ hub
SEND_HZ = 2
QUALITY = 75
# ----------------------------

def now_iso():
    return datetime.now(timezone.utc).isoformat()

# MQTT client
mq = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"pub-{CAM_ID}")
mq.connect(BROKER, 1883, 30)

# Socket.IO client
sio = socketio.Client(reconnection=True)
sio.connect(SOCKET_SERVER, transports=["websocket"])
sio.emit("join", {"cam_id": CAM_ID})

cap = cv2.VideoCapture(0)   # เปลี่ยนเป็นไฟล์/RTSP ได้
period = 1.0 / SEND_HZ
last = 0

seq = 0
while True:
    ok, frame = cap.read()
    if not ok:
        time.sleep(0.05); continue

    t = time.time()
    if t - last >= period:
        last = t

        # ---- 1) ส่ง meta ผ่าน MQTT ----
        meta = {
            "cam_id": CAM_ID,
            "timestamp": now_iso(),
            "seq": seq,
            "lat": 18.796143, "lon": 98.979263, "height_m": 120.3,
            "objects": [
                {"label": "target", "confidence": 0.92, "bbox": [100,120,80,60]}
            ]
        }
        mq.publish(TOPIC_META, json.dumps(meta), qos=1)

        # ---- 2) ส่งภาพผ่าน Socket.IO ----
        ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
        if ok2:
            img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
            sio.emit("pi:image", {
                "cam_id": CAM_ID,
                "timestamp": meta["timestamp"],
                "seq": seq,
                "image_b64": img_b64,      # (ถ้าต้อง binary จริง ๆ คุยเพิ่มได้)
                "image_format": "jpg",
                "quality": QUALITY
            })

        seq += 1
