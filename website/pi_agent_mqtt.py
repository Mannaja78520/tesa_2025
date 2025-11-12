import json, time, base64, cv2, socket
from datetime import datetime, timezone
import paho.mqtt.client as mqtt

BROKER = "192.168.50.242"   # <-- ใส่ IP คอมที่รัน mosquitto
CAM_ID = "cam_01"

TOPIC_DATA = f"od/cam/{CAM_ID}"
TOPIC_CMD  = f"cmd/cam/{CAM_ID}"
TOPIC_STAT = f"status/cam/{CAM_ID}"

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def on_connect(client, userdata, flags, rc, properties=None):
    print("[Pi] MQTT connect rc:", rc)
    # ประกาศสถานะ online (retain)
    client.publish(TOPIC_STAT, "online", qos=1, retain=True)
    # รอฟังคำสั่ง
    client.subscribe(TOPIC_CMD, qos=1)

def on_message(client, userdata, msg):
    print("[Pi] CMD:", msg.topic, msg.payload.decode("utf-8", errors="ignore"))
    # TODO: parse แล้วสั่ง HW ตามต้องการ
    # cmd = json.loads(msg.payload)

def main():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"pi-{CAM_ID}", clean_start=True)
    # Last Will: ถ้าหลุด ให้ broker ปล่อย offline
    client.will_set(TOPIC_STAT, "offline", qos=1, retain=True)
    client.on_connect = on_connect
    client.on_message = on_message

    # ถ้าทำ auth:
    # client.username_pw_set("user1", "pass1")

    client.connect(BROKER, 1883, keepalive=30)
    client.loop_start()

    cap = cv2.VideoCapture(0)  # เปลี่ยนเป็นไฟล์/rtsp ได้
    send_hz = 2
    period = 1.0 / send_hz
    last = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1); continue

        t = time.time()
        if t - last >= period:
            last = t

            # TODO: ใส่ผล detect จริงแทนตัวอย่างนี้
            objects = [{
                "label": "target",
                "confidence": 0.95,
                "bbox": [100,120,80,60],
                "lat": 18.796143, "lon": 98.979263, "height_m": 120.3
            }]

            payload = {
                "cam_id": CAM_ID,
                "timestamp": now_iso(),
                "objects": objects,
                # แนะนำ: ส่ง URL รูปแทน base64
                "image_url": f"http://{BROKER}/snap/{CAM_ID}.jpg"
            }

            # ส่งข้อมูล
            client.publish(TOPIC_DATA, json.dumps(payload), qos=1, retain=False)

        # ถ้าจะอัปโหลด snapshot เป็นไฟล์ HTTP ก็ทำที่นี่ (แยก thread/REST)
        # หรือบันทึกลง NGINX served dir

    # client.loop_stop(); client.disconnect()

if __name__ == "__main__":
    main()
