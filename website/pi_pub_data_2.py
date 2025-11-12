import cv2, paho.mqtt.client as mqtt

BROKER = "192.168.50.242"
CAM_ID = "cam_01"
TOPIC_IMG = f"od/cam/{CAM_ID}/image"

cli = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"pubimg-{CAM_ID}")
cli.connect(BROKER, 1883, keepalive=30)

frame = cv2.imread("img_0004.jpg")
ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
if not ok:
    raise RuntimeError("imencode failed")

# publish เป็น bytes ตรง ๆ (ไม่ต้อง base64)
cli.publish(TOPIC_IMG, buf.tobytes(), qos=1, retain=False)
cli.disconnect()
print("✅ published binary image")
