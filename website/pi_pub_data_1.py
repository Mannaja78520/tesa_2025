# pub_pack.py
import cv2, base64, json
from datetime import datetime, timezone
import paho.mqtt.client as mqtt

BROKER="192.168.50.242"; CAM_ID="cam_01"
TOPIC=f"od/cam/{CAM_ID}/pack"

def now_iso(): from datetime import datetime, timezone; return datetime.now(timezone.utc).isoformat()

cli=mqtt.Client(mqtt.CallbackAPIVersion.VERSION2); cli.connect(BROKER,1883,30)
frame=cv2.imread("img_0004.jpg")
ok,buf=cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY,75])
img_b64=base64.b64encode(buf.tobytes()).decode("utf-8")

payload={
  "cam_id": CAM_ID,
  "timestamp": now_iso(),
  "objects":[{"label":"drone","confidence":0.9,"bbox":[100,120,80,60],
              "lat":18.796143,"lon":98.979263,"height_m":120.3}],
  "image_b64": img_b64,
  "image_format": "jpg"
}
cli.publish(TOPIC, json.dumps(payload), qos=1)
cli.disconnect()
