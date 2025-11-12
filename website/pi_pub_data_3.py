# pub_meta.py
import json, paho.mqtt.client as mqtt
from datetime import datetime, timezone

BROKER="192.168.50.242"; CAM_ID="cam_01"
TOPIC=f"od/cam/{CAM_ID}/meta"
now=lambda: datetime.now(timezone.utc).isoformat()

payload={
  "cam_id": CAM_ID,
  "timestamp": now(),
  "objects":[{"label":"drone","confidence":0.88,"bbox":[90,110,70,50],
              "lat":18.7961,"lon":98.9792,"height_m":118.0}]
}
cli=mqtt.Client(mqtt.CallbackAPIVersion.VERSION2); cli.connect(BROKER,1883,30)
cli.publish(TOPIC, json.dumps(payload), qos=1)
cli.disconnect()
