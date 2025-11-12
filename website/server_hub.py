from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room
import json, eventlet, paho.mqtt.client as mqtt

# ---------- Config ----------
BROKER = "192.168.50.242"              # IP broker MQTT
MQTT_TOPIC = "od/cam/+/meta"           # MQTT meta
HTTP_HOST, HTTP_PORT = "0.0.0.0", 3000 # Socket.IO server
# ----------------------------

app = Flask(__name__)
CORS(app)
io = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# ===== Socket.IO (รับภาพจาก Pi และ broadcast ไป web) =====
@io.on("join")
def on_join(data):
    cam_id = data.get("cam_id", "default")
    join_room(cam_id)
    emit("joined", {"room": cam_id})

@io.on("pi:image")  # Pi ส่งภาพ (binary หรือ base64) + cam_id + timestamp
def on_pi_image(payload):
    # payload: {"cam_id": "...", "timestamp": "...", "image_b64": "..."} (หรือ "image_bytes")
    cam_id = payload.get("cam_id", "default")
    io.emit("image", payload, to=cam_id)  # กระจายให้ผู้ดูในห้องนั้น

# ===== MQTT (รับ meta แล้ว broadcast ไป web) =====
def mqtt_on_connect(client, userdata, flags, rc, props=None):
    print("MQTT connected rc=", rc)
    client.subscribe(MQTT_TOPIC, qos=1)

def mqtt_on_message(client, userdata, msg):
    # msg.topic = od/cam/<cam_id>/meta
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
    except Exception as e:
        print("MQTT decode error:", e); return

    cam_id = payload.get("cam_id", "default")
    # โยน meta ไปหน้าเว็บตามห้อง cam_id
    io.emit("meta", payload, to=cam_id)

def start_mqtt():
    cli = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="hub-server")
    cli.on_connect = mqtt_on_connect
    cli.on_message = mqtt_on_message
    cli.connect(BROKER, 1883, 30)
    cli.loop_start()
    return cli

if __name__ == "__main__":
    _mqtt = start_mqtt()
    io.run(app, host=HTTP_HOST, port=HTTP_PORT)
