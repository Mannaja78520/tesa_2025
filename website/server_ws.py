from flask import Flask, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room
import eventlet
eventlet.monkey_patch()

app = Flask(__name__)
CORS(app)
io = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

pi_sid = {}

@io.on("connect")
def _on_connect():
    emit("connected", {"ok": True})

@io.on("join")
def _on_join(data):
    role = data.get("role", "web")
    cam_id = data.get("cam_id", "default")
    join_room(cam_id)
    if role == "pi":
        pi_sid[cam_id] = request.sid     # << ต้องมี from flask import request
    emit("joined", {"role": role, "room": cam_id})
    
@io.on("disconnect")
def _on_disconnect():
    # ถ้าต้องการลบ mapping pi_sid เมื่อ pi หลุด อาจต้องไล่หาและลบ
    pass

# ========== FROM PI ==========
@io.on("pi:meta")
def _pi_meta(payload):
    """
    payload ตัวอย่าง:
    {
      "cam_id": "cam_01",
      "timestamp": "...",
      "lat": 18.79, "lon": 98.97, "height_m": 120.3,
      "objects": [...]
    }
    """
    cam_id = payload.get("cam_id", "default")
    io.emit("meta", payload, to=cam_id)

@io.on("pi:image")
def _pi_image(payload):
    """
    payload ตัวอย่าง:
    {
      "cam_id": "cam_01",
      "timestamp": "...",
      "image_b64": "<...>",
      "image_format": "jpg"
    }
    """
    cam_id = payload.get("cam_id", "default")
    io.emit("image", payload, to=cam_id)

@io.on("pi:pack")
def _pi_pack(payload):
    """
    รวม meta + image ใน event เดียว
    payload: {cam_id, timestamp, ... meta fields ..., image_b64, image_format}
    """
    cam_id = payload.get("cam_id", "default")
    io.emit("pack", payload, to=cam_id)

# ========== COMMANDS ==========
@io.on("web:video_status")
def _web_video_status(data):
    """
    web ส่งคำสั่งไปหา Pi ผ่าน server
    data: {cam_id:"cam_01", status:"start"|"stop"|"pause"|...}
    """
    cam_id = data.get("cam_id", "default")
    # ส่งเข้าห้อง (broadcast ถึง web ด้วย) และพยายามเจาะไปที่ Pi ถ้าเคยจำ sid ไว้
    io.emit("server:video_status", data, to=cam_id)
    sid = pi_sid.get(cam_id)
    if sid:
        io.emit("server:video_status", data, to=sid)

if __name__ == "__main__":
    io.run(app, host="0.0.0.0", port=3000)
