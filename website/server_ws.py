# --- Eventlet version (monkey-patch first line) ---
import eventlet
eventlet.monkey_patch()  # 👈 ต้องเป็นบรรทัดแรกสุด

from flask import Flask, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room

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
        pi_sid[cam_id] = request.sid
    emit("joined", {"role": role, "room": cam_id})

@io.on("pi:meta")
def _pi_meta(payload):
    cam_id = payload.get("cam_id", "default")
    io.emit("meta", payload, to=cam_id)

@io.on("pi:image")
def _pi_image(payload):
    cam_id = payload.get("cam_id", "default")
    io.emit("image", payload, to=cam_id)

@io.on("pi:pack")
def _pi_pack(payload):
    cam_id = payload.get("cam_id", "default")
    io.emit("pack", payload, to=cam_id)

@io.on("web:video_status")
def _web_video_status(data):
    cam_id = data.get("cam_id", "default")
    io.emit("server:video_status", data, to=cam_id)
    sid = pi_sid.get(cam_id)
    if sid:
        io.emit("server:video_status", data, to=sid)

if __name__ == "__main__":
    io.run(app, host="0.0.0.0", port=3000)
