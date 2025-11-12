# --- Eventlet version (monkey-patch first line) ---
import eventlet
eventlet.monkey_patch()  # 👈 ต้องเป็นบรรทัดแรกสุด

import os, json, base64, pathlib
from datetime import datetime, timezone
from flask import Flask, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room

# ===== Config =====
SAVE_DIR = "inbox"          # โฟลเดอร์ปลายทางเก็บไฟล์
IMG_EXT  = "jpg"            # นามสกุลรูปที่บันทึก
# ===================

app = Flask(__name__)
CORS(app)
io = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

pi_sid = {}       # cam_id -> sid ของ Pi
last_meta = {}    # meta ล่าสุดต่อ cam_id (ไว้จับคู่ตอนมารูปอย่างเดียว)

# ---------- Utils ----------
def now_iso():
    return datetime.now(timezone.utc).isoformat()

def safe_ts(ts: str) -> str:
    # แปลง timestamp ให้กลายเป็นชื่อไฟล์ที่ปลอดภัย
    return ts.replace(":", "-").replace("/", "_").replace(" ", "_")

def ensure_dir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_image_b64(path, b64str):
    data = base64.b64decode(b64str)
    with open(path, "wb") as f:
        f.write(data)
# ----------------------------

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

# ====== FROM PI ======
@io.on("pi:meta")
def _pi_meta(payload):
    cam_id = payload.get("cam_id", "default")
    ts = payload.get("timestamp") or now_iso()
    tss = safe_ts(ts)

    # 1) broadcast ไป client
    io.emit("meta", payload, to=cam_id)

    # 2) save meta
    cam_dir = os.path.join(SAVE_DIR, cam_id)
    ensure_dir(cam_dir)
    meta_path = os.path.join(cam_dir, f"{tss}.json")
    save_json(meta_path, payload)

    # เก็บ meta ล่าสุดไว้จับคู่
    last_meta[cam_id] = payload

@io.on("pi:image")
def _pi_image(payload):
    cam_id = payload.get("cam_id", "default")
    ts = payload.get("timestamp") or now_iso()
    tss = safe_ts(ts)

    # 1) broadcast ไป client
    io.emit("image", payload, to=cam_id)

    # 2) save image
    img_b64 = payload.get("image_b64")
    if img_b64:
        cam_dir = os.path.join(SAVE_DIR, cam_id)
        ensure_dir(cam_dir)
        img_path = os.path.join(cam_dir, f"{tss}.{payload.get('image_format', IMG_EXT)}")
        save_image_b64(img_path, img_b64)

        # 3) ถ้ามี meta ล่าสุด → เซฟ meta คู่ชื่อเดียวกันด้วย
        if cam_id in last_meta:
            meta = dict(last_meta[cam_id])  # copy
            meta["_paired_with"] = os.path.basename(img_path)
            meta_path = os.path.join(cam_dir, f"{tss}.json")
            save_json(meta_path, meta)

@io.on("pi:pack")
def _pi_pack(payload):
    cam_id = payload.get("cam_id", "default")
    ts = payload.get("timestamp") or now_iso()
    tss = safe_ts(ts)

    # 1) broadcast ไป client
    io.emit("pack", payload, to=cam_id)

    # 2) save image + meta
    cam_dir = os.path.join(SAVE_DIR, cam_id)
    ensure_dir(cam_dir)

    # meta ก่อน
    meta_path = os.path.join(cam_dir, f"{tss}.json")
    save_json(meta_path, payload)

    # รูป
    img_b64 = payload.get("image_b64")
    if img_b64:
        img_path = os.path.join(cam_dir, f"{tss}.{payload.get('image_format', IMG_EXT)}")
        save_image_b64(img_path, img_b64)

    # อัปเดต meta ล่าสุด
    last_meta[cam_id] = payload

# ====== COMMANDS FROM WEB ======
@io.on("web:video_status")
def _web_video_status(data):
    cam_id = data.get("cam_id", "default")
    # broadcast ไปทุก client ในห้อง
    io.emit("server:video_status", data, to=cam_id)
    # ส่งเจาะไปที่ Pi ถ้ารู้ sid
    sid = pi_sid.get(cam_id)
    if sid:
        io.emit("server:video_status", data, to=sid)

if __name__ == "__main__":
    io.run(app, host="0.0.0.0", port=3000)
