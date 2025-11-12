# pip install python-socketio[client] opencv-python
import socketio, time, base64, cv2
from datetime import datetime, timezone

SERVER = "http://YOUR_SERVER_IP:3000"  # แก้เป็น IP server
CAM_ID = "cam_01"
QUALITY = 75
SEND_FPS = 2

def now_iso():
    return datetime.now(timezone.utc).isoformat()

sio = socketio.Client(reconnection=True, reconnection_delay=1, reconnection_attempts=0)

@sio.event
def connect():
    print("[PI] connected")
    sio.emit("join", {"role": "pi", "cam_id": CAM_ID})

@sio.event
def disconnect():
    print("[PI] disconnected")

@sio.on("server:video_status")
def on_video_status(data):
    # data: {cam_id, status: "start"|"stop"|...}
    print("[PI] video_status:", data)
    # TODO: ใส่โค้ดควบคุมการส่งภาพ/อัตราเฟรมจริงตาม status

def send_meta_only(seq):
    meta = {
        "cam_id": CAM_ID,
        "timestamp": now_iso(),
        "seq": seq,
        "lat": 18.796143, "lon": 98.979263, "height_m": 120.3,
        "objects": [{"label":"target","confidence":0.92,"bbox":[100,120,80,60]}]
    }
    sio.emit("pi:meta", meta)

def send_image_only(frame, ts, seq):
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
    if not ok: return
    img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    payload = {
        "cam_id": CAM_ID,
        "timestamp": ts,
        "seq": seq,
        "image_b64": img_b64,
        "image_format": "jpg"
    }
    sio.emit("pi:image", payload)

def send_pack(frame, seq):
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
    if not ok: return
    img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    payload = {
        "cam_id": CAM_ID,
        "timestamp": now_iso(),
        "seq": seq,
        "lat": 18.796143, "lon": 98.979263, "height_m": 120.3,
        "objects": [{"label":"target","confidence":0.93,"bbox":[100,120,80,60]}],
        "image_b64": img_b64,
        "image_format": "jpg"
    }
    sio.emit("pi:pack", payload)

def main():
    sio.connect(SERVER, transports=["websocket"])
    cap = cv2.VideoCapture(0)  # หรือเปลี่ยนเป็นไฟล์/RTSP
    period = 1.0 / max(SEND_FPS, 1)
    seq = 0
    mode = "pack"  # "image" | "meta" | "pack"

    last = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05); continue

        t = time.time()
        if t - last >= period:
            last = t
            if mode == "image":
                ts = now_iso()
                send_image_only(frame, ts, seq)
            elif mode == "meta":
                send_meta_only(seq)
            else:
                send_pack(frame, seq)
            seq += 1

if __name__ == "__main__":
    main()
