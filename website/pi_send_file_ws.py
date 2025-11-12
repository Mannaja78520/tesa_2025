# pip install python-socketio[client] opencv-python
import socketio, base64, cv2, os, time
from datetime import datetime, timezone

SERVER = "http://192.168.50.242:3000"   # แก้ IP เซิร์ฟเวอร์
CAM_ID = "cam_01"
QUALITY = 80

def now_iso(): return datetime.now(timezone.utc).isoformat()

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
    print("[PI] video_status:", data)
    # TODO: handle start/stop/pause ตามต้องการ

def send_image_file(path, with_meta=False):
    if not os.path.exists(path):
        print("file not found:", path); return
    img = cv2.imread(path)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
    if not ok:
        print("encode failed"); return
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    ts = now_iso()

    if with_meta:
        payload = {
            "cam_id": CAM_ID, "timestamp": ts, "seq": int(time.time()),
            "lat": 18.796143, "lon": 98.979263, "height_m": 120.3,
            "objects": [{"label":"target","confidence":0.9,"bbox":[100,120,80,60]}],
            "image_b64": b64, "image_format": "jpg"
        }
        sio.emit("pi:pack", payload)
        print("sent PACK")
    else:
        payload = {"cam_id": CAM_ID, "timestamp": ts, "image_b64": b64, "image_format": "jpg"}
        sio.emit("pi:image", payload)
        print("sent IMAGE")

def send_meta_only():
    payload = {
        "cam_id": CAM_ID, "timestamp": now_iso(), "seq": int(time.time()),
        "lat": 18.796143, "lon": 98.979263, "height_m": 120.3,
        "objects": [{"label":"target","confidence":0.9,"bbox":[100,120,80,60]}],
    }
    sio.emit("pi:meta", payload)
    print("sent META")

def main():
    try:
        sio.connect(SERVER, transports=["websocket"], wait_timeout=5)
    except Exception as e:
        print("connect failed:", e); return

    print("Commands:")
    print("  1 <path>     -> send IMAGE only, e.g. 1 img_0004.jpg")
    print("  2 <path>     -> send PACK (meta+image), e.g. 2 img_0004.jpg")
    print("  3            -> send META only")
    print("  q            -> quit")

    while True:
        cmd = input(">> ").strip().split()
        if not cmd: continue
        if cmd[0] == "q": break
        elif cmd[0] == "1" and len(cmd) >= 2:
            send_image_file(cmd[1], with_meta=False)
        elif cmd[0] == "2" and len(cmd) >= 2:
            send_image_file(cmd[1], with_meta=True)
        elif cmd[0] == "3":
            send_meta_only()
        else:
            print("unknown command")

if __name__ == "__main__":
    main()
