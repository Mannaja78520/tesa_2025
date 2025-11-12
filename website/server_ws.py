# server_subscriber.py
import os, json, base64, pathlib
from datetime import datetime
import paho.mqtt.client as mqtt

BROKER = "192.168.50.242"
BASE_DIR = "inbox"  # โฟลเดอร์ปลายทาง

last_meta = {}  # เก็บ meta ล่าสุดต่อ cam_id เพื่อจับคู่กับรูป

def ts_safe(s: str) -> str:
    # ทำ timestamp ให้เป็นชื่อไฟล์ได้
    return s.replace(":", "-").replace("/", "_").replace(" ", "_")

def ensure_dir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def on_message(cli, userdata, msg):
    topic = msg.topic  # เช่น od/cam/cam_01/pack
    parts = topic.split("/")
    if len(parts) < 3 or parts[0] != "od" or parts[1] != "cam":
        print("skip", topic); return

    cam_id = parts[2]
    kind = parts[3] if len(parts) > 3 else "meta"  # เผื่อ /od/cam/<id> เฉยๆ
    cam_dir = os.path.join(BASE_DIR, cam_id)
    ensure_dir(cam_dir)

    if kind == "pack":
        # JSON รวม (อาจมี image_b64 หรือ image_url)
        payload = json.loads(msg.payload.decode("utf-8"))
        ts = payload.get("timestamp") or datetime.utcnow().isoformat()
        tss = ts_safe(ts)

        # 1) เซฟ meta (.json)
        meta_path = os.path.join(cam_dir, f"{tss}.json")
        save_json(meta_path, payload)

        # 2) ถ้ามีรูปใน JSON (image_b64) ให้เซฟด้วย
        img_b64 = payload.get("image_b64")
        if img_b64:
            img_bytes = base64.b64decode(img_b64)
            img_ext = payload.get("image_format", "jpg")
            img_path = os.path.join(cam_dir, f"{tss}.{img_ext}")
            with open(img_path, "wb") as f:
                f.write(img_bytes)
            print(f"[{cam_id}] pack: saved {img_path} + {meta_path}")
        else:
            print(f"[{cam_id}] pack: saved meta only {meta_path}")

        # เก็บไว้เป็น meta ล่าสุด
        last_meta[cam_id] = payload

    elif kind == "meta":
        # JSON meta อย่างเดียว
        payload = json.loads(msg.payload.decode("utf-8"))
        ts = payload.get("timestamp") or datetime.utcnow().isoformat()
        tss = ts_safe(ts)
        meta_path = os.path.join(cam_dir, f"{tss}.json")
        save_json(meta_path, payload)
        last_meta[cam_id] = payload
        print(f"[{cam_id}] meta: saved {meta_path}")

    elif kind == "image":
        # รูปอย่างเดียว (bytes)
        ts = datetime.utcnow().isoformat()
        tss = ts_safe(ts)
        img_path = os.path.join(cam_dir, f"{tss}.jpg")
        with open(img_path, "wb") as f:
            f.write(msg.payload)
        print(f"[{cam_id}] image: saved {img_path}")

        # ถ้ามี meta ล่าสุด → เซฟ meta คู่ชื่อเดียวกัน (optional)
        if cam_id in last_meta:
            meta = dict(last_meta[cam_id])  # copy
            meta["_paired_with"] = os.path.basename(img_path)
            pair_meta_path = os.path.join(cam_dir, f"{tss}.json")
            save_json(pair_meta_path, meta)
            print(f"[{cam_id}] image: paired meta -> {pair_meta_path}")

    else:
        # รองรับกรณีส่งมาที่ od/cam/<id> โดยไม่มี suffix → ถือเป็น meta
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except:
            print(f"[{cam_id}] unknown kind={kind}, raw bytes len={len(msg.payload)}")
            return
        ts = payload.get("timestamp") or datetime.utcnow().isoformat()
        tss = ts_safe(ts)
        meta_path = os.path.join(cam_dir, f"{tss}.json")
        save_json(meta_path, payload)
        last_meta[cam_id] = payload
        print(f"[{cam_id}] meta(default): saved {meta_path}")

def main():
    cli = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="server-sub")
    cli.on_message = on_message
    cli.connect(BROKER, 1883, 30)
    # ฟังทุกกล้อง ทุกชนิด
    cli.subscribe("od/cam/+/#", qos=1)
    print("listening on od/cam/+/# …")
    cli.loop_forever()

if __name__ == "__main__":
    main()
