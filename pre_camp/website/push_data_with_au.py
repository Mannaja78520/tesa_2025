# push_data_with_au.py
import requests
import json
from datetime import datetime, timezone
from pathlib import Path

CAM_ID = "bb120e02-dc26-48ae-a876-f17e7fb2373a"
CAM_TOKEN = "94bb9161d9bb03fb0d4382ec7a2413379e183c386ccbc89934f83d8cb6c2a651"

API_URL = f"https://tesa-api.crma.dev/api/object-detection/{CAM_ID}"

IMAGE_PATH = Path("test_0009.jpg")
OBJECT_JSON_PATH = Path("data.json")

def main():
    # 1) โหลด objects จากไฟล์ JSON (เช่น เป็น list ของกล่องที่ detect แล้ว)
    if not OBJECT_JSON_PATH.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ {OBJECT_JSON_PATH}")

    with OBJECT_JSON_PATH.open("r", encoding="utf-8") as f:
        objects = json.load(f)   # ต้องมั่นใจว่าไฟล์เป็น JSON ถูกต้อง

    # 2) เวลา timestamp ตอนนี้ (รูปแบบ ISO 8601)
    timestamp = datetime.now(timezone.utc).isoformat()

    # 3) เตรียมไฟล์รูป
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ {IMAGE_PATH}")

    files = {
        "image": (IMAGE_PATH.name, IMAGE_PATH.open("rb"), "image/jpeg")
    }

    # 4) ฟิลด์ปกติ — objects ส่งเป็น string (JSON) ผ่าน form
    data = {
        "objects": json.dumps(objects),
        "timestamp": timestamp,
    }

    headers = {
        "x-camera-token": CAM_TOKEN
    }

    print("👉 ส่งข้อมูลไปยัง:", API_URL)
    resp = requests.post(API_URL, headers=headers, files=files, data=data)

    print("Status code:", resp.status_code)
    print("Response:", resp.text)

    if resp.status_code in (200, 201):
        print("✅ ส่ง object detection data สำเร็จ")
    elif resp.status_code == 400:
        print("❌ Bad request — ลองเช็ค format ของ objects / timestamp / image")
    elif resp.status_code == 401:
        print("❌ Unauthorized — cam_id หรือ token ผิด")
    else:
        print("⚠️ Unknown error")

if __name__ == "__main__":
    main()
