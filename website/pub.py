# publisher.py — ถ้าจะ pub จากอีกเครื่อง: call HTTP endpoint ที่ server ทำไว้หรือยิง WS ก็ได้
# วิธีง่าย: เพิ่ม REST บน server รับ JSON แล้ว server emit ต่อ
# ตัวอย่าง endpoint (เพิ่มเข้า ws_server.py):
#
# @app.route("/pub/<cam_id>", methods=["POST"])
# def pub(cam_id):
#     publish(cam_id, request.get_json(force=True))
#     return {"ok": True}
#
# จากนั้น publisher ยิง:
import requests, time
while True:
    requests.post("http://192.168.50.50:3000/pub/cam01", json={"ts": time.time(), "objects":[]})
    time.sleep(1)
