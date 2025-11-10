#!/usr/bin/env python3
import cv2
import numpy as np
import time
import os
from flask import Flask, Response
import tensorflow.lite as tflite
from threading import Thread, Lock

# ==================== CONFIG ====================
MODEL_PATH = "trainvschair_float16.tflite"   # ชื่อไฟล์โมเดล .tflite
VIDEO_PATH = "basketball_hoop.mp4"           # วิดีโอ, หรือใช้ 0 ถ้าเป็นกล้องจริง
INPUT_SIZE = (256, 256)                      # ตามที่ใช้ train / ROS
CONF_THRESH = 0.35                           # confidence threshold
LOOP_VIDEO = True                            # ถ้าใช้ไฟล์ video แล้วอยากวนลูปให้เป็น True
JPEG_QUALITY = 50                            # คุณภาพ JPEG (0-100 ยิ่งต่ำยิ่งเร็ว/เบา)
OUTPUT_SIZE = (320, 320)                     # ขนาดภาพที่ stream ออกไป
INFER_EVERY_N = 1                            # รัน TFLite ทุกกี่เฟรม (ยิ่งมากยิ่งลื่นขึ้น แต่กล่องอัปเดตช้าลง)
# ===============================================

# -------- เตรียม TFLite model ----------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
print("Output details:", output_details)

# -------- เตรียม VideoCapture ----------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

# -------- Flask app ----------
app = Flask(__name__)

# -------- ตัวแปรใช้ร่วมระหว่าง worker กับ Flask ----------
latest_jpeg = None
jpeg_lock = Lock()


def run_inference_and_draw(frame):
    """
    รับ frame (BGR), ย่อเป็น 256x256, รัน TFLite, วาดกล่องแล้วคืนภาพ (BGR)
    """
    frame_resized = cv2.resize(frame, INPUT_SIZE)
    h_resized, w_resized = frame_resized.shape[:2]

    # Preprocess
    inp = frame_resized.astype(np.float32) / 255.0
    inp = np.expand_dims(inp, axis=0)  # NHWC

    if input_details[0]["dtype"] == np.uint8:
        inp = (inp * 255).astype(np.uint8)

    # Run TFLite (เรียกแค่ครั้งเดียว)
    interpreter.set_tensor(input_details[0]["index"], inp)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"]).copy()  # copy กัน reference ใน interpreter

    # Postprocess
    output = np.squeeze(output, axis=0).T  # shape: (N, 5) = [x, y, w, h, conf]

    detections = [det for det in output if det[4] > CONF_THRESH]

    if detections:
        best_det = max(detections, key=lambda d: d[4])
        x, y, w, h, conf = best_det

        x1 = int((x - w / 2) * w_resized)
        y1 = int((y - h / 2) * h_resized)
        x2 = int((x + w / 2) * w_resized)
        y2 = int((y + h / 2) * h_resized)

        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame_resized,
            f"hoop: {conf:.2f}",
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            2,
        )

    return frame_resized


def worker_loop():
    """
    thread นี้จะ:
    - อ่าน frame จาก video
    - รัน TFLite เฉพาะบางเฟรม (INFER_EVERY_N)
    - resize เป็น OUTPUT_SIZE
    - encode JPEG
    - เก็บไว้ใน latest_jpeg
    """
    global latest_jpeg, cap
    frame_index = 0
    last_time = time.time()
    last_annotated = None  # เก็บเฟรมที่มีกรอบล่าสุด

    while True:
        ret, frame = cap.read()
        if not ret:
            if LOOP_VIDEO:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                print("End of video.")
                break

        frame_index += 1
        start = time.time()

        # ตัดสินใจว่าจะรันโมเดลไหม
        if (frame_index % INFER_EVERY_N == 0) or (last_annotated is None):
            # รัน TFLite + วาดกล่อง
            annotated = run_inference_and_draw(frame)
            last_annotated = annotated
        else:
            # ไม่รัน TFLite → ใช้เฟรมปกติหรือใช้เฟรมที่ annotate ครั้งล่าสุดก็ได้
            # ถ้าอยากให้วิดีโอลื่นสุด ๆ (กล่องไม่ค้าง) ใช้ภาพดิบ:
            annotated = cv2.resize(frame, INPUT_SIZE)
            # ถ้าอยาก reuse กล่องเดิม:
            # annotated = last_annotated

        # resize สำหรับแสดง
        display_frame = cv2.resize(annotated, OUTPUT_SIZE)

        # Encode เป็น JPEG
        ret2, buffer = cv2.imencode(
            ".jpg",
            display_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
        )
        if not ret2:
            continue

        frame_bytes = buffer.tobytes()

        # update latest_jpeg
        with jpeg_lock:
            latest_jpeg = frame_bytes

        # log ประมาณ FPS ของ loop (ไม่ใช่ infer ทุกเฟรมแล้ว)
        now = time.time()
        loop_time = (now - start) * 1000.0
        if now - last_time >= 1.0:
            fps = frame_index / (now - last_time)
            print(f"[worker] LOOP FPS: {fps:.2f}, last loop time: {loop_time:.2f} ms")
            frame_index = 0
            last_time = now


def gen_frames():
    """
    generator สำหรับ stream MJPEG
    ดึง latest_jpeg จาก worker ส่งออกไป
    """
    global latest_jpeg
    while True:
        with jpeg_lock:
            frame_bytes = latest_jpeg

        if frame_bytes is None:
            # ยังไม่มี frame เลย (worker อาจยังไม่ทันอ่าน) → รอแป๊บ
            time.sleep(0.01)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.route("/video")
def video():
    """
    endpoint หลักสำหรับ stream
    browser/React: <img src="http://SERVER_IP:5000/video" />
    """
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/")
def index():
    """
    หน้า demo เล็ก ๆ แค่โชว์ video stream
    """
    return """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Hoop Detection Stream</title>
</head>
<body style="margin:0; background:#000; display:flex; justify-content:center; align-items:center; height:100vh;">
  <img src="/video" style="max-width:100vw; max-height:100vh; border:2px solid #22c55e;" />
</body>
</html>
    """


if __name__ == "__main__":
    # start worker thread ก่อน
    t = Thread(target=worker_loop, daemon=True)
    t.start()

    # Flask แค่ส่ง latest_jpeg ไม่ยุ่งกับ interpreter
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=False)
