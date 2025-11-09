#!/usr/bin/env python3
import cv2
import numpy as np
import time
import os
import tensorflow.lite as tflite

# ==================== CONFIG ====================
MODEL_PATH = "trainvschair_float16.tflite"   # ไฟล์โมเดล .tflite
VIDEO_PATH = "basketball_hoop.mp4"                     # วิดีโอที่ต้องการ detect
INPUT_SIZE = (256, 256)                      # ตามที่ใช้ใน ROS node
CONF_THRESH = 0.35                           # threshold เลือกกล่อง
# ===============================================


def load_interpreter(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    interpreter = tflite.Interpreter(model_path=model_path,
                                     num_threads=4
                                    )
    interpreter.allocate_tensors()
    return interpreter


def main():
    # ---- Load model ----
    interpreter = load_interpreter(MODEL_PATH)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Input details:", input_details)
    print("Output details:", output_details)

    # ---- Open video file ----
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Cannot open video: {VIDEO_PATH}")
        return

    frame_count = 0
    last_fps_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read frame.")
            break

        start_time = time.time()

        # Resize ให้ตรงกับ input model
        frame_resized = cv2.resize(frame, INPUT_SIZE)
        h_resized, w_resized = frame_resized.shape[:2]

        # -------- Preprocess ----------
        inp = frame_resized.astype(np.float32) / 255.0
        inp = np.expand_dims(inp, axis=0)  # NHWC

        # ถ้าโมเดล dtype = uint8 ก็แปลงไป
        if input_details[0]["dtype"] == np.uint8:
            inp = (inp * 255).astype(np.uint8)

        interpreter.set_tensor(input_details[0]["index"], inp)
        interpreter.invoke()

        # -------- Postprocess ----------
        # จากโค้ดเดิมคุณ: output shape: (1, 5, N) → squeeze → (5, N) → .T → (N, 5)
        output = interpreter.get_tensor(output_details[0]["index"])
        output = np.squeeze(output, axis=0).T   # (N, 5) = [x, y, w, h, conf]

        detections = [det for det in output if det[4] > CONF_THRESH]

        x_det = y_det = 0.0

        if detections:
            best_det = max(detections, key=lambda d: d[4])
            x, y, w, h, conf = best_det

            # แปลงเป็นพิกัด pixel
            x1 = int((x - w / 2) * w_resized)
            y1 = int((y - h / 2) * h_resized)
            x2 = int((x + w / 2) * w_resized)
            y2 = int((y + h / 2) * h_resized)

            x_det = x * w_resized
            y_det = y * h_resized

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
        else:
            x_det, y_det = 0.0, 0.0

        center_x = w_resized / 2.0
        center_y = h_resized / 2.0

        elapsed_ms = (time.time() - start_time) * 1000.0
        print(
            f"Process time: {elapsed_ms:.2f} ms, "
            f"det=({x_det:.1f},{y_det:.1f}), "
            f"center=({center_x:.1f},{center_y:.1f})"
        )

        # FPS counter
        frame_count += 1 
        now = time.time()
        if now - last_fps_time >= 1.0:
            fps = frame_count / (now - last_fps_time)
            print(f"FPS: {fps:.2f}")
            frame_count = 0
            last_fps_time = now

        # Show
        cv2.imshow("Hoop detection (TFLite, mp4)", frame_resized)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):  # ESC or q
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
