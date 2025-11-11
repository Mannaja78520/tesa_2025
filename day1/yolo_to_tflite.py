from ultralytics import YOLO
import os
import tensorflow

# Load the YOLO11 model
username = os.getenv("USER")
model_path = f"drone_lastest.pt"
model = YOLO(model_path)
picture_path = f"P1_DATASET/test_0003.jpg"

# Export the model to TFLite format
model.export(format="tflite",
            #  format="TensorRT",
            # opset=12,
            simplify=True,
            # imgsz=(320, 240),
            imgsz = 256,
            optimize = True,
            int8 = True,
            device = "cpu",
            # Specifies the device for exporting: GPU (device=0), CPU (device=cpu), MPS for Apple silicon (device=mps) or
            # DLA for NVIDIA Jetson (device=dla:0 or device=dla:1). TensorRT exports automatically use GPU.
            # nms = True,
            )

# # Load the exported TFLite model
# tflite_model = YOLO("trainvschair_float32.tflite")

# # Run inference
# results = tflite_model(picture_path)