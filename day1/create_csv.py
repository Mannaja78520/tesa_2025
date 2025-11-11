import os
import csv
import random

## replace with real detection
def predict_bbox(image_path):
    boxes = []
    for _ in range(random.randint(1,3)):
        center_x = random.randint(100, 400)
        center_y = random.randint(100, 400)
        width    = random.randint(80, 300)
        height   = random.randint(80, 300)
        boxes.append((center_x, center_y, width, height))
    return boxes

folder = "TEST_DATA"
files = os.listdir(folder)
files = [x for x in files if x.endswith('.jpg')]

with open("output.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image_name", "center_x", "center_y", "width", "height"])

for fname in files:
    image_path = os.path.join(folder, fname)
    bboxs = predict_bbox(image_path)

    for center_x, center_y, width, height in bboxs:
        row = [fname, center_x, center_y, width, height]
        print("-" * 10)
        print(row)

        with open("output.csv", "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)
