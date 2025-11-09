"""
source_dir/
├── cls.txt                 ← Class names file (e.g., pi4, pi5, jetson)
├── images/                 ← Image files (all in one place)
│   ├── img001.png
│   ├── img002.png
│   ├── ...
├── labels/                 ← Corresponding YOLO label files
│   ├── img001.txt
│   ├── img002.txt
│   ├── ...

"""

import os
import shutil
import random
from pathlib import Path

def prepare_yolo_dataset(source_dir, output_dir, class_file='cls.txt', train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    image_dir = Path(source_dir) / "images"
    label_dir = Path(source_dir) / "labels"

    # Load class names
    with open(class_file, 'r') as f:
        class_names = [line.strip() for line in f if line.strip()]
    nc = len(class_names)

    image_files = list(image_dir.glob("*.*"))
    image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]

    random.shuffle(image_files)
    total = len(image_files)

    train_split = int(train_ratio * total)
    val_split = int(val_ratio * total)

    subsets = {
        'train': image_files[:train_split],
        'val': image_files[train_split:train_split + val_split],
        'test': image_files[train_split + val_split:]
    }

    for subset, files in subsets.items():
        for subfolder in ['images', 'labels']:
            os.makedirs(Path(output_dir) / subfolder / subset, exist_ok=True)
        for image_path in files:
            label_path = label_dir / (image_path.stem + ".txt")
            shutil.copy(image_path, Path(output_dir) / "images" / subset / image_path.name)
            if label_path.exists():
                shutil.copy(label_path, Path(output_dir) / "labels" / subset / label_path.name)

    # Write data.yaml for YOLOv8
    data_yaml_path = Path(output_dir) / "data.yaml"
    with open(data_yaml_path, "w") as f:
        f.write(f"train: {Path(output_dir) / 'images/train'}\n")
        f.write(f"val: {Path(output_dir) / 'images/val'}\n")
        f.write(f"test: {Path(output_dir) / 'images/test'}\n\n")
        f.write(f"nc: {nc}\n")
        f.write("names: [")
        f.write(", ".join([f"'{name}'" for name in class_names]))
        f.write("]\n")

    print(f"✅ Dataset prepared with {nc} classes. `data.yaml` created at {data_yaml_path}")
    

# Example usage -----------------------------------------

prepare_yolo_dataset(source_dir="source_dir", output_dir= "datasets", class_file="source_dir/cls.txt", train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)

# --------------------------------------------------------