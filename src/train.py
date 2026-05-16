# ============================================================
# SAIE 2026
# Super AI Engineer Thailand
# ------------------------------------------------------------
# Topic:
# AI in the Real World:
# Trade-offs Behind Fast & Scalable Object Detection
#
# Training Script:
# SAIE_train.py
#
# Author:
# Super AI Engineer Workshop
# ============================================================

# ============================================================
# STEP 0 — INSTALL REQUIRED PACKAGES
# ============================================================

# Recommended:
# pip install ultralytics gdown seaborn opencv-python matplotlib

# ============================================================
# STEP 1 — IMPORT LIBRARIES
# ============================================================

import random
import zipfile
from pathlib import Path
from collections import Counter

import cv2
import gdown
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ultralytics import YOLO


# ============================================================
# STEP 2 — GLOBAL CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# Dataset Configuration
# ------------------------------------------------------------

DATASET_URL = (
    "https://github.com/kaopanboonyuen/SAIE2026/raw/main/dataset/"
    "SAE_TinyVisDroneFinal.zip"
)

DATASET_ZIP = "dataset.zip"

DATA_ROOT = Path("data/SAE_TinyVisDroneFinal")

# ------------------------------------------------------------
# Pretrained Model
# ------------------------------------------------------------

# Instead of yolov8s.pt, we use our custom SAIE pretrained model
# trained specifically for Tiny VisDrone scenarios.

PRETRAINED_MODEL = (
    "https://github.com/kaopanboonyuen/SAIE2026/raw/main/weights/"
    "SAIE_TinyVisDrone_8s_50E.pt"
)

PRETRAINED_WEIGHT_NAME = "SAIE_TinyVisDrone_8s_50E.pt"

# ------------------------------------------------------------
# Training Configuration
# ------------------------------------------------------------

EPOCHS = 5000
IMAGE_SIZE = 416
BATCH_SIZE = 16
DEVICE = 0

PROJECT_NAME = "SAIE2026"
RUN_NAME = "SAIE_TinyVisDrone_Training"

# ------------------------------------------------------------
# Class Names
# ------------------------------------------------------------

CLASS_NAMES = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor"
}


# ============================================================
# STEP 3 — DOWNLOAD DATASET
# ============================================================

print("\n================================================")
print("STEP 3 — DOWNLOADING DATASET")
print("================================================\n")

gdown.download(DATASET_URL, DATASET_ZIP, quiet=False)

print("\nDataset download completed.")


# ============================================================
# STEP 4 — EXTRACT DATASET
# ============================================================

print("\n================================================")
print("STEP 4 — EXTRACTING DATASET")
print("================================================\n")

with zipfile.ZipFile(DATASET_ZIP, "r") as zip_ref:
    zip_ref.extractall("data")

print("Dataset extracted successfully.")

# Auto-fix YAML path

yaml_path = DATA_ROOT / "data.yaml"

yaml_text = """
path: data/SAE_TinyVisDroneFinal

train: images/train
val: images/val

names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
"""

with open(yaml_path, "w") as f:
    f.write(yaml_text)

print("data.yaml updated.")

# ============================================================
# STEP 5 — VERIFY DATASET STRUCTURE
# ============================================================

print("\n================================================")
print("STEP 5 — VERIFYING DATASET")
print("================================================\n")

print("Dataset root exists:", DATA_ROOT.exists())

print("\nDataset folders:")

for folder in DATA_ROOT.iterdir():
    print(" -", folder)


# ============================================================
# STEP 6 — LOAD IMAGE AND LABEL PATHS
# ============================================================

train_imgs = list((DATA_ROOT / "images/train").glob("*.jpg"))
val_imgs = list((DATA_ROOT / "images/val").glob("*.jpg"))

train_lbls = list((DATA_ROOT / "labels/train").glob("*.txt"))
val_lbls = list((DATA_ROOT / "labels/val").glob("*.txt"))


# ============================================================
# STEP 7 — DATASET STATISTICS
# ============================================================

print("\n================================================")
print("STEP 7 — DATASET STATISTICS")
print("================================================\n")

print(f"Train images : {len(train_imgs)}")
print(f"Validation images : {len(val_imgs)}")

print(f"Train labels : {len(train_lbls)}")
print(f"Validation labels : {len(val_lbls)}")


# ============================================================
# STEP 8 — CLASS DISTRIBUTION ANALYSIS
# ============================================================

print("\n================================================")
print("STEP 8 — CLASS DISTRIBUTION")
print("================================================\n")

counter = Counter()

all_labels = train_lbls + val_lbls

for label_file in all_labels:

    with open(label_file) as f:

        for line in f:

            if line.strip():

                cls = int(line.split()[0])

                counter[cls] += 1

for cls_id, count in sorted(counter.items()):

    class_name = CLASS_NAMES.get(cls_id, "unknown")

    print(f"{class_name:20s} ({cls_id}) : {count}")


# ============================================================
# STEP 9 — VISUALIZE RANDOM TRAINING SAMPLES
# ============================================================

print("\n================================================")
print("STEP 9 — VISUALIZING RANDOM SAMPLES")
print("================================================\n")


def show_sample(img_path, label_path):

    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]

    with open(label_path) as f:
        lines = f.readlines()

    for line in lines:

        cls, x, y, bw, bh = map(float, line.split())

        x1 = int((x - bw / 2) * w)
        y1 = int((y - bh / 2) * h)

        x2 = int((x + bw / 2) * w)
        y2 = int((y + bh / 2) * h)

        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            img,
            str(int(cls)),
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1
        )

    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.show()


# ------------------------------------------------------------
# Show 3 random training examples
# ------------------------------------------------------------

for _ in range(3):

    img_file = random.choice(train_imgs)

    lbl_file = (
        DATA_ROOT /
        "labels/train" /
        f"{img_file.stem}.txt"
    )

    show_sample(img_file, lbl_file)


# ============================================================
# STEP 10 — DOWNLOAD PRETRAINED SAIE MODEL
# ============================================================

print("\n================================================")
print("STEP 10 — DOWNLOADING SAIE PRETRAINED MODEL")
print("================================================\n")

gdown.download(
    PRETRAINED_MODEL,
    PRETRAINED_WEIGHT_NAME,
    quiet=False
)

print("\nPretrained model downloaded successfully.")


# ============================================================
# STEP 11 — LOAD MODEL
# ============================================================

print("\n================================================")
print("STEP 11 — LOADING MODEL")
print("================================================\n")

model = YOLO(PRETRAINED_WEIGHT_NAME)

print("Model loaded successfully.")


# ============================================================
# STEP 12 — TRAIN MODEL
# ============================================================

print("\n================================================")
print("STEP 12 — TRAINING STARTED")
print("================================================\n")

print("Training Configuration:")
print(f"Epochs     : {EPOCHS}")
print(f"Image Size : {IMAGE_SIZE}")
print(f"Batch Size : {BATCH_SIZE}")
print(f"Device     : {DEVICE}")

# ------------------------------------------------------------
# Main Training
# ------------------------------------------------------------

results = model.train(

    data=str(yaml_path),

    epochs=5000,

    imgsz=416,

    batch=16,

    device=0,

    workers=8,

    cache="ram",

    amp=True,

    pretrained=True,

    save=True,

    save_period=25,

    cos_lr=True,

    patience=300,

    project="SAIE2026",

    name="TinyVisDrone_5000E"
)

print("\nTraining completed successfully.")


# ============================================================
# STEP 13 — SAVE FINAL MODEL
# ============================================================

print("\n================================================")
print("STEP 13 — SAVING FINAL MODEL")
print("================================================\n")

FINAL_MODEL_PATH = "SAIE_TinyVisDrone_Final.pt"

model.save(FINAL_MODEL_PATH)

print(f"Final model saved to: {FINAL_MODEL_PATH}")


# ============================================================
# STEP 14 — VALIDATION
# ============================================================

print("\n================================================")
print("STEP 14 — MODEL VALIDATION")
print("================================================\n")

metrics = model.val()

print("\nValidation completed.")


# ============================================================
# STEP 15 — CONFUSION MATRIX
# ============================================================

print("\n================================================")
print("STEP 15 — CONFUSION MATRIX")
print("================================================\n")

cm = metrics.confusion_matrix.matrix

print("Confusion Matrix Shape:", cm.shape)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=False,
    cmap="Blues",
    xticklabels=list(CLASS_NAMES.values()),
    yticklabels=list(CLASS_NAMES.values())
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Ground Truth")

plt.show()


# ============================================================
# STEP 16 — PRECISION / RECALL / F1 SCORE
# ============================================================

print("\n================================================")
print("STEP 16 — METRICS ANALYSIS")
print("================================================\n")

p = metrics.box.mp
r = metrics.box.mr

f1 = 2 * (p * r) / (p + r + 1e-6)

print(f"Mean Precision : {p:.4f}")
print(f"Mean Recall    : {r:.4f}")
print(f"Mean F1-score  : {f1:.4f}")


# ============================================================
# STEP 17 — PER-CLASS PERFORMANCE
# ============================================================

print("\n================================================")
print("STEP 17 — PER-CLASS PERFORMANCE")
print("================================================\n")

precision = metrics.box.p
recall = metrics.box.r

for i, name in enumerate(CLASS_NAMES.values()):

    p_i = precision[i]

    r_i = recall[i]

    f1_i = 2 * (p_i * r_i) / (p_i + r_i + 1e-6)

    print(
        f"{name:20s} | "
        f"P:{p_i:.3f} "
        f"R:{r_i:.3f} "
        f"F1:{f1_i:.3f}"
    )


# ============================================================
# STEP 18 — INFERENCE DEMO
# ============================================================

print("\n================================================")
print("STEP 18 — INFERENCE DEMO")
print("================================================\n")

test_imgs = list((DATA_ROOT / "images/val").glob("*.jpg"))

for i in range(3):

    img_path = random.choice(test_imgs)

    print(f"\nRunning inference on: {img_path.name}")

    results = model.predict(

        img_path,

        imgsz=IMAGE_SIZE,

        conf=0.25
    )

    for r in results:

        im = r.plot()

        plt.figure(figsize=(10, 6))

        plt.imshow(im)

        plt.axis("off")

        plt.show()


# ============================================================
# STEP 19 — EXPORT MODEL
# ============================================================

print("\n================================================")
print("STEP 19 — EXPORTING MODEL")
print("================================================\n")

# ------------------------------------------------------------
# Export to ONNX
# ------------------------------------------------------------

model.export(format="onnx")

print("ONNX export completed.")

# ------------------------------------------------------------
# Export to TensorRT
# ------------------------------------------------------------

try:

    model.export(format="engine")

    print("TensorRT export completed.")

except Exception as e:

    print("\nTensorRT export skipped.")
    print("Reason:", e)


# ============================================================
# STEP 20 — FINAL SUMMARY
# ============================================================

print("\n================================================")
print("FINAL SUMMARY")
print("================================================\n")

print("Dataset READY          :", True)

print("Training images        :", len(train_imgs))

print("Validation images      :", len(val_imgs))

print("Number of classes      :", len(counter))

print("Training epochs        :", EPOCHS)

print("Model exported         :", True)

print("\nSAIE 2026 training pipeline completed successfully.")
print("Ready for real-world object detection deployment.")
print("================================================\n")