"""
SAIE2026 Workshop Training Pipeline
Theme:
AI in the Real World: Trade-offs Behind Fast & Scalable Object Detection

This script demonstrates:
1. Dataset download and validation
2. EDA (counts and class distribution)
3. Model training using SAIE pretrained weights
4. Evaluation (metrics, confusion matrix, F1, Precision, Recall, mAP)
5. Final summary and time logging

Dataset: SAE_TinyVisDroneFinal
Pretrained model: SAIE_TinyVisDrone_8s_50E.pt
"""

# =========================
# 1. IMPORTS
# =========================
import gdown
import zipfile
import time
from pathlib import Path
from collections import Counter

from ultralytics import YOLO

# =========================
# 2. DOWNLOAD DATASET
# =========================
print("[INFO] Downloading dataset...")
url = "https://github.com/kaopanboonyuen/SAIE2026/raw/main/dataset/SAE_TinyVisDroneFinal.zip"
output = "dataset.zip"
gdown.download(url, output, quiet=False)

with zipfile.ZipFile(output, "r") as zip_ref:
    zip_ref.extractall("data")
print("[INFO] Dataset extracted successfully.")

# =========================
# 3. LOAD DATASET
# =========================
root = Path("data/SAE_TinyVisDroneFinal")
print(f"[INFO] Dataset root exists: {root.exists()}")
print(f"[INFO] Dataset folders: {list(root.iterdir())}")

train_imgs = list((root / "images/train").glob("*.jpg"))
val_imgs = list((root / "images/val").glob("*.jpg"))
train_lbls = list((root / "labels/train").glob("*.txt"))
val_lbls = list((root / "labels/val").glob("*.txt"))

print("\n===== DATASET STATS =====")
print(f"Train images: {len(train_imgs)}")
print(f"Val images: {len(val_imgs)}")
print(f"Train labels: {len(train_lbls)}")
print(f"Val labels: {len(val_lbls)}")

# =========================
# 4. CLASS DISTRIBUTION
# =========================
CLASS_NAMES = {
    0: "pedestrian", 1: "people", 2: "bicycle", 3: "car", 4: "van",
    5: "truck", 6: "tricycle", 7: "awning-tricycle", 8: "bus", 9: "motor"
}

counter = Counter()
for lf in train_lbls + val_lbls:
    with open(lf, "r") as f:
        for line in f:
            if line.strip():
                cls = int(line.split()[0])
                counter[cls] += 1

print("\n===== CLASS DISTRIBUTION =====")
for k, v in sorted(counter.items()):
    print(f"{CLASS_NAMES.get(k,'unknown'):20s} ({k}): {v}")

# =========================
# 5. LOAD PRETRAINED MODEL
# =========================
print("\n[INFO] Loading pretrained model from GitHub...")
model = YOLO("https://github.com/kaopanboonyuen/SAIE2026/raw/main/weights/SAIE_TinyVisDrone_8s_50E.pt")

# =========================
# 6. TRAINING WITH TIME LOGGING
# =========================
print("\n==============================")
print("🚀 START TRAINING")
print("==============================\n")

start_time = time.time()

results = model.train(
    data=str(root / "data.yaml"),
    epochs=2000,
    imgsz=416,
    batch=16,
    device=0,
    verbose=True
)

end_time = time.time()
elapsed = end_time - start_time
days = int(elapsed // 86400)
hours = int((elapsed % 86400) // 3600)
minutes = int((elapsed % 3600) // 60)
seconds = int(elapsed % 60)

print("\n==============================")
print("🏁 TRAINING COMPLETED")
print("==============================")
print(f"Total training time: {days}d {hours}h {minutes}m {seconds}s")

# =========================
# 7. VALIDATION
# =========================
print("\n[INFO] Running validation...")
metrics = model.val()

# =========================
# 8. METRICS SUMMARY
# =========================
precision = metrics.box.mp
recall = metrics.box.mr
f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
map50 = metrics.box.map50
map95 = metrics.box.map

print("\n==============================")
print("📊 FINAL MODEL PERFORMANCE")
print("==============================")
print(f"Precision (P) : {precision:.4f}")
print(f"Recall    (R) : {recall:.4f}")
print(f"F1-score      : {f1:.4f}")
print(f"mAP@50        : {map50:.4f}")
print(f"mAP@50-95     : {map95:.4f}")

# =========================
# 9. PER-CLASS METRICS
# =========================
print("\n===== PER-CLASS METRICS =====")
for i, name in CLASS_NAMES.items():
    p_i = metrics.box.p[i]
    r_i = metrics.box.r[i]
    f1_i = 2 * (p_i * r_i) / (p_i + r_i + 1e-6)
    print(f"{name:20s} | P:{p_i:.3f} R:{r_i:.3f} F1:{f1_i:.3f}")

# =========================
# 10. FINAL CHECK
# =========================
print("\n===== FINAL CHECK =====")
print(f"Train images: {len(train_imgs)}")
print(f"Val images  : {len(val_imgs)}")
print(f"Classes     : {len(counter)}")
print("Pipeline ready: ✅")
print("SAIE2026 workshop training pipeline completed successfully.")