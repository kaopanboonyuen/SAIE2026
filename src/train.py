"""
SAIE2026 Workshop Training Pipeline (FULL PRODUCTION READY)

Theme:
AI in the Real World: Trade-offs Behind Fast & Scalable Object Detection

This script demonstrates:
1. Dataset auto-download and validation
2. Exploratory Data Analysis (EDA)
3. Auto-fix data.yaml for Colab/EC2 compatibility
4. Pretrained YOLO model training (5000 epochs)
5. Metrics logging: Precision, Recall, F1, mAP
6. Final report with ETA and runtime summary

Designed for EC2 / Colab GPU runs for workshop demonstration
"""

# =========================
# 1. IMPORTS
# =========================
import gdown
import zipfile
import time
import yaml, os
from pathlib import Path
from collections import Counter
from datetime import datetime

from ultralytics import YOLO

# =========================
# 2. DOWNLOAD DATASET
# =========================
print("\n[INFO] Downloading SAE TinyVisDrone dataset...")
url = "https://github.com/kaopanboonyuen/SAIE2026/raw/main/dataset/SAE_TinyVisDroneFinal.zip"
output = "SAE_TinyVisDroneFinal.zip"
gdown.download(url, output, quiet=False)

# Extract ZIP directly to current folder
with zipfile.ZipFile(output, "r") as zip_ref:
    zip_ref.extractall(".")
print("[INFO] Dataset extracted to SAE_TinyVisDroneFinal/")

# =========================
# 3. LOAD DATASET
# =========================
root = Path("SAE_TinyVisDroneFinal")

train_imgs = list((root / "images/train").glob("*.jpg"))
val_imgs = list((root / "images/val").glob("*.jpg"))
train_lbls = list((root / "labels/train").glob("*.txt"))
val_lbls = list((root / "labels/val").glob("*.txt"))

print("\n===== DATASET STATS =====")
print(f"Train images: {len(train_imgs)}")
print(f"Val images  : {len(val_imgs)}")

# =========================
# 4. AUTO-FIX data.yaml
# =========================
yaml_path = root / "data.yaml"

with open(yaml_path, 'r') as f:
    cfg = yaml.safe_load(f)

# Remove any existing 'path' to avoid double-folder issue
cfg.pop('path', None)

# Set absolute paths for train/val to current root folder
cfg['train'] = os.path.join(str(root), 'images/train')
cfg['val']   = os.path.join(str(root), 'images/val')

with open(yaml_path, 'w') as f:
    yaml.dump(cfg, f)

print(f"[INFO] data.yaml auto-fixed: train -> {cfg['train']}, val -> {cfg['val']}")

# =========================
# 5. CLASS DISTRIBUTION
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
                counter[int(line.split()[0])] += 1

print("\n===== CLASS DISTRIBUTION =====")
for k, v in sorted(counter.items()):
    print(f"{CLASS_NAMES[k]:20s} ({k}): {v}")

# =========================
# 6. LOAD PRETRAINED MODEL
# =========================
print("\n[INFO] Loading SAIE pretrained YOLO model...")
model = YOLO("https://github.com/kaopanboonyuen/SAIE2026/raw/main/weights/SAIE_TinyVisDrone_8s_50E.pt")

# =========================
# 7. TRAINING (5000 epochs)
# =========================
print("\n==============================")
print("🚀 START TRAINING (5000 EPOCHS)")
print("==============================")

start_time = time.time()
start_datetime = datetime.now()
print(f"Start time: {start_datetime}")

import yaml, os

yaml_path = root / "data.yaml"

with open(yaml_path,'r') as f:
    cfg = yaml.safe_load(f)

# Remove path if exists
cfg.pop('path', None)

# Update train/val to absolute path from current root
cfg['train'] = os.path.join(str(root), 'images/train')
cfg['val']   = os.path.join(str(root), 'images/val')

with open(yaml_path,'w') as f:
    yaml.dump(cfg, f)

print(f"[INFO] data.yaml auto-fixed: train -> {cfg['train']}, val -> {cfg['val']}")

# NOTE: Ultralytics YOLO internally uses tqdm progress bar
results = model.train(
    data=str(root / "data.yaml"),
    epochs=5000,
    imgsz=416,
    batch=16,
    device=0,
    verbose=True
)

end_time = time.time()

# =========================
# 8. TRAINING TIME SUMMARY
# =========================
total_sec = end_time - start_time
days = int(total_sec // 86400)
hours = int((total_sec % 86400) // 3600)
minutes = int((total_sec % 3600) // 60)

# Rough ETA for full 5000 epochs based on first run
sec_per_epoch = total_sec / max(1, 50)  # assume first 50 epochs baseline
eta_sec = sec_per_epoch * (5000 - 50)
eta_hours = int(eta_sec // 3600)

print("\n==============================")
print("🏁 TRAINING COMPLETED")
print("==============================")
print(f"Total runtime       : {days}d {hours}h {minutes}m")
print(f"Estimated full run  : ~{eta_hours} hours (if linear scaling)")
print(f"Finish time         : {datetime.now()}")

# =========================
# 9. VALIDATION & METRICS
# =========================
print("\n[INFO] Running validation...")

metrics = model.val()

p = metrics.box.mp
r = metrics.box.mr
f1 = 2 * (p * r) / (p + r + 1e-6)

print("\n==============================")
print("📊 FINAL MODEL PERFORMANCE")
print("==============================")
print(f"Precision : {p:.4f}")
print(f"Recall    : {r:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"mAP@50    : {metrics.box.map50:.4f}")
print(f"mAP@50-95 : {metrics.box.map:.4f}")

# =========================
# 10. PER-CLASS METRICS
# =========================
print("\n===== PER-CLASS METRICS =====")
for i, name in CLASS_NAMES.items():
    p_i = metrics.box.p[i]
    r_i = metrics.box.r[i]
    f1_i = 2 * (p_i * r_i) / (p_i + r_i + 1e-6)
    print(f"{name:20s} | P:{p_i:.3f} R:{r_i:.3f} F1:{f1_i:.3f}")

# =========================
# 11. FINAL CHECK & REPORT
# =========================
print("\n==============================")
print("🎯 SAIE WORKSHOP SUMMARY")
print("==============================")

if f1 > 0.6:
    status = "🔥 PRODUCTION READY"
elif f1 > 0.4:
    status = "⚠️ GOOD BUT NEED OPTIMIZATION"
else:
    status = "❌ UNDERFITTED MODEL"

print(f"Status: {status}")
print(f"Dataset: Tiny VisDrone (Edge AI Scenario)")
print(f"Model: SAIE pretrained → fine-tuned")
print(f"Total classes: {len(CLASS_NAMES)}")
print("✨ Training pipeline completed successfully.")
print("💡 This is a real-world AI system training flow (not just a notebook).")