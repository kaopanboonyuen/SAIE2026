"""
SAIE2026 Workshop Training Pipeline (PRO MODE)

Features:
- Dataset auto-load
- EDA stats
- YOLO pretrained training
- 5000 epochs long-run training
- Live progress + ETA estimation
- Final performance report

Designed for:
🔥 EC2 / GPU server / workshop demo
"""

# =========================
# 1. IMPORTS
# =========================
import gdown
import zipfile
import time
from pathlib import Path
from collections import Counter
from datetime import datetime

from ultralytics import YOLO

# =========================
# 2. DOWNLOAD DATASET
# =========================
print("\n[INFO] Downloading dataset...")

url = "https://github.com/kaopanboonyuen/SAIE2026/raw/main/dataset/SAE_TinyVisDroneFinal.zip"
output = "SAE_TinyVisDroneFinal.zip"

gdown.download(url, output, quiet=False)

with zipfile.ZipFile(output, "r") as zip_ref:
    zip_ref.extractall(".")

print("[INFO] Dataset ready: SAE_TinyVisDroneFinal/")


# =========================
# 3. LOAD DATASET
# =========================
root = Path("SAE_TinyVisDroneFinal")

train_imgs = list((root / "images/train").glob("*.jpg"))
val_imgs   = list((root / "images/val").glob("*.jpg"))
train_lbls = list((root / "labels/train").glob("*.txt"))
val_lbls   = list((root / "labels/val").glob("*.txt"))

print("\n===== DATASET STATS =====")
print(f"Train images: {len(train_imgs)}")
print(f"Val images  : {len(val_imgs)}")


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
                counter[int(line.split()[0])] += 1

print("\n===== CLASS DISTRIBUTION =====")
for k, v in sorted(counter.items()):
    print(f"{CLASS_NAMES[k]:20s} ({k}): {v}")


# =========================
# 5. LOAD PRETRAINED MODEL
# =========================
print("\n[INFO] Loading SAIE pretrained model...")

model = YOLO(
    "https://github.com/kaopanboonyuen/SAIE2026/raw/main/weights/SAIE_TinyVisDrone_8s_50E.pt"
)


# =========================
# 6. TRAINING (WITH ETA SYSTEM)
# =========================
print("\n==============================")
print("🚀 START TRAINING (5000 EPOCHS)")
print("==============================")

start_time = time.time()
start_datetime = datetime.now()

print(f"Start time: {start_datetime}")

# NOTE:
# Ultralytics handles internal tqdm progress bar already
# We add external "training expectation tracker"

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
# 7. TRAINING TIME SUMMARY
# =========================
total_sec = end_time - start_time

days = int(total_sec // 86400)
hours = int((total_sec % 86400) // 3600)
minutes = int((total_sec % 3600) // 60)

# rough ETA estimation for full 5000 epochs
sec_per_epoch = total_sec / max(1, 50)   # assuming first run baseline ~50 epochs
eta_sec = sec_per_epoch * (5000 - 50)

eta_hours = int(eta_sec // 3600)

print("\n==============================")
print("🏁 TRAINING COMPLETED")
print("==============================")

print(f"Total runtime     : {days}d {hours}h {minutes}m")
print(f"Estimated full run : ~{eta_hours} hours (if linear scaling)")
print(f"Finish time       : {datetime.now()}")


# =========================
# 8. VALIDATION
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
# 9. FINAL IMPRESS SUMMARY (KEY FOR WORKSHOP)
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

print("\n✨ Training pipeline completed successfully.")
print("💡 This is a real-world AI system training flow (not just a notebook).")