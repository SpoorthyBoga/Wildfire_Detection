import os
import csv
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DATASET_DIR = "landsat8_train_dataset"
B7_DIR = os.path.join(DATASET_DIR, "band7_swir")
B10_DIR = os.path.join(DATASET_DIR, "band10_thermal")
LABEL_FILE = os.path.join(DATASET_DIR, "labels.csv")

MODEL_PATH = "fire_cnn_heavy.pth"
OUT_FILE = "onboard_model.npy"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_DN = 65535.0

print("Using device:", DEVICE)

# -------------------------------------------------
# TEACHER MODEL (MUST MATCH TRAINING EXACTLY)
# -------------------------------------------------
class FireCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),

            nn.Flatten(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# -------------------------------------------------
# LOAD TRAINED TEACHER
# -------------------------------------------------
print("Loading trained teacher CNN...")

teacher = FireCNN().to(DEVICE)
teacher.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
teacher.eval()

print("Teacher model loaded successfully.")

# -------------------------------------------------
# FEATURE EXTRACTION (ONBOARD-COMPATIBLE)
# -------------------------------------------------
def extract_features(b7_dn, b10_dn):
    """
    Physics-aware, lightweight features
    """
    b7 = b7_dn / MAX_DN
    b10 = b10_dn / MAX_DN

    return np.array([
        b10.mean(),                 # overall thermal level
        b10.max(),                  # hotspot detection
        b10.std(),                  # thermal variability
        b7.max(),                   # SWIR fire response
        (b7 / (b10 + 1e-6)).max()   # SWIR–thermal contrast
    ])

# -------------------------------------------------
# LOAD DATASET
# -------------------------------------------------
with open(LABEL_FILE) as f:
    rows = list(csv.DictReader(f))

print("Total samples for distillation:", len(rows))

X = []
y_soft = []

# -------------------------------------------------
# DISTILL KNOWLEDGE
# -------------------------------------------------
print("Extracting teacher knowledge...")

with torch.no_grad():
    for i, r in enumerate(rows):
        sid = r["sample_id"]

        b7 = np.load(os.path.join(B7_DIR, f"{sid}.npy"))
        b10 = np.load(os.path.join(B10_DIR, f"{sid}.npy"))

        # Student features
        feats = extract_features(b7, b10)
        X.append(feats)

        # Teacher prediction
        x = np.stack([b7 / MAX_DN, b10 / MAX_DN], axis=0)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        prob = teacher(x).item()
        y_soft.append(prob)

        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1} samples")

X = np.array(X)
y_soft = np.array(y_soft)

# -------------------------------------------------
# TRAIN STUDENT MODEL (REGRESSION ON SOFT LABELS)
# -------------------------------------------------
print("\nTraining lightweight onboard student model...")

student = LinearRegression()
student.fit(X, y_soft)

W = student.coef_
B = student.intercept_

print("Distillation complete.")
print("Onboard weights:", W)
print("Onboard bias:", B)

# -------------------------------------------------
# SAVE ONBOARD MODEL
# -------------------------------------------------
np.save(OUT_FILE, {"W": W, "B": B})

print("\nOnboard model saved as:", OUT_FILE)
print("READY FOR SATELLITE DEPLOYMENT")
