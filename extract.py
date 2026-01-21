import os
import csv
import numpy as np

DATASET_DIR = "landsat8_train_dataset"
B10_DIR = os.path.join(DATASET_DIR, "band10_thermal")
LABEL_FILE = os.path.join(DATASET_DIR, "labels.csv")

temps_bg = []
temps_fire = []

with open(LABEL_FILE) as f:
    reader = csv.DictReader(f)
    for row in reader:
        sid = row["sample_id"]
        b10 = np.load(os.path.join(B10_DIR, f"{sid}.npy"))

        temp_max = (b10 / 65535.0 * 330.0).max()

        if row["fire_label"] == "0":
            temps_bg.append(temp_max)
        else:
            temps_fire.append(temp_max)

print("\n=== THERMAL STATISTICS (K) ===")
print("Background max temp:")
print("  mean :", round(np.mean(temps_bg), 2))
print("  max  :", round(np.max(temps_bg), 2))

print("\nFire max temp:")
print("  mean :", round(np.mean(temps_fire), 2))
print("  max  :", round(np.max(temps_fire), 2))
