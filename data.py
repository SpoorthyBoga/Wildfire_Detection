import numpy as np
import csv
import os

OUT_DIR = "landsat8_train_dataset"
os.makedirs(f"{OUT_DIR}/band7_swir", exist_ok=True)
os.makedirs(f"{OUT_DIR}/band10_thermal", exist_ok=True)

PATCH = 32
NUM_SAMPLES = 3000

rows = []

def generate_background():
    b7 = np.random.normal(8000, 1500, (PATCH, PATCH))
    b10 = np.random.normal(12000, 2000, (PATCH, PATCH))
    return b7, b10

def generate_early_fire():
    b7, b10 = generate_background()
    r, c = np.random.randint(8, 24), np.random.randint(8, 24)

    for i in range(-2, 3):
        for j in range(-2, 3):
            b10[r+i, c+j] += np.random.uniform(8000, 12000)
            b7[r+i, c+j] += np.random.uniform(4000, 6000)

    return b7, b10

def generate_active_fire():
    b7, b10 = generate_background()
    r, c = np.random.randint(10, 22), np.random.randint(10, 22)

    for i in range(-5, 6):
        for j in range(-5, 6):
            dist = np.sqrt(i*i + j*j)
            heat = np.exp(-dist / 3)
            b10[r+i, c+j] += heat * np.random.uniform(15000, 25000)
            b7[r+i, c+j] += heat * np.random.uniform(8000, 12000)

    return b7, b10

sample_id = 0

for _ in range(NUM_SAMPLES):
    if _ < NUM_SAMPLES * 0.4:
        b7, b10 = generate_background()
        label = 0
    elif _ < NUM_SAMPLES * 0.7:
        b7, b10 = generate_early_fire()
        label = 1
    else:
        b7, b10 = generate_active_fire()
        label = 1

    b7 = np.clip(b7, 0, 65535)
    b10 = np.clip(b10, 0, 65535)

    sid = f"L8_{sample_id:05d}"
    np.save(f"{OUT_DIR}/band7_swir/{sid}.npy", b7)
    np.save(f"{OUT_DIR}/band10_thermal/{sid}.npy", b10)

    rows.append({"sample_id": sid, "fire_label": label})
    sample_id += 1

with open(f"{OUT_DIR}/labels.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["sample_id", "fire_label"])
    writer.writeheader()
    writer.writerows(rows)

print("Spectrally realistic dataset generated.")
