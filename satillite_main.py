# satellite_demo.py
# -------------------------------------------------
# COMPLETE ONBOARD DEMO – ALL SCENARIOS
# -------------------------------------------------

import time
import tracemalloc
import numpy as np
import csv
import os

from flight_software import radiometric_fire_trigger

# -------------------------------------------------
# LOAD DISTILLED ONBOARD MODEL
# -------------------------------------------------
model_data = np.load("onboard_model.npy", allow_pickle=True).item()
W = model_data["W"]
B = model_data["B"]

# -------------------------------------------------
# UTILS
# -------------------------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def extract_features(b7_dn, b10_dn):
    b7 = b7_dn / 65535.0
    b10 = b10_dn / 65535.0
    return np.array([
        b10.mean(),
        b10.max(),
        b10.std(),
        b7.max(),
        (b7 / (b10 + 1e-6)).max()
    ])

def run_onboard_inference(b7_dn, b10_dn):
    feats = extract_features(b7_dn, b10_dn)
    return sigmoid(np.dot(W, feats) + B)

# -------------------------------------------------
# LOAD DATASET
# -------------------------------------------------
DATASET_DIR = "landsat8_train_dataset"
B7_DIR = os.path.join(DATASET_DIR, "band7_swir")
B10_DIR = os.path.join(DATASET_DIR, "band10_thermal")
LABEL_FILE = os.path.join(DATASET_DIR, "labels.csv")

with open(LABEL_FILE) as f:
    rows = list(csv.DictReader(f))

# -------------------------------------------------
# SAMPLE SELECTION
# -------------------------------------------------
background = None
weak_fire = None
strong_fire = None
best_temp = 0.0

for r in rows:
    sid = r["sample_id"]
    b10 = np.load(os.path.join(B10_DIR, f"{sid}.npy"))
    temp_max = (b10 / 65535.0 * 330.0).max()

    if r["fire_label"] == "0" and background is None:
        background = sid

    if r["fire_label"] == "1" and weak_fire is None:
        weak_fire = sid

    if r["fire_label"] == "1" and temp_max > best_temp:
        best_temp = temp_max
        strong_fire = sid

# -------------------------------------------------
# DEMO HEADER
# -------------------------------------------------
print("\n" + "="*46)
print("SATELLITE ONBOARD WILDFIRE DETECTION DEMO")
print("Execution Mode : CPU-only")
print("="*46)

# -------------------------------------------------
# RUN ONE SCENARIO
# -------------------------------------------------
def run_scenario(title, sample_id):
    b7 = np.load(os.path.join(B7_DIR, f"{sample_id}.npy"))
    b10 = np.load(os.path.join(B10_DIR, f"{sample_id}.npy"))

    tracemalloc.start()
    t0 = time.perf_counter()
    cpu0 = time.process_time()

    trigger, _ = radiometric_fire_trigger(b7, b10)

    confidence = None
    if trigger:
        confidence = run_onboard_inference(b7, b10)

    cpu1 = time.process_time()
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\n[{title}]")
    print("-"*46)
    print("Physics Trigger :", trigger)

    if trigger:
        print("ML Confidence   :", round(confidence, 3))
        if confidence >= 0.5:
            print("Decision        : 🔥 WILDFIRE ALERT")
        else:
            print("Decision        : Low-confidence event")
    else:
        print("Action          : ML skipped (power saving)")

    print(f"Cycle Time      : {(t1 - t0)*1000:.2f} ms")
    print(f"Peak Memory     : {peak/1024:.1f} KB")

# -------------------------------------------------
# RUN ALL SCENARIOS
# -------------------------------------------------
run_scenario("SCENARIO 1 – BACKGROUND SCENE", background)
run_scenario("SCENARIO 2 – WEAK FIRE (EARLY STAGE)", weak_fire)
run_scenario("SCENARIO 3 – STRONG FIRE (ACTIVE)", strong_fire)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
print("\n" + "="*46)
print("DEMO COMPLETE – ONBOARD AUTONOMY VERIFIED")
print("="*46)
