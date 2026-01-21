import numpy as np
import csv
import os
import shutil

# --- CONFIGURATION ---
K1_CONST = 774.8853
K2_CONST = 1321.0789
PATCH_SIZE = 32
NUM_SAMPLES = 3000
OUT_DIR = "landsat8_dataset"

def setup_directories():
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(f"{OUT_DIR}/band7_swir")
    os.makedirs(f"{OUT_DIR}/band10_thermal")
    print(f"✔ Directories created at ./{OUT_DIR}")

def temp_to_radiance(T):
    return K1_CONST / (np.exp(K2_CONST / T) - 1)

def generate_scene(scene_type):
    # Base Background (Land Surface Temp ~285K)
    T = np.random.normal(285, 5, (PATCH_SIZE, PATCH_SIZE))
    b10 = temp_to_radiance(T)
    b7 = np.random.normal(0.15, 0.03, (PATCH_SIZE, PATCH_SIZE)) # SWIR Reflectance

    if scene_type == "background":
        return b7, b10, 0

    # Inject Fire
    r, c = np.random.randint(8, 24), np.random.randint(8, 24)
    intensity = "early" if scene_type == "early_fire" else "active"
    
    radius = 2 if intensity == "early" else 5
    temp_base = 310 if intensity == "early" else 340
    
    for i in range(-radius, radius+1):
        for j in range(-radius, radius+1):
            dist = np.sqrt(i*i + j*j)
            if dist > radius: continue
            
            heat_decay = np.exp(-dist / 2)
            T_pixel = temp_base + (np.random.uniform(0, 20) if intensity=="early" else heat_decay * 120)
            
            # Update bands
            b10[r+i, c+j] = temp_to_radiance(T_pixel)
            b7[r+i, c+j] += np.random.uniform(0.15, 0.25) if intensity=="early" else (heat_decay * 0.4)

    return b7, b10, 1

# --- MAIN EXECUTION ---
setup_directories()
rows = []

print("Generating synthetic Landsat 8 data...")
for i in range(NUM_SAMPLES):
    if i < NUM_SAMPLES * 0.4:
        sType = "background"
    elif i < NUM_SAMPLES * 0.7:
        sType = "early_fire"
    else:
        sType = "active_fire"

    b7, b10, label = generate_scene(sType)

    # Normalize/Clip to realistic sensor bounds
    b7 = np.clip(b7, 0, 1)
    b10 = np.clip(b10, 0, None)

    name = f"L8_{i:05d}"
    np.save(f"{OUT_DIR}/band7_swir/{name}.npy", b7)
    np.save(f"{OUT_DIR}/band10_thermal/{name}.npy", b10)
    rows.append({"sample_id": name, "fire_label": label})

with open(f"{OUT_DIR}/labels.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["sample_id", "fire_label"])
    writer.writeheader()
    writer.writerows(rows)

print(f"✔ Dataset Complete: {NUM_SAMPLES} scenes generated.")