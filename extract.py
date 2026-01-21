import os, csv, numpy as np
from math import log

K1 = 774.8853
K2 = 1321.0789

def radiance_to_temp(L):
    return K2 / np.log((K1 / L) + 1)

DATASET = "landsat8_train_dataset/band10_thermal"

bg, fire = [], []

with open("landsat8_train_dataset/labels.csv") as f:
    for r in csv.DictReader(f):
        b10 = np.load(f"{DATASET}/{r['sample_id']}.npy")
        T = radiance_to_temp(b10.max())

        if r["fire_label"] == "0":
            bg.append(T)
        else:
            fire.append(T)

print("\nTHERMAL CALIBRATION (K)")
print("Background max:", round(np.mean(bg),2), "/", round(np.max(bg),2))
print("Fire max      :", round(np.mean(fire),2), "/", round(np.max(fire),2))
