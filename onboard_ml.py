# onboard_ml.py
# ------------------------------------
# Lightweight Onboard ML Inference
# ------------------------------------

import numpy as np

# Pretend these came from offline training
W = np.array([0.55, 0.30, 0.15])
B = -0.35

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def extract_features(img):
    """
    Very cheap features suitable for onboard inference
    """
    mean_intensity = img.mean()
    red_mean = img[:, :, 0].mean()
    green_mean = img[:, :, 1].mean()
    return np.array([mean_intensity, red_mean, green_mean])

def run_inference(img):
    features = extract_features(img)
    score = np.dot(W, features) + B
    return sigmoid(score)
