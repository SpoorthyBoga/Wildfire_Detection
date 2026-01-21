import numpy as np

class EarlyExitModel:
    """
    Simulates MobileNetV2 with Early Exits.
    Ref: Section III.B (Adaptive Inference)
    """
    def __init__(self, weights_path="student_model.npy"):
        data = np.load(weights_path, allow_pickle=True).item()
        self.W = data["W"]
        self.B = data["B"]

    def sigmoid(self, x): return 1 / (1 + np.exp(-x))

    def extract_features(self, b7, b10):
        return np.array([
            b10.mean(), b10.max(), b10.std(),
            b7.max(), (b7/(b10+1e-6)).max()
        ])

    def infer(self, b7, b10):
        """
        Returns: (Confidence, Exit_Stage, Simulated_Latency_ms)
        """
        feats = self.extract_features(b7, b10)
        
        # --- EXIT 1: Shallow (Obvious Background) ---
        # If max temp is very low, exit immediately
        if feats[1] < 0.1: 
            return 0.05, "EXIT_1 (Shallow - Layer 3)", 15
            
        # --- EXIT 2: Medium (Ambiguous/Clouds) ---
        # If hot but no SWIR (likely cloud/rock), exit
        if feats[1] > 0.4 and feats[3] < 0.1:
             return 0.12, "EXIT_2 (Medium - Layer 12)", 45

        # --- EXIT 3: Deep (Full Computation) ---
        # Run the distilled student model to get final score
        score = np.dot(self.W, feats) + self.B
        conf = self.sigmoid(score)
        
        return conf, "EXIT_3 (Deep - Final Layer)", 98