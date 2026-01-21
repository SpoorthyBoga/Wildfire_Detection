import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import csv
from sklearn.linear_model import LinearRegression

# --- CONFIG ---
DEVICE = "cpu" # Safe for all machines
DATA_DIR = "landsat8_dataset"

# --- 1. DEFINE TEACHER MODEL (Heavy CNN) ---
class FireCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

# --- 2. DATA LOADER ---
class FireDataset(Dataset):
    def __init__(self):
        with open(f"{DATA_DIR}/labels.csv") as f:
            self.rows = list(csv.DictReader(f))
    def __getitem__(self, i):
        sid = self.rows[i]["sample_id"]
        b7 = np.load(f"{DATA_DIR}/band7_swir/{sid}.npy")
        b10 = np.load(f"{DATA_DIR}/band10_thermal/{sid}.npy")
        x = np.stack([b7, b10])
        y = float(self.rows[i]["fire_label"])
        return torch.tensor(x, dtype=torch.float32), torch.tensor([y]), b7, b10
    def __len__(self): return len(self.rows)

# --- 3. TRAINING LOOP ---
print(">>> Training Teacher Model (CNN)...")
dataset = FireDataset()
loader = DataLoader(dataset, batch_size=32, shuffle=True)
teacher = FireCNN().to(DEVICE)
opt = optim.Adam(teacher.parameters(), lr=1e-3)
loss_fn = nn.BCELoss()

for epoch in range(5):
    total_loss = 0
    for x, y, _, _ in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = teacher(x)
        loss = loss_fn(pred, y)
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item()
    print(f"    Epoch {epoch+1}: Loss {total_loss/len(loader):.4f}")

# --- 4. DISTILLATION (Teacher -> Student) ---
print("\n>>> Distilling to Student Model (Linear)...")
teacher.eval()
X_student, y_teacher = [], []

def extract_features(b7, b10):
    # Lightweight features for the onboard CPU
    return [b10.mean(), b10.max(), b10.std(), b7.max(), (b7/(b10+1e-6)).max()]

with torch.no_grad():
    for x, _, b7_batch, b10_batch in loader:
        teacher_preds = teacher(x.to(DEVICE)).cpu().numpy().flatten()
        
        for i in range(len(teacher_preds)):
            feats = extract_features(b7_batch[i].numpy(), b10_batch[i].numpy())
            X_student.append(feats)
            y_teacher.append(teacher_preds[i])

# Fit Linear Regression to mimic the Teacher
student = LinearRegression().fit(X_student, y_teacher)
weights = {"W": student.coef_, "B": student.intercept_}
np.save("student_model.npy", weights)

print(f"✔ Distillation Complete. Student weights saved.")
print(f"    Weights: {student.coef_}")