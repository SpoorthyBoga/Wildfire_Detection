import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DATASET_DIR = "landsat8_train_dataset"
B7_DIR = os.path.join(DATASET_DIR, "band7_swir")
B10_DIR = os.path.join(DATASET_DIR, "band10_thermal")
LABEL_FILE = os.path.join(DATASET_DIR, "labels.csv")

BATCH_SIZE = 32
EPOCHS = 12
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)

# -------------------------------------------------
# DATASET
# -------------------------------------------------
class LandsatFireDataset(Dataset):
    def __init__(self):
        with open(LABEL_FILE) as f:
            self.rows = list(csv.DictReader(f))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        sid = r["sample_id"]
        label = float(r["fire_label"])

        b7 = np.load(os.path.join(B7_DIR, f"{sid}.npy"))
        b10 = np.load(os.path.join(B10_DIR, f"{sid}.npy"))

        # Normalize to [0,1]
        b7 = b7 / 65535.0
        b10 = b10 / 65535.0

        x = np.stack([b7, b10], axis=0)
        y = np.array([label], dtype=np.float32)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y)

# -------------------------------------------------
# MODEL (IMPROVED TEACHER CNN)
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
# TRAINING SETUP
# -------------------------------------------------
dataset = LandsatFireDataset()
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = FireCNN().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.BCELoss()

print("Total training samples:", len(dataset))
print("Starting training...\n")

# -------------------------------------------------
# TRAIN LOOP
# -------------------------------------------------
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        preds = model(x)
        loss = loss_fn(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * x.size(0)

        predicted = (preds > 0.5).float()
        correct += (predicted == y).sum().item()
        total += y.numel()

    avg_loss = epoch_loss / len(dataset)
    acc = correct / total * 100

    print(f"Epoch [{epoch+1}/{EPOCHS}] | "
          f"Loss: {avg_loss:.4f} | "
          f"Accuracy: {acc:.2f}%")

# -------------------------------------------------
# SAVE MODEL
# -------------------------------------------------
torch.save(model.state_dict(), "fire_cnn_heavy.pth")
print("\nTraining complete.")
print("Teacher model saved as fire_cnn_heavy.pth")
