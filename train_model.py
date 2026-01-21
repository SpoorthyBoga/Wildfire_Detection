import torch, csv, os, numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class FireDataset(Dataset):
    def __init__(self):
        with open("landsat8_train_dataset/labels.csv") as f:
            self.rows = list(csv.DictReader(f))

    def __getitem__(self, i):
        sid = self.rows[i]["sample_id"]
        y = float(self.rows[i]["fire_label"])
        b7 = np.load(f"landsat8_train_dataset/band7_swir/{sid}.npy")
        b10 = np.load(f"landsat8_train_dataset/band10_thermal/{sid}.npy")
        x = np.stack([b7, b10])
        return torch.tensor(x, dtype=torch.float32), torch.tensor([y])

    def __len__(self): return len(self.rows)

class FireCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Linear(64,1), nn.Sigmoid()
        )
    def forward(self,x): return self.net(x)

loader = DataLoader(FireDataset(), batch_size=32, shuffle=True)
model = FireCNN().to(DEVICE)
opt = optim.Adam(model.parameters(), 1e-3)
loss_fn = nn.BCELoss()

for e in range(10):
    loss_sum = 0
    for x,y in loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        p = model(x)
        loss = loss_fn(p,y)
        opt.zero_grad(); loss.backward(); opt.step()
        loss_sum += loss.item()
    print(f"Epoch {e+1} Loss {loss_sum/len(loader):.4f}")

torch.save(model.state_dict(),"fire_cnn_heavy.pth")
print("✔ Teacher model saved")
