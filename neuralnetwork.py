#!/usr/bin/env python3

import os
import joblib
import numpy as np
import pandas as pd
from numerapi import NumerAPI
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = "v5.1"
TRAIN_FILE = os.path.join(DATA_DIR, "train.parquet")
LIVE_FILE  = os.path.join(DATA_DIR, "live.parquet")
live = pd.read_parquet(LIVE_FILE)
print(live.columns)
print(live.head())

MODEL_FILE = "numerai_model.pt"
SUBMISSION_FILE = "submission.csv"

os.makedirs(DATA_DIR, exist_ok=True)

# ---------- Download train (only needs to change infrequently) ----------
napi = NumerAPI()
print("Downloading train data…")
napi.download_dataset("v5.1/train.parquet", TRAIN_FILE)

# ---------- Load training data ----------
train = pd.read_parquet(TRAIN_FILE)
features = [c for c in train.columns if c.startswith("feature")]
X = train[features].astype(np.float32).values
y = train["target"].astype(np.float32).values

# ---------- Train/validation split ----------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- Prepare PyTorch datasets ----------
train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).unsqueeze(1))
val_dataset   = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val).unsqueeze(1))

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=1024)

# ---------- Define neural network ----------
class NumeraiNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NumeraiNN(len(features)).to(device)

# ---------- Training setup ----------
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 50

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
    train_loss /= len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            val_loss += loss.item() * xb.size(0)
    val_loss /= len(val_loader.dataset)
    print(f"Epoch {epoch+1}/{epochs} — Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

# ---------- Save model ----------
torch.save(model.state_dict(), MODEL_FILE)
print("Model saved.")

# ---------- Download latest live data ----------
print("Downloading latest live data…")
napi.download_dataset("v5.1/live.parquet", LIVE_FILE)
live = pd.read_parquet(LIVE_FILE)
X_live = torch.from_numpy(live[features].astype(np.float32).values).to(device)

# ---------- Predict ----------
model.eval()
with torch.no_grad():
    preds = model(X_live).cpu().numpy().flatten()

submission = pd.DataFrame({
    "id": live.index,
    "prediction": preds
})
submission.to_csv(SUBMISSION_FILE, index=False)
print(f"Saved {SUBMISSION_FILE}")
