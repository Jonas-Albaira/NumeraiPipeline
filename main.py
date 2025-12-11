#!/usr/bin/env python3
"""
Numerai end-to-end pipeline with live parquet download:
- download latest v5.1 train & tournament datasets
- train & validate LGBM model
- save model
- generate submission.csv
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from numerapi import NumerAPI

# ---------- Config ----------
DATA_DIR = "v5.1"
TRAIN_FILE = os.path.join(DATA_DIR, "train.parquet")
TOURNAMENT_FILE = os.path.join(DATA_DIR, "tournament.parquet")
MODEL_FILE = "numerai_model.joblib"
SUBMISSION_FILE = "submission.csv"
RANDOM_STATE = 42

# create data directory if missing
os.makedirs(DATA_DIR, exist_ok=True)

# ---------- Metrics ----------
def corr(pred, target):
    pred = pred - pred.mean()
    target = target - target.mean()
    cov = np.cov(pred, target)[0, 1]
    denom = pred.std() * target.std()
    return cov / denom if denom != 0 else 0.0

def sharpe(pred):
    std = pred.std()
    return pred.mean() / std if std != 0 else 0.0

def mmc(pred, target, baseline=None):
    if baseline is None:
        baseline = np.zeros_like(pred)
    pred_res = pred - baseline
    return corr(pred_res, target)

# ---------- Download live parquet ----------
print("Downloading latest Numerai datasets...")
napi = NumerAPI()

napi.download_dataset("v5.1/train.parquet", dest_path=TRAIN_FILE)
napi.download_dataset("v5.1/tournament.parquet", dest_path=TOURNAMENT_FILE)

# ---------- Load training data ----------
print("Loading training data...")
train = pd.read_parquet(TRAIN_FILE)
features = [c for c in train.columns if c.startswith("feature")]
y = train["target"].astype(np.float32)
X = train[features].astype(np.float32)

# ---------- Train/validation split ----------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
print("Train size:", X_train.shape, "Val size:", X_val.shape)

# ---------- Model ----------
model = lgb.LGBMRegressor(
    n_estimators=5000,
    learning_rate=0.01,
    num_leaves=128,
    colsample_bytree=0.2,
    subsample=0.8,
    reg_alpha=5,
    reg_lambda=5,
    min_child_weight=50,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
model.set_params(verbose=-1)

# ---------- Fit with early stopping ----------
print("Training model...")
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="l2",
    early_stopping_rounds=200,
    callbacks=[lgb.reset_parameter(learning_rate=lambda iter: 0.01)]
)

# ---------- Validation metrics ----------
print("\n--- VALIDATION METRICS ---")
val_preds = model.predict(X_val)
print(f"Correlation: {corr(val_preds, y_val):.6f}")
print(f"Sharpe:      {sharpe(val_preds):.6f}")
print(f"MMC:         {mmc(val_preds, y_val):.6f}")

# ---------- Save model ----------
joblib.dump(model, MODEL_FILE)
print(f"\nSaved model -> {MODEL_FILE}")

# ---------- Tournament predictions ----------
print("\nLoading tournament data...")
tournament = pd.read_parquet(TOURNAMENT_FILE)
X_tournament = tournament[features].astype(np.float32)
tournament_preds = model.predict(X_tournament)

submission = pd.DataFrame({
    "id": tournament["id"],
    "prediction": tournament_preds
})
submission.to_csv(SUBMISSION_FILE, index=False)
print(f"Submission saved -> {SUBMISSION_FILE}")
print("All done!")
