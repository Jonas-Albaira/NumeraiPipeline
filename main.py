#!/usr/bin/env python3

import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from numerapi import NumerAPI
from sklearn.model_selection import train_test_split

DATA_DIR = "v5.1"
TRAIN_FILE = os.path.join(DATA_DIR, "train.parquet")
LIVE_FILE  = os.path.join(DATA_DIR, "live.parquet")
live = pd.read_parquet(LIVE_FILE)
print(live.columns)
print(live.head())
MODEL_FILE = "numerai_model.joblib"
SUBMISSION_FILE = "submission.csv"

os.makedirs(DATA_DIR, exist_ok=True)

# ---------- Download train (only needs to change infrequently) ----------
napi = NumerAPI()
print("Downloading train data…")
napi.download_dataset("v5.1/train.parquet", TRAIN_FILE)

# ---------- Load training data ----------
train = pd.read_parquet(TRAIN_FILE)
features = [c for c in train.columns if c.startswith("feature")]
X = train[features].astype(np.float32)
y = train["target"].astype(np.float32)

# ---------- Train/validation split ----------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- Train model ----------
model = lgb.LGBMRegressor(
    n_estimators=1200,
    learning_rate=0.01,
    num_leaves=128,
    colsample_bytree=0.2,
    subsample=0.8,
    random_state=42,
    n_jobs=-1
)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="l2",
    callbacks=[lgb.log_evaluation(0)]  # 0 means silence LightGBM output
)

joblib.dump(model, MODEL_FILE)
print("Model saved.")

# ---------- Download latest live data ----------
print("Downloading latest live data…")
napi.download_dataset("v5.1/live.parquet", LIVE_FILE)

live = pd.read_parquet(LIVE_FILE)
X_live = live[features].astype(np.float32)

# ---------- Predict & save submission ----------
preds = model.predict(X_live)
submission = pd.DataFrame({
    "id": live["t_id"],
    "prediction": preds
})
submission.to_csv(SUBMISSION_FILE, index=False)
print("Saved submission.csv")
