import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from numerapi import NumerAPI

# -----------------------
# Paths
# -----------------------
TRAIN_FILE = "v5.1/train.parquet"
LIVE_FILE  = "v5.1/live.parquet"
MODEL_PATH = "numerai_model.joblib"
SUB_FILE   = "submission.csv"

# -----------------------
# Load training data
# -----------------------
train = pd.read_parquet(TRAIN_FILE)
features = [c for c in train.columns if c.startswith("feature")]

X = train[features].astype(np.float32)
y = train["target"].astype(np.float32)

# -----------------------
# Train/validation split
# -----------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Train size:", X_train.shape, "Val size:", X_val.shape)

# -----------------------
# LGBMRegressor Model
# -----------------------
model = lgb.LGBMRegressor(
    n_estimators=1200,
    learning_rate=0.01,
    num_leaves=128,
    colsample_bytree=0.2,
    subsample=0.8,
    reg_alpha=5,
    reg_lambda=5,
    min_child_weight=50,
    random_state=42,
    n_jobs=-1
)

model.set_params(verbose=-1)  # Silence LightGBM output
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="l2")

# Save model
joblib.dump(model, MODEL_PATH)
print("Model saved.")

# -----------------------
# Validation metrics
# -----------------------
def corr(pred, target):
    pred = pred - pred.mean()
    target = target - target.mean()
    return np.cov(pred, target)[0, 1] / (pred.std() * target.std())

def sharpe(pred):
    return pred.mean() / pred.std()

def mmc(pred, target, baseline=None):
    if baseline is None:
        baseline = np.zeros_like(pred)
    pred_res = pred - baseline
    return corr(pred_res, target)

val_preds = model.predict(X_val)
print("\n--- VALIDATION METRICS ---")
print(f"Validation correlation: {corr(val_preds, y_val):.4f}")
print(f"Validation Sharpe:      {sharpe(val_preds):.4f}")
print(f"Validation MMC:         {mmc(val_preds, y_val):.4f}")
print("\nTraining + validation complete.")

# -----------------------
# Load latest live dataset
# -----------------------
sapi = NumerAPI(
    public_id=os.getenv("NUMERAI_PUBLIC_ID"),
    secret_key=os.getenv("NUMERAI_SECRET_KEY")
)
sapi.download_dataset("v5.1/live.parquet")
live = pd.read_parquet(LIVE_FILE)
features_live = [c for c in live.columns if c.startswith("feature")]

# -----------------------
# Predict on live data
# -----------------------
live_preds = model.predict(live[features_live])
print(f"Predictions done for {len(live_preds)} rows.")

# -----------------------
# Prepare submission
# -----------------------
id_column = None
for col in live.columns:
    if "id" in col.lower():
        id_column = col
        break

if id_column is None:
    raise ValueError("No ID column found in live dataset!")

submission = live[[id_column]].copy()
submission["prediction"] = live_preds
submission.rename(columns={id_column: "id"}, inplace=True)
submission.to_csv(SUB_FILE, index=False)
print(f"Submission saved to {SUB_FILE}.")

# -----------------------
# Upload predictions
# -----------------------
model_uuid = os.getenv("NUMERAI_MODEL_UUID")
if not model_uuid:
    raise ValueError("Missing Numerai MODEL UUID. Set NUMERAI_MODEL_UUID env variable.")

sapi.upload_predictions(file_path=SUB_FILE, model_id=model_uuid)
print("Uploaded predictions successfully!")
