import os
import pandas as pd
import lightgbm as lgb
import numpy as np
import joblib
from sklearn.model_selection import train_test_split


# -----------------------
# Load training data
# -----------------------
TRAIN_FILE = "v5.1/train.parquet"
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
# Improved LGBM Model
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

model.set_params(verbose=-1)   # Silence warnings (optional)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="l2"
)

# Save model
joblib.dump(model, "numerai_model.joblib")
print("Model saved.")


# -----------------------------------
# VALIDATION METRICS (Very Important)
# -----------------------------------

def corr(pred, target):
    """Pearson correlation"""
    pred = pred - pred.mean()
    target = target - target.mean()
    return np.cov(pred, target)[0, 1] / (pred.std() * target.std())


def sharpe(pred):
    return pred.mean() / pred.std()


def mmc(pred, target, baseline=None):
    """MMC metric used by Numerai."""
    if baseline is None:
        baseline = np.zeros_like(pred)
    pred_res = pred - baseline
    return corr(pred_res, target)


print("\n--- VALIDATION METRICS ---")

val_preds = model.predict(X_val)

val_corr = corr(val_preds, y_val)
val_sharpe = sharpe(val_preds)
val_mmc = mmc(val_preds, y_val)

print(f"Validation correlation: {val_corr:.4f}")
print(f"Validation Sharpe:      {val_sharpe:.4f}")
print(f"Validation MMC:         {val_mmc:.4f}")

print("\nTraining + validation complete.")
