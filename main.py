import os
import shutil
import pandas as pd
import lightgbm as lgb
from numerapi import NumerAPI
import joblib

# -----------------------
# Numerai credentials
# -----------------------
NUMERAI_PUBLIC_ID = "TOMHYZX6NX2PPSZ2LRSF2WLKP6G4XJWF"
NUMERAI_SECRET_KEY = "OA4FBTNIF2Q4AV3QN5WZIKHAJYD4DAMYE4K3JIABL3V34Y7N2USM3MG53Q65HR4B"
NUMERAI_MODEL_UUID = "32601aab-e6d4-432c-aca4-6c36e8ab3691"

# -----------------------
# Data directory
# -----------------------


# -----------------------
# Initialize NumerAPI
# -----------------------
sapi = NumerAPI(public_id=NUMERAI_PUBLIC_ID, secret_key=NUMERAI_SECRET_KEY)

# -----------------------
# Hardcoded latest dataset version
# -----------------------
TRAIN_FILE = "v5.1/train.parquet"        # update version if Numerai releases a new one
LIVE_FILE = "v5.1/live.parquet"

# -----------------------
# Download datasets
# -----------------------
print(f"Downloading training dataset: {TRAIN_FILE} …")
sapi.download_dataset(TRAIN_FILE)

print(f"Downloading live dataset: {LIVE_FILE} …")
sapi.download_dataset(LIVE_FILE)
print("Downloads complete.")

# -----------------------
# Load training data
# -----------------------
train_path = os.path.basename(TRAIN_FILE)
train = pd.read_parquet("v5.1/train.parquet")

features = [col for col in train.columns if col.startswith("feature")]
X_train = train[features]
y_train = train["target"]

# -----------------------
# Train LightGBM model
# -----------------------
model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)
print(f"Training done on {X_train.shape[0]} rows and {X_train.shape[1]} features")

# Save model
model_path = os.path.join(DATA_DIR, "numerai_model.joblib")
joblib.dump(model, model_path)
print("Model saved to:", model_path)

# -----------------------
# Predict on live data and prepare submission
# -----------------------
live_path = os.path.join(DATA_DIR, os.path.basename(LIVE_FILE))
live = pd.read_parquet(live_path)
X_live = live[[col for col in live.columns if col.startswith("feature")]]
preds = model.predict_proba(X_live)[:, 1]

submission = live[["id"]].copy()
submission["prediction"] = preds
sub_file = os.path.join(DATA_DIR, "submission.csv")
submission.to_csv(sub_file, index=False)
print("Submission saved to:", sub_file)

# -----------------------
# Upload predictions
# -----------------------
sapi.upload_predictions(file_path=sub_file, model_id=NUMERAI_MODEL_UUID)
print("Uploaded predictions successfully!")
