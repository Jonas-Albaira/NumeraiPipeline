import os
import pandas as pd
import joblib
from numerapi import NumerAPI

# -----------------------
# Paths
# -----------------------
MODEL_PATH = "numerai_model.joblib"
LIVE_FILE  = "v5.1/live.parquet"
SUB_FILE   = "submission.csv"

# -----------------------
# Load trained model
# -----------------------
model = joblib.load(MODEL_PATH)
print("Model loaded.")

# -----------------------
# Load live dataset
# -----------------------
live = pd.read_parquet(LIVE_FILE)
features = [c for c in live.columns if c.startswith("feature")]

# -----------------------
# Predict
# -----------------------
preds = model.predict(live[features])
print(f"Predictions done for {len(preds)} rows.")

# -----------------------
# Prepare submission
# -----------------------
# Dynamically detect ID column
id_column = None
for col in live.columns:
    if "id" in col.lower():
        id_column = col
        break

if id_column is None:
    raise ValueError("No ID column found in live dataset!")

submission = live[[id_column]].copy()
submission["prediction"] = preds

# Numerai requires column 'id'
submission.rename(columns={id_column: "id"}, inplace=True)
submission.to_csv(SUB_FILE, index=False)
print(f"Submission saved to {SUB_FILE}.")
# -----------------------
# Upload predictions
# -----------------------
# WARNING: never hardcode keys in production. Use environment variables for security:
# export NUMERAI_PUBLIC_ID="your_id"
# export NUMERAI_SECRET_KEY="your_secret"
public_id = os.getenv("LXRJTLL43UJTW27LDO4PWUYBRSW5MQAC")
secret_key = os.getenv("EJLIK4JF343K224QHSNFMI73WNR55RMEEU3ESFLCITY2B2J5UG4J7Q6U2ZEQ4XIR")
model_uuid = os.getenv("32601aab-e6d4-432c-aca4-6c36e8ab3691")  # store your Numerai model UUID here

if public_id is None or secret_key is None or model_uuid is None:
    raise ValueError("Missing Numerai API credentials. Set environment variables.")

sapi = NumerAPI(public_id=public_id, secret_key=secret_key)
sapi.upload_predictions(file_path=SUB_FILE, model_id=model_uuid)
print("Uploaded predictions successfully!")
