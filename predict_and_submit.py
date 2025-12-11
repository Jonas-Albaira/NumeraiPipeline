import pandas as pd
import joblib
from numerapi import NumerAPI

MODEL_PATH = "numerai_model.joblib"
LIVE_FILE  = "v5.1/live.parquet"

model = joblib.load(MODEL_PATH)

live = pd.read_parquet(LIVE_FILE)
features = [c for c in live.columns if c.startswith("feature")]

preds = model.predict(live[features])

submission = live[["row_id"]].copy()
submission["prediction"] = preds
submission.rename(columns={"row_id": "id"}, inplace=True)

submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")

# ---- Upload ----
sapi = NumerAPI(public_id="YOUR_ID", secret_key="YOUR_SECRET")
sapi.upload_predictions("submission.csv", model_id="YOUR_MODEL_UUID")

print("Uploaded!")
