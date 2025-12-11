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
sapi = NumerAPI(public_id="DXU2EQ3UTLV4XBZ2NFKNXP3T5UE5I7Q3", secret_key="ABPTLCXD4OQL2YJ4TRO755474E3C5D2KSQQA7WKGWSDXF2M3ZUMEA6ZKVVUXRIS2")
sapi.upload_predictions("submission.csv", model_id="32601aab-e6d4-432c-aca4-6c36e8ab3691")

print("Uploaded!")
