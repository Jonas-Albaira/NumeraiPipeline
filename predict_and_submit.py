from numerapi import NumerAPI

model_uuid = ""  # store your Numerai model UUID here
napi = NumerAPI(public_id="", secret_key="")

# Upload predictions
napi.upload_predictions("submission.csv", model_id=model_uuid)
print("Submission uploaded!")
