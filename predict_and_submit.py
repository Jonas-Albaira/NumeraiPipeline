from numerapi import NumerAPI

model_uuid = "32601aab-e6d4-432c-aca4-6c36e8ab3691"  # store your Numerai model UUID here
napi = NumerAPI(public_id="LXRJTLL43UJTW27LDO4PWUYBRSW5MQAC", secret_key="EJLIK4JF343K224QHSNFMI73WNR55RMEEU3ESFLCITY2B2J5UG4J7Q6U2ZEQ4XIR")

# Upload predictions
napi.upload_predictions("submission.csv", model_id=model_uuid)
print("Submission uploaded!")