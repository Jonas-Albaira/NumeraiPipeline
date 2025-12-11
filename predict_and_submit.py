import os
import sys
import argparse
import logging
from pathlib import Path
import joblib
import pandas as pd
from numerapi import NumerAPI

os.environ["NUMERAI_PUBLIC_ID"] = "LXRJTLL43UJTW27LDO4PWUYBRSW5MQAC"
os.environ["NUMERAI_SECRET_KEY"] = "EJLIK4JF343K224QHSNFMI73WNR55RMEEU3ESFLCITY2B2J5UG4J7Q6U2ZEQ4XIR"
os.environ["NUMERAI_MODEL_UUID"] = "32601aab-e6d4-432c-aca4-6c36e8ab3691"
public_id = "LXRJTLL43UJTW27LDO4PWUYBRSW5MQAC"
secret_key = "EJLIK4JF343K224QHSNFMI73WNR55RMEEU3ESFLCITY2B2J5UG4J7Q6U2ZEQ4XIR"
model_uuid = "32601aab-e6d4-432c-aca4-6c36e8ab3691"  # store your Numerai model UUID here
"""
predict_and_submit.py

Robust Numerai submission script.

Features:
- Optionally downloads the latest v5.1/live.parquet using numerapi
- Loads a joblib model and predicts on live features
- Robust validation (id matching, duplicates, NaNs, shape checks)
- Saves a CSV with columns (id, prediction) and uploads via numerapi

Usage (safe defaults):
  export NUMERAI_PUBLIC_ID="..."
  export NUMERAI_SECRET_KEY="..."
  export NUMERAI_MODEL_UUID="..."     # optional, can pass --model-uuid

  python predict_and_submit.py --model-path numerai_model.joblib

Author: ChatGPT
"""


LOG = logging.getLogger("predict_and_submit")


def setup_logging():
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def find_id_column(df: pd.DataFrame) -> str:
    # Prefer exact 'id'
    if "id" in df.columns:
        return "id"
    # fallback: any column with 'id' in its name
    for c in df.columns:
        if "id" in c.lower():
            return c
    raise ValueError("No id column found in live dataset.")


def select_feature_columns(df: pd.DataFrame) -> list:
    feats = [c for c in df.columns if c.startswith("feature")]
    if not feats:
        # fallback: heuristics (exclude id, era, target, data-type columns)
        candidates = [c for c in df.columns if c not in ("id", "era", "data_type", "target")]
        # only keep numeric-like columns
        numeric = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
        if numeric:
            return numeric
        raise ValueError("No feature columns found. Expect columns starting with 'feature'.")
    return sorted(feats)


def validate_submission(live_df: pd.DataFrame, submission_df: pd.DataFrame, id_col: str) -> list:
    errs = []

    # 1) length
    if len(submission_df) != len(live_df):
        errs.append(f"Submission length {len(submission_df)} != live length {len(live_df)}")

    # normalize ids to strings for comparison
    live_ids = live_df[id_col].astype(str)
    sub_ids = submission_df["id"].astype(str)

    set_live = set(live_ids)
    set_sub = set(sub_ids)

    missing = sorted(list(set_live - set_sub))
    extra = sorted(list(set_sub - set_live))
    if missing:
        errs.append(f"{len(missing)} ids missing (examples: {missing[:10]})")
    if extra:
        errs.append(f"{len(extra)} extra ids (examples: {extra[:10]})")

    # duplicates
    dup_ids = submission_df["id"][submission_df["id"].duplicated()].unique()
    if len(dup_ids) > 0:
        errs.append(f"Found {len(dup_ids)} duplicate ids in submission (examples: {dup_ids[:5]})")

    # NaNs
    if submission_df["id"].isna().any():
        errs.append("Found NaN in id column")
    if submission_df["prediction"].isna().any():
        errs.append("Found NaN in prediction column")

    return errs


def main(argv=None):
    setup_logging()

    p = argparse.ArgumentParser(description="Predict on Numerai live and upload submission")
    p.add_argument("--model-path", default="numerai_model.joblib", help="Path to joblib model")
    p.add_argument("--live-file", default="v5.1/live.parquet", help="Local live parquet file")
    p.add_argument("--submission", default="submission.csv", help="Output submission CSV path")
    p.add_argument("--download-live", action="store_true", help="Download latest live.parquet from Numerai before predicting")
    p.add_argument("--model-uuid", default=None, help="Numerai model UUID (overrides env NUMERAI_MODEL_UUID)")
    p.add_argument("--no-upload", action="store_true", help="Don't actually upload to Numerai (useful for testing)")
    p.add_argument("--force-reshape", action="store_true", help="Force ravel predictions to 1D")
    args = p.parse_args(argv)

    # Check credentials exist
    public_id = os.environ.get("NUMERAI_PUBLIC_ID")
    secret_key = os.environ.get("NUMERAI_SECRET_KEY")
    env_model_uuid = os.environ.get("NUMERAI_MODEL_UUID")

    if not public_id or not secret_key:
        LOG.error("NUMERAI_PUBLIC_ID and NUMERAI_SECRET_KEY must be set as environment variables")
        sys.exit(2)

    model_uuid = args.model_uuid or env_model_uuid
    if not model_uuid and not args.no_upload:
        LOG.error("NUMERAI_MODEL_UUID must be set as env var or passed with --model-uuid to upload")
        sys.exit(2)

    # create NumerAPI client
    napi = NumerAPI(public_id=public_id, secret_key=secret_key)

    local_live_path = Path(args.live_file)

    if args.download_live:
        LOG.info("Downloading latest v5.1/live.parquet from Numerai...")
        try:
            napi.download_dataset("v5.1/live.parquet", str(local_live_path))
            LOG.info("Downloaded live.parquet")
        except Exception as e:
            LOG.warning("Failed to download live.parquet from Numerai: %s", e)
            if not local_live_path.exists():
                LOG.error("No local live.parquet available to continue. Exiting.")
                sys.exit(3)

    if not local_live_path.exists():
        LOG.error("Live file not found: %s", local_live_path)
        sys.exit(3)

    # Load model
    LOG.info("Loading model from %s", args.model_path)
    model = joblib.load(args.model_path)
    LOG.info("Model loaded")

    # Load live
    LOG.info("Loading live dataset from %s", local_live_path)
    live = pd.read_parquet(local_live_path)
    LOG.info("Live dataset: %d rows, %d columns", len(live), len(live.columns))

    id_col = "object_id"
    LOG.info("Using id column: %s", id_col)

    features = select_feature_columns(live)
    LOG.info("Using %d feature columns", len(features))

    # If model exposes expected input shape, compare
    if hasattr(model, "n_features_in_"):
        expected = int(model.n_features_in_)
        if expected != len(features):
            LOG.warning("Model expects %d features but %d features found in live. This may be an issue.", expected, len(features))

    # Predict
    X = live[features]
    LOG.info("Running predictions on shape %s", X.shape)
    preds = model.predict(X)

    # Force to 1D if needed
    if args.force_reshape:
        preds = getattr(preds, "ravel", lambda: preds)()
    # If preds is 2D with single col, flatten it
    if hasattr(preds, "shape") and len(getattr(preds, "shape", ())) == 2 and preds.shape[1] == 1:
        preds = preds.ravel()

    LOG.info("Predictions produced with shape %s", getattr(preds, "shape", None))

    # Build submission
    submission = pd.DataFrame({"id": live[id_col].values, "prediction": preds})

    # Validate
    errors = validate_submission(live, submission, id_col)
    if errors:
        LOG.error("Submission validation failed with %d error(s):", len(errors))
        for e in errors:
            LOG.error(" - %s", e)
        LOG.error("Aborting upload. Fix issues and retry.")
        sys.exit(4)

    # Save
    submission.to_csv(args.submission, index=False)
    LOG.info("Saved submission to %s (%d rows)", args.submission, len(submission))

    # Upload
    if args.no_upload:
        LOG.info("Skipping upload (--no-upload). Done.")
        return

    LOG.info("Uploading submission to Numerai for model %s...", model_uuid)
    try:
        napi.upload_predictions(file_path=args.submission, model_id=model_uuid)
        LOG.info("Upload successful.")
    except Exception as e:
        LOG.exception("Upload failed: %s", e)
        sys.exit(5)


if __name__ == "__main__":
    main()
