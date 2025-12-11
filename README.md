📈 Numerai Automated Pipeline

A fully automated Numerai v5.1 pipeline that:

Downloads the latest live dataset

Trains or loads a LightGBM model

Generates predictions

Uploads submissions to Numerai automatically

Optional: Can be deployed on GCP for continuous operation

🛠 Features

✅ End-to-end workflow from training → predicting → submitting

✅ Uses LightGBM for regression modeling

✅ Supports live data download

✅ Automatic submission with Numerai API keys and model UUID

✅ Validation metrics (correlation, Sharpe, MMC)

✅ Ready for GCP deployment

🎨 Design

Pipeline stages:

Data Stage 📥

Download train.parquet (static)

Download live.parquet (current round)

Preprocess features

Model Stage 🤖

Train LightGBM regressor

Optional early stopping on validation split

Save model locally (numerai_model.joblib)

Prediction Stage 🔮

Generate predictions on live data

Save submission as submission.csv

Handles id via DataFrame index

Submission Stage 🚀

Upload submission to Numerai using numerapi

Requires API keys and model UUID

☁️ GCP Deployment Stages

You can automate the pipeline using Google Cloud Platform:

Cloud Storage 🗄

Store train.parquet, live.parquet, models, and submissions

Cloud Functions ⚡

Trigger pipeline on schedule (e.g., daily or per tournament round)

Cloud Scheduler ⏰

Schedule the pipeline to run automatically

Ensures you never miss a submission round

Logging & Monitoring 📊

Use Cloud Logging to track pipeline runs and errors

Send email or Slack notifications after successful submission

📝 Requirements

Python 3.11+

Packages:

pandas
numpy
lightgbm
scikit-learn
numerapi
joblib


Numerai API keys (public & secret)

Numerai model UUID

⚡ Usage
# Clone repo
git clone https://github.com/yourusername/numerai-pipeline.git
cd numerai-pipeline

# Install dependencies
pip install -r requirements.txt

# Run pipeline
python main.py

📂 Project Structure
numerai-pipeline/
│
├─ main.py                # Main pipeline script
├─ requirements.txt       # Python dependencies
├─ v5.1/                  # Data directory (train + live parquet)
├─ numerai_model.joblib    # Saved LightGBM model
└─ submission.csv         # Generated Numerai submission

🧪 Example Output
Train size: (100000, 3000)  Val size: (25000, 3000)
Training model...
Validation correlation: 0.054321
Validation Sharpe:      0.032145
Validation MMC:         0.048765
Saved model -> numerai_model.joblib
Downloading latest live data...
Saved submission.csv
Submission uploaded!

💡 Tips

Retrain your model weekly or monthly to incorporate latest patterns

Always use the latest live.parquet for submission

Monitor validation metrics to avoid overfitting

🎯 Roadmap

 End-to-end pipeline with LightGBM

 Automatic submission using Numerai API

 Validation metrics logging

 GCP deployment with Cloud Functions + Scheduler

 Slack/Email notifications for submission status

🔐 Security Notes

Never commit your Numerai API keys to GitHub

Use environment variables or GCP Secret Manager for credentials

📌 References

Numerai Docs

LightGBM Docs

NumerAPI Python Package
