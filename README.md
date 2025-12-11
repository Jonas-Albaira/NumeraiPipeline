# 📈 Numerai Automated Pipeline

![Numerai Logo](https://numer.ai/images/numerai-logo.png)  

A fully automated **Numerai v5.1 pipeline** that:

- Downloads the latest live dataset  
- Trains or loads a LightGBM model  
- Generates predictions  
- Uploads submissions to Numerai automatically  
- Optional: Can be deployed on **GCP** for continuous operation  

---

## 🛠 Features

- ✅ End-to-end workflow: **training → predicting → submitting**  
- ✅ Uses **LightGBM** for regression modeling  
- ✅ Supports **live data** download  
- ✅ Automatic submission with **Numerai API keys** and **model UUID**  
- ✅ Validation metrics (correlation, Sharpe, MMC)  
- ✅ Ready for **GCP deployment**  

---

## 🎨 Design

**Pipeline stages**:

1. **Data Stage** 📥  
   - Download `train.parquet` (static)  
   - Download `live.parquet` (current round)  
   - Preprocess features  

2. **Model Stage** 🤖  
   - Train LightGBM regressor  
   - Optional early stopping on validation split  
   - Save model locally (`numerai_model.joblib`)  

3. **Prediction Stage** 🔮  
   - Generate predictions on live data  
   - Save submission as `submission.csv`  
   - Handles `id` via DataFrame index  

4. **Submission Stage** 🚀  
   - Upload submission to Numerai using `numerapi`  
   - Requires **API keys** and **model UUID**  

---

## ☁️ GCP Deployment Stages

You can automate the pipeline using **Google Cloud Platform**:

1. **Cloud Storage** 🗄  
   - Store `train.parquet`, `live.parquet`, models, and submissions  

2. **Cloud Functions** ⚡  
   - Trigger pipeline on schedule (e.g., daily or per tournament round)  

3. **Cloud Scheduler** ⏰  
   - Schedule the pipeline to run automatically  
   - Ensures you never miss a submission round  

4. **Logging & Monitoring** 📊  
   - Use **Cloud Logging** to track pipeline runs and errors  
   - Send email or Slack notifications after successful submission  

---

## 📝 Requirements

- Python 3.11+  
- Packages:  

```bash
pip install pandas numpy lightgbm scikit-learn numerapi joblib
