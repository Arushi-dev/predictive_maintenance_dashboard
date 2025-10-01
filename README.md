# 🔧 Predictive Maintenance Dashboard using Machine Learning

This project predicts Remaining Useful Life (RUL) and engine failure risk using sensor data from turbofan engines. It features a complete ML pipeline and an interactive Streamlit dashboard for diagnostics and anomaly detection.

---

## 📌 Project Overview

The dashboard supports:
- **📉 RUL Prediction** – XGBoost regression per unit
- **⚠️ Failure Risk Classification** – Binary classification with XGBoost and Random Forest
- **🧠 Model Explainability** – SHAP values, ROC curve, feature importance
- **📉 Unit-wise ROC Viewer** – Dynamic ROC, confusion matrix and thresholds per engine
- **🚨 Alerts** – Highlights high/medium risk engines based on failure probability and RUL
- **📥 Download Center** – Export all key results (CSVs, images)

---

## 💾 Dataset

- Source: [NASA C-MAPSS](https://data.nasa.gov/dataset/C-MAPSS-Dataset/sdww-tz42)
- Simulated turbofan engine degradation
- Preprocessed with RUL labeling:  
  `label = 1 if RUL ≤ 20 else 0`

---

## 🧠 Modeling Summary

- **RUL Regression**: XGBoostRegressor
- **Binary Classifier**: XGBoostClassifier & RandomForestClassifier
- **Features**: Sensor values + rolling means & std devs

---

## 📊 Dashboard Navigation

```bash
streamlit run dashboards/dashboard_predictive_maintenance.py
Sections:

Engine Summary

RUL Prediction

Failure Risk

Model Explainability

Compare Classifiers

Unit-wise ROC Viewer

Alerts & High-Risk Engines

Download Center

📁 Folder Structure
css
Copy code
predictive_maintenance/
├── dashboards/
│   └── dashboard_predictive_maintenance.py
├── data/
│   └── processed_sensor_data.csv
├── models/
│   ├── xgb_rul_model.pkl
│   ├── xgb_binary_classifier.pkl
│   └── rf_binary_classifier.pkl
├── outputs/
│   ├── rul_predictions.csv
│   ├── binary_predictions.csv
│   └── [SHAP/ROC/CM images]
├── notebooks/
│   └── [EDA, preprocessing, model training]
├── utils/
│   └── [feature_engineering.py, preprocessing.py]
├── requirements.txt
└── README.md
✅ How to Run
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Launch the dashboard:

bash
Copy code
streamlit run dashboards/dashboard_predictive_maintenance.py
📥 Outputs
CSVs:
rul_predictions.csv
binary_predictions.csv

Visuals (PNG):
SHAP, ROC curves, Confusion Matrices

🧪 Skills Demonstrated
Time-series feature engineering

Supervised ML (XGBoost, RF)

Model evaluation & explainability

Anomaly alert logic

Streamlit dashboarding

GitHub portfolio readiness

👤 Author
Arushi Sharma
M.Eng. Machine Intelligence & Data Science (MIND)
Arcada UAS, Finland | 2025
