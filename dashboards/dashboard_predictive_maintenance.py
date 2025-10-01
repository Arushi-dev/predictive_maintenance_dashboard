# dashboard_predictive_maintenance.py

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report, accuracy_score
import os

# -------------------------------
# 📁 Setup Paths & Load Data
# -------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

@st.cache_data
def load_data_and_models():
    df = pd.read_csv(DATA_DIR / "processed_sensor_data.csv")
    rul_model = joblib.load(MODELS_DIR / "xgb_rul_model.pkl")
    clf_model = joblib.load(MODELS_DIR / "xgb_binary_classifier.pkl")
    return df, rul_model, clf_model

df, rul_model, clf_model = load_data_and_models()

# ✅ Add label column safely
if "label" not in df.columns:
    df["label"] = (df["RUL"] <= 20).astype(int)

# -------------------------------
# 🎛️ Sidebar Navigation
# -------------------------------
st.sidebar.title("Failure Risk Dashboard")
section = st.sidebar.radio(
    "Select Analysis:",
    [
        "Engine Summary",
        "RUL Prediction",
        "Failure Risk",
        "Model Explainability",
        "Compare Classifiers (XGBoost vs RF)",
        "Unit-wise ROC Viewer",
        "Alerts & High-Risk Engines",
        "Download Center"

    ]
)

# -------------------------------
# 📊 Section 1: Engine Summary
# -------------------------------
if section == "Engine Summary":
    st.subheader("Engine Summary")
    st.dataframe(df.head())

# -------------------------------
# 📉 Section 2: RUL Prediction
# -------------------------------
elif section == "RUL Prediction":
    st.subheader("Remaining Useful Life (RUL) Prediction")
    unit_ids = df['unit'].unique()
    selected_unit = st.selectbox("Select Engine Unit:", unit_ids)
    unit_df = df[df['unit'] == selected_unit].copy()
    features = [col for col in unit_df.columns if col not in ['unit', 'cycle', 'RUL', 'label']]
    unit_df['Predicted RUL'] = rul_model.predict(unit_df[features])
    st.line_chart(unit_df[['cycle', 'RUL', 'Predicted RUL']].set_index('cycle'))
    st.success(f"Latest Predicted RUL for unit {selected_unit}: {unit_df['Predicted RUL'].iloc[-1]:.1f} cycles")
    unit_df.to_csv(OUTPUTS_DIR / "rul_predictions.csv", index=False)

# -------------------------------
# ⚠️ Section 3: Failure Risk
# -------------------------------
elif section == "Failure Risk":
    st.subheader("Binary Failure Risk Classifier")
    unit_ids = df['unit'].unique()
    selected_unit = st.selectbox("Select Engine Unit:", unit_ids, key="binary_unit")
    unit_df = df[df['unit'] == selected_unit].copy()
    features = [col for col in df.columns if col not in ['unit', 'cycle', 'RUL', 'label']]
    probs = clf_model.predict_proba(unit_df[features])[:, 1]
    preds = clf_model.predict(unit_df[features])
    unit_df['Failure Risk (%)'] = probs * 100
    unit_df['Will Fail'] = preds
    st.metric(label="Latest Failure Risk", value=f"{unit_df['Failure Risk (%)'].iloc[-1]:.1f}%")
    if unit_df['Will Fail'].iloc[-1]:
        st.error("Prediction: Likely to fail within 20 cycles.")
    else:
        st.success("Prediction: Operating normally.")
    st.line_chart(unit_df[['cycle', 'Failure Risk (%)']].set_index('cycle'))
    unit_df.to_csv(OUTPUTS_DIR / "binary_predictions.csv", index=False)

# -------------------------------
# 🧠 Section 4: Explainability
# -------------------------------
elif section == "Model Explainability":
    st.subheader("Model Explainability")
    st.image(str(OUTPUTS_DIR / "shap_summary_beeswarm.png"), caption="SHAP Beeswarm", use_column_width=True)
    st.image(str(OUTPUTS_DIR / "shap_summary_bar.png"), caption="SHAP Bar", use_column_width=True)
    st.image(str(OUTPUTS_DIR / "shap_waterfall_row10.png"), caption="SHAP Waterfall", use_column_width=True)
    st.image(str(OUTPUTS_DIR / "roc_auc_curve.png"), caption="ROC-AUC Curve", use_column_width=True)
    st.image(str(OUTPUTS_DIR / "feature_importance.png"), caption="Feature Importance", use_column_width=True)

elif section == "Compare Classifiers (XGBoost vs RF)":
    st.subheader("Binary Classifier Comparison: XGBoost vs Random Forest")

    # ✅ Only use features used during training (exclude label!)
    features = [col for col in df.columns if col not in ['unit', 'cycle', 'RUL', 'label']]
    X = df[features].copy()
    y = df['label']  # we still use label as target

    # ✅ Load Random Forest model
    rf_model = joblib.load(MODELS_DIR / "rf_binary_classifier.pkl")

    # ✅ Predict with both models
    xgb_probs = clf_model.predict_proba(X)[:, 1]
    xgb_preds = clf_model.predict(X)

    rf_probs = rf_model.predict_proba(X)[:, 1]
    rf_preds = rf_model.predict(X)

    # ✅ ROC Curve
    fpr_xgb, tpr_xgb, _ = roc_curve(y, xgb_probs)
    fpr_rf, tpr_rf, _ = roc_curve(y, rf_probs)

    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC={roc_auc_score(y, xgb_probs):.2f})")
    ax_roc.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={roc_auc_score(y, rf_probs):.2f})", linestyle="--")
    ax_roc.plot([0, 1], [0, 1], "k--")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curves")
    ax_roc.legend()
    st.pyplot(fig_roc)

    # ✅ Accuracy & AUC Table
    st.markdown("### Accuracy & ROC-AUC Comparison")
    compare_df = pd.DataFrame({
        "Model": ["XGBoost", "Random Forest"],
        "Accuracy": [accuracy_score(y, xgb_preds), accuracy_score(y, rf_preds)],
        "ROC-AUC": [roc_auc_score(y, xgb_probs), roc_auc_score(y, rf_probs)]
    })
    st.dataframe(compare_df)

    # ✅ Confusion Matrix
    st.markdown("### Confusion Matrix")
    model_choice = st.selectbox("Select model", ["XGBoost", "Random Forest"])
    cm = confusion_matrix(y, xgb_preds if model_choice == "XGBoost" else rf_preds)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
    ax_cm.set_title(f"Confusion Matrix: {model_choice}")
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

# -------------------------------
# 📁 Section 6: Download Center
# -------------------------------
elif section == "Download Center":
    st.subheader("Download Center")
    st.markdown("Export key prediction results and visuals.")
    for file_name in ["rul_predictions.csv", "binary_predictions.csv"]:
        file_path = OUTPUTS_DIR / file_name
        if file_path.exists():
            with open(file_path, "rb") as f:
                st.download_button(f"Download {file_name}", f, file_name=file_name)

    st.markdown("### Visuals (PNG)")
    for img in ["shap_summary_beeswarm.png", "shap_summary_bar.png",
                "shap_waterfall_row10.png", "roc_auc_curve.png", "feature_importance.png"]:
        img_path = OUTPUTS_DIR / img
        if img_path.exists():
            with open(img_path, "rb") as f:
                st.download_button(f"Download {img}", f, file_name=img)

# --------------------------------------------
# 📄 Section 6: Unit-wise ROC Viewer
# --------------------------------------------
elif section == "Unit-wise ROC Viewer":
    st.subheader("Unit-wise ROC and Risk Viewer")

    unit_ids = df["unit"].unique()
    selected_unit = st.selectbox("Select Engine Unit:", unit_ids, key="roc_unit")

    unit_df = df[df["unit"] == selected_unit].copy()
    features = [col for col in df.columns if col not in ['unit', 'cycle', 'RUL', 'label']]

    y_true = unit_df["label"]
    y_probs = clf_model.predict_proba(unit_df[features])[:, 1]

    # Optional threshold slider
    threshold = st.slider("Select classification threshold", 0.0, 1.0, 0.5, 0.01)
    y_pred = (y_probs >= threshold).astype(int)

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label="ROC Curve")
    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(f"ROC Curve for Unit {selected_unit}")
    ax_roc.legend()
    st.pyplot(fig_roc)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", ax=ax_cm)
    ax_cm.set_title(f"Confusion Matrix @ Threshold = {threshold}")
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

    # Alert suggestion
    latest_rul = unit_df["RUL"].iloc[-1]
    latest_risk = y_probs[-1] * 100
    st.metric("Latest Failure Risk (%)", f"{latest_risk:.1f}%")
    st.metric("Latest Predicted RUL", f"{latest_rul:.1f} cycles")

    if latest_risk > 80 and latest_rul < 20:
        st.error("🚨 Alert: High risk of failure soon!")
    elif latest_risk > 60:
        st.warning("⚠️ Warning: Medium risk level.")
    else:
        st.success("✅ Engine currently healthy.")

# -------------------------------
# 🚨 Section: Alerts & High-Risk Engines
# -------------------------------
elif section == "Alerts & High-Risk Engines":
    st.subheader("Alerts: Engines with High or Medium Failure Risk")

    # ✅ Define features (exclude unit, cycle, RUL, label)
    features = [col for col in df.columns if col not in ['unit', 'cycle', 'RUL', 'label']]

    # Prepare latest prediction per unit
    alert_units = []
    for uid in df["unit"].unique():
        unit_df = df[df["unit"] == uid]
        latest = unit_df.iloc[-1]
        risk = clf_model.predict_proba([latest[features]])[0][1] * 100
        rul = latest["RUL"]

        if risk > 80 and rul < 20:
            risk_level = "HIGH"
        elif risk > 60:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        alert_units.append({
            "Unit": uid,
            "Latest RUL": rul,
            "Latest Risk (%)": round(risk, 1),
            "Risk Level": risk_level
        })

    alert_df = pd.DataFrame(alert_units)
    high_risk_count = (alert_df["Risk Level"] == "HIGH").sum()
    medium_risk_count = (alert_df["Risk Level"] == "MEDIUM").sum()
    low_risk_count = (alert_df["Risk Level"] == "LOW").sum()

    # KPI Summary
    st.markdown("### ⚙️ Summary KPIs")
    col1, col2, col3 = st.columns(3)
    col1.metric("🔴 High Risk", high_risk_count)
    col2.metric("🟠 Medium Risk", medium_risk_count)
    col3.metric("🟢 Low Risk", low_risk_count)

    # Filtered display
    filtered_df = alert_df[alert_df["Risk Level"].isin(["HIGH", "MEDIUM"])]
    st.markdown("### 📋 High & Medium Risk Engines")
    st.dataframe(filtered_df)

    # Download button
    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Alert Table", csv_bytes, file_name="alert_risks.csv")


