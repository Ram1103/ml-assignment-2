"""
app.py - Streamlit demo app for the Breast Cancer classification models.

Features:
  a. Dataset upload option (CSV) - upload test data with a 'target' column
  b. Model selection dropdown
  c. Display of evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
  d. Confusion matrix + classification report
"""

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

st.set_page_config(page_title="ML Assignment 2 - Classification Demo", layout="wide")

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "kNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Random Forest (Ensemble)": "model/random_forest_ensemble.joblib",
}


@st.cache_resource
def load_scaler():
    return joblib.load("model/scaler.joblib")


@st.cache_resource
def load_model(name):
    return joblib.load(MODEL_FILES[name])


@st.cache_data
def load_meta():
    with open("model/meta.json") as f:
        return json.load(f)


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    st.title("🔬 Breast Cancer Classification - Model Demo")
    st.markdown(
        "This app demonstrates **5 classification models** trained on the "
        "Breast Cancer Wisconsin (Diagnostic) dataset (30 features, 569 instances). "
        "Upload the provided `test_data.csv`, pick a model, and view its performance."
    )

    meta = load_meta()
    feature_names = meta["feature_names"]

    # (a) Dataset upload
    st.sidebar.header("1. Upload Test Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload test_data.csv (must include a 'target' column)", type=["csv"]
    )

    # (b) Model selection dropdown
    st.sidebar.header("2. Select Model")
    model_choice = st.sidebar.selectbox("Choose a classification model", list(MODEL_FILES.keys()))

    if uploaded_file is None:
        st.info("👈 Upload the `test_data.csv` file from the repo to see results.")
        st.stop()

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    if "target" not in df.columns:
        st.error("Uploaded CSV must contain a 'target' column (0 = malignant, 1 = benign).")
        st.stop()

    missing_cols = [c for c in feature_names if c not in df.columns]
    if missing_cols:
        st.error(f"Uploaded CSV is missing required feature columns: {missing_cols}")
        st.stop()

    X = df[feature_names]
    y_true = df["target"]

    scaler = load_scaler()
    X_scaled = scaler.transform(X)

    model = load_model(model_choice)
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]

    st.subheader(f"Results for: {model_choice}")

    # (c) Display evaluation metrics
    metrics = compute_metrics(y_true, y_pred, y_proba)
    cols = st.columns(6)
    for col, (metric_name, value) in zip(cols, metrics.items()):
        col.metric(metric_name, f"{value:.4f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    # (d) Confusion matrix
    with col1:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Malignant (0)", "Benign (1)"],
            yticklabels=["Malignant (0)", "Benign (1)"],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    # (d) Classification report
    with col2:
        st.markdown("**Classification Report**")
        report = classification_report(y_true, y_pred, target_names=["Malignant", "Benign"], output_dict=True)
        report_df = pd.DataFrame(report).transpose().round(3)
        st.dataframe(report_df, use_container_width=True)

    st.markdown("---")
    with st.expander("Preview uploaded data"):
        st.dataframe(df.head(20), use_container_width=True)

    st.markdown("---")
    st.markdown("### Compare All Models")
    if st.checkbox("Run all 5 models on this uploaded data and compare"):
        all_results = []
        for name in MODEL_FILES:
            m = load_model(name)
            pred = m.predict(X_scaled)
            proba = m.predict_proba(X_scaled)[:, 1]
            res = compute_metrics(y_true, pred, proba)
            res["Model"] = name
            all_results.append(res)
        comp_df = pd.DataFrame(all_results).set_index("Model").round(4)
        st.dataframe(comp_df, use_container_width=True)
        st.bar_chart(comp_df[["Accuracy", "AUC", "F1"]])


if __name__ == "__main__":
    main()
