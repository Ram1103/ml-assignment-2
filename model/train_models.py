"""
train_models.py
----------------
Trains 5 classification models on the Breast Cancer Wisconsin (Diagnostic) dataset,
evaluates them on a held-out test split, and saves:
  - Each trained model (model/*.joblib)
  - The fitted StandardScaler (model/scaler.joblib)
  - test_data.csv  -> held-out test set (features + true label), used by the Streamlit app
  - metrics_summary.csv -> comparison table used in README.md

Dataset: sklearn's built-in Breast Cancer Wisconsin dataset
  - 569 instances (> 500 required)
  - 30 numeric features (> 12 required)
  - Binary classification: malignant (0) vs benign (1)

Run:  python train_models.py
"""

import json
import numpy as np
import pandas as pd
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

RANDOM_STATE = 42


def load_data():
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target  # 0 = malignant, 1 = benign
    return X, y, list(X.columns)


def main():
    X, y, feature_names = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Scale features (helps LR / KNN especially); tree-based models are unaffected
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, "model/scaler.joblib")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "kNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE
        ),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        metrics = {
            "ML Model Name": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "AUC": round(roc_auc_score(y_test, y_proba), 4),
            "Precision": round(precision_score(y_test, y_pred), 4),
            "Recall": round(recall_score(y_test, y_pred), 4),
            "F1": round(f1_score(y_test, y_pred), 4),
            "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
        }
        results.append(metrics)

        # Save each model as its own file, e.g. model/logistic_regression.joblib
        fname = "model/" + name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".joblib"
        joblib.dump(model, fname)
        print(f"Saved {fname} -> {metrics}")

    # Save comparison table for README
    results_df = pd.DataFrame(results)
    results_df.to_csv("metrics_summary.csv", index=False)
    print("\nComparison table:\n", results_df.to_string(index=False))

    # Save held-out test data (features + true label) for the Streamlit app
    test_df = X_test.copy()
    test_df["target"] = y_test.values
    test_df.to_csv("test_data.csv", index=False)
    print(f"\nSaved test_data.csv with shape {test_df.shape}")

    # Save feature names + target mapping for the app to use
    meta = {
        "feature_names": feature_names,
        "target_names": {"0": "malignant", "1": "benign"},
    }
    with open("model/meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
