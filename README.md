# ML Assignment 2 — Breast Cancer Classification

## a. Problem Statement

The goal of this assignment is to build, evaluate, and deploy multiple binary
classification models that predict whether a breast tumor is **malignant** or
**benign** based on measurements computed from digitized images of a fine needle
aspirate (FNA) of a breast mass. Five classification algorithms are trained on
the same dataset, compared using six evaluation metrics, and demonstrated through
an interactive Streamlit web application.

## b. Dataset Description

- **Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset
- **Source:** UCI Machine Learning Repository (also bundled with `scikit-learn`
  via `sklearn.datasets.load_breast_cancer`, originally from
  https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- **Instances:** 569 (exceeds the minimum requirement of 500)
- **Features:** 30 numeric features (exceeds the minimum requirement of 12) —
  mean, standard error, and "worst" values of 10 real-valued measurements
  (radius, texture, perimeter, area, smoothness, compactness, concavity,
  concave points, symmetry, fractal dimension) computed from cell nuclei.
- **Target variable:** Binary — `0 = malignant`, `1 = benign`
- **Class balance:** 212 malignant, 357 benign
- **Train/test split:** 80% train (455 rows) / 20% test (114 rows), stratified
  by class, `random_state=42`. Features were standardized using
  `StandardScaler` (fit on the training split only).

## c. GitHub Repository Link

**https://github.com/Ram1103/ml-assignment-2**

## d. Models Used

All 5 models were trained on the same 80/20 train-test split of the dataset
above, and evaluated on the held-out 114-row test set.

### Comparison Table

| ML Model Name             | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|----------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression        | 0.9825   | 0.9954 | 0.9861    | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree               | 0.9123   | 0.9157 | 0.9559    | 0.9028 | 0.9286 | 0.8174 |
| kNN                         | 0.9737   | 0.9884 | 0.9600    | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes                 | 0.9298   | 0.9868 | 0.9444    | 0.9444 | 0.9444 | 0.8492 |
| Random Forest (Ensemble)    | 0.9474   | 0.9937 | 0.9583    | 0.9583 | 0.9583 | 0.8869 |

### Observations

| ML Model Name             | Observation about model performance |
|----------------------------|--------------------------------------|
| Logistic Regression        | Best overall performer on this dataset. The classes are close to linearly separable after standardization, so a linear decision boundary generalizes very well. Highest accuracy, F1, and MCC of all five models. |
| Decision Tree               | Weakest performer. A single unpruned tree overfits the training data (visible in the lowest AUC and MCC), and is sensitive to small variations in the split — variance is the main issue. |
| kNN                         | Very strong performer, achieving perfect recall (no malignant case was misclassified as benign in the test set). Performs well because the features are on a comparable, standardized scale and the classes form fairly compact clusters. |
| Naive Bayes                 | Solid but not top-tier. The Gaussian independence assumption is violated somewhat (several features, e.g. radius/perimeter/area, are highly correlated), which caps its accuracy relative to Logistic Regression and kNN. |
| Random Forest (Ensemble)    | Clear improvement over the single Decision Tree — bagging reduces variance and pushes AUC close to the best models, though it still trails Logistic Regression and kNN slightly on this particular test split. |
| **Overall Winner for your dataset?** | **Logistic Regression** — highest Accuracy (0.9825), AUC (0.9954), F1 (0.9861), and MCC (0.9623) among all 5 models on this dataset. |

> **Note:** Exact metric values can vary slightly depending on `random_state`
> and train/test split. Re-run `model/train_models.py` to regenerate results.

## Repository Structure

```
project-folder/
|-- app.py                 # Streamlit application
|-- requirements.txt       # Python dependencies
|-- README.md              # This file
|-- test_data.csv          # Held-out test data (114 rows, 30 features + target)
|-- metrics_summary.csv    # Generated comparison table (raw)
|-- model/
|   |-- train_models.py            # Training script (trains all 5 models)
|   |-- logistic_regression.joblib
|   |-- decision_tree.joblib
|   |-- knn.joblib
|   |-- naive_bayes.joblib
|   |-- random_forest_ensemble.joblib
|   |-- scaler.joblib               # Fitted StandardScaler
|   |-- meta.json                   # Feature names / target label mapping
```

## How to Run Locally

```bash
pip install -r requirements.txt
python model/train_models.py   # (optional) regenerate models + test_data.csv
streamlit run app.py
```

Then, in the running app: upload `test_data.csv` in the sidebar, pick a model
from the dropdown, and view the metrics, confusion matrix, and classification
report.

## How to Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to https://streamlit.io/cloud and sign in with GitHub.
3. Click **New App** → select this repository and branch (`main`).
4. Set the main file path to `app.py`.
5. Click **Deploy**.
6. Once live, upload `test_data.csv` through the app's sidebar to see results.

## Live App Link

**https://ml-assignment-2-c8m6tjuwgscn2xehwvxbks.streamlit.app**
