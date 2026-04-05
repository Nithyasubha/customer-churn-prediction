# train.py

import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

from preprocessing import load_and_preprocess

# Load Data
X, y = load_and_preprocess(
    "D:\\Al Data Analytics\\Interview\\churn_prediction_project\\Data\\telecom_churn.csv"
)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
models = {
    "Logistic": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

best_model = None
best_score = 0

print("\n===== MODEL PERFORMANCE =====\n")

for name, model in models.items():
    # Train
    model.fit(X_train, y_train)

    # Predictions
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, pred)
    roc = roc_auc_score(y_test, prob)
    cm = confusion_matrix(y_test, pred)

    print(f"{name} Accuracy: {acc:.4f}")
    print(f"{name} ROC-AUC: {roc:.4f}")
    print(f"{name} Confusion Matrix:\n{cm}\n")

    # Best Model Selection
    if roc > best_score:   # Use ROC instead of accuracy (better)
        best_score = roc
        best_model = model

# Create model folder if not exists
os.makedirs(
    "D:\\Al Data Analytics\\Interview\\churn_prediction_project\\model",
    exist_ok=True
)

# Save Best Model
joblib.dump(
    best_model,
    "D:\\Al Data Analytics\\Interview\\churn_prediction_project\\model\\churn_model.pkl"
)

print("Best model saved successfully!")