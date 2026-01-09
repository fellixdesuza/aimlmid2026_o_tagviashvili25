"""
Train and evaluate Logistic Regression for spam detection.

Uses:
- 70% training data
- 30% testing data

Saves:
- models/logreg_spam.joblib
- outputs/metrics.json
- outputs/coefficients.csv
"""

from __future__ import annotations

import json
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

import joblib


HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "o_tagviashvili25_61845.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "logreg_spam.joblib")
METRICS_PATH = os.path.join(PROJECT_ROOT, "outputs", "metrics.json")
COEF_PATH = os.path.join(PROJECT_ROOT, "outputs", "coefficients.csv")


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    feature_cols = ["words", "links", "capital_words", "spam_word_count"]
    label_col = "is_spam"

    X = df[feature_cols].copy()
    y = df[label_col].copy()

    # 70/30 split with stratification to keep class ratio similar
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    # Pipeline: scaling + logistic regression
    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000, random_state=42))
    ])

    model.fit(X_train, y_train)

    # Validation on the 30% test set
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)  # [[TN, FP],[FN, TP]]
    acc = accuracy_score(y_test, y_pred)

    # Coefficients (after scaling). Intercept is also included.
    logreg = model.named_steps["logreg"]
    coef = logreg.coef_[0].tolist()
    intercept = float(logreg.intercept_[0])

    coef_df = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": coef
    })

    # Save artifacts
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    coef_df.to_csv(COEF_PATH, index=False)

    metrics = {
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "accuracy": float(acc),
        "confusion_matrix": {
            "TN": int(cm[0, 0]),
            "FP": int(cm[0, 1]),
            "FN": int(cm[1, 0]),
            "TP": int(cm[1, 1]),
        },
        "intercept": intercept,
        "coefficients": {feature_cols[i]: float(coef[i]) for i in range(len(feature_cols))}
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("✅ Training complete.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Accuracy on test set: {acc:.4f}")
    print("Confusion Matrix [[TN, FP],[FN, TP]]:")
    print(cm)
    print("\nCoefficients:")
    print(coef_df)
    print(f"\nIntercept: {intercept:.6f}")


if __name__ == "__main__":
    main()
