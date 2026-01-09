"""
Generate required visualizations for the report.

Creates:
1) reports/figures/class_distribution.png
2) reports/figures/confusion_matrix_heatmap.png
3) reports/figures/feature_coefficients.png

Run:
python -m src.visualize
"""

from __future__ import annotations

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "o_tagviashvili25_61845.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "logreg_spam.joblib")
FIG_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")


def plot_class_distribution(df: pd.DataFrame) -> None:
    counts = df["is_spam"].value_counts().sort_index()
    labels = ["Legitimate (0)", "Spam (1)"]
    values = [counts.get(0, 0), counts.get(1, 0)]

    plt.figure()
    plt.bar(labels, values, label="Email count")
    plt.title("Class Distribution: Spam vs Legitimate")
    plt.xlabel("Class")
    plt.ylabel("Number of emails")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "class_distribution.png"), dpi=200)
    plt.close()


def plot_confusion_matrix_heatmap(df: pd.DataFrame) -> None:
    # Recreate the same test split used in training (random_state=42, test_size=0.3, stratify)
    feature_cols = ["words", "links", "capital_words", "spam_word_count"]
    X = df[feature_cols].copy()
    y = df["is_spam"].copy()

    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)  # [[TN, FP],[FN, TP]]

    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix Heatmap (Test Set)")
    plt.colorbar(label="Count")

    class_names = ["Legit (0)", "Spam (1)"]
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    # Labels and annotations
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "confusion_matrix_heatmap.png"), dpi=200)
    plt.close()


def plot_feature_coefficients() -> None:
    model = joblib.load(MODEL_PATH)
    logreg = model.named_steps["logreg"]
    feature_cols = ["words", "links", "capital_words", "spam_word_count"]
    coefs = logreg.coef_[0]

    plt.figure()
    plt.bar(feature_cols, coefs, label="Coefficient")
    plt.axhline(0)
    plt.title("Logistic Regression Feature Coefficients")
    plt.xlabel("Feature")
    plt.ylabel("Coefficient value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "feature_coefficients.png"), dpi=200)
    plt.close()


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Run: python -m src.train_logreg")

    plot_class_distribution(df)
    plot_confusion_matrix_heatmap(df)
    plot_feature_coefficients()

    print("✅ Figures generated in reports/figures/")


if __name__ == "__main__":
    main()
