"""
Predict spam/legitimate for a custom email text using the trained model.

Examples:
python -m src.predict --text "URGENT! Win cash now, click https://bit.ly/xxx"
python -m src.predict --file samples/spam_email.txt
"""

from __future__ import annotations

import argparse
import os
import joblib
import numpy as np
import pandas as pd

from src.features import extract_features


HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "logreg_spam.joblib")


def load_text_from_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default=None, help="Email text to classify")
    parser.add_argument("--file", type=str, default=None, help="Path to a .txt file containing email text")
    args = parser.parse_args()

    if args.file:
        email_text = load_text_from_file(args.file)
    elif args.text:
        email_text = args.text
    else:
        print("Paste your email text below. Finish with Ctrl+Z (Windows) or Ctrl+D (Linux/Mac):")
        email_text = ""
        try:
            while True:
                line = input()
                email_text += line + "\n"
        except EOFError:
            pass

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run src/train_logreg.py first.")

    model = joblib.load(MODEL_PATH)

    feats = extract_features(email_text)
    X = pd.DataFrame([feats])  # same columns as training
    proba_spam = float(model.predict_proba(X)[0, 1])
    pred = int(model.predict(X)[0])

    label = "SPAM" if pred == 1 else "LEGITIMATE"
    print("\n--- Extracted Features ---")
    for k, v in feats.items():
        print(f"{k:16s}: {v}")

    print("\n--- Prediction ---")
    print(f"Predicted class: {label}")
    print(f"Spam probability: {proba_spam:.4f}")


if __name__ == "__main__":
    main()
