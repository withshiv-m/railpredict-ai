"""
Standalone evaluation script — reloads the saved model and re-evaluates it
on a fresh train/test split of the current dataset. Useful for sanity
checking after regenerating data.

Run:
    python ml/evaluate.py
"""
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.config import DATA_CSV_PATH  # noqa: E402
from ml.feature_engineering import load_training_frame  # noqa: E402
from ml.predict import load_model  # noqa: E402


def evaluate():
    model, columns = load_model()
    X, y = load_training_frame(DATA_CSV_PATH)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preds = model.predict(X_test[columns])
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"MAE  : {mae:.2f} minutes")
    print(f"RMSE : {rmse:.2f} minutes")
    print(f"R2   : {r2:.3f}")


if __name__ == "__main__":
    evaluate()
