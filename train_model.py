"""
Trains the ETA prediction model.

Run:
    python ml/train_model.py

Creates:
    models/eta_model.joblib
    models/feature_columns.joblib
    models/metrics.joblib   (MAE / RMSE / R2 computed on a held-out test split)

The architecture is intentionally swappable: change MODEL_FACTORY below to
plug in GradientBoostingRegressor, XGBoost, LightGBM, etc. without touching
any other file, since backend/services/prediction_service.py only depends on
the saved joblib artifacts and the FEATURE_COLUMNS contract.
"""
import os
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.config import (  # noqa: E402
    DATA_CSV_PATH, MODEL_PATH, FEATURE_COLUMNS_PATH, METRICS_PATH, FEATURE_COLUMNS
)
from ml.feature_engineering import load_training_frame  # noqa: E402


def MODEL_FACTORY():
    return RandomForestRegressor(
        n_estimators=250,
        max_depth=14,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )


def train():
    if not os.path.exists(DATA_CSV_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_CSV_PATH}. "
            "Run `python scripts/generate_dataset.py` first."
        )

    X, y = load_training_frame(DATA_CSV_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = MODEL_FACTORY()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))

    print("Model evaluation on held-out test set:")
    print(f"  MAE  : {mae:.2f} minutes")
    print(f"  RMSE : {rmse:.2f} minutes")
    print(f"  R2   : {r2:.3f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(FEATURE_COLUMNS, FEATURE_COLUMNS_PATH)
    joblib.dump(
        {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "r2": round(r2, 3),
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_train": len(X_train),
            "n_test": len(X_test),
        },
        METRICS_PATH,
    )
    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved feature columns -> {FEATURE_COLUMNS_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    train()
