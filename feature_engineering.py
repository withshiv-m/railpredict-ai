"""
Shared feature engineering used by both training (train_model.py) and
inference (predict.py / backend prediction_service.py), so train and serve
never drift apart.
"""
import pandas as pd

from backend.config import FEATURE_COLUMNS, TARGET_COLUMN


def build_feature_frame(records: list) -> pd.DataFrame:
    """
    Convert a list of dicts (or a single dict) with raw input fields into a
    DataFrame with columns in the exact order the model was trained on.
    """
    if isinstance(records, dict):
        records = [records]
    df = pd.DataFrame(records)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    return df[FEATURE_COLUMNS]


def load_training_frame(csv_path: str):
    """Load and lightly clean the CSV for training. Never crashes hard —
    raises a clear error instead so callers can show a friendly message."""
    df = pd.read_csv(csv_path)

    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = df.dropna(subset=required)
    for col in required:
        df = df[df[col] >= 0] if col != TARGET_COLUMN else df

    X = df[FEATURE_COLUMNS].astype(float)
    y = df[TARGET_COLUMN].astype(float)
    return X, y
