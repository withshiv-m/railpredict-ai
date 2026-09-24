"""
Standalone prediction helper. Loads the trained model once (cached) and
exposes predict_remaining_time() plus a simple feature-impact explainer.

This module is imported by backend/services/prediction_service.py — the
FastAPI backend never retrains or reloads the model per-request.
"""
import os
import joblib
import numpy as np

from backend.config import MODEL_PATH, FEATURE_COLUMNS_PATH, METRICS_PATH
from ml.feature_engineering import build_feature_frame

_model_cache = {"model": None, "columns": None}


class ModelNotAvailableError(Exception):
    pass


def load_model():
    """Load model + feature columns once, cache in-process."""
    if _model_cache["model"] is not None:
        return _model_cache["model"], _model_cache["columns"]

    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURE_COLUMNS_PATH):
        raise ModelNotAvailableError(
            "ML model not found. Run `python ml/train_model.py` after "
            "`python scripts/generate_dataset.py`."
        )

    model = joblib.load(MODEL_PATH)
    columns = joblib.load(FEATURE_COLUMNS_PATH)
    _model_cache["model"] = model
    _model_cache["columns"] = columns
    return model, columns


def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    return joblib.load(METRICS_PATH)


def predict_remaining_time(record: dict) -> float:
    model, columns = load_model()
    X = build_feature_frame(record)[columns]
    pred = model.predict(X)[0]
    return float(max(pred, 1.0))


def baseline_remaining_time(record: dict) -> float:
    """Simple physics baseline: distance / speed, adjusted by current delay."""
    speed = max(record.get("speed", 0), 1)
    distance = max(record.get("distance_remaining", 0), 0)
    base = (distance / speed) * 60.0
    return float(base + 0.5 * record.get("current_delay", 0))


def confidence_score(model, X_row) -> float:
    """
    Estimate confidence using agreement across the RandomForest's trees:
    lower variance among tree predictions -> higher confidence.
    """
    if not hasattr(model, "estimators_"):
        return 85.0
    tree_preds = np.array([tree.predict(X_row)[0] for tree in model.estimators_])
    mean_pred = max(tree_preds.mean(), 1.0)
    spread_ratio = tree_preds.std() / mean_pred
    confidence = 100.0 * np.exp(-2.5 * spread_ratio)
    return float(np.clip(confidence, 40.0, 99.0))


def explain_feature_impacts(record: dict, predicted_minutes: float) -> list:
    """
    Feature-impact estimates (NOT exact SHAP values unless SHAP is wired in).
    Approach: perturb each feature to a "neutral"/favourable baseline value
    and measure the resulting change in prediction. This is a lightweight,
    dependency-free stand-in for SHAP that is honestly labelled as such.
    """
    model, columns = load_model()
    baseline_record = dict(record)
    neutral_values = {
        "current_delay": 0,
        "congestion_level": 0.1,
        "rain": 0.0,
        "historical_delay": 0,
        "previous_train_delay": 0,
        "unscheduled_stop": 0,
        "speed_restriction": 0,
    }
    impacts = []
    labels = {
        "current_delay": "Current delay",
        "congestion_level": "Section congestion",
        "rain": "Rainfall / weather",
        "historical_delay": "Historical running delay",
        "previous_train_delay": "Preceding train's delay",
        "unscheduled_stop": "Unscheduled stoppage",
        "speed_restriction": "Speed restriction",
        "speed": "Current speed",
    }

    for feature, neutral_value in neutral_values.items():
        modified = dict(baseline_record)
        modified[feature] = neutral_value
        X_mod = build_feature_frame(modified)[columns]
        pred_mod = float(model.predict(X_mod)[0])
        delta = predicted_minutes - pred_mod
        if abs(delta) < 0.3:
            continue
        direction = "increase" if delta > 0 else "decrease"
        impacts.append(
            {
                "factor": labels.get(feature, feature),
                "direction": direction,
                "minutes": round(abs(delta), 1),
                "description": (
                    f"{labels.get(feature, feature)} is estimated to "
                    f"{'increase' if delta > 0 else 'reduce'} ETA by "
                    f"about {abs(delta):.1f} min (feature impact estimate, not exact SHAP)."
                ),
            }
        )

    # Speed impact: compare against a favourably higher speed
    modified = dict(baseline_record)
    modified["speed"] = min(baseline_record.get("speed", 60) + 25, 130)
    modified["section_average_speed"] = min(
        baseline_record.get("section_average_speed", 60) + 25, 130
    )
    X_mod = build_feature_frame(modified)[columns]
    pred_mod = float(model.predict(X_mod)[0])
    delta = predicted_minutes - pred_mod
    if abs(delta) >= 0.3:
        impacts.append(
            {
                "factor": "Current speed",
                "direction": "increase" if delta > 0 else "decrease",
                "minutes": round(abs(delta), 1),
                "description": (
                    f"Lower speed is estimated to "
                    f"{'increase' if delta > 0 else 'reduce'} ETA by about "
                    f"{abs(delta):.1f} min versus a faster-running scenario "
                    "(feature impact estimate, not exact SHAP)."
                ),
            }
        )

    impacts.sort(key=lambda x: x["minutes"], reverse=True)
    return impacts[:6]
