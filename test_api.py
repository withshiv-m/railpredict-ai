"""
Backend test suite.

Run (from project root, with venv activated and dataset/model already
generated/trained):
    pytest tests/ -v
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)

VALID_PREDICT_PAYLOAD = {
    "train_id": "TR00001",
    "distance_remaining": 85,
    "speed": 68,
    "current_delay": 12,
    "rain": 0.3,
    "temperature": 28,
    "hour": 18,
    "day_of_week": 2,
    "historical_delay": 8,
    "congestion_level": 0.5,
    "previous_train_delay": 5,
    "scheduled_travel_time": 100,
    "section_average_speed": 70,
    "unscheduled_stop": 0,
    "speed_restriction": 0,
}


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_list_trains():
    r = client.get("/api/trains")
    assert r.status_code == 200
    data = r.json()
    assert data["is_demo"] is True
    assert len(data["trains"]) > 0


def test_train_detail_valid():
    r = client.get("/api/trains/TR00001")
    assert r.status_code == 200
    assert r.json()["train_id"] == "TR00001"


def test_train_detail_invalid_id():
    r = client.get("/api/trains/NOT_A_REAL_TRAIN")
    assert r.status_code == 404


def test_predict_valid():
    r = client.post("/api/predict", json=VALID_PREDICT_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert "predicted_eta" in body
    assert "confidence_score" in body
    assert 0 <= body["confidence_score"] <= 100


def test_predict_invalid_input():
    bad_payload = dict(VALID_PREDICT_PAYLOAD)
    bad_payload["speed"] = -50  # invalid: below allowed range
    r = client.post("/api/predict", json=bad_payload)
    assert r.status_code == 422


def test_predict_missing_field():
    bad_payload = dict(VALID_PREDICT_PAYLOAD)
    del bad_payload["distance_remaining"]
    r = client.post("/api/predict", json=bad_payload)
    assert r.status_code == 422


def test_model_metrics():
    r = client.get("/api/model/metrics")
    # Passes whether or not a model has been trained yet in this environment.
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        body = r.json()
        assert "mae" in body and "rmse" in body and "r2" in body


def test_alerts_endpoint():
    r = client.get("/api/alerts")
    assert r.status_code == 200
    assert "alerts" in r.json()


def test_weather_demo():
    r = client.get("/api/weather/demo?latitude=19.07&longitude=72.87")
    assert r.status_code == 200
    body = r.json()
    assert "temperature" in body and "rain" in body
