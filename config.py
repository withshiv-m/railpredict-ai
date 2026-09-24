"""
Central configuration for RailPredict AI backend.
Loads settings from environment variables (.env) with safe fallbacks so the
app never crashes just because a config value is missing.
"""
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Database ---
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'database.db')}")

# --- Weather ---
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "").strip()

# --- CORS ---
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if origin.strip()
]

# --- Paths ---
DATA_CSV_PATH = os.path.join(BASE_DIR, "data", "train_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "eta_model.joblib")
FEATURE_COLUMNS_PATH = os.path.join(BASE_DIR, "models", "feature_columns.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.joblib")

# --- ML feature schema (order matters — must match training) ---
FEATURE_COLUMNS = [
    "distance_remaining",
    "speed",
    "current_delay",
    "rain",
    "temperature",
    "hour",
    "day_of_week",
    "historical_delay",
    "congestion_level",
    "previous_train_delay",
    "scheduled_travel_time",
    "section_average_speed",
    "unscheduled_stop",
    "speed_restriction",
]

TARGET_COLUMN = "remaining_travel_time"

# --- App metadata ---
APP_NAME = "RailPredict AI"
APP_VERSION = "1.0.0"
DEMO_DATA_NOTICE = (
    "DEMO DATA — this prototype uses a simulated data layer and is NOT "
    "connected to live Indian Railways / NTES / GPS / signalling systems."
)
