"""
Generates a realistic SYNTHETIC dataset for the RailPredict AI prototype.

This is demo/simulation data only — it is NOT real Indian Railways data.
It encodes realistic relationships (e.g. higher congestion/delay/rain ->
higher remaining travel time; higher speed -> lower remaining travel time)
so the downstream ML model has genuine signal to learn from.

Run:
    python scripts/generate_dataset.py
Creates:
    data/train_data.csv
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.config import DATA_CSV_PATH  # noqa: E402

N_RECORDS = 6000
RNG_SEED = 42

DEMO_TRAINS = [
    ("TR00001", "12123", "Mumbai - Pune Express", "Mumbai CST", "Pune Jn"),
    ("TR00002", "11010", "Deccan Express", "Mumbai CST", "Pune Jn"),
    ("TR00003", "12010", "Shatabdi Express", "Mumbai Central", "Ahmedabad Jn"),
    ("TR00004", "12951", "Mumbai Rajdhani", "Mumbai Central", "New Delhi"),
    ("TR00005", "12622", "Tamil Nadu Express", "New Delhi", "Chennai Central"),
    ("TR00006", "12295", "Sanghamitra Express", "Bengaluru", "Patna Jn"),
]

STATIONS_BY_ROUTE = {
    "TR00001": ["Mumbai CST", "Dadar", "Kalyan", "Karjat", "Lonavala", "Pune Jn"],
    "TR00002": ["Mumbai CST", "Dadar", "Kalyan", "Lonavala", "Pune Jn"],
    "TR00003": ["Mumbai Central", "Borivali", "Vapi", "Surat", "Vadodara", "Ahmedabad Jn"],
    "TR00004": ["Mumbai Central", "Surat", "Vadodara", "Kota Jn", "New Delhi"],
    "TR00005": ["New Delhi", "Bhopal Jn", "Nagpur", "Vijayawada", "Chennai Central"],
    "TR00006": ["Bengaluru", "Guntakal Jn", "Vijayawada", "Gaya Jn", "Patna Jn"],
}


def generate():
    rng = np.random.default_rng(RNG_SEED)
    rows = []

    for i in range(N_RECORDS):
        train_id, train_number, train_name, source, destination = DEMO_TRAINS[
            i % len(DEMO_TRAINS)
        ]
        stations = STATIONS_BY_ROUTE[train_id]
        seg = rng.integers(0, len(stations) - 1)
        current_station = stations[seg]
        next_station = stations[seg + 1]

        distance_remaining = float(np.clip(rng.gamma(4.0, 60), 1, 1500))
        speed = float(np.clip(rng.normal(75, 22), 20, 130))
        congestion_level = float(np.clip(rng.beta(2, 5), 0, 1))
        rain = float(np.clip(rng.beta(1.5, 6), 0, 1))
        temperature = float(np.clip(rng.normal(28, 7), 5, 48))
        hour = int(rng.integers(0, 24))
        day_of_week = int(rng.integers(0, 7))
        historical_delay = float(np.clip(rng.gamma(2.0, 8), 0, 180))
        previous_train_delay = float(np.clip(rng.gamma(1.5, 6), 0, 180))
        unscheduled_stop = int(rng.random() < 0.08)
        speed_restriction = int(rng.random() < 0.12)
        section_average_speed = float(np.clip(speed + rng.normal(0, 8), 20, 130))
        scheduled_travel_time = float(np.clip(distance_remaining / 70 * 60, 10, 1200))

        # current_delay correlates with historical_delay, congestion, rain,
        # previous train delay, and unscheduled stop / speed restriction.
        current_delay = max(
            0.0,
            0.35 * historical_delay
            + 0.25 * previous_train_delay
            + 25 * congestion_level
            + 18 * rain
            + (15 if unscheduled_stop else 0)
            + (10 if speed_restriction else 0)
            + rng.normal(0, 6),
        )
        current_delay = float(np.clip(current_delay, 0, 180))

        # Effective speed drops with rain / congestion / speed restriction.
        effective_speed = speed * (
            1 - 0.25 * congestion_level - 0.15 * rain - (0.2 if speed_restriction else 0)
        )
        effective_speed = max(effective_speed, 10)

        base_time = (distance_remaining / effective_speed) * 60  # minutes
        remaining_travel_time = (
            base_time
            + 0.5 * current_delay
            + (12 if unscheduled_stop else 0)
            + rng.normal(0, 4)
        )
        remaining_travel_time = float(max(remaining_travel_time, 1))

        now = datetime(2026, 1, 1) + timedelta(
            days=int(rng.integers(0, 240)), hours=hour
        )
        actual_arrival_time = now + timedelta(minutes=remaining_travel_time)

        rows.append(
            {
                "train_id": train_id,
                "train_number": train_number,
                "train_name": train_name,
                "source": source,
                "destination": destination,
                "current_station": current_station,
                "next_station": next_station,
                "distance_remaining": round(distance_remaining, 2),
                "speed": round(speed, 2),
                "current_delay": round(current_delay, 2),
                "rain": round(rain, 3),
                "temperature": round(temperature, 1),
                "hour": hour,
                "day_of_week": day_of_week,
                "historical_delay": round(historical_delay, 2),
                "congestion_level": round(congestion_level, 3),
                "previous_train_delay": round(previous_train_delay, 2),
                "scheduled_travel_time": round(scheduled_travel_time, 2),
                "section_average_speed": round(section_average_speed, 2),
                "unscheduled_stop": unscheduled_stop,
                "speed_restriction": speed_restriction,
                "remaining_travel_time": round(remaining_travel_time, 2),
                "actual_arrival_time": actual_arrival_time.strftime("%Y-%m-%d %H:%M"),
            }
        )

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(DATA_CSV_PATH), exist_ok=True)
    df.to_csv(DATA_CSV_PATH, index=False)
    print(f"Generated {len(df)} synthetic records -> {DATA_CSV_PATH}")
    print(df.describe(include="all").T[["count"]].to_string())


if __name__ == "__main__":
    generate()
