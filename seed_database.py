"""
Seeds the SQLite database with the demo trains and stations so the
`trains` and `stations` tables are populated (used for reference data;
live-changing state such as position/speed/delay is served from the
in-memory simulation layer — see backend/services/simulation_service.py).

Run:
    python scripts/seed_database.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import SessionLocal, init_db  # noqa: E402
from backend.models import Train, Station  # noqa: E402
from backend.services.simulation_service import DEMO_TRAINS, STATION_COORDS  # noqa: E402


def seed():
    init_db()
    db = SessionLocal()
    try:
        for train_id, meta in DEMO_TRAINS.items():
            existing = db.query(Train).filter_by(train_id=train_id).first()
            if existing:
                continue
            db.add(Train(
                train_id=train_id,
                train_number=meta["train_number"],
                train_name=meta["train_name"],
                source=meta["source"],
                destination=meta["destination"],
                scheduled_travel_time=meta["scheduled_travel_time"],
            ))

        for name, (lat, lon) in STATION_COORDS.items():
            code = name.replace(" ", "_").upper()[:8]
            existing = db.query(Station).filter_by(code=code).first()
            if existing:
                continue
            db.add(Station(code=code, name=name, latitude=lat, longitude=lon))

        db.commit()
        print(f"Seeded {len(DEMO_TRAINS)} trains and {len(STATION_COORDS)} stations.")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
