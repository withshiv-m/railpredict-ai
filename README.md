# RailPredict AI — Dynamic Train ETA Forecasting System

A working prototype built for **Smart India Hackathon (SIH) 2026**.

- **Problem Statement ID:** 26028
- **Title:** Dynamic Forecast of Expected Time of Arrival (ETA) for Coaching Trains
- **Organization:** Ministry of Railways
- **Category:** Software · **Theme:** Disaster Management

> **DEMO DATA NOTICE:** This project uses a realistic **synthetic / simulated** data layer
> (demo trains, demo weather, demo GPS-style positions and demo congestion). It is **NOT**
> connected to live Indian Railways, NTES, GPS, or signalling systems. The backend is
> deliberately structured so authorized real data sources can be integrated later — see
> [Live Data Integration Plan](#live-data-integration-plan).

---

## 1. Project Overview

RailPredict AI dynamically forecasts a coaching train's Expected Time of Arrival (ETA) using
a trained machine learning model, rather than a static timetable. It considers current
position, delay, distance remaining, speed, historical running patterns, weather, time of
day/week, section congestion, the preceding train's delay, unscheduled stoppages and speed
restrictions — and recalculates the ETA whenever any of these inputs change.

It ships with:
- A **baseline** ETA (distance ÷ speed, delay-adjusted) shown next to the **AI** ETA, so the
  model's value is transparent rather than assumed.
- A **What-If simulator** (sliders) and a **Live Simulation** (WebSocket) mode to demonstrate
  the "dynamic" part of the problem statement live, in the browser.
- A lightweight, honestly-labelled **explainability** feature ("why did AI predict this ETA?")
  — feature-impact *estimates*, not exact SHAP values, unless SHAP is wired in later.

## 2. Features

- AI + baseline ETA comparison, confidence score, delay-vs-schedule
- Operations dashboard with live-computed stats and a train monitoring table
- Per-train detail page: route timeline, demo map, explainable AI panel
- Live Simulation over WebSocket (ticks train state every 3s, recalculates ETA)
- ETA What-If Simulator (drag sliders → live re-prediction)
- Analytics page (prediction vs actual, delay distribution, weather vs delay, congestion vs
  travel time, train performance, ETA error distribution) — all computed from the real dataset
  and the currently trained model, never hard-coded
- Alerts (high delay, severe congestion, low confidence, speed restriction, weather impact)
- `/api/model/metrics` reporting MAE / RMSE / R² computed from an actual held-out test split

## 3. Architecture

```
Data Sources (GPS/telemetry, historical data, weather, signalling, congestion, schedule)
        |
Data Ingestion  ->  Data Cleaning  ->  Feature Engineering
        |
ML Prediction Engine (RandomForestRegressor, swappable for XGBoost/LightGBM/PyTorch)
        |
ETA Calculation  (ETA = current_time + predicted_remaining_travel_time)
        |
FastAPI Backend  ->  React Dashboard  ->  Passengers / Railway Operations
```

For the hackathon prototype, a demo simulator (`backend/services/simulation_service.py`)
replaces unavailable live feeds. It generates plausible in-memory train state and ticks it
forward, so the rest of the pipeline (features → model → ETA → API → UI) behaves exactly as
it would with a real feed plugged in.

## 4. Technology Stack

| Layer      | Technology |
|------------|------------|
| Backend    | Python, FastAPI, SQLAlchemy, SQLite, Pydantic |
| ML         | Pandas, NumPy, Scikit-learn (RandomForestRegressor), Joblib |
| Frontend   | React, Vite, Chart.js (via react-chartjs-2), Leaflet (via react-leaflet) |
| Testing    | Pytest |

## 5. Folder Structure

```
RailPredict-AI/
├── backend/            FastAPI app, DB models, schemas, API routers, services
├── ml/                 Training, prediction, evaluation, feature engineering
├── scripts/            Dataset generation + DB seeding
├── data/                train_data.csv (generated)
├── models/              eta_model.joblib, feature_columns.joblib, metrics.joblib (generated)
├── frontend/            React + Vite app
├── tests/                Pytest backend tests
├── .env.example
├── run.bat               Windows one-click helper
└── README.md
```

## 6. Installation (Windows, beginner-friendly)

Do not assume you're already inside the project folder — `cd` into it first:

```
cd "C:\Users\YOUR_NAME\Downloads\RailPredict-AI"
```

### 6.1 Backend setup

```
python -m venv venv
venv\Scripts\activate
pip install -r backend\requirements.txt
```

> If PowerShell blocks `venv\Scripts\activate` with an execution-policy error, run PowerShell
> as Administrator once and execute:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
> then reopen your terminal and try again. Alternatively, use **Command Prompt (cmd.exe)**
> instead of PowerShell, where this restriction does not apply.

### 6.2 Generate the dataset

```
python scripts\generate_dataset.py
```
Creates `data\train_data.csv` (6,000 synthetic, realistic records).

### 6.3 Train the model

```
python ml\train_model.py
```
Creates `models\eta_model.joblib`, `models\feature_columns.joblib`, `models\metrics.joblib`,
and prints MAE / RMSE / R² computed on a held-out test split.

### 6.4 Start the backend

```
uvicorn backend.main:app --reload
```
Backend runs at `http://localhost:8000` (interactive docs at `http://localhost:8000/docs`).

### 6.5 Start the frontend (in a **second** terminal)

```
cd frontend
npm install
npm run dev
```
Open the URL Vite prints (usually `http://localhost:5173`).

### 6.6 Or just double-click `run.bat`

`run.bat` automates steps 6.1–6.4 (venv, install, dataset, training, backend start). You still
need to start the frontend yourself in a second terminal (step 6.5) — printed on screen when
`run.bat` runs.

## 7. API Reference

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | App metadata + demo-data notice |
| GET | `/api/health` | Health check |
| GET | `/api/trains` | List all demo trains with live AI ETA |
| GET | `/api/trains/{train_id}` | Full detail + explainable prediction for one train |
| GET | `/api/trains/{train_id}/route` | Route stations with coordinates for the map |
| POST | `/api/predict` | Run a prediction for arbitrary input features |
| POST | `/api/simulate` | Advance a demo train's state N ticks and re-predict |
| GET | `/api/model/metrics` | MAE / RMSE / R² from the currently trained model |
| GET | `/api/analytics` | Chart-ready aggregates from the dataset + model |
| GET | `/api/alerts` | Live-computed operational alerts |
| GET | `/api/weather/demo` | Demo (or live, if configured) weather for a point |
| WS | `/ws/simulation/{train_id}` | Live-ticking simulation feed for the dashboard |

Example `/api/predict` request:
```json
{
  "train_id": "TR00001",
  "distance_remaining": 85, "speed": 68, "current_delay": 12,
  "rain": 0.3, "temperature": 28, "hour": 18, "day_of_week": 2,
  "historical_delay": 8, "congestion_level": 0.5, "previous_train_delay": 5,
  "scheduled_travel_time": 100, "section_average_speed": 70,
  "unscheduled_stop": 0, "speed_restriction": 0
}
```

## 8. ML Explanation

- **Model:** `RandomForestRegressor` (`ml/train_model.py`) — swap `MODEL_FACTORY()` in that
  file for GradientBoostingRegressor / XGBoost / LightGBM / PyTorch without touching any other
  file, since the rest of the app only depends on the saved `.joblib` artifacts and the shared
  `FEATURE_COLUMNS` contract in `backend/config.py`.
- **Target:** `remaining_travel_time` (minutes). `ETA = current_time + predicted_remaining_travel_time`.
- **Baseline:** `distance_remaining / speed`, adjusted by current delay — always computed and
  shown alongside the AI prediction for honest comparison.
- **Confidence score:** derived from agreement across the RandomForest's individual trees
  (lower spread → higher confidence), not a hard-coded number.
- **Explainability:** `ml/predict.py::explain_feature_impacts()` perturbs each feature to a
  neutral/favourable value and measures the resulting change in prediction — labelled as
  "feature impact estimates," not exact SHAP values.
- **Metrics:** MAE, RMSE, R² are computed on a real held-out test split every time you run
  `python ml/train_model.py`, saved to `models/metrics.joblib`, and served (never hard-coded)
  via `GET /api/model/metrics`.

## 9. Running Tests

```
venv\Scripts\activate
pytest tests\ -v
```
Covers health, train listing/detail (valid + invalid IDs), prediction (valid + invalid +
missing input), model metrics, alerts, and weather.

## 10. Demo Workflow

1. Run `run.bat` (or the manual steps above) and start the frontend.
2. Open the **Home** page, click **Check Train ETA**.
3. Pick a demo train, tweak conditions, submit — see baseline vs AI ETA and the explanation.
4. Go to **Simulation** → **Start Simulation** to watch a live-ticking WebSocket feed update
   the ETA in real time; or drag the **What-If** sliders for instant re-prediction.
5. Visit **Dashboard** for the operations overview, **Analytics** for charts, and **Alerts**
   for live-computed operational warnings.

## 11. Error Handling

The app is designed not to crash when the model file, CSV, or database is missing, when the
weather API key is absent, or when an invalid train ID / prediction input is supplied — it
returns clear HTTP error messages (e.g. `503` with a message telling you which script to run)
instead.

## 12. Security Notes

- No API keys are hard-coded; `WEATHER_API_KEY` is read from `.env` (see `.env.example`) and
  the app **automatically falls back to demo weather data** if it's blank.
- CORS is restricted to the origins listed in `.env` (`CORS_ORIGINS`), defaulting to the local
  Vite dev server.
- All API input is validated with Pydantic (see `backend/schemas.py`).

## 13. Future Improvements

- Swap SQLite for PostgreSQL (the database layer is already isolated in `backend/database.py`)
- Plug in a real, authorized weather provider in `backend/services/weather_service.py`
- Replace the demo simulator with a real GPS/signalling feed once access is authorized
- Swap RandomForest for XGBoost/LightGBM/a temporal model, and add real SHAP explanations
- Introduce Kafka/Redis for higher-throughput real-time ingestion at scale

## 14. Limitations

This is a hackathon prototype: all positions, weather, congestion and GPS-style coordinates
are simulated for demo purposes; it should not be interpreted as, or connected to, live
railway operational systems without proper authorization and integration work.

## Live Data Integration Plan

Real-time architecture is designed for but intentionally **not required to run** the MVP:
- `backend/services/weather_service.py` isolates the weather call — add a real
  provider call there and set `WEATHER_API_KEY` in `.env`.
- `backend/services/simulation_service.py` is the seam where a real GPS/signalling feed would
  replace the in-memory demo state — the rest of the pipeline (features → model → API → UI)
  does not need to change.
- The database layer (`backend/database.py`) can be pointed at PostgreSQL by changing
  `DATABASE_URL` in `.env` — no other code changes required.
