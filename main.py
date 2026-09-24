"""
RailPredict AI — FastAPI backend entrypoint.

Run (from project root, with venv activated):
    uvicorn backend.main:app --reload
"""
import asyncio
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.config import APP_NAME, APP_VERSION, CORS_ORIGINS, DEMO_DATA_NOTICE
from backend.database import init_db
from backend.api import trains, prediction, analytics, alerts, weather
from backend.services.simulation_service import (
    DEMO_TRAINS, tick, build_prediction_record, generate_alerts_for_state,
)
from backend.services.prediction_service import run_prediction, ModelNotAvailableError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("railpredict")

app = FastAPI(title=APP_NAME, version=APP_VERSION, description=DEMO_DATA_NOTICE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    # Never crash startup because of a missing DB file — init_db creates it.
    try:
        init_db()
        logger.info("Database initialized.")
    except Exception as e:
        logger.warning(f"Database init warning: {e}")


@app.get("/")
def root():
    return {
        "app": APP_NAME,
        "version": APP_VERSION,
        "notice": DEMO_DATA_NOTICE,
        "docs": "/docs",
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "app": APP_NAME, "version": APP_VERSION}


app.include_router(trains.router)
app.include_router(prediction.router)
app.include_router(analytics.router)
app.include_router(alerts.router)
app.include_router(weather.router)


@app.websocket("/ws/simulation/{train_id}")
async def simulation_socket(websocket: WebSocket, train_id: str):
    """
    Live Simulation feed. Every 3 seconds, advances the demo train's state
    one tick and pushes back the updated state + AI prediction + alerts.
    DEMO DATA — not a live GPS/signalling feed.
    """
    await websocket.accept()
    if train_id not in DEMO_TRAINS:
        await websocket.send_json({"error": f"Unknown train_id '{train_id}'"})
        await websocket.close()
        return
    try:
        while True:
            state = tick(train_id)
            try:
                record = build_prediction_record(train_id)
                prediction_result = run_prediction(record, explain=False)
                alerts_list = generate_alerts_for_state(train_id, state, prediction_result)
            except ModelNotAvailableError as e:
                await websocket.send_json({"error": str(e)})
                await asyncio.sleep(3)
                continue

            await websocket.send_json({
                "train_id": train_id,
                "is_demo": True,
                "state": state,
                "prediction": prediction_result,
                "alerts": alerts_list,
            })
            await asyncio.sleep(3)
    except WebSocketDisconnect:
        logger.info(f"Simulation socket disconnected for {train_id}")
