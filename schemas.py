"""
Pydantic models used for request validation and response shaping.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    train_id: str = Field(..., example="TR00001")
    distance_remaining: float = Field(..., ge=0, le=2000)
    speed: float = Field(..., ge=0, le=200)
    current_delay: float = Field(..., ge=0, le=600)
    rain: float = Field(..., ge=0, le=1)
    temperature: float = Field(..., ge=-10, le=55)
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    historical_delay: float = Field(..., ge=0, le=600)
    congestion_level: float = Field(..., ge=0, le=1)
    previous_train_delay: float = Field(..., ge=0, le=600)
    scheduled_travel_time: float = Field(..., ge=1, le=3000)
    section_average_speed: float = Field(..., ge=0, le=200)
    unscheduled_stop: int = Field(..., ge=0, le=1)
    speed_restriction: int = Field(..., ge=0, le=1)


class FeatureImpact(BaseModel):
    factor: str
    direction: str  # "increase" or "decrease"
    minutes: float
    description: str


class PredictionResponse(BaseModel):
    train_id: str
    predicted_remaining_time: float
    predicted_eta: str
    baseline_eta: str
    scheduled_eta: str
    delay_prediction: float
    confidence_score: float
    is_demo: bool = True
    feature_impacts: Optional[List[FeatureImpact]] = None


class TrainSummary(BaseModel):
    train_id: str
    train_number: str
    train_name: str
    source: str
    destination: str
    current_station: str
    next_station: str
    speed: float
    current_delay: float
    distance_remaining: float
    predicted_eta: Optional[str] = None
    confidence_score: Optional[float] = None
    status: str = "ON TIME"


class RouteStation(BaseModel):
    code: str
    name: str
    latitude: float
    longitude: float
    is_current: bool = False


class ModelMetrics(BaseModel):
    mae: float
    rmse: float
    r2: float
    trained_at: Optional[str] = None
    is_demo: bool = True


class AlertOut(BaseModel):
    train_id: str
    alert_type: str
    message: str
    severity: str
    created_at: str


class SimulateRequest(BaseModel):
    train_id: str
    ticks: int = Field(default=1, ge=1, le=50)
