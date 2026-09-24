"""
SQLAlchemy ORM models for RailPredict AI.

Tables: trains, stations, train_positions, predictions, alerts, weather_data
"""
from datetime import datetime

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey, Boolean
)
from sqlalchemy.orm import relationship

from backend.database import Base


class Train(Base):
    __tablename__ = "trains"

    id = Column(Integer, primary_key=True, index=True)
    train_id = Column(String, unique=True, index=True, nullable=False)
    train_number = Column(String, nullable=False)
    train_name = Column(String, nullable=False)
    source = Column(String, nullable=False)
    destination = Column(String, nullable=False)
    scheduled_travel_time = Column(Float, default=120.0)  # minutes

    positions = relationship("TrainPosition", back_populates="train")
    predictions = relationship("Prediction", back_populates="train")
    alerts = relationship("Alert", back_populates="train")


class Station(Base):
    __tablename__ = "stations"

    id = Column(Integer, primary_key=True, index=True)
    code = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)


class TrainPosition(Base):
    __tablename__ = "train_positions"

    id = Column(Integer, primary_key=True, index=True)
    train_id = Column(String, ForeignKey("trains.train_id"))
    current_station = Column(String)
    next_station = Column(String)
    distance_remaining = Column(Float)
    speed = Column(Float)
    current_delay = Column(Float)
    congestion_level = Column(Float, default=0.0)
    unscheduled_stop = Column(Boolean, default=False)
    speed_restriction = Column(Boolean, default=False)
    updated_at = Column(DateTime, default=datetime.utcnow)

    train = relationship("Train", back_populates="positions")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    train_id = Column(String, ForeignKey("trains.train_id"))
    predicted_remaining_time = Column(Float)
    predicted_eta = Column(String)
    baseline_eta = Column(String)
    delay_prediction = Column(Float)
    confidence_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    train = relationship("Train", back_populates="predictions")


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    train_id = Column(String, ForeignKey("trains.train_id"))
    alert_type = Column(String)  # HIGH_DELAY, SEVERE_CONGESTION, LOW_CONFIDENCE, SPEED_RESTRICTION, WEATHER_IMPACT
    message = Column(String)
    severity = Column(String, default="MEDIUM")  # LOW, MEDIUM, HIGH, CRITICAL
    created_at = Column(DateTime, default=datetime.utcnow)

    train = relationship("Train", back_populates="alerts")


class WeatherData(Base):
    __tablename__ = "weather_data"

    id = Column(Integer, primary_key=True, index=True)
    station_code = Column(String)
    temperature = Column(Float)
    rain = Column(Float)
    is_demo = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
