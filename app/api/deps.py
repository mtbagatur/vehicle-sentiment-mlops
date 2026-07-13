from app.services.vehicle_db import VehicleDatabase
from app.services.sentiment_model import SentimentModel

_vehicle_db: VehicleDatabase | None = None
_sentiment_model: SentimentModel | None = None


def get_vehicle_db() -> VehicleDatabase:
    global _vehicle_db
    if _vehicle_db is None:
        _vehicle_db = VehicleDatabase()
    return _vehicle_db


def get_sentiment_model() -> SentimentModel:
    global _sentiment_model
    if _sentiment_model is None:
        _sentiment_model = SentimentModel()
    return _sentiment_model


def set_sentiment_model(model: SentimentModel) -> None:
    global _sentiment_model
    _sentiment_model = model
