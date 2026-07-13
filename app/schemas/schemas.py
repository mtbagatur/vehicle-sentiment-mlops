from pydantic import BaseModel, field_validator
from typing import Literal, Optional


# ── Vehicle schemas ──────────────────────────────────────────────────────────

class VehicleBase(BaseModel):
    vehicle_id: str
    name: str
    quantity: int
    price: float


class CarCreate(VehicleBase):
    type: Literal["car"] = "car"
    num_doors: int = 4
    is_electric: bool = False


class MotorcycleCreate(VehicleBase):
    type: Literal["motorcycle"] = "motorcycle"
    engine_cc: int = 600
    has_sidecar: bool = False


class VehicleResponse(BaseModel):
    vehicle_id: str
    name: str
    quantity: int
    price: float
    type: str

    model_config = {"from_attributes": True}


class SellRequest(BaseModel):
    quantity: int


class RestockRequest(BaseModel):
    quantity: int


# ── Sentiment schemas ─────────────────────────────────────────────────────────

class PredictionInput(BaseModel):
    texts: list[str]

    @field_validator("texts")
    @classmethod
    def texts_not_empty(cls, v):
        if not v:
            raise ValueError("texts list cannot be empty")
        return v


class PredictionOutput(BaseModel):
    text: str
    sentiment: Literal["positive", "negative"]


class ModelMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    is_trained: bool


# ── Health schema ─────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    vehicle_count: int
