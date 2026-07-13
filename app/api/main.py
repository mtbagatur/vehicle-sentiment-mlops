import os
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.api.vehicles import router as vehicle_router
from app.api.sentiment import router as sentiment_router
from app.api.deps import get_sentiment_model, get_vehicle_db
from app.schemas import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: train sentiment model
    model = get_sentiment_model()
    data_path = os.path.join(os.path.dirname(__file__), "../../data/training.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        metrics = model.train(df)
        print(f"Model trained — accuracy: {metrics['accuracy']:.3f}")
    else:
        print("Warning: training.csv not found, model not trained")
    yield
    # Shutdown
    print("Application shutting down")


app = FastAPI(
    title="Vehicle & Sentiment MLOps API",
    description="Production-ready MLOps system combining vehicle inventory management and sentiment analysis.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(vehicle_router)
app.include_router(sentiment_router)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    model = get_sentiment_model()
    db = get_vehicle_db()
    return HealthResponse(
        status="ok",
        model_ready=model.is_trained,
        vehicle_count=db.count,
    )


@app.get("/prometheus", tags=["system"], include_in_schema=False)
def prometheus_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
