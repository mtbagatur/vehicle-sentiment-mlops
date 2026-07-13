from fastapi import APIRouter, HTTPException
from app.schemas import PredictionInput, PredictionOutput, ModelMetrics
from app.api.deps import get_sentiment_model
from app.models.errors import ModelNotReadyError
from prometheus_client import Counter, Histogram
import time

router = APIRouter(prefix="/sentiment", tags=["sentiment"])

PREDICTIONS = Counter("sentiment_predictions_total", "Total predictions", ["label"])
PREDICTION_LATENCY = Histogram("sentiment_prediction_duration_seconds", "Prediction latency")
PREDICTION_ERRORS = Counter("sentiment_prediction_errors_total", "Prediction errors")


@router.post("/predict", response_model=list[PredictionOutput])
def predict(payload: PredictionInput):
    model = get_sentiment_model()
    start = time.time()
    try:
        sentiments = model.predict_batch(payload.texts)
        results = []
        for text, sentiment in zip(payload.texts, sentiments):
            PREDICTIONS.labels(label=sentiment).inc()
            results.append(PredictionOutput(text=text, sentiment=sentiment))
        return results
    except ModelNotReadyError:
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=503, detail="Model is not ready yet. Try again later.")
    except Exception as e:
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        PREDICTION_LATENCY.observe(time.time() - start)


@router.get("/metrics", response_model=ModelMetrics)
def get_metrics():
    model = get_sentiment_model()
    try:
        m = model.metrics
        return ModelMetrics(
            accuracy=m["accuracy"],
            precision=m["precision"],
            recall=m["recall"],
            f1_score=m["f1_score"],
            is_trained=model.is_trained,
        )
    except ModelNotReadyError:
        raise HTTPException(status_code=503, detail="Model is not ready yet.")
