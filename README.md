# Vehicle & Sentiment MLOps

Production-ready MLOps system combining vehicle inventory management and sentiment analysis.
Built from a real ING Hubs Turkey Codility assessment, extended into a full MLOps pipeline.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   FastAPI Application                   │
│  /vehicles/*  (CRUD + sell/restock)                    │
│  /sentiment/predict   /sentiment/metrics               │
│  /health   /prometheus                                  │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
   ┌────▼───┐  ┌─────▼────┐ ┌───▼──────┐
   │MLflow  │  │Prometheus│ │ Grafana  │
   │:5000   │  │:9090     │ │ :3000    │
   └────────┘  └──────────┘ └──────────┘
```

## Stack

| Layer | Tool |
|---|---|
| API | FastAPI + Pydantic |
| ML | scikit-learn (TF-IDF + LogReg) |
| Experiment tracking | MLflow |
| Containerization | Docker + docker-compose |
| CI/CD | GitHub Actions |
| Metrics | Prometheus |
| Dashboards | Grafana |

## Quick Start

```bash
# Clone and run
git clone https://github.com/your-username/vehicle-sentiment-mlops
cd vehicle-sentiment-mlops
docker compose up --build

# API docs
open http://localhost:8000/docs

# MLflow experiments
open http://localhost:5000

# Grafana dashboard (admin / mlops123)
open http://localhost:3000
```

## API Endpoints

### Vehicle Inventory

```bash
# Create a car
curl -X POST http://localhost:8000/vehicles/cars \
  -H "Content-Type: application/json" \
  -d '{"vehicle_id":"c1","name":"Honda Civic","quantity":10,"price":25000,"num_doors":4,"is_electric":false}'

# Sell 3 units
curl -X POST http://localhost:8000/vehicles/c1/sell \
  -H "Content-Type: application/json" \
  -d '{"quantity":3}'

# List all
curl http://localhost:8000/vehicles/
```

### Sentiment Analysis

```bash
# Predict
curl -X POST http://localhost:8000/sentiment/predict \
  -H "Content-Type: application/json" \
  -d '{"texts":["I love this product","Terrible experience"]}'

# Model metrics
curl http://localhost:8000/sentiment/metrics
```

## Run Tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

## CI/CD

On every push to `main`:
1. Run tests
2. Build Docker image
3. Push to GitHub Container Registry
4. Deploy to server via SSH

Set these GitHub secrets: `SERVER_HOST`, `SERVER_USER`, `SSH_PRIVATE_KEY`
