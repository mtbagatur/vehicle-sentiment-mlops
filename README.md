# Vehicle & Sentiment MLOps

Production-ready MLOps system combining vehicle inventory management and sentiment analysis.

[![CI/CD](https://github.com/mtbagatur/vehicle-sentiment-mlops/actions/workflows/pipeline.yml/badge.svg)](https://github.com/mtbagatur/vehicle-sentiment-mlops/actions)

> Read the full article on Medium: [Building a Production-Ready MLOps System from Scratch](https://medium.com/@mtbagatur/building-a-production-ready-mlops-system-from-scratch-4d91f75be673)

---

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
git clone https://github.com/mtbagatur/vehicle-sentiment-mlops
cd vehicle-sentiment-mlops
docker compose up --build
```

| Service | URL |
|---|---|
| API docs | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |
| Grafana | http://localhost:3000 (admin / mlops123) |
| Prometheus | http://localhost:9090 |

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

# List all vehicles
curl http://localhost:8000/vehicles/
```

### Sentiment Analysis

```bash
# Predict sentiment
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
1. Run tests — pipeline stops if any test fails
2. Build Docker image tagged with commit SHA
3. Push to GitHub Container Registry
4. Deploy to server via SSH

Required GitHub secrets: `SERVER_HOST`, `SERVER_USER`, `SSH_PRIVATE_KEY`

---

*Built by [Taha Bagatur](https://www.linkedin.com/in/taha-bagatur) — MLOps & DevOps Engineer*