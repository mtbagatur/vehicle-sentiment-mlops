# Codility'den Production'a: Gerçek Bir MLOps Projesi Nasıl Kurulur
### ING Hubs Turkey mülakatında çözdüğüm iki görevi production-ready bir sisteme dönüştürme hikayesi

*Taha Bagatur — MLOps & DevOps Engineer*

---

Geçen hafta ING Hubs Turkey için bir Codility teknik değerlendirmesine girdim. İki görev, 105 dakika, %98 sonuç.

Ama sınav bittikten sonra kendime bir soru sordum: **"Bu kodu gerçekten production'a almak isteseydim ne yapardım?"**

Bu yazı o sorunun cevabı. Codility'de çözdüğüm iki görevi alıp sıfırdan gerçek bir MLOps sistemi kuruyorum — Docker, MLflow, Prometheus, Grafana ve GitHub Actions ile.

Kodu GitHub'da bulabilirsiniz: `github.com/your-username/vehicle-sentiment-mlops`

---

## Codility'de Ne Yaptım?

**Görev 1 — Vehicle Inventory Management:**
Python `@dataclass` ile araç envanteri sistemi. `Vehicle` temel sınıfı, `Car` ve `Motorcycle` alt sınıfları, validation mantığı, CRUD operasyonları, sell/restock işlemleri.

**Görev 2 — Sentiment Analysis API:**
FastAPI + scikit-learn ile duygu analizi. Metin temizleme pipeline'ı, TF-IDF vektörizasyon, lojistik regresyon, `/predict` ve `/metrics` endpoint'leri.

Her ikisi de Codility'de iyi çalıştı. Ama ikisi birbirinden izole sistemlerdi ve production'a alınmaya hazır değildi.

---

## Proje Mimarisi

```
vehicle-sentiment-mlops/
├── app/
│   ├── api/
│   │   ├── main.py          # FastAPI + lifespan
│   │   ├── vehicles.py      # Vehicle router
│   │   ├── sentiment.py     # Sentiment router
│   │   └── deps.py          # Dependency injection
│   ├── models/
│   │   ├── vehicle.py       # Dataclasses
│   │   └── errors.py        # Custom exceptions
│   ├── services/
│   │   ├── vehicle_db.py    # In-memory database
│   │   ├── sentiment_model.py  # MLflow + sklearn
│   │   └── preprocessor.py  # Text cleaning
│   └── schemas/
│       └── schemas.py       # Pydantic models
├── monitoring/
│   ├── prometheus/
│   └── grafana/
├── .github/workflows/
│   └── pipeline.yml
├── data/training.csv
├── Dockerfile
└── docker-compose.yml
```

Tek komutla tüm sistem ayağa kalkar:
```bash
docker compose up --build
```

---

## Katman Katman: Nasıl Çalışıyor?

### 1. Veri Modelleri — Codility'den Daha Güçlü

Codility'de `Vehicle` ve `Car` sınıflarını yazmıştım. Burada aynı mantığı aldım ama production için güçlendirdim:

```python
@dataclass
class Car(Vehicle):
    num_doors: int = 4
    is_electric: bool = False

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.num_doors, int) or isinstance(self.num_doors, bool) or self.num_doors <= 0:
            raise VehicleValidationError("num_doors must be a positive integer")
        if not isinstance(self.is_electric, bool):
            raise VehicleValidationError("is_electric must be a boolean")
```

`isinstance(x, bool)` kontrolü kritik: Python'da `bool`, `int`'in alt sınıfıdır. `num_doors=True` geçmek `num_doors=1` olarak geçer — bunu engellemek için bool'u ayrıca reddetmek gerekiyor.

### 2. Servis Katmanı — Bağımlılık Yönetimi

Codility'de her şey tek dosyadaydı. Burada dependency injection ile ayırdım:

```python
# deps.py
_vehicle_db: VehicleDatabase | None = None
_sentiment_model: SentimentModel | None = None

def get_vehicle_db() -> VehicleDatabase:
    global _vehicle_db
    if _vehicle_db is None:
        _vehicle_db = VehicleDatabase()
    return _vehicle_db
```

Bu pattern test yazılabilirliği açısından kritik. Testlerde mock bir database enjekte edebiliyorsun.

### 3. MLflow — Experiment Tracking

Codility'de model startup'ta eğitilip bellekte saklanıyordu. Yeterli ama yetersiz. Gerçekte her eğitim run'ının takip edilmesi gerekiyor:

```python
def train(self, data: pd.DataFrame) -> dict:
    mlflow.set_experiment("sentiment-analysis")

    with mlflow.start_run():
        # Parametreleri kaydet
        mlflow.log_param("vectorizer", "TfidfVectorizer")
        mlflow.log_param("max_features", 5000)

        # Eğit
        X_train_vec = self.vectorizer.fit_transform(X_train)
        self.estimator.fit(X_train_vec, y_train)

        # Metrikleri kaydet
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))

        # Modeli kaydet
        mlflow.sklearn.log_model(self.estimator, "model")
```

`http://localhost:5000` adresinde her run'ın parametrelerini, metriklerini ve model artifactlarını görebiliyorsun. "v1.2 neden v1.1'den kötü?" sorusuna bir bakışta cevap veriyorsun.

### 4. Prometheus Metrics — Production'da Ne Oluyor?

Codility'de monitoring yoktu. Production'da olmadan olmaz:

```python
# sentiment.py router'ında
PREDICTIONS = Counter("sentiment_predictions_total", "Total predictions", ["label"])
PREDICTION_LATENCY = Histogram("sentiment_prediction_duration_seconds", "Prediction latency")

@router.post("/predict")
def predict(payload: PredictionInput):
    start = time.time()
    sentiments = model.predict_batch(payload.texts)
    for s in sentiments:
        PREDICTIONS.labels(label=s).inc()  # Her tahmin sayılıyor
    PREDICTION_LATENCY.observe(time.time() - start)  # Latency ölçülüyor
    return results
```

`/prometheus` endpoint'i Prometheus'a bu metrikleri sağlıyor. Grafana bunu görselleştiriyor.

### 5. GitHub Actions — Her Push Otomatik Deploy

```yaml
jobs:
  test:
    steps:
      - run: pytest tests/ -v

  build:
    needs: test  # Test geçmeden devam etmez
    steps:
      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          push: true
          tags: ghcr.io/${{ github.repository }}:${{ github.sha }}

  deploy:
    needs: build
    steps:
      - name: Deploy via SSH
        run: |
          docker compose pull api
          docker compose up -d api
```

Test geçmeden image build olmaz. Build olmadan deploy olmaz. Bu zincirleme garantiyi manuel süreçlerle sağlamak neredeyse imkansız.

---

## Testler — Codility'de Yoktu, Burada Var

```bash
$ pytest tests/ -v

tests/test_vehicle_db.py::test_create_and_read PASSED
tests/test_vehicle_db.py::test_create_duplicate_raises PASSED
tests/test_vehicle_db.py::test_sell_insufficient_raises PASSED
tests/test_sentiment.py::test_model_train_and_predict PASSED
tests/test_sentiment.py::test_prepare_data_filters_shipping PASSED
...
```

Her edge case test edilmiş durumda. Yeni bir geliştirici gelip `Car(num_doors=True)` gönderirse test bunu yakalar, CI pipeline'ı kırar ve deploy olmaz.

---

## Çalıştırmak İçin

```bash
git clone https://github.com/your-username/vehicle-sentiment-mlops
cd vehicle-sentiment-mlops
docker compose up --build
```

- **API docs:** http://localhost:8000/docs
- **MLflow:** http://localhost:5000
- **Grafana:** http://localhost:3000 (admin / mlops123)
- **Prometheus:** http://localhost:9090

---

## Codility vs Production: Gerçek Fark

| | Codility | Bu proje |
|---|---|---|
| Validation | `__post_init__` | `__post_init__` + edge case testleri |
| Model saklama | In-memory | MLflow registry |
| Experiment tracking | Yok | MLflow runs + parameters + metrics |
| API | FastAPI skeleton | FastAPI + Pydantic + dependency injection |
| Monitoring | Yok | Prometheus + Grafana dashboards |
| CI/CD | Yok | GitHub Actions: test → build → deploy |
| Test coverage | Yok | Unittest suite |
| Hata yönetimi | Basic | Custom exceptions + HTTP status codes |

---

## Ne Öğrendim?

Codility sınavı bir enstantane. Düşünme hızını, temel kavramları ve kod yazabilme yeteneğini ölçüyor. %98 almak güzel.

Ama MLOps mühendisliği farklı bir oyun. Kodun çalışması değil, **güvenilir kalması** hedef. Model restartta kaybolmaz, deploy pipeline'ı insana bağlı değil, accuracy düştüğünde Grafana alarm veriyor.

Codility beni test etti. Bu proje beni yetiştirdi.

---

*Taha Bagatur, Senior MLOps & DevOps Engineer. CI/CD pipeline'larından NVIDIA Jetson GPU altyapısına kadar production AI sistemleri üzerine çalışıyor. LinkedIn: linkedin.com/in/taha-bagatur*

---
---

# From Codility to Production: How to Build a Real MLOps Project
### How I transformed two interview tasks into a production-ready system

*Taha Bagatur — MLOps & DevOps Engineer*

---

Last week I took a Codility technical assessment for ING Hubs Turkey. Two tasks, 105 minutes, 98% result.

But after the assessment ended, I asked myself one question: **"What would I actually need to take this code to production?"**

This article is the answer. I'm taking the two Codility tasks and building a real MLOps system from scratch — with Docker, MLflow, Prometheus, Grafana, and GitHub Actions.

Code on GitHub: `github.com/your-username/vehicle-sentiment-mlops`

---

## What Did I Build on Codility?

**Task 1 — Vehicle Inventory Management:**
Python `@dataclass` vehicle inventory system. `Vehicle` base class, `Car` and `Motorcycle` subclasses, validation logic, CRUD operations, sell/restock functionality.

**Task 2 — Sentiment Analysis API:**
FastAPI + scikit-learn sentiment analysis. Text cleaning pipeline, TF-IDF vectorization, logistic regression, `/predict` and `/metrics` endpoints.

Both worked on Codility. But they were isolated systems and not production-ready.

---

## Project Architecture

```
vehicle-sentiment-mlops/
├── app/
│   ├── api/          # FastAPI routers + dependency injection
│   ├── models/       # Dataclasses + custom exceptions
│   ├── services/     # VehicleDatabase + SentimentModel
│   └── schemas/      # Pydantic models
├── monitoring/       # Prometheus + Grafana config
├── .github/workflows/  # CI/CD pipeline
├── data/training.csv
├── Dockerfile
└── docker-compose.yml
```

The entire system starts with one command:
```bash
docker compose up --build
```

---

## Layer by Layer: How It Works

### 1. Data Models — Stronger Than Codility

The `Vehicle` and `Car` classes are the same logic from Codility, hardened for production:

```python
@dataclass
class Car(Vehicle):
    num_doors: int = 4
    is_electric: bool = False

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.num_doors, int) or isinstance(self.num_doors, bool) or self.num_doors <= 0:
            raise VehicleValidationError("num_doors must be a positive integer")
```

The `isinstance(x, bool)` check is critical: in Python, `bool` is a subclass of `int`. Passing `num_doors=True` would pass as `num_doors=1` without this guard.

### 2. Service Layer — Dependency Injection

On Codility everything was in one file. Here I separated concerns with dependency injection:

```python
def get_vehicle_db() -> VehicleDatabase:
    global _vehicle_db
    if _vehicle_db is None:
        _vehicle_db = VehicleDatabase()
    return _vehicle_db
```

This pattern is critical for testability — in tests you can inject a mock database.

### 3. MLflow — Experiment Tracking

On Codility the model was trained on startup and kept in memory. Fine but not enough. In production, every training run needs to be tracked:

```python
def train(self, data: pd.DataFrame) -> dict:
    with mlflow.start_run():
        mlflow.log_param("max_features", 5000)
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.sklearn.log_model(self.estimator, "model")
```

At `http://localhost:5000` you can see every run's parameters, metrics, and model artifacts. "Why is v1.2 worse than v1.1?" — answered in one glance.

### 4. Prometheus Metrics — What's Happening in Production?

Codility had no monitoring. Production can't survive without it:

```python
PREDICTIONS = Counter("sentiment_predictions_total", "Total predictions", ["label"])
PREDICTION_LATENCY = Histogram("sentiment_prediction_duration_seconds", "Prediction latency")

@router.post("/predict")
def predict(payload: PredictionInput):
    start = time.time()
    sentiments = model.predict_batch(payload.texts)
    for s in sentiments:
        PREDICTIONS.labels(label=s).inc()
    PREDICTION_LATENCY.observe(time.time() - start)
```

The `/prometheus` endpoint feeds these metrics to Prometheus. Grafana visualizes them. When the positive/negative prediction ratio starts shifting, you see it on the dashboard before users complain.

### 5. GitHub Actions — Every Push Auto-Deploys

```yaml
jobs:
  test:
    steps:
      - run: pytest tests/ -v

  build:
    needs: test  # Won't continue if tests fail
    steps:
      - uses: docker/build-push-action@v5
        with:
          push: true
          tags: ghcr.io/${{ github.repository }}:${{ github.sha }}

  deploy:
    needs: build
    steps:
      - run: docker compose up -d api
```

No image build without passing tests. No deploy without a successful build. Enforcing this chain manually is nearly impossible at scale.

---

## Tests — Didn't Exist on Codility, Exist Here

```bash
$ pytest tests/ -v

tests/test_vehicle_db.py::test_create_and_read PASSED
tests/test_vehicle_db.py::test_sell_insufficient_raises PASSED
tests/test_sentiment.py::test_model_train_and_predict PASSED
tests/test_sentiment.py::test_prepare_data_filters_shipping PASSED
```

Every edge case is covered. A new developer passing `Car(num_doors=True)` will hit a failing test, which breaks the CI pipeline, which stops the deploy.

---

## Running It

```bash
git clone https://github.com/your-username/vehicle-sentiment-mlops
cd vehicle-sentiment-mlops
docker compose up --build
```

- **API docs:** http://localhost:8000/docs
- **MLflow:** http://localhost:5000
- **Grafana:** http://localhost:3000 (admin / mlops123)
- **Prometheus:** http://localhost:9090

---

## Codility vs Production: The Real Difference

| | Codility | This project |
|---|---|---|
| Validation | `__post_init__` | `__post_init__` + edge case tests |
| Model storage | In-memory | MLflow registry |
| Experiment tracking | None | MLflow runs + parameters + metrics |
| API | FastAPI skeleton | FastAPI + Pydantic + dependency injection |
| Monitoring | None | Prometheus + Grafana dashboards |
| CI/CD | None | GitHub Actions: test → build → deploy |
| Test coverage | None | Full unittest suite |
| Error handling | Basic | Custom exceptions + HTTP status codes |

---

## What I Learned

A Codility assessment is a snapshot. It measures thinking speed, fundamentals, and the ability to write code. Scoring 98% is nice.

But MLOps engineering is a different game. The goal isn't for code to work — it's for it to **stay reliable**. The model doesn't disappear on restart. The deploy pipeline doesn't depend on a person remembering the right steps. When accuracy drops, Grafana alerts before users notice.

Codility tested me. This project trained me.

---

*Taha Bagatur is a Senior MLOps & DevOps Engineer working on production AI systems — from CI/CD pipelines and NVIDIA Jetson GPU infrastructure to FastAPI services and Prometheus monitoring. LinkedIn: linkedin.com/in/taha-bagatur*
