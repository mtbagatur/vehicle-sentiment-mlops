import mlflow
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from app.services.preprocessor import prepare_data, clean_text
from app.models.errors import ModelNotReadyError


class SentimentModel:
    """Sentiment classifier using TF-IDF + Logistic Regression with MLflow tracking."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.estimator = LogisticRegression(max_iter=1000, random_state=42)
        self._metrics: dict | None = None
        self._is_trained = False

    def train(self, data: pd.DataFrame, experiment_name: str = "sentiment-analysis") -> dict:
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run():
            texts, labels = prepare_data(data)

            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels, test_size=0.2, random_state=42
            )

            mlflow.log_param("vectorizer", "TfidfVectorizer")
            mlflow.log_param("max_features", 5000)
            mlflow.log_param("model", "LogisticRegression")
            mlflow.log_param("max_iter", 1000)
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))

            X_train_vec = self.vectorizer.fit_transform(X_train)
            self.estimator.fit(X_train_vec, y_train)

            X_test_vec = self.vectorizer.transform(X_test)
            y_pred = self.estimator.predict(X_test_vec)

            self._metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            }

            for k, v in self._metrics.items():
                mlflow.log_metric(k, v)

            mlflow.sklearn.log_model(self.estimator, "model")
            mlflow.sklearn.log_model(self.vectorizer, "vectorizer")

            self._is_trained = True
            return self._metrics

    def predict(self, text: str) -> str:
        if not self._is_trained:
            raise ModelNotReadyError("Model has not been trained yet")
        cleaned = clean_text(text)
        vec = self.vectorizer.transform([cleaned])
        prediction = self.estimator.predict(vec)[0]
        return "positive" if prediction == 1 else "negative"

    def predict_batch(self, texts: list[str]) -> list[str]:
        if not self._is_trained:
            raise ModelNotReadyError("Model has not been trained yet")
        cleaned = [clean_text(t) for t in texts]
        vecs = self.vectorizer.transform(cleaned)
        predictions = self.estimator.predict(vecs)
        return ["positive" if p == 1 else "negative" for p in predictions]

    @property
    def metrics(self) -> dict:
        if self._metrics is None:
            raise ModelNotReadyError("Model has not been trained yet")
        return self._metrics

    @property
    def is_trained(self) -> bool:
        return self._is_trained
