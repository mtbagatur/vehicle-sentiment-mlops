import pytest
import pandas as pd
from app.services.sentiment_model import SentimentModel
from app.services.preprocessor import clean_text, prepare_data
from app.models.errors import ModelNotReadyError


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "text": [
            "I love this product it is amazing",
            "This is terrible I hate it",
            "Great quality highly recommend",
            "Worst purchase ever total waste",
            "Excellent service very happy",
            "Broken arrived damaged awful",
            "Best product outstanding quality",
            "Never buying again complete disappointment",
        ],
        "sentiment": [
            "positive", "negative", "positive", "negative",
            "positive", "negative", "positive", "negative",
        ]
    })


def test_clean_text_lowercase():
    assert clean_text("HELLO WORLD") == "hello world"


def test_clean_text_removes_punctuation():
    result = clean_text("Hello, world!")
    assert "," not in result
    assert "!" not in result


def test_clean_text_removes_numbers():
    result = clean_text("I rated it 5 out of 10")
    assert "5" not in result
    assert "10" not in result


def test_prepare_data_filters_shipping(sample_df):
    df_with_shipping = pd.concat([
        sample_df,
        pd.DataFrame({"text": ["shipping was fast"], "sentiment": ["positive"]})
    ])
    texts, labels = prepare_data(df_with_shipping)
    assert not any("shipping" in t for t in texts)


def test_prepare_data_filters_long_text(sample_df):
    long_text = "a" * 150
    df_with_long = pd.concat([
        sample_df,
        pd.DataFrame({"text": [long_text], "sentiment": ["positive"]})
    ])
    texts, labels = prepare_data(df_with_long)
    assert not any(len(t) > 100 for t in texts)


def test_model_not_trained_raises():
    model = SentimentModel()
    with pytest.raises(ModelNotReadyError):
        model.predict("test")


def test_model_train_and_predict(sample_df):
    model = SentimentModel()
    metrics = model.train(sample_df)
    assert model.is_trained
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    result = model.predict("I love this amazing product")
    assert result in ["positive", "negative"]


def test_model_predict_batch(sample_df):
    model = SentimentModel()
    model.train(sample_df)
    results = model.predict_batch([
        "I love this product",
        "This is terrible",
    ])
    assert len(results) == 2
    assert all(r in ["positive", "negative"] for r in results)
