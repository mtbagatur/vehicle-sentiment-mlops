import re
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

FILTER_KEYWORD = "shipping"
MAX_TEXT_LENGTH = 100


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    words = [w for w in text.split() if w not in ENGLISH_STOP_WORDS]
    return ' '.join(words).strip()


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    df = df[~df['text'].str.contains(FILTER_KEYWORD, case=False, na=False)]
    df = df[df['text'].str.len() <= MAX_TEXT_LENGTH]
    return df.reset_index(drop=True)


def prepare_data(df: pd.DataFrame) -> tuple[list[str], list[int]]:
    df = filter_dataframe(df.copy())
    texts = df['text'].apply(clean_text).tolist()
    labels = df['sentiment'].map(
        lambda s: 1 if s == 'positive' else 0
    ).tolist()
    return texts, labels
