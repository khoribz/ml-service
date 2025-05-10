import joblib
from pathlib import Path
from typing import Tuple

MODEL_PATH = Path(__file__).parents[1] / "models" / "spam_classifier.joblib"
_spam_model = joblib.load(MODEL_PATH)

def predict_text(text: str) -> Tuple[int, float]:
    """Возвращает (label, probability). label: 1 — spam, 0 — ham"""
    proba = _spam_model.predict_proba([text])[0]
    label = int(proba.argmax())
    return label, float(proba.max())
