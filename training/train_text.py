import subprocess
from datetime import datetime, UTC

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
import onnx

DATA_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
MODEL_DIR = Path(__file__).parents[1] / "models"
MODEL_DIR.mkdir(exist_ok=True)
LABEL_MAP = {"ham": 0, "spam": 1}
REVERSE_LABEL_MAP = {0: "ham", 1: "spam"}

def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_URL, sep="\t", names=["label", "text"])
    df["label"] = df["label"].map(LABEL_MAP)
    return df

def train() -> Pipeline:
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
    )
    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english")),
            ("clf", MultinomialNB()),
        ]
    )
    pipe.fit(X_train, y_train)
    print(classification_report(y_test, pipe.predict(X_test)))
    return pipe

def git_hash() -> str:
    """short SHA of current HEAD"""
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode()
        .strip()
    )

def export_onnx(pipe: Pipeline, exp_name: str):
    onx = convert_sklearn(
        pipe,
        initial_types=[("input", StringTensorType([None, 1]))],
        options={id(pipe): {"zipmap": False}},
        target_opset=12
    )
    meta = {
        "git_commit": git_hash(),
        "saved_at": datetime.now(UTC).isoformat(),
        "experiment": exp_name,
    }
    for k, v in meta.items():
        p = onx.metadata_props.add()
        p.key, p.value = k, v
    onnx.save(onx, MODEL_DIR / "spam_classifier.onnx")

if __name__ == "__main__":
    pipe = train()
    export_onnx(pipe, exp_name="baseline-tfidf-nb")
