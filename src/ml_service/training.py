import subprocess
from datetime import datetime, UTC
from pathlib import Path
import json
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from .config import settings
from .logging import logger
from .data import load_split

MODELS_DIR = Path("models"); MODELS_DIR.mkdir(exist_ok=True)
METRICS_DIR = Path("metrics"); METRICS_DIR.mkdir(exist_ok=True)
PLOTS_DIR = Path("plots/tb_logs"); PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _build_pipeline(max_iter: int) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english")),
            (
                "clf",
                LogisticRegression(max_iter=max_iter, n_jobs=-1, solver="lbfgs"),
            ),
        ]
    )


def git_hash() -> str:
    """short SHA of current HEAD"""
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode()
        .strip()
    )

def train() -> Path:
    X_tr, X_val, y_tr, y_val = load_split()
    pipe = _build_pipeline(max_iter=settings.LOGISTIC_REGRESSION_ITERATIONS)

    writer = SummaryWriter(PLOTS_DIR / settings.EXPERIMENT)
    logger.info("training_started", experiment=settings.EXPERIMENT)

    pipe.fit(X_tr, y_tr)
    prob = pipe.predict_proba(X_val)[:, 1]
    writer.add_scalar("val/roc_auc", roc_auc_score(y_val, prob), 0)
    writer.add_scalar(
        "val/accuracy", accuracy_score(y_val, pipe.predict(X_val)), 0
    )
    writer.flush(); writer.close()

    model_path = MODELS_DIR / f"{settings.EXPERIMENT}.onnx"

    onx = convert_sklearn(
        pipe,
        initial_types=[("input", StringTensorType([None, 1]))],
        options={id(pipe): {"zipmap": False}},
        target_opset=settings.ONNX_OPSET,
    )

    for k, v in {
        "git_commit": git_hash(),
        "saved_at": datetime.now(UTC).isoformat(),
        "experiment": settings.EXPERIMENT,
    }.items():
        p = onx.metadata_props.add(); p.key, p.value = k, v

    onnx.save(onx, model_path)
    logger.info("model_saved", path=str(model_path))

    metrics = {
        "roc_auc": roc_auc_score(y_val, prob),
        "accuracy": accuracy_score(y_val, pipe.predict(X_val)),
    }
    with open(METRICS_DIR / "train.json", "w") as fp:
        json.dump(metrics, fp, indent=2)

    return model_path


if __name__ == "__main__":
    train()