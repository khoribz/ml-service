from pathlib import Path
from typing import Sequence

import numpy as np

import onnxruntime as ort
from .config import settings


class SpamONNX:
    """Lightweight ONNXRuntime wrapper for spam classifier."""

    def __init__(self, model_path: Path):
        self.path = model_path
        self.sess = ort.InferenceSession(
            model_path.as_posix(), providers=["CPUExecutionProvider"]
        )
        self.proba_out = self.sess.get_outputs()[1].name

    def predict(self, text: str):
        probas = self.sess.run(
            [self.proba_out], {"input": [[text]]}
        )[0][0]
        label = int(probas.argmax())
        return label, float(probas.max())

    def predict_batch(self, texts: Sequence[str]) -> tuple[list[int], np.ndarray]:
        """Return (labels, probabilities) for a batch."""
        probas = self.sess.run(
            [self.proba_out],
            {"input": np.array(texts, dtype=object).reshape(-1, 1)}
        )[0]
        labels = probas.argmax(1).tolist()
        confs = probas.max(1).tolist()
        return labels, confs

MODEL = SpamONNX(Path("models") / f"{settings.EXPERIMENT}.onnx")