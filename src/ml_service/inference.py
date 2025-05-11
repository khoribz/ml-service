from pathlib import Path
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


MODEL = SpamONNX(Path("models") / f"{settings.EXPERIMENT}.onnx")