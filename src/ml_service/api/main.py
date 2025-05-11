from pathlib import Path
import onnx
from fastapi import FastAPI, HTTPException
from fastapi import UploadFile, File, status
from fastapi.responses import JSONResponse

from src.ml_service.api.file_process import load_dataset
from src.ml_service.api.models import (
    TextRequest, PredictionResponse, MetaResponse, ForwardBatchResponse,
    EvaluationResponse,
)
from src.ml_service.inference import MODEL
from src.ml_service.logging import logger

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)
app = FastAPI(title="Spam‑ONNX API", version="1.0.0")


@app.post("/forward", response_model=PredictionResponse)
async def forward(payload: TextRequest):
    try:
        label, prob = MODEL.predict(payload.text)
        return {"label": label, "probability": prob}
    except Exception:
        logger.exception("inference_failed")
        raise HTTPException(403, "model can not process data")


@app.get("/metadata", response_model=MetaResponse)
async def metadata():
    meta = {
        p.key: p.value
        for p in onnx.load(Path(MODEL.path)).metadata_props
    }
    return meta


@app.post("/forward_batch", response_model=ForwardBatchResponse)
async def forward_batch(file: UploadFile = File(...)):
    try:
        df = load_dataset(file)
    except Exception as exc:
        logger.warning("dataset_load_failed", error=str(exc))
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "bad dataset") from exc

    if "text" not in df.columns:
        raise HTTPException(400, "column 'text' missing")

    labels, probs = MODEL.predict_batch(df["text"].tolist())
    preds = [{"index": i, "label": l, "probability": p}
             for i, (l, p) in enumerate(zip(labels, probs))]
    return JSONResponse({"predictions": preds})


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(file: UploadFile = File(...)):
    df = load_dataset(file)
    if {"text", "label"} - set(df.columns):
        raise HTTPException(400, "columns 'text' and 'label' required")
    y_true = df["label"].tolist()
    y_pred, y_prob = MODEL.predict_batch(df["text"].tolist())

    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary"
    )
    cm = confusion_matrix(y_true, y_pred).tolist()

    preds = [{"index": i, "label": y, "probability": p}
             for i, (y, p) in enumerate(zip(y_pred, y_prob))]
    metrics = {"accuracy": acc, "precision": pr, "recall": rc,
               "f1": f1, "confusion": cm}
    return JSONResponse({"predictions": preds, "metrics": metrics})