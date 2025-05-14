import io
from pathlib import Path

import onnx
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.ml_service.api.file_process import load_dataset
from src.ml_service.api.metrics import ACTIVE_EXPERIMENT, BATCH_SIZE, PRED_LATENCY
from src.ml_service.api.models import (
    EvaluationResponse,
    ForwardBatchResponse,
    MetaResponse,
    PredictionResponse,
    TextRequest,
)
from src.ml_service.experiments import deploy, load_metrics, save_experiment
from src.ml_service.inference import MODEL
from src.ml_service.logging import logger
from src.ml_service.training import train

app = FastAPI(title="Spam‑ONNX API", version="1.0.0")

Instrumentator().instrument(app).expose(app)

DATA_CSV = Path("data/train_data.csv")
DATA_CSV.parent.mkdir(parents=True, exist_ok=True)
if not DATA_CSV.exists():
    DATA_CSV.write_text("text,label\n")


# ---- PUT /add_data --------------------------------------------------
@app.put("/add_data", status_code=200)
async def add_data(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only .csv accepted")
    df_new = pd.read_csv(io.BytesIO(await file.read()))
    if {"text", "label"} - set(df_new.columns):
        raise HTTPException(400, "Columns text,label required")
    df_new.to_csv(DATA_CSV, mode="a", index=False, header=False)
    return {"rows_added": len(df_new)}


# ---- PUT /retrain ---------------------------------------------------
@app.put("/retrain", status_code=200)
async def retrain(background_tasks: BackgroundTasks):
    def _job():
        df = pd.read_csv(DATA_CSV)
        model_path, metrics = train(df)
        eid = save_experiment(model_path, metrics, {"algo": "LogReg"})
        logger.info("experiment_saved", id=eid)

    background_tasks.add_task(_job)
    return {"message": "training started"}


# ---- GET /metrics/{id} ---------------------------------------------
@app.get("/metrics/{experiment_id}", status_code=200)
async def get_metrics(experiment_id: int):
    try:
        return load_metrics(experiment_id)
    except FileNotFoundError:
        raise HTTPException(404, "experiment not found")


# ---- POST /deploy/{id} ---------------------------------------------
@app.post("/deploy/{experiment_id}", status_code=200)
async def deploy_experiment(experiment_id: int):
    try:
        deploy(experiment_id)
        MODEL.__init__(Path("models/current.onnx"))
        return {"status": "deployed", "active_id": experiment_id}
    except FileNotFoundError:
        raise HTTPException(404, "experiment not found")


@app.post("/forward", response_model=PredictionResponse, status_code=200)
async def forward(payload: TextRequest):
    try:
        with PRED_LATENCY.time():
            label, prob = MODEL.predict(payload.text)
        ACTIVE_EXPERIMENT.set(MODEL.current_id)
        return {"label": label, "probability": prob}
    except Exception:
        logger.exception("inference_failed")
        raise HTTPException(403, "model can not process data")


@app.get("/metadata", response_model=MetaResponse, status_code=200)
async def metadata():
    meta = {p.key: p.value for p in onnx.load(Path(MODEL.path)).metadata_props}
    return meta


@app.post("/forward_batch", response_model=ForwardBatchResponse, status_code=200)
async def forward_batch(file: UploadFile = File(...)):
    try:
        df = load_dataset(file)
    except Exception as exc:
        logger.warning("dataset_load_failed", error=str(exc))
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "bad dataset") from exc

    BATCH_SIZE.set(len(df))

    if "text" not in df.columns:
        raise HTTPException(400, "column 'text' missing")

    labels, probs = MODEL.predict_batch(df["text"].tolist())
    preds = [
        {"index": i, "label": l, "probability": p}
        for i, (l, p) in enumerate(zip(labels, probs))
    ]
    return JSONResponse({"predictions": preds})


@app.post("/evaluate", response_model=EvaluationResponse, status_code=200)
async def evaluate(file: UploadFile = File(...)):
    df = load_dataset(file)
    BATCH_SIZE.set(len(df))

    if {"text", "label"} - set(df.columns):
        raise HTTPException(400, "columns 'text' and 'label' required")

    y_true = df["label"].tolist()
    y_pred, y_prob = MODEL.predict_batch(df["text"].tolist())

    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    cm = confusion_matrix(y_true, y_pred).tolist()

    preds = [
        {"index": i, "label": y, "probability": p}
        for i, (y, p) in enumerate(zip(y_pred, y_prob))
    ]
    metrics = {
        "accuracy": acc,
        "precision": pr,
        "recall": rc,
        "f1": f1,
        "confusion": cm,
    }
    return JSONResponse({"predictions": preds, "metrics": metrics})
