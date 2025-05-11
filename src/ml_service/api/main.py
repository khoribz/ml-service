from pathlib import Path
import onnx
from fastapi import FastAPI, HTTPException
from src.ml_service.api.models import TextRequest, PredictionResponse, MetaResponse
from src.ml_service.inference import MODEL
from src.ml_service.logging import logger

app = FastAPI(title="Spam‑ONNX API", version="1.0.0")


@app.post("/forward", response_model=PredictionResponse)
async def forward(payload: TextRequest):
    try:
        label, prob = MODEL.predict(payload.text)
        return {"label": label, "probability": prob}
    except Exception:
        logger.exception("inference_failed")
        raise HTTPException(403, "модель не смогла обработать данные")


@app.get("/metadata", response_model=MetaResponse)
async def metadata():
    meta = {
        p.key: p.value
        for p in onnx.load(Path(MODEL.path)).metadata_props
    }
    return meta