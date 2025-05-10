import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

import onnxruntime as ort
import onnx

from training.train_text import REVERSE_LABEL_MAP

app = FastAPI(title="ML Forward Service")

MODEL_PATH = Path(__file__).parents[1] / "models" / "spam_classifier.onnx"

RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)


def dump_result(payload: dict) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    fname = f"{stamp}_{uuid.uuid4().hex}.json"
    with open(RESULT_DIR / fname, "w") as f:
        f.write(json.dumps(payload, ensure_ascii=False, indent=2))


@app.post("/forward", response_class=JSONResponse)
async def forward(req: Request):
    data = await req.json()
    if "text" not in data:
        raise HTTPException(400, "bad request")
    input_text = data["text"]

    try:
        sess = ort.InferenceSession(MODEL_PATH.as_posix(), providers=['CPUExecutionProvider'])
        outputs = sess.run(None, {"input": [[input_text]]})
        probas = outputs[1][0]
        label = int(probas.argmax())
        result = {"label": REVERSE_LABEL_MAP[label], "probability": float(probas.max())}
        dump_result(result)
        return result
    except Exception as e:
        print(e)
        raise HTTPException(403, "модель не смогла обработать данные")


@app.get("/metadata", response_class=JSONResponse)
async def metadata():
    return {p.key: p.value for p in onnx.load(MODEL_PATH).metadata_props}
