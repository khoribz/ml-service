import json
import threading
from pathlib import Path

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from src.ml_service.config import settings

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "experiments"
EXP_DIR.mkdir(exist_ok=True)

_next_id_lock = threading.Lock()


def _next_id() -> int:
    path = EXP_DIR / "next_id.txt"
    with _next_id_lock:
        current = int(path.read_text()) if path.exists() else 1
        path.write_text(str(current + 1))
    return current


def save_experiment(model_path: Path, metrics: dict, params: dict) -> int:
    eid = _next_id()
    run = EXP_DIR / f"{eid:04d}"
    run.mkdir()
    (run / "model.onnx").write_bytes(model_path.read_bytes())
    json.dump(metrics, open(run / "metrics.json", "w"), indent=2)
    json.dump(params, open(run / "params.yaml", "w"))

    reg = CollectorRegistry()

    Gauge(
        "experiment_roc_auc",
        "roc_auc for given experiment",
        ["experiment_id"],
        registry=reg,
    ).labels(experiment_id=eid).set(metrics["roc_auc"])

    Gauge(
        "experiment_accuracy",
        "accuracy for given experiment",
        ["experiment_id"],
        registry=reg,
    ).labels(experiment_id=eid).set(metrics["accuracy"])

    Gauge("experiment_id", "Index of experiment", registry=reg).set(eid)

    push_to_gateway(
        settings.PUSHGATEWAY_URL,
        job="retrain_job",
        grouping_key={"experiment_id": str(eid)},
        registry=reg,
    )

    return eid


def load_metrics(eid: int) -> dict:
    return json.load(open(EXP_DIR / f"{eid:04d}" / "metrics.json"))


def deploy(eid: int) -> None:
    run = EXP_DIR / f"{eid:04d}"
    prod_path = ROOT / "models/current.onnx"
    prod_path.write_bytes((run / "model.onnx").read_bytes())
    (ROOT / "models/active_experiment.txt").write_text(str(eid))
