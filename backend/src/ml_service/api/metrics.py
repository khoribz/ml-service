from prometheus_client import Summary, Gauge

PRED_LATENCY = Summary(
    "prediction_latency_seconds",
    "Time spent in MODEL.predict",
)
BATCH_SIZE = Gauge(
    "batch_request_size",
    "Rows per /forward_batch or /evaluate call",
)
ACTIVE_EXPERIMENT = Gauge(
    "active_experiment_id",
    "ID of model that served the last request",
)