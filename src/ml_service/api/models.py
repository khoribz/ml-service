from pydantic import BaseModel, Field


class TextRequest(BaseModel):
    text: str = Field(..., example="Free prize! Click here.")


class PredictionResponse(BaseModel):
    label: int
    probability: float


class MetaResponse(BaseModel):
    git_commit: str
    saved_at: str
    experiment: str


class BatchPrediction(BaseModel):
    index: int
    label: int
    probability: float

class ForwardBatchResponse(BaseModel):
    predictions: list[BatchPrediction]

class EvaluationResponse(ForwardBatchResponse):
    metrics: dict[str, float]