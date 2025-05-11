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