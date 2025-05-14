from pathlib import Path

from pydantic.v1 import BaseSettings


class Settings(BaseSettings):
    """App configuration pulled from environment variables or .env file."""

    DATA_URL: str = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    EXPERIMENT: str = "lr-tfidf-baseline"
    ONNX_OPSET: int = 12
    LOGISTIC_REGRESSION_ITERATIONS = 400
    PUSHGATEWAY_URL: str = "pushgateway:9091"
    GIT_HASH: str = "unknown"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
