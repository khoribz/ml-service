from pathlib import Path

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from .config import settings
from .logging import logger

RAW_PATH = Path("data/raw/sms.tsv")


def download_raw() -> Path:
    """Download dataset if absent and return path."""
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not RAW_PATH.exists():
        logger.info("downloading_dataset", url=settings.DATA_URL)
        resp = requests.get(settings.DATA_URL, timeout=30)
        resp.raise_for_status()
        RAW_PATH.write_bytes(resp.content)
    return RAW_PATH


def load_split(test_size: float = 0.2, random_state: int = 42):
    df = pd.read_csv(download_raw(), sep="\t", names=["label", "text"])
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    return train_test_split(
        df["text"],
        df["label"],
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state,
    )


def load_df(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(
        df["text"],
        df["label"],
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state,
    )
