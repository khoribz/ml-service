import io

import pandas as pd
from fastapi import UploadFile


def _load_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def load_dataset(upload: UploadFile) -> pd.DataFrame:
    if upload.filename.endswith(".csv"):
        return _load_csv(upload.file.read())
    raise ValueError("Only .csv  accepted")
