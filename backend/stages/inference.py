"""DVC stage — run model on full dataset and dump confusion‑matrix"""

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix

from src.ml_service.data import download_raw
from src.ml_service.inference import MODEL

raw = pd.read_csv(download_raw(), sep="\t", names=["label", "text"])
raw["label"] = raw["label"].map({"ham": 0, "spam": 1})

y_pred = []
for txt in raw["text"]:
    lbl, _ = MODEL.predict(txt)
    y_pred.append(lbl)

cm = confusion_matrix(raw["label"], y_pred).tolist()
Path("metrics").mkdir(exist_ok=True)
json.dump({"confusion": cm}, open("metrics/infer.json", "w"))
json.dump({"cm": cm}, open("plots/confusion.json", "w"))
