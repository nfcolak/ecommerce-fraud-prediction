"""Per-model artefacts: booster and per-row scores under models_dir()/<name>/."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]


def models_dir() -> Path:
    return Path(os.environ.get("MODELS_DIR", REPO / "models"))


def save_artifacts(name, booster_or_None, ids_valid, scores_valid, ids_test_or_None,
                   scores_test_or_None):
    out = models_dir() / name
    out.mkdir(parents=True, exist_ok=True)
    if booster_or_None is not None:
        booster_or_None.save_model(str(out / "booster.txt"))
    pd.DataFrame({"TransactionID": ids_valid, "score": scores_valid}).to_parquet(
        out / "scores_valid.parquet", index=False)
    if ids_test_or_None is not None:
        pd.DataFrame({"TransactionID": ids_test_or_None, "score": scores_test_or_None}
                     ).to_parquet(out / "scores_test.parquet", index=False)
    return out
