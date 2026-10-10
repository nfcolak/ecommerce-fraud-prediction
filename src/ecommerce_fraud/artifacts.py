"""Persist boosters and per-row scores under models_dir() (env MODELS_DIR, default <repo>/models)."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

WORKTREE = Path(__file__).resolve().parents[2]


def models_dir() -> Path:
    return Path(os.environ.get("MODELS_DIR", WORKTREE / "models"))


def _write_scores(path: Path, ids, scores) -> None:
    pd.DataFrame({"TransactionID": list(ids) if not hasattr(ids, "to_numpy") else ids.to_numpy(),
                  "score": scores}).to_parquet(path, index=False)


def save_artifacts(name, booster_or_None, ids_valid, scores_valid, ids_test_or_None,
                   scores_test_or_None) -> Path:
    out = models_dir() / name
    out.mkdir(parents=True, exist_ok=True)
    if booster_or_None is not None:
        booster_or_None.save_model(str(out / "booster.txt"))
    _write_scores(out / "scores_valid.parquet", ids_valid, scores_valid)
    if ids_test_or_None is not None and scores_test_or_None is not None:
        _write_scores(out / "scores_test.parquet", ids_test_or_None, scores_test_or_None)
    return out
