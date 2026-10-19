"""Block graph_emb: 32-dim inductive temporal-GraphSAGE embedding + score (graph_encoder.py).

USES_LABELS is False: the encoder never receives a label as an input or message. Labels are only the
BCE training target of the fit-window rows of the run whose embed.parquet is loaded here; every
embedding/score is computed from the row's own features and its strictly earlier entity neighbourhood.

build(df, fit_end_dt) loads models_dir()/<run>/embed.parquet of the run whose run_info.json has
fit_end_dt equal to the requested one (several valid seeds share it: the lowest seed is used unless env
GRAPH_EMB_RUN names a run). Rows the run did not embed (gap rows, rows outside fit/scored windows) get
NaN. No matching run raises FileNotFoundError.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

NAME = "graph_emb"
DESCRIPTION = ("GraphSAGE (hetero, temporal, past-only) 32-dim embedding + score; labels only as the "
               "encoder's fit-window training target, never as input")
USES_LABELS = False


def find_run(fit_end_dt: int):
    """(run_name, run_dir) for fit_end_dt, or raises FileNotFoundError."""
    from ecommerce_fraud.artifacts import models_dir
    root = models_dir()
    forced = os.environ.get("GRAPH_EMB_RUN")
    cands = []
    for info in sorted(root.glob("*/run_info.json")):
        try:
            meta = json.loads(info.read_text())
        except ValueError:
            continue
        if meta.get("fit_end_dt") is None or "epochs_run" not in meta:
            continue
        if int(meta["fit_end_dt"]) == int(fit_end_dt) and (info.parent / "embed.parquet").exists():
            cands.append((int(meta.get("seed", 0)), info.parent.name, info.parent))
    if forced:
        cands = [c for c in cands if c[1] == forced]
    if not cands:
        raise FileNotFoundError(
            f"graph_emb: no graph_encoder run with fit_end_dt={int(fit_end_dt)} and embed.parquet under {root}; "
            f"train one first: python -m ecommerce_fraud.graph_encoder --window valid|fold:S-E --name gnn_...")
    cands.sort()
    return cands[0][1], cands[0][2]


def build(df: pd.DataFrame, fit_end_dt: int) -> pd.DataFrame:
    _, run_dir = find_run(fit_end_dt)
    emb = pd.read_parquet(run_dir / "embed.parquet")
    emb.columns = ["TransactionID"] + [c.replace("gnn_emb__", f"{NAME}__", 1) for c in emb.columns[1:]]
    ids = df["TransactionID"].to_numpy()
    out = emb.set_index("TransactionID").reindex(ids)
    out = out.astype(np.float32).reset_index()
    out.columns = ["TransactionID"] + list(emb.columns[1:])
    return out
