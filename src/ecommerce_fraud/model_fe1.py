"""LightGBM with past-only entity-history features (lgbm_fe1).

Same split, categorical handling, params, rounds and early stopping as baseline_lgbm.
Run: python -m ecommerce_fraud.model_fe1  (PYTHONPATH=src)
"""
from __future__ import annotations

import json
import platform
import resource
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow
import sklearn

from ecommerce_fraud.baseline_lgbm import (EARLY_STOP, MAX_ROUNDS, PARAMS, git_sha, metrics,
                                           prepare)
from ecommerce_fraud.data import RAW_DIR, WORKTREE, load_joined, time_split
from ecommerce_fraud.features_history import add_history_features, leakage_check

OUT_DIR = WORKTREE / "results" / "lgbm_fe1"


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_joined()
    # time_split sorts by TransactionDT (stable) and resets the index; do the same first
    # so row labels of the featured frame match the split frames.
    df = df.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    df, added = add_history_features(df)
    train, valid, test, stats = time_split(df)
    print("split", json.dumps(stats), flush=True)

    check = leakage_check(df, test.index, seed=0, n=3)
    print(check, flush=True)
    (OUT_DIR / "leakage_check.txt").write_text(check + "\n")

    X, feats, obj_cols = prepare(train, valid, test)
    y = {"train": train["isFraud"].to_numpy(), "valid": valid["isFraud"].to_numpy(),
         "test": test["isFraud"].to_numpy()}
    print(f"{len(feats)} features ({len(added)} new), {len(obj_cols)} categorical", flush=True)

    dtrain = lgb.Dataset(X["train"], y["train"])
    dvalid = lgb.Dataset(X["valid"], y["valid"], reference=dtrain)
    booster = lgb.train(
        PARAMS, dtrain, num_boost_round=MAX_ROUNDS, valid_sets=[dvalid], valid_names=["valid"],
        callbacks=[lgb.early_stopping(EARLY_STOP), lgb.log_evaluation(50)],
    )
    best = int(booster.best_iteration)
    m_valid = metrics(y["valid"], booster.predict(X["valid"], num_iteration=best))
    m_test = metrics(y["test"], booster.predict(X["test"], num_iteration=best))  # scored once
    result = {"model": "lgbm_fe1", "best_iteration": best, "split": stats,
              "valid": m_valid, "test": m_test, "features_added": added}
    (OUT_DIR / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")

    imp = pd.DataFrame({
        "feature": booster.feature_name(),
        "gain": booster.feature_importance("gain", iteration=best),
        "split": booster.feature_importance("split", iteration=best),
    }).sort_values("gain", ascending=False).head(50)
    imp.to_csv(OUT_DIR / "feature_importance.csv", index=False)

    raw_sizes = {f: (RAW_DIR / f).stat().st_size
                 for f in ("train_transaction.csv", "train_identity.csv")}
    info = {
        "git_sha": git_sha(),
        "versions": {"python": platform.python_version(), "lightgbm": lgb.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__,
                     "scikit-learn": sklearn.__version__, "pyarrow": pyarrow.__version__},
        "params": {**PARAMS, "max_rounds": MAX_ROUNDS, "early_stopping_rounds": EARLY_STOP},
        "n_features": len(feats),
        "n_features_added": len(added),
        "n_categorical_features": len(obj_cols),
        "runtime_seconds": round(time.time() - t0, 1),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 1),
        "data_file_sizes_bytes": raw_sizes,
    }
    (OUT_DIR / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print(f"runtime {info['runtime_seconds']}s peak_rss_mb {info['peak_rss_mb']}", flush=True)


if __name__ == "__main__":
    main()
