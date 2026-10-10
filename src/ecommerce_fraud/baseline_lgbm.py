"""Raw-feature LightGBM baseline with a time split.

Run: python -m ecommerce_fraud.baseline_lgbm  (PYTHONPATH=src)
"""
from __future__ import annotations

import json
import platform
import resource
import subprocess
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

from ecommerce_fraud.data import RAW_DIR, WORKTREE, load_joined, time_split

OUT_DIR = WORKTREE / "results" / "baseline_lgbm"
DROP = ["TransactionID", "isFraud", "TransactionDT"]
PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "seed": 42,
    "learning_rate": 0.05,
    "num_leaves": 256,
    "min_child_samples": 100,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "n_jobs": 12,
    "verbosity": -1,
}
MAX_ROUNDS = 3000
EARLY_STOP = 100


def prepare(train, valid, test):
    feats = [c for c in train.columns if c not in DROP]
    obj_cols = [c for c in feats if not pd.api.types.is_numeric_dtype(train[c])
                and not pd.api.types.is_bool_dtype(train[c])]
    X = {k: v[feats].copy() for k, v in (("train", train), ("valid", valid), ("test", test))}
    for c in obj_cols:
        cats = pd.Index(train[c].dropna().unique()).sort_values()
        dtype = pd.CategoricalDtype(categories=cats)
        for k in X:
            X[k][c] = X[k][c].astype(dtype)  # unseen values -> NaN
    return X, feats, obj_cols


def top_k(y, p, frac):
    k = max(1, int(np.ceil(len(y) * frac)))
    idx = np.argsort(-p, kind="stable")[:k]
    tp = float(y[idx].sum())
    return tp / k, tp / float(y.sum())


def metrics(y, p):
    out = {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "fraud_rate": float(y.mean()),
        "n_rows": int(len(y)),
    }
    for frac, tag in ((0.01, "1pct"), (0.05, "5pct")):
        pr, rc = top_k(y, p, frac)
        out[f"precision_at_top_{tag}"] = pr
        out[f"recall_at_top_{tag}"] = rc
    return out


def git_sha():
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=WORKTREE,
                          capture_output=True, text=True).stdout.strip()


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_joined()
    train, valid, test, stats = time_split(df)
    print("split", json.dumps(stats), flush=True)
    X, feats, obj_cols = prepare(train, valid, test)
    y = {"train": train["isFraud"].to_numpy(), "valid": valid["isFraud"].to_numpy(),
         "test": test["isFraud"].to_numpy()}
    print(f"{len(feats)} features, {len(obj_cols)} categorical", flush=True)

    dtrain = lgb.Dataset(X["train"], y["train"])
    dvalid = lgb.Dataset(X["valid"], y["valid"], reference=dtrain)
    booster = lgb.train(
        PARAMS, dtrain, num_boost_round=MAX_ROUNDS, valid_sets=[dvalid], valid_names=["valid"],
        callbacks=[lgb.early_stopping(EARLY_STOP), lgb.log_evaluation(50)],
    )
    best = int(booster.best_iteration)
    m_valid = metrics(y["valid"], booster.predict(X["valid"], num_iteration=best))
    m_test = metrics(y["test"], booster.predict(X["test"], num_iteration=best))
    result = {"model": "lgbm_raw", "best_iteration": best, "split": stats,
              "valid": m_valid, "test": m_test}
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
        "n_categorical_features": len(obj_cols),
        "runtime_seconds": round(time.time() - t0, 1),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 1),
        "data_file_sizes_bytes": raw_sizes,
    }
    (OUT_DIR / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print(f"runtime {info['runtime_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
