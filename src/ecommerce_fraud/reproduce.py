"""Retrain the three committed models with unchanged code paths/parameters, persist boosters and
per-row scores, and compare against the committed metrics.

Run: python -m ecommerce_fraud.reproduce  (PYTHONPATH=src). Writes only results/repro/.
"""
from __future__ import annotations

import gc
import json
import platform
import resource
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow
import sklearn

from ecommerce_fraud.artifacts import models_dir, save_artifacts
from ecommerce_fraud.baseline_lgbm import (DROP, EARLY_STOP, MAX_ROUNDS, PARAMS, git_sha,
                                           metrics, prepare)
from ecommerce_fraud.data import WORKTREE, load_joined, time_split
from ecommerce_fraud.features_history import add_history_features
from ecommerce_fraud.features_uid_agg import add_uid_aggregates
from ecommerce_fraud.model_fe2 import train_one

OUT_DIR = WORKTREE / "results" / "repro"
TOL = 1e-6


def committed(model_dir):
    m = json.loads((WORKTREE / "results" / model_dir / "metrics.json").read_text())
    return {"best_iteration": int(m["best_iteration"]), "valid_pr_auc": m["valid"]["pr_auc"],
            "test_pr_auc": m["test"]["pr_auc"]}


def score_and_save(name, booster, best, X, train, valid, test):
    p_valid = booster.predict(X["valid"], num_iteration=best)
    p_test = booster.predict(X["test"], num_iteration=best)
    y_valid, y_test = valid["isFraud"].to_numpy(), test["isFraud"].to_numpy()
    mv, mt = metrics(y_valid, p_valid), metrics(y_test, p_test)
    save_artifacts(name, booster, valid["TransactionID"], p_valid, test["TransactionID"], p_test)
    return {"best_iteration": int(best), "valid_pr_auc": mv["pr_auc"], "test_pr_auc": mt["pr_auc"]}


def fit_plain(train, valid, test):
    """Exactly the training block of baseline_lgbm.main / model_fe1.main."""
    X, feats, _ = prepare(train, valid, test)
    dtrain = lgb.Dataset(X["train"], train["isFraud"].to_numpy())
    dvalid = lgb.Dataset(X["valid"], valid["isFraud"].to_numpy(), reference=dtrain)
    booster = lgb.train(
        PARAMS, dtrain, num_boost_round=MAX_ROUNDS, valid_sets=[dvalid], valid_names=["valid"],
        callbacks=[lgb.early_stopping(EARLY_STOP), lgb.log_evaluation(50)],
    )
    return booster, int(booster.best_iteration), X


def compare(c, r):
    return bool(c["best_iteration"] == r["best_iteration"]
                and abs(c["valid_pr_auc"] - r["valid_pr_auc"]) <= TOL
                and abs(c["test_pr_auc"] - r["test_pr_auc"]) <= TOL)


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("models_dir", models_dir(), flush=True)
    raw = load_joined()
    models, timings = {}, {}

    def record(name, committed_dir, retrained, t):
        c = committed(committed_dir) if committed_dir else None
        models[name] = {"committed": c, "retrained": retrained, "exact": compare(c, retrained)}
        timings[name] = round(time.time() - t, 1)
        print(name, json.dumps(models[name]), flush=True)

    # (a) baseline_lgbm: time_split on the raw frame
    t = time.time()
    train, valid, test, _ = time_split(raw)
    booster, best, X = fit_plain(train, valid, test)
    r = score_and_save("baseline_lgbm", booster, best, X, train, valid, test)
    record("baseline_lgbm", "baseline_lgbm", r, t)
    del booster, X, train, valid, test
    gc.collect()

    # (b) lgbm_fe1: sort, add history features, split (as model_fe1.main)
    t = time.time()
    df = raw.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    df, _added = add_history_features(df)
    train, valid, test, _ = time_split(df)
    booster, best, X = fit_plain(train, valid, test)
    r = score_and_save("lgbm_fe1", booster, best, X, train, valid, test)
    record("lgbm_fe1", "lgbm_fe1", r, t)
    del booster, X, train, valid, test, df
    gc.collect()

    # (c) lgbm_fe2 candidate A_hist, seed 42 (as model_fe2.main)
    t = time.time()
    df = raw.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    base_cols = [c for c in df.columns if c not in DROP]
    df, hist_added = add_history_features(df)
    df, _uid_added = add_uid_aggregates(df)  # kept: same frame/column order as model_fe2
    train, valid, test, _ = time_split(df)
    cols = base_cols + [c for c in hist_added if c != "day"]
    assert not {"day", "TransactionDT"} & set(cols)
    booster, best, _pr = train_one(cols, train, valid, 42)
    X, _, _ = prepare(train[cols], valid[cols], test[cols])
    r = score_and_save("lgbm_fe2_A_hist_s42", booster, best, X, train, valid, test)
    fe2 = json.loads((WORKTREE / "results" / "lgbm_fe2" / "metrics.json").read_text())
    assert fe2["selection"]["chosen"] == "A_hist"
    record("lgbm_fe2_A_hist_s42", "lgbm_fe2", r, t)

    (OUT_DIR / "repro.json").write_text(json.dumps({"models": models}, indent=2) + "\n")
    info = {
        "git_sha": git_sha(),
        "versions": {"python": platform.python_version(), "lightgbm": lgb.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__,
                     "scikit-learn": sklearn.__version__, "pyarrow": pyarrow.__version__},
        "params": {**PARAMS, "max_rounds": MAX_ROUNDS, "early_stopping_rounds": EARLY_STOP},
        "tolerance": TOL,
        "per_model_seconds": timings,
        "runtime_seconds": round(time.time() - t0, 1),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 1),
    }
    (OUT_DIR / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2), flush=True)


if __name__ == "__main__":
    main()
