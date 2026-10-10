"""LightGBM lgbm_fe2: no absolute time, past-only per-client aggregates, feature set chosen
on VALID only (3 seeds per candidate), test scored once for the chosen candidate (seed 42).

Same split, categorical handling, params, rounds and early stopping as baseline_lgbm.
Run: python -m ecommerce_fraud.model_fe2  (PYTHONPATH=src)
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
from sklearn.metrics import average_precision_score

from ecommerce_fraud.baseline_lgbm import (DROP, EARLY_STOP, MAX_ROUNDS, PARAMS, git_sha,
                                           metrics, prepare)
from ecommerce_fraud.data import RAW_DIR, WORKTREE, load_joined, time_split
from ecommerce_fraud.features_history import add_history_features
from ecommerce_fraud.features_uid_agg import add_uid_aggregates, leakage_check

OUT_DIR = WORKTREE / "results" / "lgbm_fe2"
SEEDS = (42, 43, 44)


def train_one(cols, train, valid, seed):
    """Train on `cols`, early-stop on valid. Returns (booster, best_iter, valid_pr_auc).
    No test data is touched."""
    X, _, _ = prepare(train[cols], valid[cols], valid[cols].iloc[:0])
    dtrain = lgb.Dataset(X["train"], train["isFraud"].to_numpy())
    dvalid = lgb.Dataset(X["valid"], valid["isFraud"].to_numpy(), reference=dtrain)
    booster = lgb.train(
        {**PARAMS, "seed": seed}, dtrain, num_boost_round=MAX_ROUNDS, valid_sets=[dvalid],
        valid_names=["valid"], callbacks=[lgb.early_stopping(EARLY_STOP), lgb.log_evaluation(100)],
    )
    best = int(booster.best_iteration)
    pr = float(average_precision_score(valid["isFraud"].to_numpy(),
                                       booster.predict(X["valid"], num_iteration=best)))
    return booster, best, pr


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_joined()
    df = df.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    base_cols = [c for c in df.columns if c not in DROP]
    df, hist_added = add_history_features(df)
    df, uid_added = add_uid_aggregates(df)
    print(f"uid aggregates: {len(uid_added)} columns", flush=True)
    train, valid, test, stats = time_split(df)
    print("split", json.dumps(stats), flush=True)

    hist_cols = [c for c in hist_added if c != "day"]
    extra = {
        "A_hist": hist_cols,
        "B_hist_uidagg": hist_cols + uid_added,
        "C_uidagg": ["D1n", "D4n", "D10n", "D15n", "amt_log", "amt_cents"] + uid_added,
    }
    cands = {k: base_cols + v for k, v in extra.items()}
    for k, cols in cands.items():
        assert not {"day", "TransactionDT"} & set(cols), k

    # Candidate selection: VALID only.
    sel, boosters = {}, {}
    for name, cols in cands.items():
        prs = []
        for seed in SEEDS:
            booster, best, pr = train_one(cols, train, valid, seed)
            prs.append(pr)
            print(f"[select] {name} seed {seed} best_iter {best} valid_pr_auc {pr:.5f}", flush=True)
            if seed == SEEDS[0]:
                boosters[name] = (booster, best)
            del booster
            gc.collect()
        sel[name] = {
            "valid_pr_auc_per_seed": prs,
            "valid_pr_auc_mean": float(np.mean(prs)),
            "valid_pr_auc_sd": float(np.std(prs, ddof=1)),
            "n_features": len(cols),
        }
    chosen = max(sel, key=lambda k: sel[k]["valid_pr_auc_mean"])
    print("chosen", chosen, flush=True)

    # Test: chosen candidate, seed 42, scored once.
    cols = cands[chosen]
    booster, best = boosters[chosen]
    X, feats, obj_cols = prepare(train[cols], valid[cols], test[cols])
    y_valid, y_test = valid["isFraud"].to_numpy(), test["isFraud"].to_numpy()
    m_valid = metrics(y_valid, booster.predict(X["valid"], num_iteration=best))
    m_test = metrics(y_test, booster.predict(X["test"], num_iteration=best))
    added = [c for c in cols if c not in base_cols]
    result = {"model": "lgbm_fe2", "best_iteration": best, "split": stats,
              "valid": m_valid, "test": m_test, "features_added": added,
              "selection": {"chosen": chosen, "candidates": sel}}
    (OUT_DIR / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")

    imp_all = pd.DataFrame({
        "feature": booster.feature_name(),
        "gain": booster.feature_importance("gain", iteration=best),
        "split": booster.feature_importance("split", iteration=best),
    }).sort_values("gain", ascending=False)
    imp_all.head(50).to_csv(OUT_DIR / "feature_importance.csv", index=False)
    top_new = imp_all[imp_all["feature"].isin(added)].head(15)

    check = leakage_check(df, test.index, seed=0, n=3)
    print(check, flush=True)
    (OUT_DIR / "leakage_check.txt").write_text(check + "\n")

    raw_sizes = {f: (RAW_DIR / f).stat().st_size
                 for f in ("train_transaction.csv", "train_identity.csv")}
    info = {
        "git_sha": git_sha(),
        "versions": {"python": platform.python_version(), "lightgbm": lgb.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__,
                     "scikit-learn": sklearn.__version__, "pyarrow": pyarrow.__version__},
        "params": {**PARAMS, "max_rounds": MAX_ROUNDS, "early_stopping_rounds": EARLY_STOP,
                   "selection_seeds": list(SEEDS), "test_seed": SEEDS[0]},
        "chosen": chosen,
        "n_features": len(feats),
        "n_features_added": len(added),
        "n_categorical_features": len(obj_cols),
        "top15_new_features_by_gain": [{"feature": r.feature, "gain": float(r.gain),
                                        "split": int(r.split)} for r in top_new.itertuples()],
        "runtime_seconds": round(time.time() - t0, 1),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 1),
        "data_file_sizes_bytes": raw_sizes,
    }
    (OUT_DIR / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(result["selection"], indent=2), flush=True)
    print(f"test pr_auc {m_test['pr_auc']:.5f}", flush=True)
    print(f"runtime {info['runtime_seconds']}s peak_rss_mb {info['peak_rss_mb']}", flush=True)


if __name__ == "__main__":
    main()
