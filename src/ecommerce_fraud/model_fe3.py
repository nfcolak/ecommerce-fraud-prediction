"""LightGBM lgbm_fe3_graph: A_hist without all-time counts, plus past-only graph features.

VALID ONLY: candidates are compared on the validation period; the test split is never
predicted or scored. Same split, categorical handling, params, rounds, early stopping and
seeds as model_fe2. The control arm (A_hist) is copied from the committed lgbm_fe2 metrics.
Run: python -m ecommerce_fraud.model_fe3  (PYTHONPATH=src)
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

from ecommerce_fraud.artifacts import save_artifacts
from ecommerce_fraud.baseline_lgbm import (DROP, EARLY_STOP, MAX_ROUNDS, PARAMS, git_sha,
                                           prepare)
from ecommerce_fraud.data import WORKTREE, load_joined, time_split
from ecommerce_fraud.features_graph import add_graph_features, graph_leakage_check
from ecommerce_fraud.features_history import add_history_features

OUT_DIR = WORKTREE / "results" / "lgbm_fe3_graph"
FE2_METRICS = WORKTREE / "results" / "lgbm_fe2" / "metrics.json"
SEEDS = (42, 43, 44)
NOALL_DROP = ["P_emaildomain_cnt_all", "card1_cnt_all", "card1_addr1_cnt_all"]


def train_one(cols, train, valid, seed):
    """Train on `cols`, early-stop on valid. Returns (booster, best_iter, pr_auc, valid_scores)."""
    X, _, _ = prepare(train[cols], valid[cols], valid[cols].iloc[:0])
    dtrain = lgb.Dataset(X["train"], train["isFraud"].to_numpy())
    dvalid = lgb.Dataset(X["valid"], valid["isFraud"].to_numpy(), reference=dtrain)
    booster = lgb.train(
        {**PARAMS, "seed": seed}, dtrain, num_boost_round=MAX_ROUNDS, valid_sets=[dvalid],
        valid_names=["valid"], callbacks=[lgb.early_stopping(EARLY_STOP), lgb.log_evaluation(100)],
    )
    best = int(booster.best_iteration)
    scores = booster.predict(X["valid"], num_iteration=best)
    return booster, best, float(average_precision_score(valid["isFraud"].to_numpy(), scores)), scores


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_joined()
    df = df.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    base_cols = [c for c in df.columns if c not in DROP]
    df, hist_added = add_history_features(df)
    t_g = time.time()
    df, graph_added = add_graph_features(df)
    graph_seconds = round(time.time() - t_g, 1)
    train, valid, _test, stats = time_split(df)  # test is never used below
    del _test
    print("split", json.dumps(stats), flush=True)

    check = graph_leakage_check(df, valid.index, seed=0, n=3)
    print(check, flush=True)
    (OUT_DIR / "leakage_check.txt").write_text(check + "\n")
    assert "ALL MATCH" in check

    hist_cols = [c for c in hist_added if c != "day"]
    a_hist = base_cols + hist_cols
    assert all(c in a_hist for c in NOALL_DROP)
    noall = [c for c in a_hist if c not in NOALL_DROP]
    cands = {"A_hist_noall": noall, "A_hist_noall_graph": noall + graph_added}
    for k, cols in cands.items():
        assert not {"day", "TransactionDT", "isFraud", "TransactionID"} & set(cols), k

    fe2 = json.loads(FE2_METRICS.read_text())["selection"]["candidates"]["A_hist"]
    assert fe2["n_features"] == len(a_hist), (fe2["n_features"], len(a_hist))
    sel = {"A_hist": {**{k: fe2[k] for k in ("valid_pr_auc_per_seed", "valid_pr_auc_mean",
                                              "valid_pr_auc_sd", "n_features")},
                      "source": "committed lgbm_fe2"}}

    imp_df = None
    ids_valid = valid["TransactionID"].to_numpy()
    for name, cols in cands.items():
        prs, bests = [], []
        for seed in SEEDS:
            ts = time.time()
            booster, best, pr, scores = train_one(cols, train, valid, seed)
            prs.append(pr)
            bests.append(best)
            print(f"[select] {name} seed {seed} best_iter {best} valid_pr_auc {pr:.5f} "
                  f"({time.time() - ts:.0f}s)", flush=True)
            if seed == SEEDS[0]:
                save_artifacts(f"{name}_s42", booster, ids_valid, scores, None, None)
                if name == "A_hist_noall_graph":
                    imp_df = pd.DataFrame({
                        "feature": booster.feature_name(),
                        "gain": booster.feature_importance("gain", iteration=best),
                        "split": booster.feature_importance("split", iteration=best),
                    }).sort_values("gain", ascending=False)
            del booster
            gc.collect()
        sel[name] = {
            "valid_pr_auc_per_seed": prs,
            "valid_pr_auc_mean": float(np.mean(prs)),
            "valid_pr_auc_sd": float(np.std(prs, ddof=1)),
            "n_features": len(cols),
            "best_iteration_per_seed": bests,
        }
    best_name = max(sel, key=lambda k: sel[k]["valid_pr_auc_mean"])
    print("best_on_valid", best_name, flush=True)

    result = {"model": "lgbm_fe3_graph", "split": stats,
              "selection": {"candidates": sel, "best_on_valid": best_name},
              "graph_features": graph_added, "dropped_noall": NOALL_DROP}
    (OUT_DIR / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    imp_df.to_csv(OUT_DIR / "feature_importance.csv", index=False)
    top_graph = imp_df[imp_df["feature"].isin(graph_added)].head(15)

    info = {
        "git_sha": git_sha(),
        "versions": {"python": platform.python_version(), "lightgbm": lgb.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__,
                     "scikit-learn": sklearn.__version__, "pyarrow": pyarrow.__version__},
        "params": {**PARAMS, "max_rounds": MAX_ROUNDS, "early_stopping_rounds": EARLY_STOP,
                   "seeds": list(SEEDS)},
        "graph_build_seconds": graph_seconds,
        "top_graph_features_by_gain": [{"feature": r.feature, "gain": float(r.gain),
                                        "split": int(r.split)} for r in top_graph.itertuples()],
        "runtime_seconds": round(time.time() - t0, 1),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 1),
    }
    (OUT_DIR / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({k: {m: v[m] for m in ("valid_pr_auc_mean", "valid_pr_auc_sd")}
                      for k, v in sel.items()}, indent=2), flush=True)
    print(f"runtime {info['runtime_seconds']}s peak_rss_mb {info['peak_rss_mb']}", flush=True)


if __name__ == "__main__":
    main()
