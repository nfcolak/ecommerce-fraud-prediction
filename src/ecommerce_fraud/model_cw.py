"""LightGBM lgbm_cw: NaN-key fix for the history features, then class weight on the winner.

VALID ONLY: nothing here predicts or scores the test split. Same split, categorical handling,
params, rounds, early stopping and seeds as model_fe2. The control arm (A_hist, spw 1) is
copied from the committed lgbm_fe2 metrics.
Run: python -m ecommerce_fraud.model_cw  (PYTHONPATH=src)
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
from sklearn.metrics import average_precision_score, brier_score_loss

from ecommerce_fraud.artifacts import save_artifacts
from ecommerce_fraud.baseline_lgbm import (DROP, EARLY_STOP, MAX_ROUNDS, PARAMS, git_sha,
                                           prepare)
from ecommerce_fraud.data import WORKTREE, load_joined, time_split
from ecommerce_fraud.features_history import add_history_features, make_keys

OUT_DIR = WORKTREE / "results" / "lgbm_cw"
FE2_METRICS = WORKTREE / "results" / "lgbm_fe2" / "metrics.json"
SEEDS = (42, 43, 44)
SPW = (5, 28)


def train_one(cols, train, valid, seed, spw=1.0):
    """Train on `cols`, early-stop on valid (average_precision, weight-free).
    Returns (booster, best_iter, pr_auc, brier, valid_scores)."""
    X, _, _ = prepare(train[cols], valid[cols], valid[cols].iloc[:0])
    dtrain = lgb.Dataset(X["train"], train["isFraud"].to_numpy())
    dvalid = lgb.Dataset(X["valid"], valid["isFraud"].to_numpy(), reference=dtrain)
    params = {**PARAMS, "seed": seed}
    if spw != 1:
        params["scale_pos_weight"] = float(spw)
    booster = lgb.train(
        params, dtrain, num_boost_round=MAX_ROUNDS, valid_sets=[dvalid],
        valid_names=["valid"], callbacks=[lgb.early_stopping(EARLY_STOP), lgb.log_evaluation(100)],
    )
    best = int(booster.best_iteration)
    scores = booster.predict(X["valid"], num_iteration=best)
    y = valid["isFraud"].to_numpy()
    return (booster, best, float(average_precision_score(y, scores)),
            float(brier_score_loss(y, scores)), scores)


def run_arm(label, cols, train, valid, ids_valid, save_name):
    prs, bests, briers = [], [], []
    for seed in SEEDS:
        ts = time.time()
        booster, best, pr, brier, scores = train_one(
            cols, train, valid, seed, SPW_BY_LABEL.get(label, 1.0))
        prs.append(pr)
        bests.append(best)
        briers.append(brier)
        print(f"[{label}] seed {seed} best_iter {best} valid_pr_auc {pr:.5f} brier {brier:.5f} "
              f"({time.time() - ts:.0f}s)", flush=True)
        if seed == SEEDS[0] and save_name:
            save_artifacts(save_name, booster, ids_valid, scores, None, None)
        del booster
        gc.collect()
    return {
        "valid_pr_auc_per_seed": prs,
        "valid_pr_auc_mean": float(np.mean(prs)),
        "valid_pr_auc_sd": float(np.std(prs, ddof=1)),
        "best_iteration_per_seed": bests,
        "valid_brier_per_seed": briers,
        "n_features": len(cols),
    }


SPW_BY_LABEL = {}


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_joined()
    raw = raw.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    base_cols = [c for c in raw.columns if c not in DROP]

    keys = make_keys(raw)
    nan_share = {k: float(keys[k].isna().mean()) for k in
                 ("uid", "card1_addr1", "uid_email", "P_emaildomain")}
    print("nan_key_share", json.dumps(nan_share), flush=True)

    fe2 = json.loads(FE2_METRICS.read_text())["selection"]["candidates"]["A_hist"]
    control = {**{k: fe2[k] for k in ("valid_pr_auc_per_seed", "valid_pr_auc_mean",
                                      "valid_pr_auc_sd", "n_features")},
               "source": "committed lgbm_fe2"}

    # a. NaN-key candidate
    df, hist_added = add_history_features(raw, nan_keys="nan")
    train, valid, _test, stats = time_split(df)  # test is never used below
    del _test
    print("split", json.dumps(stats), flush=True)
    hist_cols = [c for c in hist_added if c != "day"]
    cols = base_cols + hist_cols
    assert not {"day", "TransactionDT", "isFraud", "TransactionID"} & set(cols)
    assert fe2["n_features"] == len(cols), (fe2["n_features"], len(cols))
    ids_valid = valid["TransactionID"].to_numpy()

    nanfix = run_arm("A_hist_nanfix", cols, train, valid, ids_valid, "A_hist_nanfix_s42")
    winner = ("A_hist_nanfix" if nanfix["valid_pr_auc_mean"] > control["valid_pr_auc_mean"]
              else "A_hist")
    print("winner", winner, flush=True)

    # b. winner feature set
    if winner == "A_hist":
        del df, train, valid
        gc.collect()
        df, hist_added = add_history_features(raw, nan_keys="lump")
        train, valid, _test, _ = time_split(df)
        del _test
        assert [c for c in hist_added if c != "day"] == hist_cols
        # Per-seed Brier for spw1 is not in lgbm_fe2: retrain (no spw); PR-AUC must match.
        spw1 = run_arm("spw1", cols, train, valid, ids_valid, None)
        same = bool(np.allclose(spw1["valid_pr_auc_per_seed"],
                                control["valid_pr_auc_per_seed"], atol=1e-9))
        spw1["note"] = "retrained for Brier only"
        spw1["pr_auc_matches_committed_fe2"] = same
        print("spw1 retrain matches committed lgbm_fe2:", same, flush=True)
    else:
        spw1 = dict(nanfix)
        spw1["note"] = "same run as A_hist_nanfix"

    # c. class weight
    cands = {"spw1": spw1}
    for w in SPW:
        label = f"spw{w}"
        SPW_BY_LABEL[label] = float(w)
        cands[label] = run_arm(label, cols, train, valid, ids_valid, f"{winner}_{label}_s42")
        cands[label]["scale_pos_weight"] = float(w)
    best_name = max(cands, key=lambda k: cands[k]["valid_pr_auc_mean"])
    print("best_on_valid", best_name, flush=True)

    result = {
        "model": "lgbm_cw", "split": stats,
        "nan_fix": {"A_hist": control, "A_hist_nanfix": nanfix, "winner": winner},
        "selection": {"candidates": cands, "best_on_valid": best_name},
        "nan_key_share": nan_share,
    }
    assert "test" not in result
    (OUT_DIR / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")

    info = {
        "git_sha": git_sha(),
        "versions": {"python": platform.python_version(), "lightgbm": lgb.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__,
                     "scikit-learn": sklearn.__version__, "pyarrow": pyarrow.__version__},
        "params": {**PARAMS, "max_rounds": MAX_ROUNDS, "early_stopping_rounds": EARLY_STOP,
                   "seeds": list(SEEDS), "scale_pos_weights": [1] + list(SPW)},
        "winner_feature_set": winner,
        "runtime_seconds": round(time.time() - t0, 1),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 1),
    }
    (OUT_DIR / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({k: {m: v[m] for m in ("valid_pr_auc_mean", "valid_pr_auc_sd")}
                      for k, v in cands.items()}, indent=2), flush=True)
    print(f"runtime {info['runtime_seconds']}s peak_rss_mb {info['peak_rss_mb']}", flush=True)


if __name__ == "__main__":
    main()
