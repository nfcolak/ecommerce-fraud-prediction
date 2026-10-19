"""final_bag5: 5-seed rank-averaged final, and its single paired test view (5th view of the test period).

python -m ecommerce_fraud.final_bag
Approved by the user 2026-10-17 after bag_check passed ADR-005 (+0.0068, 6/6 windows).
Trains final's exact config (final.build / prepare / PARAMS, spw 28, ES 100 on valid) with seeds
42-46 on the main split, scores valid and test, rank-averages the seeds, and compares on the same
test rows against the existing single-seed `final` (models/final/scores_test.parquet). The seed-42
retrain must reproduce the existing final test scores (control). Paired 7-day-block bootstrap.
Writes results/final_bag5/test_view.json; per-seed scores in models/final_bag5_s<seed>/.
"""
from __future__ import annotations

import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from ecommerce_fraud.artifacts import models_dir, save_artifacts
from ecommerce_fraud.baseline_lgbm import EARLY_STOP, PARAMS, git_sha, metrics, prepare
from ecommerce_fraud.data import WORKTREE
from ecommerce_fraud.final import MAX_ROUNDS, SPW, build, rss_mb

SEEDS = (42, 43, 44, 45, 46)
OUT = WORKTREE / "results" / "final_bag5"
DAY, BLOCK_DAYS, N_BOOT, BOOT_SEED = 86400, 7, 2000, 20261017


def ap_weighted(score, y, w):
    o = np.argsort(-score, kind="stable")
    s, ys, ws = score[o], y[o].astype(float), w[o]
    ge = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s) - 1]
    tp, cnt = np.cumsum(ws * ys)[ge], np.cumsum(ws)[ge]
    prec = np.divide(tp, cnt, out=np.zeros_like(tp), where=cnt > 0)
    return float(np.sum(np.diff(tp / tp[-1], prepend=0.0) * prec))


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    train, valid, test, cols, stats = build()
    X, _, _ = prepare(train[cols], valid[cols], test[cols])
    y_tr, y_va, y_te = (d["isFraud"].to_numpy() for d in (train, valid, test))
    ids_va, ids_te = valid["TransactionID"].to_numpy(), test["TransactionID"].to_numpy()
    day_te = (test["TransactionDT"].to_numpy() // DAY).astype(int)
    del train, valid, test
    runs = {}
    for seed in SEEDS:
        name = f"final_bag5_s{seed}"
        d = models_dir() / name
        if (d / "scores_test.parquet").exists() and (OUT / f"{name}.json").exists():
            runs[seed] = json.loads((OUT / f"{name}.json").read_text())
            print(f"{name}: exists", flush=True)
            continue
        t1 = time.time()
        params = {**PARAMS, "seed": seed, "scale_pos_weight": SPW}
        dtrain = lgb.Dataset(X["train"], y_tr)
        dvalid = lgb.Dataset(X["valid"], y_va, reference=dtrain)
        booster = lgb.train(params, dtrain, num_boost_round=MAX_ROUNDS, valid_sets=[dvalid], valid_names=["valid"],
                            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(500)])
        best = int(booster.best_iteration)
        s_va = booster.predict(X["valid"], num_iteration=best)
        s_te = booster.predict(X["test"], num_iteration=best)
        save_artifacts(name, booster, pd.Series(ids_va), s_va, pd.Series(ids_te), s_te)
        runs[seed] = {"seed": seed, "best_iteration": best, "seconds": round(time.time() - t1, 1), "rss_mb": rss_mb(),
                      "valid": metrics(y_va, s_va), "test": metrics(y_te, s_te)}
        (OUT / f"{name}.json").write_text(json.dumps(runs[seed], indent=2))
        print(f"{name}: best_it {best} valid {runs[seed]['valid']['pr_auc']:.5f} ({runs[seed]['seconds']}s)", flush=True)
        del booster

    def load(name, which, ids):
        s = pd.read_parquet(models_dir() / name / f"scores_{which}.parquet").set_index("TransactionID")["score"].reindex(ids)
        assert not s.isna().any(), f"{name} {which} misses rows"
        return s.to_numpy()

    te = {s: load(f"final_bag5_s{s}", "test", ids_te) for s in SEEDS}
    va = {s: load(f"final_bag5_s{s}", "valid", ids_va) for s in SEEDS}
    fin_te = load("final", "test", ids_te)
    control = {"max_abs_diff_s42_vs_final_test": float(np.max(np.abs(te[42] - fin_te)))}
    control["reproduces_final"] = bool(control["max_abs_diff_s42_vs_final_test"] < 1e-9)
    rank = lambda a: pd.Series(a).rank(pct=True).to_numpy()
    bag_te = np.mean([rank(te[s]) for s in SEEDS], axis=0)
    bag_va = np.mean([rank(va[s]) for s in SEEDS], axis=0)
    m_bag, m_fin = metrics(y_te, bag_te), metrics(y_te, fin_te)
    # paired 7-day-block bootstrap over test days
    blk = (day_te - day_te.min()) // BLOCK_DAYS
    nb = int(blk.max()) + 1
    rng = np.random.default_rng(BOOT_SEED)
    deltas = np.empty(N_BOOT)
    for r in range(N_BOOT):
        c = np.bincount(rng.integers(0, nb, nb), minlength=nb).astype(float)[blk]
        deltas[r] = ap_weighted(bag_te, y_te, c) - ap_weighted(fin_te, y_te, c)
    res = {"note": "5th view of the test period (days 141-182); comparative, not an unbiased absolute estimate. "
                   "Candidate motivated post hoc by round p4 Stage B scores; selected under ADR-005 (bag_check).",
           "git_sha": git_sha(), "split": stats, "n_features": len(cols), "seeds": SEEDS, "control": control,
           "per_seed": runs,
           "valid": {"bag5": metrics(y_va, bag_va)["pr_auc"],
                     "single_seed_mean": float(np.mean([runs[s]["valid"]["pr_auc"] for s in SEEDS]))},
           "test": {"final_bag5": m_bag, "final": m_fin,
                    "single_seed_mean_pr_auc": float(np.mean([runs[s]["test"]["pr_auc"] for s in SEEDS]))},
           "paired_bootstrap": {"block_days": BLOCK_DAYS, "n_reps": N_BOOT, "seed": BOOT_SEED,
                                "delta_pr_auc": m_bag["pr_auc"] - m_fin["pr_auc"],
                                "ci95": [float(x) for x in np.percentile(deltas, [2.5, 97.5])],
                                "p_delta_le_0": float((deltas <= 0).mean())},
           "runtime_seconds": round(time.time() - t0, 1)}
    (OUT / "test_view.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: res[k] for k in ("control", "valid", "paired_bootstrap")}, indent=2, default=float))
    print("test PR-AUC bag5", m_bag["pr_auc"], "final", m_fin["pr_auc"])


if __name__ == "__main__":
    main()
