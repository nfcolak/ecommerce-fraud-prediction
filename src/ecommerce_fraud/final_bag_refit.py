"""final_bag5_refit: final_bag5 refit on train + valid, and its single paired test view (6th view of the test period).

python -m ecommerce_fraud.final_bag_refit
Approved by the user 2026-10-19 ("ac") after round p5 Stage B: refit bag5 vs final bag5 +0.0556 [+0.0489, +0.0618],
6/6 windows; label-lag robustness +0.0431, 5/5 (PREREGISTRATION p5, amendments 2026-10-18/19).
Recipe = arm refit_x100 transferred to the main split: for each seed 42-46 take final_bag5's early-stopped round
count B (results/final_bag5/final_bag5_s<seed>.json; ES 100 on valid), retrain final's exact config (final.build,
PARAMS, spw 28) on train + valid for B rounds, no early stopping. Categorical codes are fitted on train + valid, as the
protocol runner fits them on fit + ES rows. Scores test only; the 7-day gap between valid and test is unchanged.
Candidate = rank average of the 5 refit seeds; reference = final_bag5 (existing test scores, same rows).
Paired 7-day-block bootstrap over test days. Writes results/final_bag5_refit/test_view.json; per-seed boosters and
test scores in models/final_bag5_refit_s<seed>/ (scores_valid.parquet there holds IN-SAMPLE valid scores).
"""
from __future__ import annotations

import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from ecommerce_fraud.artifacts import models_dir, save_artifacts
from ecommerce_fraud.baseline_lgbm import PARAMS, git_sha, metrics, prepare
from ecommerce_fraud.data import WORKTREE
from ecommerce_fraud.final import SPW, build, rss_mb
from ecommerce_fraud.final_bag import ap_weighted

SEEDS = (42, 43, 44, 45, 46)
REF_JSON = WORKTREE / "results" / "final_bag5"
OUT = WORKTREE / "results" / "final_bag5_refit"
DAY, BLOCK_DAYS, N_BOOT, BOOT_SEED = 86400, 7, 2000, 20261019


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    train, valid, test, cols, stats = build()
    assert train["TransactionDT"].max() <= valid["TransactionDT"].min() and valid["TransactionDT"].max() \
        < test["TransactionDT"].min(), "split order"
    gap_days = (test["TransactionDT"].min() - valid["TransactionDT"].max()) / DAY
    trv = pd.concat([train, valid], axis=0, ignore_index=True)
    n_tr, n_va = len(train), len(valid)
    X, _, _ = prepare(trv[cols], test[cols], test[cols].iloc[:0])
    Xa, Xt = X["train"], X["valid"]
    assert list(Xa.columns) == list(cols) and len(Xa) == n_tr + n_va
    y_a, y_te = trv["isFraud"].to_numpy(), test["isFraud"].to_numpy()
    ids_a, ids_te = trv["TransactionID"].to_numpy(), test["TransactionID"].to_numpy()
    assert not np.isin(ids_te, ids_a).any(), "test rows in the training set"
    day_te = (test["TransactionDT"].to_numpy() // DAY).astype(int)
    del train, valid, test, trv
    runs = {}
    for seed in SEEDS:
        name = f"final_bag5_refit_s{seed}"
        B = int(json.loads((REF_JSON / f"final_bag5_s{seed}.json").read_text())["best_iteration"])
        if (models_dir() / name / "scores_test.parquet").exists() and (OUT / f"{name}.json").exists():
            runs[seed] = json.loads((OUT / f"{name}.json").read_text())
            assert runs[seed]["rounds"] == B
            print(f"{name}: exists", flush=True)
            continue
        t1 = time.time()
        params = {**PARAMS, "seed": seed, "scale_pos_weight": SPW}
        booster = lgb.train(params, lgb.Dataset(Xa, y_a), num_boost_round=B)
        s_te = booster.predict(Xt)
        s_in = booster.predict(Xa.iloc[n_tr:])  # in-sample valid scores (trained on), kept for the artifact contract
        save_artifacts(name, booster, pd.Series(ids_a[n_tr:]), s_in, pd.Series(ids_te), s_te)
        runs[seed] = {"seed": seed, "rounds": B, "n_train_rows": int(len(Xa)), "seconds": round(time.time() - t1, 1),
                      "rss_mb": rss_mb(), "test": metrics(y_te, s_te)}
        (OUT / f"{name}.json").write_text(json.dumps(runs[seed], indent=2))
        print(f"{name}: rounds {B} test-scored ({runs[seed]['seconds']}s)", flush=True)
        del booster

    def load(name):
        s = pd.read_parquet(models_dir() / name / "scores_test.parquet").set_index("TransactionID")["score"]
        s = s.reindex(ids_te)
        assert not s.isna().any(), f"{name} misses test rows"
        return s.to_numpy()

    rank = lambda a: pd.Series(a).rank(pct=True).to_numpy()
    cand = np.mean([rank(load(f"final_bag5_refit_s{s}")) for s in SEEDS], axis=0)
    ref = np.mean([rank(load(f"final_bag5_s{s}")) for s in SEEDS], axis=0)
    fin = load("final")
    m_c, m_r, m_f = metrics(y_te, cand), metrics(y_te, ref), metrics(y_te, fin)
    blk = (day_te - day_te.min()) // BLOCK_DAYS
    nb = int(blk.max()) + 1
    rng = np.random.default_rng(BOOT_SEED)
    d_ref, d_fin = np.empty(N_BOOT), np.empty(N_BOOT)
    for r in range(N_BOOT):
        c = np.bincount(rng.integers(0, nb, nb), minlength=nb).astype(float)[blk]
        a = ap_weighted(cand, y_te, c)
        d_ref[r], d_fin[r] = a - ap_weighted(ref, y_te, c), a - ap_weighted(fin, y_te, c)
    ci = lambda d: [float(x) for x in np.percentile(d, [2.5, 97.5])]
    res = {"note": "6th view of the test period (days 141-182); comparative, not an unbiased absolute estimate. "
                   "Recipe selected under ADR-005 extended to bags (round p5 Stage B, PREREGISTRATION p5).",
           "git_sha": git_sha(), "split": stats, "gap_days_valid_to_test": float(gap_days), "n_features": len(cols),
           "seeds": SEEDS, "per_seed": runs, "n_train_rows": int(len(Xa)),
           "test": {"final_bag5_refit": m_c, "final_bag5": m_r, "final": m_f,
                    "single_seed_mean_pr_auc": float(np.mean([runs[s]["test"]["pr_auc"] for s in SEEDS]))},
           "paired_bootstrap": {"block_days": BLOCK_DAYS, "n_reps": N_BOOT, "seed": BOOT_SEED,
                                "vs_final_bag5": {"delta_pr_auc": m_c["pr_auc"] - m_r["pr_auc"], "ci95": ci(d_ref),
                                                  "p_delta_le_0": float((d_ref <= 0).mean())},
                                "vs_final": {"delta_pr_auc": m_c["pr_auc"] - m_f["pr_auc"], "ci95": ci(d_fin),
                                             "p_delta_le_0": float((d_fin <= 0).mean())}},
           "runtime_seconds": round(time.time() - t0, 1)}
    (OUT / "test_view.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: res[k] for k in ("paired_bootstrap",)}, indent=2, default=float))
    print("test PR-AUC refit_bag5", m_c["pr_auc"], "final_bag5", m_r["pr_auc"], "final", m_f["pr_auc"])


if __name__ == "__main__":
    main()
