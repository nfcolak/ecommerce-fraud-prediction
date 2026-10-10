"""Forward-fold backtest before the test period + label-delay simulation (valid/fold data only).

Run (PYTHONPATH=src): python -m ecommerce_fraud.backtest
History features are built once on the full sorted frame (past-only). The test split (day >= 141)
is dropped right after the main time split and never scored.

Folds: score window [S, S+20] (days = TransactionDT // 86400), train = days 1 .. S-8 (7-day gap),
early stopping on the last 14 days of that range (not used for fitting). Models per fold:
baseline_lgbm config (raw features, no class weight) and the final config (A_hist_nanfix, spw 28).
Label delay: main split, final config; train rows of the last N days get label 0 (variant B) or
are dropped (variant A); early stopping and evaluation on the unchanged valid labels.
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

from ecommerce_fraud.artifacts import save_artifacts
from ecommerce_fraud.baseline_lgbm import DROP, EARLY_STOP, PARAMS, git_sha, metrics, prepare
from ecommerce_fraud.data import WORKTREE, load_joined, time_split
from ecommerce_fraud.features_history import add_history_features

OUT_DIR = WORKTREE / "results" / "backtest"
SEED = 42
SPW = 28.0
MAX_ROUNDS = 5000
DAY = 86400
FOLDS = [(63, 83), (87, 107), (111, 131)]
GAP_DAYS = 7
ES_DAYS = 14
TEST_START_DAY = 141
DELAYS = [7, 14, 30]
VARIANT_A_DEADLINE_S = 40 * 60  # start variant A trainings only while elapsed time is below this
T0 = time.time()


def rss_mb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 1)


def log(*a):
    print(f"[{time.time() - T0:7.0f}s rss {rss_mb():.0f}MB]", *a, flush=True)


def fit(X_fit, y_fit, X_es, y_es, spw):
    params = {**PARAMS, "seed": SEED}
    if spw:
        params["scale_pos_weight"] = spw
    dtrain = lgb.Dataset(X_fit, y_fit)
    dval = lgb.Dataset(X_es, y_es, reference=dtrain)
    booster = lgb.train(params, dtrain, num_boost_round=MAX_ROUNDS, valid_sets=[dval],
                        valid_names=["es"],
                        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                                   lgb.log_evaluation(500)])
    return booster, int(booster.best_iteration)


def run_fold(df, day, base_cols, final_cols, s, e):
    tr_mask = (day <= s - GAP_DAYS - 1).to_numpy()
    sc_mask = ((day >= s) & (day <= e)).to_numpy()
    tr_all, sc = df[tr_mask], df[sc_mask]
    d_tr = day[tr_mask].to_numpy()
    es_start = (s - GAP_DAYS - 1) - ES_DAYS + 1
    is_es = d_tr >= es_start
    X, _, _ = prepare(tr_all[final_cols], sc[final_cols], sc[final_cols].iloc[:0])
    y_tr, y_sc = tr_all["isFraud"].to_numpy(), sc["isFraud"].to_numpy()
    out = {"score_start_day": s, "score_end_day": e, "train_days": [1, s - GAP_DAYS - 1],
           "es_days": [es_start, s - GAP_DAYS - 1],
           "n_train": int(len(tr_all)), "n_fraud_train": int(y_tr.sum()),
           "n_fit": int((~is_es).sum()), "n_es": int(is_es.sum()),
           "share_train_after_day30": float((d_tr > 30).mean()),
           "n_score": int(len(sc)), "n_fraud_score": int(y_sc.sum())}
    for tag, cols, spw in (("baseline", base_cols, 0.0), ("final", final_cols, SPW)):
        t1 = time.time()
        Xt, Xs = X["train"][cols], X["valid"][cols]
        booster, best = fit(Xt[~is_es], y_tr[~is_es], Xt[is_es], y_tr[is_es], spw)
        s_sc = booster.predict(Xs, num_iteration=best)
        m = metrics(y_sc, s_sc)
        m["best_iteration"] = best
        m["fit_seconds"] = round(time.time() - t1, 1)
        out[tag] = m
        save_artifacts(f"backtest_d{s}_{tag}", booster, sc["TransactionID"], s_sc, None, None)
        log(f"fold {s}-{e} {tag}: PR-AUC {m['pr_auc']:.4f} p@1% {m['precision_at_top_1pct']:.4f} "
            f"best_it {best} ({m['fit_seconds']}s)")
        del booster, Xt, Xs
        gc.collect()
    out["delta_pr_auc"] = out["final"]["pr_auc"] - out["baseline"]["pr_auc"]
    out["delta_p_at_1pct"] = (out["final"]["precision_at_top_1pct"]
                              - out["baseline"]["precision_at_top_1pct"])
    return out


def delay_run(variant, n_days, X, y_tr, dt_tr, y_va, ids_va, dt_last):
    t1 = time.time()
    late = dt_tr > dt_last - n_days * DAY
    if variant == "B":
        y_fit = np.where(late, 0, y_tr)
        booster, best = fit(X["train"], y_fit, X["valid"], y_va, SPW)
        n_rows, n_fraud = len(y_tr), int(y_fit.sum())
    else:
        keep = ~late
        booster, best = fit(X["train"][keep], y_tr[keep], X["valid"], y_va, SPW)
        n_rows, n_fraud = int(keep.sum()), int(y_tr[keep].sum())
    s_va = booster.predict(X["valid"], num_iteration=best)
    m = metrics(y_va, s_va)
    m.update({"best_iteration": best, "n_train_rows_used": n_rows, "n_fraud_train_used": n_fraud,
              "n_late_rows": int(late.sum()), "n_late_fraud_true": int(y_tr[late].sum()),
              "fit_seconds": round(time.time() - t1, 1)})
    save_artifacts(f"label_delay_{variant}{n_days}", booster, ids_va, s_va, None, None)
    log(f"delay {variant}{n_days}: valid PR-AUC {m['pr_auc']:.4f} best_it {best}")
    return m


def write(result, info_extra=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    info = {"git_sha": git_sha(),
            "versions": {"python": platform.python_version(), "lightgbm": lgb.__version__,
                         "pandas": pd.__version__, "numpy": np.__version__},
            "params": {**PARAMS, "seed": SEED, "max_rounds": MAX_ROUNDS,
                       "early_stopping_rounds": EARLY_STOP, "final_scale_pos_weight": SPW},
            "runtime_seconds": round(time.time() - T0, 1), "peak_rss_mb": rss_mb(),
            **(info_extra or {})}
    (OUT_DIR / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")


def main():
    final_metrics = json.loads((WORKTREE / "results" / "final" / "metrics.json").read_text())
    cfg = final_metrics["config"]
    assert cfg["feature_set"] == "A_hist_nanfix" and cfg["scale_pos_weight"] == SPW

    raw = load_joined().sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    base_cols = [c for c in raw.columns if c not in DROP]
    df, hist_added = add_history_features(raw, nan_keys="nan")
    del raw
    final_cols = base_cols + [c for c in hist_added if c != "day"]
    assert len(final_cols) == cfg["n_features"], (len(final_cols), cfg["n_features"])
    assert not {"day", "TransactionDT", "isFraud", "TransactionID"} & set(final_cols)
    train, valid, test, stats = time_split(df)
    del test  # never scored in this unit
    df = df[(df["TransactionDT"] // DAY) < TEST_START_DAY].reset_index(drop=True)
    day = df["TransactionDT"] // DAY
    log("frame", df.shape, "valid max day", int(valid["TransactionDT"].max() // DAY))

    result = {"folds": [], "label_delay": {"B": {}, "A": {}}}
    for s, e in FOLDS:
        assert e < valid["TransactionDT"].max() // DAY
        result["folds"].append(run_fold(df, day, base_cols, final_cols, s, e))
        write(result)
    del df, day
    gc.collect()

    # ------------------------------------------------------------ label delay
    X, _, _ = prepare(train[final_cols], valid[final_cols], valid[final_cols].iloc[:0])
    y_tr, dt_tr = train["isFraud"].to_numpy(), train["TransactionDT"].to_numpy()
    y_va, ids_va = valid["isFraud"].to_numpy(), valid["TransactionID"]
    dt_last = int(dt_tr.max())
    ref = final_metrics["valid"]["pr_auc"]
    result["label_delay"]["B"]["0"] = ref
    result["label_delay"]["note"] = (
        "valid PR-AUC; N=0 is the final model's value (results/final, full-valid early stopping). "
        "B: train rows in the last N days keep features, label 0. A: those rows dropped. "
        "Early stopping and evaluation on valid with unchanged labels.")
    result["label_delay"]["details"] = {}
    write(result)
    for variant in ("B", "A"):
        for n in DELAYS:
            if variant == "A" and time.time() - T0 > VARIANT_A_DEADLINE_S:
                log(f"skip A{n}: time budget")
                continue
            m = delay_run(variant, n, X, y_tr, dt_tr, y_va, ids_va, dt_last)
            result["label_delay"][variant][str(n)] = m["pr_auc"]
            result["label_delay"]["details"][f"{variant}{n}"] = m
            write(result)
    if not result["label_delay"]["A"]:
        del result["label_delay"]["A"]
    write(result, {"time_split": stats, "n_features_final": len(final_cols),
                   "n_features_baseline": len(base_cols)})
    log("done")


if __name__ == "__main__":
    main()
