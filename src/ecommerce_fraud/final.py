"""Final model: A_hist_nanfix + scale_pos_weight 28 (pre-registered, lgbm_cw best_on_valid).

Two stages (PYTHONPATH=src):
  python -m ecommerce_fraud.final train    # fit with full-valid early stopping, score test ONCE
  python -m ecommerce_fraud.final analyze  # calibration, capacity, cost, SHAP, bootstrap, report files
The train stage is the single new test view under ADR-002; analyze only reads the saved scores.
"""
from __future__ import annotations

import json
import platform
import resource
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow
import sklearn

from ecommerce_fraud.artifacts import models_dir, save_artifacts
from ecommerce_fraud.baseline_lgbm import DROP, EARLY_STOP, PARAMS, git_sha, metrics, prepare
from ecommerce_fraud.data import WORKTREE, load_joined, time_split
from ecommerce_fraud.features_history import add_history_features

OUT_DIR = WORKTREE / "results" / "final"
NAME = "final"
SEED = 42
SPW = 28.0
MAX_ROUNDS = 5000
PRIOR_BEST_ITER = 2822  # seed 42, MAX_ROUNDS 3000, results/lgbm_cw
MAIN_MODELS = WORKTREE.parents[1] / "models"  # main checkout (worktrees live in <main>/.worktrees)


def rss_mb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 1)


def versions():
    return {"python": platform.python_version(), "lightgbm": lgb.__version__,
            "pandas": pd.__version__, "numpy": np.__version__,
            "scikit-learn": sklearn.__version__, "pyarrow": pyarrow.__version__}


def build():
    """Same construction as model_cw (A_hist_nanfix): returns train, valid, test, cols, split stats."""
    raw = load_joined().sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    base_cols = [c for c in raw.columns if c not in DROP]
    df, hist_added = add_history_features(raw, nan_keys="nan")
    del raw
    train, valid, test, stats = time_split(df)
    cols = base_cols + [c for c in hist_added if c != "day"]
    assert not {"day", "TransactionDT", "isFraud", "TransactionID"} & set(cols)
    return train, valid, test, cols, stats


def train_stage():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train, valid, test, cols, stats = build()
    print("split", json.dumps(stats), f"{len(cols)} features", flush=True)
    X, _, _ = prepare(train[cols], valid[cols], test[cols])
    y_tr, y_va, y_te = (d["isFraud"].to_numpy() for d in (train, valid, test))
    dtrain = lgb.Dataset(X["train"], y_tr)
    dvalid = lgb.Dataset(X["valid"], y_va, reference=dtrain)
    params = {**PARAMS, "seed": SEED, "scale_pos_weight": SPW}
    t1 = time.time()
    booster = lgb.train(params, dtrain, num_boost_round=MAX_ROUNDS, valid_sets=[dvalid],
                        valid_names=["valid"],
                        callbacks=[lgb.early_stopping(EARLY_STOP), lgb.log_evaluation(100)])
    train_seconds = time.time() - t1
    best = int(booster.best_iteration)
    s_valid = booster.predict(X["valid"], num_iteration=best)
    s_test = booster.predict(X["test"], num_iteration=best)  # the single new test scoring
    save_artifacts(NAME, booster, valid["TransactionID"], s_valid, test["TransactionID"], s_test)
    m_valid, m_test = metrics(y_va, s_valid), metrics(y_te, s_test)
    info = {
        "git_sha": git_sha(), "versions": versions(),
        "params": {**params, "max_rounds": MAX_ROUNDS, "early_stopping_rounds": EARLY_STOP},
        "best_iteration": best, "prior_best_iteration_max3000": PRIOR_BEST_ITER,
        "stopped_later_than_prior": bool(best > PRIOR_BEST_ITER),
        "hit_max_rounds": bool(best >= MAX_ROUNDS - 1),
        "n_features": len(cols), "split": stats,
        "train_stage": {"fit_seconds": round(train_seconds, 1),
                        "runtime_seconds": round(time.time() - t0, 1), "peak_rss_mb": rss_mb()},
        "valid": m_valid, "test": m_test,
    }
    (OUT_DIR / "train_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2), flush=True)


COMPARE = ("baseline_lgbm", "lgbm_fe1", "lgbm_fe2_A_hist_s42", "lr_raw_clip", "lr_hist_clip")
PERIODS = {"d141-160": (141, 160), "d161-182": (161, 182)}
CALIB_COLORS = {"raw": "tab:gray", "platt": "tab:blue", "isotonic": "tab:red"}


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def _rank_desc(values):
    return pd.Series(values).rank(ascending=False, method="min").astype(int).to_numpy()


def _load_scores(name, ids, root):
    s = pd.read_parquet(root / name / "scores_test.parquet").set_index("TransactionID")["score"]
    return s.reindex(ids).to_numpy()


def analyze_stage():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import average_precision_score

    from ecommerce_fraud import decision as dc

    t0 = time.time()
    tinfo = json.loads((OUT_DIR / "train_info.json").read_text())
    best = tinfo["best_iteration"]
    train, valid, test, cols, stats = build()
    X, _, _ = prepare(train[cols], valid[cols], test[cols])
    del train
    y_va, y_te = valid["isFraud"].to_numpy(), test["isFraud"].to_numpy()
    amt_va, amt_te = valid["TransactionAmt"].to_numpy(), test["TransactionAmt"].to_numpy()
    day_te = (test["TransactionDT"].to_numpy() // 86400).astype(int)
    ids_te = test["TransactionID"].to_numpy()

    sv = pd.read_parquet(models_dir() / NAME / "scores_valid.parquet")
    st = pd.read_parquet(models_dir() / NAME / "scores_test.parquet")
    assert (sv["TransactionID"].to_numpy() == valid["TransactionID"].to_numpy()).all()
    assert (st["TransactionID"].to_numpy() == ids_te).all()
    s_va, s_te = sv["score"].to_numpy(), st["score"].to_numpy()
    m_valid, m_test = metrics(y_va, s_va), metrics(y_te, s_te)
    assert abs(m_test["pr_auc"] - tinfo["test"]["pr_auc"]) < 1e-12

    booster = lgb.Booster(model_file=str(models_dir() / NAME / "booster.txt"))
    chk = np.abs(booster.predict(X["test"].iloc[:3000], num_iteration=best) - s_te[:3000]).max()
    print("reload check max abs diff", chk, flush=True)
    assert chk < 1e-6

    test_days = sorted(set(day_te.tolist()))
    n_days = len(test_days)
    print(f"test days {test_days[0]}..{test_days[-1]} ({n_days})", flush=True)

    # ------------------------------------------------------------- calibration
    h = len(valid) // 2
    p1, y1, p2, y2 = s_va[:h], y_va[:h], s_va[h:], y_va[h:]
    cal_fns = {"raw": lambda q: np.asarray(q), "platt": dc.fit_platt(p1, y1),
               "isotonic": dc.fit_isotonic(p1, y1)}
    cal = {"split": {"first_half_rows": h, "second_half_rows": len(valid) - h,
                     "first_half_days": [int(valid["TransactionDT"].iloc[0] // 86400),
                                         int(valid["TransactionDT"].iloc[h - 1] // 86400)],
                     "second_half_days": [int(valid["TransactionDT"].iloc[h] // 86400),
                                          int(valid["TransactionDT"].iloc[-1] // 86400)],
                     "fraud_rate_first_half": float(y1.mean()),
                     "fraud_rate_second_half": float(y2.mean()),
                     "fraud_rate_valid": float(y_va.mean()), "fraud_rate_test": float(y_te.mean())}}
    cal["valid_second_half"] = {k: dc.calib_report(y2, f(p2)) for k, f in cal_fns.items()}
    chosen = min(("platt", "isotonic"), key=lambda k: cal["valid_second_half"][k]["brier"])
    cal["chosen"] = chosen
    cal["test"] = {k: dc.calib_report(y_te, f(s_te)) for k, f in cal_fns.items()}
    cal["prevalence_note"] = (
        f"fraud rate: valid {y_va.mean():.4f} (1st half {y1.mean():.4f}, 2nd half {y2.mean():.4f}) "
        f"vs test {y_te.mean():.4f}. Mean raw score on test {s_te.mean():.4f}; mean {chosen} "
        f"calibrated p on test {cal_fns[chosen](s_te).mean():.4f} vs observed {y_te.mean():.4f}. "
        f"The calibrator is fitted on valid 1st half only (no test information).")
    p_te = cal_fns[chosen](s_te)
    p2c = cal_fns[chosen](p2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (title, yy, pp) in zip(axes, (("valid 2nd half", y2, p2), ("test", y_te, s_te))):
        ax.plot([0, 1], [0, 1], "k:", lw=1)
        for k in ("raw", chosen):
            rel = dc.reliability(yy, cal_fns[k](pp))
            ax.plot([r[0] for r in rel], [r[1] for r in rel], "-o", color=CALIB_COLORS[k],
                    label=f"{k} (ECE {dc.ece(yy, cal_fns[k](pp)):.3f})")
        ax.set_xlabel("mean predicted probability (10 equal-width bins)")
        ax.set_ylabel("observed fraud rate")
        ax.set_title(f"{title}, n={len(yy):,}, rate {yy.mean():.2%}")
        ax.legend()
    fig.suptitle(f"Reliability: final model, calibrator = {chosen} (fit on valid 1st half)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "calibration.png", dpi=130)
    plt.close(fig)

    # ---------------------------------------------------------------- capacity
    capacity = dc.capacity_table(y_te, amt_te, s_te, day_te)
    capacity["n_test_days"] = n_days
    fr = np.array([0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05,
                   0.075, 0.1])
    curve = {m: [dc.capacity_row(y_te, amt_te, s_te, day_te, f, m == "per_day") for f in fr]
             for m in ("global", "per_day")}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, key, lab in zip(axes, ("precision", "recall", "fraud_amount_share"),
                            ("precision", "recall", "share of fraud amount caught")):
        for m, col in (("global", "tab:blue"), ("per_day", "tab:orange")):
            ax.plot(fr * 100, [r[key] for r in curve[m]], "-o", ms=3, color=col, label=m)
        for f in (0.005, 0.01, 0.02, 0.05):
            ax.axvline(f * 100, color="lightgray", lw=0.8, zorder=0)
        ax.set_xlabel("alert budget k (% of orders, top-k)")
        ax.set_ylabel(lab)
        ax.set_xscale("log")
        ax.set_xticks([0.1, 0.5, 1, 2, 5, 10])
        ax.set_xticklabels(["0.1", "0.5", "1", "2", "5", "10"])
        ax.legend()
    fig.suptitle(f"Capacity curve, test (global vs per-day top-k, {n_days} days)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "capacity_curve.png", dpi=130)
    plt.close(fig)

    # -------------------------------------------------------------------- cost
    amt2 = amt_va[h:]
    eps = 1e-9  # tie-break by the raw score (isotonic gives plateaus)
    ranks_va = {"p": p2c + eps * p2, "p_x_amt": p2c * amt2 + eps * p2}
    ranks_te = {"p": p_te + eps * s_te, "p_x_amt": p_te * amt_te + eps * s_te}
    cost = {"units": "TransactionAmt (currency units as in IEEE-CIS)", "friction": 2.0,
            "loss_no_model_test": float((amt_te * y_te).sum()),
            "loss_no_model_valid_second_half": float((amt2 * y2).sum()), "results": {}}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, c in zip(axes, (2, 5, 15)):
        for rk, col in (("p", "tab:blue"), ("p_x_amt", "tab:orange")):
            thr, loss_v = dc.choose_threshold(y2, amt2, ranks_va[rk], float(c))
            res = dc.apply_threshold(y_te, amt_te, ranks_te[rk], thr, float(c))
            res["valid_second_half_savings"] = 1.0 - loss_v / cost["loss_no_model_valid_second_half"]
            _, loss_all = dc.cost_curve(y_te, amt_te, ranks_te[rk], float(c))
            res["test_best_possible_savings"] = 1.0 - float(loss_all.min()) / loss_all[0]
            res["test_best_possible_share_held"] = int(np.argmin(loss_all)) / len(y_te)
            cost["results"][f"c{c}_{rk}"] = res
            n_plot = int(0.3 * len(y_te))
            ax.plot(np.arange(n_plot + 1) / len(y_te) * 100, 1 - loss_all[:n_plot + 1] / loss_all[0],
                    color=col, label=f"rank by {rk}")
            ax.plot(res["share_held"] * 100, res["savings"], "o", color=col, ms=8,
                    label=f"valid-chosen threshold: {res['savings']:.1%}")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"review cost c = {c} per hold (+2 friction on legit)")
        ax.set_xlabel("orders held (% of test)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("savings = 1 - loss_model / loss_no_model")
    fig.suptitle("Cost sensitivity, test (missed fraud = TransactionAmt)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cost_curve.png", dpi=130)
    plt.close(fig)

    # -------------------------------------------------------------------- SHAP
    rng = np.random.default_rng(0)
    Xv2 = X["valid"].iloc[h:]
    t_s = time.time()
    sample200 = rng.choice(len(Xv2), 200, replace=False)
    booster.predict(Xv2.iloc[sample200], num_iteration=best, pred_contrib=True)
    t200 = time.time() - t_s
    est2000 = t200 * 10
    print(f"SHAP 200 rows {t200:.1f}s (est 2000 rows {est2000:.0f}s)", flush=True)
    n_each = 1000 if est2000 <= 1500 else 500
    fraud_idx = np.flatnonzero(y2 == 1)
    legit_idx = np.flatnonzero(y2 == 0)
    n_each = min(n_each, len(fraud_idx))
    pick = np.sort(np.concatenate([rng.choice(fraud_idx, n_each, replace=False),
                                   rng.choice(legit_idx, n_each, replace=False)]))
    t_s = time.time()
    contrib = booster.predict(Xv2.iloc[pick], num_iteration=best, pred_contrib=True)
    t_shap = time.time() - t_s
    print(f"SHAP {len(pick)} rows {t_shap:.1f}s", flush=True)
    feats = booster.feature_name()
    mean_abs = np.abs(contrib[:, :-1]).mean(axis=0)
    shap_df = pd.DataFrame({"feature": feats, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False).reset_index(drop=True)
    top20 = shap_df.head(20)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top20["feature"][::-1], top20["mean_abs_shap"][::-1], color="tab:blue")
    ax.set_xlabel("mean |SHAP contribution| (log-odds of the spw-28 model)")
    ax.set_title(f"Top 20 features, {len(pick)} valid-2nd-half rows (1:1 fraud/legit)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "shap_summary.png", dpi=130)
    plt.close(fig)

    # reason codes for the 20 highest-scored test rows (no labels written)
    top_idx = np.argsort(-s_te, kind="stable")[:20]
    c20 = booster.predict(X["test"].iloc[top_idx], num_iteration=best, pred_contrib=True)
    rows = []
    for r, (i, cv) in enumerate(zip(top_idx, c20), start=1):
        for j, f_i in enumerate(np.argsort(-cv[:-1], kind="stable")[:3], start=1):
            v = test.iloc[i][feats[f_i]]
            rows.append({"alert_rank": r, "TransactionID": int(ids_te[i]), "score": s_te[i],
                         "calibrated_p": p_te[i], "reason_rank": j, "feature": feats[f_i],
                         "value": "NaN" if pd.isna(v) else v, "contribution": cv[f_i]})
    pd.DataFrame(rows).to_csv(OUT_DIR / "top_alerts_reasons.csv", index=False)

    gain = booster.feature_importance("gain", iteration=best)
    split = booster.feature_importance("split", iteration=best)
    imp = pd.DataFrame({"feature": feats, "gain": gain, "split": split}).sort_values(
        "gain", ascending=False)
    imp.head(50).to_csv(OUT_DIR / "feature_importance.csv", index=False)
    gain_rank = dict(zip(feats, _rank_desc(gain)))
    shap_rank = dict(zip(shap_df["feature"], _rank_desc(shap_df["mean_abs_shap"].to_numpy())))
    shap_val = dict(zip(shap_df["feature"], shap_df["mean_abs_shap"]))
    gain_val = dict(zip(feats, gain))
    parp_names = (["R_emaildomain", "dist1", "dist2"] + [f"M{i}" for i in range(1, 10)]
                  + [f for f in feats if "addr1" in f and f not in ("addr1",)])
    parp = {"n_features": len(feats), "shap_sample_rows": int(len(pick)),
            "note": ("rank 1 = most important of all features; addr1-sharing graph features "
                     "(lgbm_fe3_graph) are not in the final feature set, so the addr1 entries "
                     "are the history features built on the card1_addr1 / uid keys, plus addr1."),
            "signals": {}}
    for f in dict.fromkeys(parp_names + ["addr1"]):
        if f in gain_rank:
            parp["signals"][f] = {"gain": float(gain_val[f]), "gain_rank": int(gain_rank[f]),
                                  "mean_abs_shap": float(shap_val[f]),
                                  "shap_rank": int(shap_rank[f])}
        else:
            parp["signals"][f] = None
    (OUT_DIR / "parp_signals.json").write_text(json.dumps(_clean(parp), indent=2) + "\n")

    # --------------------------------------------------------------- bootstrap
    scores = {"final": s_te}
    for m in COMPARE:
        scores[m] = _load_scores(m, ids_te, MAIN_MODELS)
        assert not np.isnan(scores[m]).any(), m
    # exactness of the weighted metrics against sklearn / baseline top_k on a materialised draw
    wm = dc.WeightedMetrics(y_te, s_te)
    mult = rng.integers(0, 3, size=len(test_days)).astype(float)
    w_rows = mult[np.searchsorted(test_days, day_te)]
    rows_idx = np.repeat(np.arange(len(y_te)), w_rows.astype(int))
    ap_w, p1_w = wm.compute(w_rows)
    ap_s = average_precision_score(y_te[rows_idx], s_te[rows_idx])
    from ecommerce_fraud.baseline_lgbm import top_k
    p1_s = top_k(y_te[rows_idx], s_te[rows_idx], 0.01)[0]
    print(f"weighted-metric check AP {ap_w:.10f} vs {ap_s:.10f}; p@1% {p1_w:.10f} vs {p1_s:.10f}",
          flush=True)
    assert abs(ap_w - ap_s) < 1e-9 and abs(p1_w - p1_s) < 1e-9
    t_b = time.time()
    boot, point = dc.day_block_bootstrap(y_te, day_te, scores, n_boot=1000, seed=0)
    print(f"bootstrap {time.time() - t_b:.0f}s", flush=True)
    assert abs(point["final"][0] - m_test["pr_auc"]) < 1e-9
    assert abs(point["final"][1] - m_test["precision_at_top_1pct"]) < 1e-3
    per_model = {}
    for k, arr in boot.items():
        per_model[k] = {"point": {"pr_auc": point[k][0], "precision_at_top_1pct": point[k][1]},
                        "pr_auc": dc.quantiles(arr[:, 0]),
                        "precision_at_top_1pct": dc.quantiles(arr[:, 1])}

    def paired(other):
        d = boot["final"] - boot[other]
        out = {}
        for j, mname in enumerate(("pr_auc", "precision_at_top_1pct")):
            out[mname] = {"point": point["final"][j] - point[other][j],
                          **dc.quantiles(d[:, j]), "share_resamples_positive": float((d[:, j] > 0).mean())}
        return out

    bootstrap = {"n_resamples": 1000, "seed": 0, "unit": f"{n_days} test days",
                 "per_model": per_model,
                 "paired_delta_vs_baseline": paired("baseline_lgbm"),
                 "paired_delta_vs_fe2": paired("lgbm_fe2_A_hist_s42")}

    per_period = {}
    for pname, (lo, hi) in PERIODS.items():
        m = (day_te >= lo) & (day_te <= hi)
        per_period[pname] = {"n_days": len(set(day_te[m].tolist())),
                             **{k: metrics(y_te[m], v[m]) for k, v in scores.items()}}

    # ------------------------------------------------------------------ output
    lcw = json.loads((WORKTREE / "results" / "lgbm_cw" / "metrics.json").read_text())
    spw28 = lcw["selection"]["candidates"]["spw28"]
    config = {
        "feature_set": "A_hist_nanfix", "history_nan_keys": "nan", "n_features": len(cols),
        "scale_pos_weight": SPW, "seed": SEED, "params": PARAMS, "max_rounds": MAX_ROUNDS,
        "early_stopping_rounds": EARLY_STOP, "best_iteration": best,
        "selection_rule": "highest mean valid PR-AUC, results/lgbm_cw best_on_valid",
        "valid_pr_auc_mean_3_seeds": spw28["valid_pr_auc_mean"],
        "valid_pr_auc_sd_3_seeds": spw28["valid_pr_auc_sd"],
        "prior_best_iteration_max3000": PRIOR_BEST_ITER,
        "stopped_later_than_prior": tinfo["stopped_later_than_prior"],
        "hit_max_rounds": tinfo["hit_max_rounds"],
        "valid_note": "valid metrics use full-valid early stopping, so they are optimistic",
    }
    result = {"model": NAME, "config": config, "split": stats, "valid": m_valid, "test": m_test,
              "calibration": cal, "capacity": capacity, "cost": cost, "bootstrap": bootstrap,
              "per_period": per_period,
              "shap": {"timing_200_rows_s": round(t200, 1), "rows": int(len(pick)),
                       "seconds": round(t_shap, 1),
                       "top20": top20.to_dict(orient="records")},
              "test_view_note": "single new test scoring under ADR-002"}
    (OUT_DIR / "metrics.json").write_text(json.dumps(_clean(result), indent=2) + "\n")
    info = {k: tinfo[k] for k in ("git_sha", "versions", "params", "best_iteration",
                                  "stopped_later_than_prior", "hit_max_rounds", "n_features")}
    info["train_stage"] = tinfo["train_stage"]
    info["analyze_stage"] = {"runtime_seconds": round(time.time() - t0, 1),
                             "peak_rss_mb": rss_mb(), "shap_seconds": round(t_shap, 1)}
    info["runtime_seconds"] = round(tinfo["train_stage"]["runtime_seconds"]
                                    + info["analyze_stage"]["runtime_seconds"], 1)
    info["peak_rss_mb"] = max(tinfo["train_stage"]["peak_rss_mb"], rss_mb())
    (OUT_DIR / "run_info.json").write_text(json.dumps(_clean(info), indent=2) + "\n")
    print("done", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "train"
    {"train": train_stage, "analyze": analyze_stage}[stage]()
