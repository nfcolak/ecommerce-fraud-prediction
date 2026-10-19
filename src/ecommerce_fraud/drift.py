"""Drift scan: which features shift over time, and does removing them help?

Pre-registered in Artifacts/ecommerce-fraud-prediction/graph/PREREGISTRATION.md (amendment 2026-10-16 13:10).

  python -m ecommerce_fraud.drift --scan      per-feature drift (early vs late TRAIN rows; train vs valid for info)
  python -m ecommerce_fraud.drift --screen    valid PR-AUC (seed 42) of final minus drifting features, then
                                              every block minus its drifting columns on top of the chosen base
Results: results/drift/{feature_drift.csv, scan.json, screen.json}. Resumable (finished runs are kept).
No labels are used for the drift score; labels only train/evaluate the screen models (as in improve.py).
"""
from __future__ import annotations

import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ecommerce_fraud import blocks as blk
from ecommerce_fraud import improve as imp
from ecommerce_fraud.baseline_lgbm import metrics
from ecommerce_fraud.data import CACHE_PATH, WORKTREE, load_joined

OUT = WORKTREE / "results" / "drift"
DAY = 86400
EARLY_END_DAY = 88          # early train = days <= 88, late train = days 89..111
N_SAMPLE = 60_000
THRESHOLDS = (0.70, 0.60)
REF_FINAL = 0.6949933428746624   # results/improve/screen/final.json (seed 42)
MIN_GAIN = 0.002
ADV_PARAMS = {"objective": "binary", "num_leaves": 15, "learning_rate": 0.1, "min_child_samples": 200,
              "verbosity": -1, "n_jobs": 12, "seed": 0}


def log(*a):
    print(f"[{time.time() - imp.T0:7.0f}s]", *a, flush=True)


def day_of(ids) -> np.ndarray:
    load_joined()
    dt = pd.read_parquet(CACHE_PATH, columns=["TransactionID", "TransactionDT"]).set_index("TransactionID")
    return (dt["TransactionDT"].reindex(ids).to_numpy() // DAY).astype(int)


def adv_auc(x0: pd.Series, x1: pd.Series, seed=0) -> float:
    """Drift score max(AUC, 1-AUC) of a 1-feature LightGBM telling sample 0 from sample 1 (50/50 holdout)."""
    x = pd.concat([x0, x1], ignore_index=True).to_frame("f")
    y = np.r_[np.zeros(len(x0)), np.ones(len(x1))]
    rng = np.random.default_rng(seed)
    te = rng.random(len(y)) < 0.5
    if x["f"].nunique(dropna=False) < 2:
        return 0.5
    m = lgb.train(ADV_PARAMS, lgb.Dataset(x[~te], y[~te]), num_boost_round=50)
    a = roc_auc_score(y[te], m.predict(x[te]))
    return float(max(a, 1 - a))


def all_frames():
    """Train/valid matrices of final's features plus every block (valid-window fit_end_dt)."""
    b = imp.make_base()
    names = sorted(blk.discover())
    tr, va = [b.X["train"]], [b.X["valid"]]
    owner = {c: "final" for c in b.cols}
    for n in names:
        try:
            ftr, _ = imp.block_columns([n], b.fit_end_dt, b.ids_tr, b.X["train"].index)
            fva, _ = imp.block_columns([n], b.fit_end_dt, b.ids_va, b.X["valid"].index)
        except Exception as exc:
            log(f"block {n} skipped: {exc!r}")
            continue
        tr.append(ftr[0]); va.append(fva[0])
        owner.update({c: n for c in ftr[0].columns})
    return b, pd.concat(tr, axis=1), pd.concat(va, axis=1), owner


def cmd_scan():
    OUT.mkdir(parents=True, exist_ok=True)
    b, Xtr, Xva, owner = all_frames()
    d_tr = day_of(b.ids_tr)
    rng = np.random.default_rng(0)
    early = np.flatnonzero(d_tr <= EARLY_END_DAY)
    late = np.flatnonzero(d_tr > EARLY_END_DAY)
    e_s = rng.choice(early, min(N_SAMPLE, len(early)), replace=False)
    l_s = rng.choice(late, min(N_SAMPLE, len(late)), replace=False)
    t_s = rng.choice(len(Xtr), min(N_SAMPLE, len(Xtr)), replace=False)
    v_s = rng.choice(len(Xva), min(N_SAMPLE, len(Xva)), replace=False)
    log(f"scan {Xtr.shape[1]} features; early {len(early)} late {len(late)} rows (days <= {EARLY_END_DAY} / after)")
    rows = []
    for i, c in enumerate(Xtr.columns):
        rows.append({"feature": c, "owner": owner[c],
                     "drift_train": adv_auc(Xtr[c].iloc[e_s], Xtr[c].iloc[l_s]),
                     "drift_train_vs_valid_info": adv_auc(Xtr[c].iloc[t_s], Xva[c].iloc[v_s])})
        if i % 100 == 0:
            log(f"  {i}/{Xtr.shape[1]} {c}: {rows[-1]['drift_train']:.3f}")
    df = pd.DataFrame(rows).sort_values("drift_train", ascending=False)
    df.to_csv(OUT / "feature_drift.csv", index=False)
    summ = {"n_features": len(df), "early_end_day": EARLY_END_DAY, "n_sample": N_SAMPLE,
            "share_above": {str(t): {o: float((g["drift_train"] > t).mean()) for o, g in df.groupby("owner")}
                            for t in THRESHOLDS},
            "count_above": {str(t): int((df["drift_train"] > t).sum()) for t in THRESHOLDS},
            "top20": df.head(20)[["feature", "owner", "drift_train", "drift_train_vs_valid_info"]].to_dict("records")}
    (OUT / "scan.json").write_text(json.dumps(summ, indent=2) + "\n")
    log("scan done:", summ["count_above"])


def run_valid(b, Xtr, Xva, cols, name, done):
    if name in done:
        return done[name]
    t = time.time()
    booster, best, predict = imp.fit_lgbm(Xtr[cols], b.y_tr, Xva[cols], b.y_va, imp.SEED, log_every=500,
                                          verbose=False)
    m = metrics(b.y_va, predict(Xva[cols]))
    done[name] = {"valid_pr_auc": m["pr_auc"], "best_iteration": best, "n_features": len(cols),
                  "seconds": round(time.time() - t, 1)}
    (OUT / "screen.json").write_text(json.dumps(done, indent=2) + "\n")
    log(f"{name}: valid PR-AUC {m['pr_auc']:.5f} best_it {best} n_feat {len(cols)}")
    del booster
    return done[name]


def cmd_screen():
    drift = pd.read_csv(OUT / "feature_drift.csv").set_index("feature")["drift_train"]
    p = OUT / "screen.json"
    done = json.loads(p.read_text()) if p.exists() else {}
    b, Xtr, Xva, owner = all_frames()
    base_cols = list(b.cols)
    res = {}
    for t in THRESHOLDS:
        keep = [c for c in base_cols if drift.get(c, 0.5) <= t]
        res[t] = (keep, run_valid(b, Xtr, Xva, keep, f"final_nodrift{int(t * 100)}", done))
    best_t = max(THRESHOLDS, key=lambda t: res[t][1]["valid_pr_auc"])
    if res[best_t][1]["valid_pr_auc"] - REF_FINAL > MIN_GAIN:
        base, base_auc, thr, base_name = res[best_t][0], res[best_t][1]["valid_pr_auc"], best_t, f"final_nodrift{int(best_t * 100)}"
    else:
        base, base_auc, thr, base_name = base_cols, REF_FINAL, 0.70, "final"
    done["_choice"] = {"base": base_name, "base_valid_pr_auc": base_auc, "block_threshold": thr}
    log("base:", done["_choice"])
    for n in sorted({o for o in owner.values() if o != "final"}):
        cols = [c for c, o in owner.items() if o == n and drift.get(c, 0.5) <= thr]
        r = run_valid(b, Xtr, Xva, base + cols, f"{base_name}+{n}_nodrift", done) if cols else {"valid_pr_auc": None}
        done[f"{base_name}+{n}_nodrift"] = {**r, "n_block_cols_kept": len(cols),
                                            "n_block_cols": sum(1 for o in owner.values() if o == n),
                                            "delta_vs_base": (r["valid_pr_auc"] - base_auc) if r["valid_pr_auc"] else None}
        (OUT / "screen.json").write_text(json.dumps(done, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--screen", action="store_true")
    a = ap.parse_args()
    if a.scan:
        cmd_scan()
    if a.screen:
        cmd_screen()


if __name__ == "__main__":
    main()
