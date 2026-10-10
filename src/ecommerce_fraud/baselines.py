"""Fixed reference baselines (rule, logistic regression, isolation forest) on the time split.

Run: python -m ecommerce_fraud.baselines  (PYTHONPATH=src)
Only lr_raw_clip / lr_hist_clip: add --clip-only (DIAG_SCORES_DIR = models dir holding the
lr_raw and iforest scores, for diagnostics.json).
"""
from __future__ import annotations

import gc
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ecommerce_fraud.artifacts import models_dir, save_artifacts
from ecommerce_fraud.baseline_lgbm import DROP, git_sha, metrics
from ecommerce_fraud.data import WORKTREE, load_joined, time_split
from ecommerce_fraud.features_history import add_history_features

OUT_DIR = WORKTREE / "results" / "baselines"
HIST_INPUT = ["TransactionID", "TransactionDT", "isFraud", "TransactionAmt", "D1", "D4", "D10",
              "D15", "card1", "addr1", "P_emaildomain", "DeviceInfo"]
KS, RS = (1, 2, 3, 5), (2, 3, 5, 10)
LR_MAX_ITER = 1500
MAX_LEVELS = 20


def prf(y, flag):
    tp = float((flag & (y == 1)).sum())
    p = tp / max(1, flag.sum())
    r = tp / y.sum()
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def rank_score(cnt, ratio):
    r1 = pd.Series(cnt).fillna(-np.inf).rank(method="average").to_numpy()
    r2 = pd.Series(ratio).fillna(-np.inf).rank(method="average").to_numpy()
    return (r1 + r2) / 2


def run_rule(parts, y):
    v = parts["valid"]
    best = None
    for k in KS:
        for r in RS:
            flag = ((v["uid_cnt_1d"] >= k) | (v["uid_amt_ratio"] >= r)).to_numpy()
            f = prf(y["valid"], flag)[2]
            if best is None or f > best[0]:
                best = (f, k, r)
    _, k, r = best
    res, scores = {}, {}
    for name in ("valid", "test"):
        d = parts[name]
        s = rank_score(d["uid_cnt_1d"].to_numpy(), d["uid_amt_ratio"].to_numpy())
        res[name] = metrics(y[name], s)
        scores[name] = s
        flag = ((d["uid_cnt_1d"] >= k) | (d["uid_amt_ratio"] >= r)).to_numpy()
        p, rc, f = prf(y[name], flag)
        res[name].update({"flag_precision": p, "flag_recall": rc, "flag_f1": f,
                          "flag_rate": float(flag.mean())})
    res["params"] = {"k": k, "r": r, "valid_f1": best[0], "grid_k": list(KS), "grid_r": list(RS),
                     "score": "rank-average of uid_cnt_1d and uid_amt_ratio (NaN lowest)"}
    return res, scores


def fit_transform(X, cols, cat_cols, clip=False):
    lo = hi = None
    if clip:  # train 1st-99th percentile, fitted on train only; NaN stays NaN
        tr = X["train"][cols].to_numpy(np.float32)
        lo, hi = (np.nan_to_num(v).astype(np.float32)
                  for v in np.nanpercentile(tr, [1, 99], axis=0))
        del tr
    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    enc = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)

    def cats(k):
        c = X[k][cat_cols].astype(object)
        return c.where(c.notna(), "nan")

    enc.fit(cats("train"))
    mats = {}
    for k in ("train", "valid", "test"):
        raw = X[k][cols].to_numpy(np.float32)
        if clip:
            raw = np.clip(raw, lo, hi)
        a = imp.fit_transform(raw) if k == "train" else imp.transform(raw)
        del raw
        a = sc.fit_transform(a) if k == "train" else sc.transform(a)
        mats[k] = sp.hstack([sp.csr_matrix(a.astype(np.float32)), enc.transform(cats(k))],
                            format="csr", dtype=np.float32)
    return mats


def run_lr(X, y, cols, cat_cols, ids, name, clip=False):
    mats = fit_transform(X, cols, cat_cols, clip)
    clf = LogisticRegression(class_weight="balanced", max_iter=LR_MAX_ITER, solver="lbfgs", C=1.0)
    clf.fit(mats["train"], y["train"])
    res, scores = {}, {}
    for k in ("valid", "test"):
        s = clf.decision_function(mats[k])
        scores[k] = s
        res[k] = metrics(y[k], s)
    res["params"] = {"max_iter": LR_MAX_ITER, "n_iter_used": int(clf.n_iter_[0]),
                     "converged": bool(clf.n_iter_[0] < LR_MAX_ITER), "C": 1.0,
                     "class_weight": "balanced", "n_numeric": len(cols),
                     "n_categorical": len(cat_cols),
                     "n_matrix_columns": int(mats["train"].shape[1])}
    if clip:
        res["params"]["clip"] = "numeric columns clipped to train 1st-99th percentile"
    save_artifacts(name, None, ids["valid"], scores["valid"], ids["test"], scores["test"])
    return res


def run_iforest(X, y, cols, ids):
    imp = SimpleImputer(strategy="median")
    tr = imp.fit_transform(X["train"][cols].to_numpy(np.float32))
    m = IsolationForest(n_estimators=200, random_state=42, n_jobs=4).fit(tr)
    del tr
    res, scores = {}, {}
    for k in ("valid", "test"):
        s = -m.score_samples(imp.transform(X[k][cols].to_numpy(np.float32)))
        scores[k] = s
        res[k] = metrics(y[k], s)
    res["params"] = {"n_estimators": 200, "random_state": 42, "n_numeric": len(cols),
                     "fit_on": "train rows only"}
    save_artifacts("iforest", None, ids["valid"], scores["valid"], ids["test"], scores["test"])
    return res


def top1_diag(score, y, day):
    k = max(1, int(round(0.01 * len(score))))
    idx = np.argsort(-score, kind="stable")[:k]
    q = np.percentile(day[idx], [5, 50, 95])
    return {"n_top": int(k), "top1pct_fraud_rate": float(y[idx].mean()),
            "day_q05": float(q[0]), "day_q50": float(q[1]), "day_q95": float(q[2])}


def run_clip(parts, y, ids, num_cols, hist_cols, cat_cols, t0):
    """Run only lr_raw_clip / lr_hist_clip; merge into metrics.json, write diagnostics.json."""
    mpath = OUT_DIR / "metrics.json"
    allm = json.loads(mpath.read_text())
    new = {"lr_raw_clip": run_lr(parts, y, num_cols, cat_cols, ids, "lr_raw_clip", clip=True)}
    gc.collect()
    new["lr_hist_clip"] = run_lr(parts, y, num_cols + hist_cols, cat_cols, ids, "lr_hist_clip",
                                 clip=True)
    for n, m in new.items():
        allm["models"][n] = {"valid": m["valid"], "test": m["test"], "params": m["params"]}
        print(n, round(m["valid"]["pr_auc"], 4), round(m["test"]["pr_auc"], 4), flush=True)
    mpath.write_text(json.dumps(allm, indent=2) + "\n")
    day = (parts["test"]["TransactionDT"].to_numpy() // 86400).astype(int)
    diag = {"definition": "top 1% of test rows by score; day = TransactionDT // 86400"}
    ref = Path(os.environ.get("DIAG_SCORES_DIR", ""))
    for n in ("lr_raw", "lr_raw_clip", "iforest"):
        f = (models_dir() if n == "lr_raw_clip" else ref) / n / "scores_test.parquet"
        sc = pd.read_parquet(f).set_index("TransactionID")["score"].reindex(ids["test"])
        assert sc.notna().all()
        diag[n] = top1_diag(sc.to_numpy(), y["test"], day)
        print(n, json.dumps(diag[n]), flush=True)
    (OUT_DIR / "diagnostics.json").write_text(json.dumps(diag, indent=2) + "\n")
    print(f"runtime {time.time() - t0:.0f}s peak_rss_mb "
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2:.0f}", flush=True)


def main():
    clip_only = "--clip-only" in sys.argv
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_joined()
    df = df.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    hist_df, hist_added = add_history_features(df[HIST_INPUT].copy())
    hist_cols = [c for c in hist_added if c != "day"]
    hist_df = hist_df[hist_cols].astype(np.float32)
    feats = [c for c in df.columns if c not in DROP]
    for c in feats:
        if pd.api.types.is_bool_dtype(df[c]) or pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(np.float32)
    df = pd.concat([df, hist_df], axis=1)
    del hist_df
    gc.collect()
    train, valid, test, stats = time_split(df)
    del df
    gc.collect()
    print("split", json.dumps(stats), flush=True)
    parts = {"train": train, "valid": valid, "test": test}
    y = {k: v["isFraud"].to_numpy() for k, v in parts.items()}
    ids = {k: v["TransactionID"].to_numpy() for k, v in parts.items()}
    num_cols = [c for c in feats if pd.api.types.is_numeric_dtype(train[c])]
    obj_cols = [c for c in feats if c not in num_cols]
    cat_cols = [c for c in obj_cols if train[c].nunique() <= MAX_LEVELS]
    print(f"{len(num_cols)} numeric, {len(obj_cols)} object, {len(cat_cols)} one-hot", flush=True)

    if clip_only:
        return run_clip(parts, y, ids, num_cols, hist_cols, cat_cols, t0)
    models = {}
    r, s = run_rule(parts, y)
    models["rule"] = r
    save_artifacts("rule", None, ids["valid"], s["valid"], ids["test"], s["test"])
    print("rule", json.dumps(r["params"]), flush=True)

    models["lr_raw"] = run_lr(parts, y, num_cols, cat_cols, ids, "lr_raw")
    print("lr_raw", json.dumps(models["lr_raw"]["params"]), models["lr_raw"]["valid"]["pr_auc"],
          flush=True)
    gc.collect()
    models["lr_hist"] = run_lr(parts, y, num_cols + hist_cols, cat_cols, ids, "lr_hist")
    print("lr_hist", json.dumps(models["lr_hist"]["params"]),
          models["lr_hist"]["valid"]["pr_auc"], flush=True)
    gc.collect()
    models["iforest"] = run_iforest(parts, y, num_cols, ids)

    out_models = {n: {"valid": m["valid"], "test": m["test"], "params": m["params"]}
                  for n, m in models.items()}
    (OUT_DIR / "metrics.json").write_text(
        json.dumps({"models": out_models, "split": stats}, indent=2) + "\n")
    info = {
        "git_sha": git_sha(),
        "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                     "numpy": np.__version__, "scikit-learn": sklearn.__version__},
        "hist_columns": hist_cols,
        "runtime_seconds": round(time.time() - t0, 1),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 1),
    }
    (OUT_DIR / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({k: (round(v["valid"]["pr_auc"], 4), round(v["test"]["pr_auc"], 4))
                      for k, v in out_models.items()}), flush=True)
    print(f"runtime {info['runtime_seconds']}s peak_rss_mb {info['peak_rss_mb']}", flush=True)


if __name__ == "__main__":
    main()
