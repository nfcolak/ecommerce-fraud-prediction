"""XGBoost and CatBoost ensemble members plus a rank blend.

Frames are the output of baseline_lgbm.prepare (numeric + pandas category columns).
Each fit_* returns (model, best_iteration, predict_fn); predict_fn(X) -> fraud score.
Plumbing check (PYTHONPATH=src): python -m ecommerce_fraud.members
"""
from __future__ import annotations

import resource
import time

import numpy as np
import pandas as pd

SPW = 28.0
MAX_ROUNDS = 5000
EARLY_STOP = 200
THREADS = 12


def _cat_cols(X):
    return [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)
            or X[c].dtype == object]


def _for_xgb(X):
    X = X.copy()
    for c in _cat_cols(X):
        X[c] = X[c].astype("category")
    return X


def _for_cat(X, cat_cols):
    X = X.copy()
    for c in cat_cols:
        X[c] = X[c].astype(object).where(X[c].notna(), "nan").astype(str)
    for c in X.columns.difference(cat_cols):
        if X[c].dtype == bool:
            X[c] = X[c].astype("int8")
    return X


def fit_xgb(X_fit, y_fit, X_es, y_es, seed):
    import xgboost as xgb
    Xf, Xe = _for_xgb(X_fit), _for_xgb(X_es)
    # align category dictionaries so train/es codes agree
    for c in _cat_cols(Xf):
        Xe[c] = pd.Categorical(Xe[c], categories=Xf[c].cat.categories)
    dfit = xgb.DMatrix(Xf, np.asarray(y_fit), enable_categorical=True, nthread=THREADS)
    des = xgb.DMatrix(Xe, np.asarray(y_es), enable_categorical=True, nthread=THREADS)
    params = {"tree_method": "hist", "nthread": THREADS, "objective": "binary:logistic",
              "scale_pos_weight": SPW, "eval_metric": "aucpr", "eta": 0.03, "max_depth": 8,
              "subsample": 0.8, "colsample_bytree": 0.5, "seed": int(seed),
              "max_cat_to_onehot": 1}
    booster = xgb.train(params, dfit, num_boost_round=MAX_ROUNDS, evals=[(des, "es")],
                        early_stopping_rounds=EARLY_STOP, verbose_eval=False)
    best = int(booster.best_iteration) + 1
    cats = {c: Xf[c].cat.categories for c in _cat_cols(Xf)}

    def predict(X):
        Xp = _for_xgb(X)
        for c, k in cats.items():
            Xp[c] = pd.Categorical(Xp[c], categories=k)
        return booster.predict(xgb.DMatrix(Xp, enable_categorical=True, nthread=THREADS),
                               iteration_range=(0, best))
    return booster, best, predict


def fit_cat(X_fit, y_fit, X_es, y_es, seed):
    from catboost import CatBoostClassifier, Pool
    cat_cols = _cat_cols(X_fit)
    Xf, Xe = _for_cat(X_fit, cat_cols), _for_cat(X_es, cat_cols)
    pf = Pool(Xf, np.asarray(y_fit), cat_features=cat_cols or None)
    pe = Pool(Xe, np.asarray(y_es), cat_features=cat_cols or None)
    model = CatBoostClassifier(
        iterations=MAX_ROUNDS, learning_rate=0.05, depth=8, scale_pos_weight=SPW,
        auto_class_weights=None, eval_metric="PRAUC", loss_function="Logloss",
        od_type="Iter", od_wait=EARLY_STOP, use_best_model=True,
        thread_count=THREADS, random_seed=int(seed), verbose=False)
    model.fit(pf, eval_set=pe)
    best = int(model.get_best_iteration()) + 1

    def predict(X):
        return model.predict_proba(Pool(_for_cat(X, cat_cols), cat_features=cat_cols or None))[:, 1]
    return model, best, predict


# ---------------------------------------------------------------- round p4: xgb2 / cat2
# The graph round's xgb/cat were misconfigured (xgb never early-stopped at 5000 rounds, cat built
# CTRs on high-cardinality ids and stopped on PRAUC). xgb/cat above stay untouched (reproducibility).
# Optional keywords (params / early_stop / max_rounds) are the arm-contract overrides.
XGB2_PARAMS = {
    "tree_method": "hist", "objective": "binary:logistic", "eval_metric": "aucpr",
    "scale_pos_weight": SPW, "eta": 0.05, "grow_policy": "lossguide", "max_depth": 0,
    "max_leaves": 256, "min_child_weight": 30, "subsample": 0.8, "colsample_bytree": 0.5,
    "reg_lambda": 5.0, "max_bin": 256, "max_cat_to_onehot": 8, "max_cat_threshold": 16,
}
CAT2_PARAMS = {
    "iterations": 5000, "learning_rate": 0.08, "depth": 8, "scale_pos_weight": 8.0,
    "loss_function": "Logloss", "eval_metric": "Logloss", "l2_leaf_reg": 5.0,
    "rsm": 0.5, "border_count": 254, "one_hot_max_size": 8, "max_ctr_complexity": 1,
}
CAT2_LOWCARD = 12  # categorical columns with <= this many levels keep CatBoost's categorical handling


def fit_xgb2(X_fit, y_fit, X_es, y_es, seed, params=None, early_stop=100, max_rounds=MAX_ROUNDS):
    import xgboost as xgb
    Xf, Xe = _for_xgb(X_fit), _for_xgb(X_es)
    for c in _cat_cols(Xf):
        Xe[c] = pd.Categorical(Xe[c], categories=Xf[c].cat.categories)
    dfit = xgb.DMatrix(Xf, np.asarray(y_fit), enable_categorical=True, nthread=THREADS)
    des = xgb.DMatrix(Xe, np.asarray(y_es), enable_categorical=True, nthread=THREADS)
    p = {**XGB2_PARAMS, **(params or {}), "nthread": THREADS, "seed": int(seed)}
    booster = xgb.train(p, dfit, num_boost_round=int(max_rounds), evals=[(des, "es")],
                        early_stopping_rounds=int(early_stop), verbose_eval=False)
    best = int(booster.best_iteration) + 1
    cats = {c: Xf[c].cat.categories for c in _cat_cols(Xf)}

    def predict(X):
        Xp = _for_xgb(X)
        for c, k in cats.items():
            Xp[c] = pd.Categorical(Xp[c], categories=k)
        return booster.predict(xgb.DMatrix(Xp, enable_categorical=True, nthread=THREADS),
                               iteration_range=(0, best))
    return booster, best, predict


def _cat2_frame(X, cat_cols, num_codes, levels):
    """Low-cardinality columns -> strings (CatBoost categorical); high-cardinality -> float codes."""
    X = X.copy()
    for c in cat_cols:
        X[c] = X[c].astype(object).where(X[c].notna(), "nan").astype(str)
    for c in num_codes:  # codes from the fit dictionary, unseen/NaN -> NaN (like LightGBM)
        X[c] = pd.Categorical(X[c], categories=levels[c]).codes.astype("float32")
        X.loc[X[c] < 0, c] = np.nan
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype("int8")
    return X


def fit_cat2(X_fit, y_fit, X_es, y_es, seed, params=None, early_stop=100, max_rounds=MAX_ROUNDS):
    from catboost import CatBoostClassifier, Pool
    cats = _cat_cols(X_fit)
    levels = {c: pd.Index(X_fit[c].dropna().unique()).sort_values() for c in cats}
    low = [c for c in cats if len(levels[c]) <= CAT2_LOWCARD]
    high = [c for c in cats if c not in low]
    mk = lambda X: _cat2_frame(X, low, high, levels)
    pf = Pool(mk(X_fit), np.asarray(y_fit), cat_features=low or None)
    pe = Pool(mk(X_es), np.asarray(y_es), cat_features=low or None)
    p = {**CAT2_PARAMS, **(params or {})}
    p["iterations"] = int(max_rounds) if "iterations" not in (params or {}) else p["iterations"]
    model = CatBoostClassifier(
        **p, od_type="Iter", od_wait=int(early_stop), use_best_model=True,
        thread_count=THREADS, random_seed=int(seed), verbose=False)
    model.fit(pf, eval_set=pe)
    best = int(model.get_best_iteration()) + 1

    def predict(X):
        return model.predict_proba(Pool(mk(X), cat_features=low or None))[:, 1]
    return model, best, predict


_FITS = {"xgb": fit_xgb, "cat": fit_cat, "xgb2": fit_xgb2, "cat2": fit_cat2}


def get(name):
    """Fit function by member name: xgb, cat, xgb2, cat2 (all fit(Xf, yf, Xe, ye, seed))."""
    return _FITS[name]


def rank_blend(score_dict, weights=None):
    """Weighted mean of per-model rank-normalized scores (ranks scaled to (0, 1])."""
    names = list(score_dict)
    w = np.ones(len(names)) if weights is None else np.array(
        [weights[n] for n in names] if isinstance(weights, dict) else weights, dtype=float)
    w = w / w.sum()
    out = 0.0
    for wi, n in zip(w, names):
        s = np.asarray(score_dict[n], dtype=float)
        out = out + wi * pd.Series(s).rank(method="average").to_numpy() / len(s)
    return out


def _rss_mb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 1)


if __name__ == "__main__":
    from sklearn.metrics import average_precision_score

    from ecommerce_fraud.baseline_lgbm import prepare
    from ecommerce_fraud.final import build

    train, valid, test, cols, _ = build()
    n_full = len(train)
    sub = train.iloc[:50_000]
    del valid, test
    X, _, _ = prepare(sub[cols], sub[cols].iloc[:1], sub[cols].iloc[:1])
    Xa = X["train"]
    y = sub["isFraud"].to_numpy()
    Xf, yf, Xe, ye = Xa.iloc[:40_000], y[:40_000], Xa.iloc[40_000:], y[40_000:]
    print(f"rows fit={len(Xf)} es={len(Xe)} fraud es={int(ye.sum())} full train={n_full}", flush=True)
    scores = {}
    for name, fn in (("xgb", fit_xgb), ("cat", fit_cat)):
        t = time.time()
        _, best, pred = fn(Xf, yf, Xe, ye, 42)
        sec = time.time() - t
        scores[name] = pred(Xe)
        ap = average_precision_score(ye, scores[name])
        est = sec * n_full / len(Xf)
        print(f"{name}: fit {sec:.1f}s best_iter {best} es PR-AUC {ap:.4f} rss {_rss_mb()}MB "
              f"full-data estimate (linear in rows, same iters) {est/60:.1f} min", flush=True)
    bl = rank_blend(scores)
    print(f"blend es PR-AUC {average_precision_score(ye, bl):.4f} peak rss {_rss_mb()}MB", flush=True)
    print("MEMBERS OK")
