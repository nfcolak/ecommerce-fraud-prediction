"""Refit after early stopping (arm model 'custom').

Stage 1 is exactly protocol.fit_lgbm (baseline PARAMS + seed + spw 28, early stopping 100 on Xe, max 5000
rounds) -> best_iteration B. Stage 2: same params and seed, trained on concat(Xf, Xe) for round(B * factor)
rounds, no early stopping; scores of stage 2 on Xs are returned.

Leakage: Xe (days S-21..S-8) is labelled data strictly before the 7-day gap and the score window; training
on it is legitimate. On the 'valid' screen the ES set IS the scored window, so the fit refuses there.
Env: EF_SMOKE_N_JOBS (n_jobs, else 12); EF_SMOKE_MAX_ROUNDS caps both stages (smoke only).
Self-check: python -m ecommerce_fraud.refit --smoke   (synthetic data, no real data loaded)
"""
from __future__ import annotations

import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from ecommerce_fraud.baseline_lgbm import PARAMS

SPW = 28.0
ES = 100
MAX_ROUNDS = 5000


def _cap():
    v = os.environ.get("EF_SMOKE_MAX_ROUNDS")
    return int(v) if v else None


def fit_refit(Xf, yf, Xe, ye, Xs, seed, meta, factor=1.0, _return_stage1=False):
    m = sys.modules.get("__main__")   # `python -m ecommerce_fraud.protocol` runs as __main__ (its CFG is set there)
    protocol = m if hasattr(m, "fit_lgbm") and hasattr(m, "CFG") else __import__("ecommerce_fraud.protocol", fromlist=["x"])
    if meta.get("window") == "valid":
        raise ValueError("refit arms must not run on the 'valid' window: the ES rows are the scored rows")
    if list(Xf.columns) != list(Xe.columns) or list(Xf.columns) != list(Xs.columns):
        raise ValueError("Xf/Xe/Xs columns differ")
    if "ids_fit" in meta:   # no score row may enter the stage-2 training set
        ids_tr = np.concatenate([meta["ids_fit"], meta["ids_es"]])
        if len(ids_tr) != len(Xf) + len(Xe) or np.isin(meta["ids_score"], ids_tr).any():
            raise ValueError("fit/ES/score rows overlap or sizes mismatch")
    t0 = time.time()
    # stage 1: identical to the `final` fit (protocol.fit_lgbm honours CFG max_rounds / EF_SMOKE_N_JOBS)
    cap = _cap()
    best, _, _, predict_best = protocol.fit_lgbm({"max_rounds": cap} if cap else {}, Xf, yf, Xe, ye, seed)
    t1 = time.time()
    n2 = max(1, int(round(best * factor)))
    if cap:
        n2 = min(n2, cap)
    params = {**PARAMS, "seed": seed, "scale_pos_weight": SPW,
              "n_jobs": int(os.environ.get("EF_SMOKE_N_JOBS") or 12)}
    Xa = pd.concat([Xf, Xe], axis=0, ignore_index=True)
    ya = np.concatenate([np.asarray(yf), np.asarray(ye)])
    assert len(Xa) == len(Xf) + len(Xe) == len(ya)
    booster = lgb.train(params, lgb.Dataset(Xa, ya), num_boost_round=n2)
    s2 = np.asarray(booster.predict(Xs), dtype=float)
    t2 = time.time()
    print(f"[refit x{factor}] B={best} refit_rounds={n2} stage1 {t1 - t0:.1f}s stage2 {t2 - t1:.1f}s "
          f"rows2={len(Xa)} (fit {len(Xf)} + es {len(Xe)})", flush=True)
    if not np.isfinite(s2).all():
        raise ValueError("non-finite refit scores")
    if _return_stage1:
        return s2, np.asarray(predict_best(Xs), dtype=float), {"B": best, "rounds": n2, "rows2": len(Xa)}
    return s2


def make_arm(name, factor, description):
    def fit(Xf, yf, Xe, ye, Xs, seed, meta):
        return fit_refit(Xf, yf, Xe, ye, Xs, seed, meta, factor=factor)
    return {"name": name, "description": description, "model": "custom", "fit": fit}


def _main():
    if "--smoke" not in sys.argv:
        print(__doc__)
        return
    os.environ.setdefault("EF_SMOKE_N_JOBS", "2")
    os.environ["EF_SMOKE_MAX_ROUNDS"] = "60"
    from ecommerce_fraud import protocol
    protocol.CFG["max_rounds"] = 60
    rng = np.random.default_rng(0)
    n = 6000
    X = pd.DataFrame(rng.normal(size=(n, 8)), columns=[f"f{i}" for i in range(8)])
    y = (X.f0 + 0.5 * X.f1 + rng.normal(size=n) > 2.2).astype(int).to_numpy()
    Xf, Xe, Xs = X.iloc[:4000], X.iloc[4000:5000], X.iloc[5000:]
    yf, ye = y[:4000], y[4000:5000]
    meta = {"window": "w1", "ids_fit": np.arange(4000), "ids_es": np.arange(4000, 5000), "ids_score": np.arange(5000, n)}
    s2, s1, info = fit_refit(Xf, yf, Xe, ye, Xs, 42, meta, 1.0, _return_stage1=True)
    ref = protocol.fit_lgbm({}, Xf, yf, Xe, ye, 42)[3](Xs)
    assert np.array_equal(s1, ref), "stage 1 != protocol.fit_lgbm"
    assert info["rows2"] == 5000 and s2.shape == (1000,) and np.isfinite(s2).all()
    assert not np.array_equal(s1, s2)
    for bad in ({**meta, "window": "valid"}, {**meta, "ids_score": np.arange(4500, 5500)}):
        try:
            fit_refit(Xf, yf, Xe, ye, Xs, 42, bad, 1.0)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
    print("refit self-check OK", info)


if __name__ == "__main__":
    _main()
