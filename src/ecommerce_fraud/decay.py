"""p5 unit 'decay': time-decay sample weights and a recent-only window for final's LightGBM.

Custom fit = final's LightGBM exactly (baseline_lgbm.PARAMS + seed + scale_pos_weight 28, early stop 100 on
the ES rows Xe, max 5000 rounds) except for ONE decision variable on the fit rows:
  decay_h<h> : training Dataset weight w_i = 0.5 ** ((max(day_fit) - day_fit_i) / h), normalised to mean 1.
  recent90   : no weights; only fit rows with day_fit >= max(day_fit) - 89 are used.
ES rows are never weighted or filtered. meta['day_fit'] is used ONLY for weights / row selection, never as a feature.

Interactions (design notes)
- scale_pos_weight multiplies the weight of positives: effective positive weight = 28 * w_i. Decay therefore
  keeps the 28:1 ratio within any one day, but a window's fraud-vs-legit weight balance shifts slightly if the
  fraud rate drifts over the fit span (mean-1 normalisation is over all fit rows, not per class). Not re-tuned.
- min_child_samples (100) counts ROWS, not weights: unaffected by decay, so leaves can still be built from
  old rows whose weight is small. min_sum_hessian_in_leaf (LightGBM default 1e-3) is on weighted hessians;
  mean-1 normalisation keeps the hessian scale close to the unweighted fit, and the smallest weight stays
  >= 2**(-span/h)/mean (about 0.2 for h=45 over 110 days), far above the 1e-3 floor, so it is not binding.
- With mean-1 weights, the effective sample size drops (Kish ESS = (sum w)^2 / sum w^2); reported by the check.
- Early stopping metric (average_precision on Xe) is unweighted; with decay the ES rows are the most recent
  days, the same rows final uses. On the 'valid' screen ES = the scored window (note, as for every arm).
"""
from __future__ import annotations

import os

import numpy as np

from ecommerce_fraud.baseline_lgbm import PARAMS

SPW = 28.0
ES_ROUNDS = 100
MAX_ROUNDS = 5000
RECENT_DAYS = 90


def decay_weights(day_fit, half_life):
    """Half-life weights normalised to mean 1. half_life=inf gives all ones."""
    d = np.asarray(day_fit, dtype=float)
    if not np.all(np.isfinite(d)):
        raise ValueError("day_fit has non-finite values")
    if np.isinf(half_life):
        return np.ones(len(d))
    w = 0.5 ** ((d.max() - d) / float(half_life))
    return w / w.mean()


def _params(seed):
    p = {**PARAMS, "seed": seed, "scale_pos_weight": SPW}
    p["n_jobs"] = int(os.environ.get("EF_SMOKE_N_JOBS") or 12)
    return p


def _rounds():
    r = MAX_ROUNDS
    if os.environ.get("EF_SMOKE_MAX_ROUNDS"):       # smoke only
        r = min(r, int(os.environ["EF_SMOKE_MAX_ROUNDS"]))
    return r


def _train_predict(Xf, yf, Xe, ye, Xs, seed, weight=None):
    import lightgbm as lgb
    dtrain = lgb.Dataset(Xf, yf, weight=weight)
    dvalid = lgb.Dataset(Xe, ye, reference=dtrain)          # ES rows unweighted
    booster = lgb.train(_params(seed), dtrain, num_boost_round=_rounds(), valid_sets=[dvalid],
                        valid_names=["valid"],
                        callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False), lgb.log_evaluation(500)])
    best = int(booster.best_iteration)
    print(f"[decay] best_iteration={best} n_fit={len(Xf)}", flush=True)
    return booster.predict(Xs, num_iteration=best)


def fit_decay(half_life):
    def fit(Xf, yf, Xe, ye, Xs, seed, meta):
        day = np.asarray(meta["day_fit"], dtype=float)
        if len(day) != len(Xf):
            raise ValueError(f"day_fit length {len(day)} != fit rows {len(Xf)}")
        w = decay_weights(day, half_life)
        ess = w.sum() ** 2 / (w ** 2).sum()
        print(f"[decay] h={half_life} window={meta.get('window')} w range [{w.min():.3f}, {w.max():.3f}] "
              f"ESS={ess:.0f}/{len(w)}", flush=True)
        return _train_predict(Xf, yf, Xe, ye, Xs, seed, weight=w)
    return fit


def recent_mask(day_fit, days=RECENT_DAYS):
    d = np.asarray(day_fit, dtype=float)
    return d >= d.max() - (days - 1)


def fit_recent(days=RECENT_DAYS):
    def fit(Xf, yf, Xe, ye, Xs, seed, meta):
        day = np.asarray(meta["day_fit"], dtype=float)
        if len(day) != len(Xf):
            raise ValueError(f"day_fit length {len(day)} != fit rows {len(Xf)}")
        m = recent_mask(day, days)
        print(f"[decay] recent{days} window={meta.get('window')} keep {m.sum()}/{len(m)} rows", flush=True)
        if m.all():
            return _train_predict(Xf, yf, Xe, ye, Xs, seed)
        return _train_predict(Xf.iloc[np.flatnonzero(m)], np.asarray(yf)[m], Xe, ye, Xs, seed)
    return fit


def _main():
    """Self-checks on synthetic arrays (no transaction data). `python -m ecommerce_fraud.decay`."""
    rng = np.random.default_rng(0)
    # 1. weight properties
    day = np.sort(rng.uniform(1, 110, 5000))
    for h in (45, 90):
        w = decay_weights(day, h)
        assert np.all(np.isfinite(w)) and np.all(w > 0) and abs(w.mean() - 1) < 1e-12
        assert np.all(np.diff(w) >= -1e-15), "weights must be non-decreasing in day"
        print(f"h={h}: ok, range [{w.min():.3f}, {w.max():.3f}]")
    assert np.all(decay_weights(day, np.inf) == 1)
    # 2. h=inf reproduces final's fit exactly (ones as weights vs no weights, same params/rounds/ES)
    import pandas as pd
    n = 30000
    X = pd.DataFrame(rng.normal(size=(n, 12)), columns=[f"f{i}" for i in range(12)])
    y = (X.f0 + 0.5 * X.f1 + rng.normal(scale=2, size=n) > 3.2).astype(int).to_numpy()
    d = np.sort(rng.uniform(1, 100, n))
    nf = 24000
    Xf, yf, Xe, ye, Xs = X.iloc[:nf], y[:nf], X.iloc[nf:], y[nf:], X.iloc[nf:]
    os.environ["EF_SMOKE_MAX_ROUNDS"] = "150"
    os.environ.setdefault("EF_SMOKE_N_JOBS", "4")
    meta = {"day_fit": d[:nf], "window": "synthetic"}
    p_inf = fit_decay(np.inf)(Xf, yf, Xe, ye, Xs, 42, meta)
    # reference: protocol.fit_lgbm (final's code path)
    from ecommerce_fraud import protocol
    protocol.CFG["max_rounds"] = 150
    _, _, _, pred = protocol.fit_lgbm({}, Xf, yf, Xe, ye, 42)
    p_ref = pred(Xs)
    print("h=inf vs protocol.fit_lgbm: max abs diff", np.abs(p_inf - p_ref).max())
    assert np.array_equal(p_inf, p_ref), "h=inf must reproduce protocol.fit_lgbm exactly"
    p45 = fit_decay(45)(Xf, yf, Xe, ye, Xs, 42, meta)
    assert not np.array_equal(p45, p_ref)
    # 3. recent mask
    m = recent_mask(np.array([1, 8, 9, 97.0]), 90)
    assert m.tolist() == [False, True, True, True], m  # 8 >= 97-89
    print("ALL CHECKS OK")


if __name__ == "__main__":
    _main()
