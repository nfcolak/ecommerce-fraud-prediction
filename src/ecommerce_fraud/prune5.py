"""p5 unit 'prune': null-importance feature pruning nested inside each window's fit rows.

Selection uses ONLY fit rows (Xf, yf) and early-stopping rows (Xe, ye); the scored rows Xs are
first touched in the last step (prediction). Never reads isFraud outside yf/ye.

fit_prune_null(Xf, yf, Xe, ye, Xs, seed, meta) -> scores for Xs
  1. real model: final's params (protocol.PARAMS, spw 28), ES 100 on (Xe, ye); gain importance
     at best_iteration -> real share g_j = gain_j / sum(gain).
  2. K=4 null models on the same Xf with yf randomly permuted (rng from seed; uses only yf);
     same params but learning_rate 0.1, fixed rounds min(B, 400), no ES (B = real best iteration).
     Null share n_kj = gain_kj / sum_j gain_kj.  Shares (not raw gains) are compared because the
     real and null models run different numbers of rounds / learning rates, so raw gains are on
     different scales.
  3. score_j = log(1e-10 + g_j / (1e-6 + q75_k(n_kj)))  (null-importance kernel, shares).
     Keep score_j > 0, i.e. g_j > q75(null share) + 1e-6. Threshold fixed here, not tuned.
     Floor: if fewer than MIN_KEEP=200 pass, keep the top-200 by score (a guard against a
     degenerate real model, e.g. ES at a tiny round count, wiping out the model; the floor
     rarely binds when the real model is healthy -- it is logged when it does).
  4. final's LightGBM on kept columns, ES 100 on Xe; predict Xs[kept].
fit_prune_gain: stage 1 only; drop zero-gain features and the bottom 20% (by rank) of the rest.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

K_NULL = 4
NULL_LR = 0.1
NULL_MAX_ROUNDS = 400
MIN_KEEP = 200
EPS_NULL = 1e-6
BOTTOM_FRAC = 0.20


def _params(seed, **over):
    from ecommerce_fraud.protocol import PARAMS, SPW
    p = {**PARAMS, "seed": int(seed), "scale_pos_weight": SPW, **over}
    if os.environ.get("EF_SMOKE_N_JOBS"):
        p["n_jobs"] = int(os.environ["EF_SMOKE_N_JOBS"])
    return p


def _max_rounds(default=5000):
    from ecommerce_fraud.protocol import CFG, DEFAULT_MAX_ROUNDS
    r = DEFAULT_MAX_ROUNDS if default is None else default
    m = CFG.get("max_rounds") or os.environ.get("EF_SMOKE_MAX_ROUNDS")
    return min(r, int(m)) if m else r


def _gain(booster, iteration=None):
    return np.asarray(booster.feature_importance("gain", iteration=iteration), dtype=float)


def _train_real(Xf, yf, Xe, ye, seed):
    import lightgbm as lgb
    dtrain = lgb.Dataset(Xf, yf)
    dvalid = lgb.Dataset(Xe, ye, reference=dtrain)
    b = lgb.train(_params(seed), dtrain, num_boost_round=_max_rounds(), valid_sets=[dvalid],
                  valid_names=["valid"], callbacks=[lgb.early_stopping(100, verbose=False)])
    best = int(b.best_iteration)
    return b, best


def null_shares(Xf, yf, seed, rounds, k=K_NULL):
    """k x p matrix of gain shares from models fit on randomly permuted yf (uses only yf)."""
    import lightgbm as lgb
    yf = np.asarray(yf)
    out = []
    for i in range(k):
        rng = np.random.default_rng(int(seed) * 1000 + 17 + i)
        yp = rng.permutation(yf)
        assert yp.sum() == yf.sum() and len(yp) == len(yf)
        b = lgb.train(_params(seed + 1 + i, learning_rate=NULL_LR), lgb.Dataset(Xf, yp),
                      num_boost_round=int(rounds))
        g = _gain(b)
        out.append(g / max(g.sum(), 1e-12))
    return np.vstack(out)


def null_scores(real_gain, null_sh):
    g = np.asarray(real_gain, float)
    g = g / max(g.sum(), 1e-12)
    q75 = np.quantile(null_sh, 0.75, axis=0)
    return np.log(1e-10 + g / (EPS_NULL + q75))


def choose_kept(score, min_keep=MIN_KEEP):
    idx = np.where(score > 0)[0]
    floored = False
    if len(idx) < min(min_keep, len(score)):
        idx = np.sort(np.argsort(-score, kind="stable")[:min(min_keep, len(score))])
        floored = True
    return idx, floored


def select_null(Xf, yf, Xe, ye, seed, log, min_keep=MIN_KEEP):
    """Selection from fit/ES rows only. Returns kept column names. (Takes no scored rows.)"""
    t0 = time.time()
    b, best = _train_real(Xf, yf, Xe, ye, seed)
    g = _gain(b, best)
    log["best_iter_real"], log["sec_real"] = best, round(time.time() - t0, 1)
    t1 = time.time()
    rounds = min(max(best, 1), NULL_MAX_ROUNDS)
    sh = null_shares(Xf, yf, seed, rounds)
    log["null_rounds"], log["sec_null"] = rounds, round(time.time() - t1, 1)
    sc = null_scores(g, sh)
    idx, floored = choose_kept(sc, min_keep)
    cols = list(Xf.columns)
    log["n_in"], log["n_kept"], log["floor_applied"] = len(cols), int(len(idx)), floored
    return [cols[i] for i in idx], sc


def select_gain(Xf, yf, Xe, ye, seed, log):
    t0 = time.time()
    b, best = _train_real(Xf, yf, Xe, ye, seed)
    g = _gain(b, best)
    log["best_iter_real"], log["sec_real"] = best, round(time.time() - t0, 1)
    nz = np.where(g > 0)[0]
    order = nz[np.argsort(g[nz], kind="stable")]       # ascending gain
    drop = set(order[:int(np.floor(BOTTOM_FRAC * len(nz)))].tolist())
    cols = list(Xf.columns)
    idx = [i for i in nz if i not in drop]
    log["n_in"], log["n_kept"], log["n_zero_gain"] = len(cols), len(idx), int(len(cols) - len(nz))
    return [cols[i] for i in idx], g


def _write_kept(kept, log, meta, seed, tag):
    d = os.environ.get("MODELS_DIR")
    if not d:
        return
    p = Path(d) / "p5_prune"
    p.mkdir(parents=True, exist_ok=True)
    w = meta.get("window", "x") if isinstance(meta, dict) else "x"
    (p / f"{tag}_{w}_s{seed}.json").write_text(json.dumps({"log": log, "kept": kept}))


def _fit(selector, tag, Xf, yf, Xe, ye, Xs, seed, meta):
    import lightgbm as lgb
    t0 = time.time()
    log = meta.get("log") if isinstance(meta.get("log"), dict) else {}
    kept, _ = selector(Xf, yf, Xe, ye, seed, log)       # Xs not passed: structurally untouched
    assert kept and set(kept) <= set(Xf.columns), "kept columns must be a subset of Xf columns"
    t2 = time.time()
    dtrain = lgb.Dataset(Xf[kept], yf)
    dvalid = lgb.Dataset(Xe[kept], ye, reference=dtrain)
    b = lgb.train(_params(seed), dtrain, num_boost_round=_max_rounds(), valid_sets=[dvalid],
                  valid_names=["valid"], callbacks=[lgb.early_stopping(100, verbose=False)])
    best = int(b.best_iteration)
    log["best_iter_final"], log["sec_final"], log["sec_total"] = best, round(time.time() - t2, 1), None
    s = np.asarray(b.predict(Xs[kept], num_iteration=best))     # first use of Xs
    log["sec_total"] = round(time.time() - t0, 1)
    log["kept_head"] = kept[:5]
    print(f"[{tag}] {log}")
    _write_kept(kept, log, meta, seed, tag)
    return s


def fit_prune_null(Xf, yf, Xe, ye, Xs, seed, meta):
    return _fit(select_null, "prune_null", Xf, yf, Xe, ye, Xs, seed, meta)


def fit_prune_gain(Xf, yf, Xe, ye, Xs, seed, meta):
    return _fit(select_gain, "prune_gain", Xf, yf, Xe, ye, Xs, seed, meta)


def _main():
    """Synthetic self-check: planted noise block dropped, planted signal kept, no Xs before step 4."""
    import inspect
    import pandas as pd
    rng = np.random.default_rng(0)
    n = 20000
    sig = rng.normal(size=(n, 5))
    noise = rng.normal(size=(n, 60))
    z = sig[:, 0] * 2 + sig[:, 1] * 1.5 + sig[:, 2]
    y = (rng.random(n) < 1 / (1 + np.exp(-(z - 4)))).astype(int)
    X = pd.DataFrame(np.hstack([sig, noise]), columns=[f"sig{i}" for i in range(5)] + [f"noise{i}" for i in range(60)])
    cut = 12000
    Xf, yf, Xe, ye = X.iloc[:cut], y[:cut], X.iloc[cut:16000], y[cut:16000]
    os.environ.setdefault("EF_SMOKE_MAX_ROUNDS", "150")
    log = {}
    kept, sc = select_null(Xf, yf, Xe, ye, 42, log, min_keep=0)   # floor off: only 65 columns
    print(log)
    assert set(kept) <= set(Xf.columns)
    assert "sig0" in kept and "sig1" in kept, kept
    n_noise = sum(c.startswith("noise") for c in kept)
    print(f"kept {len(kept)}; noise kept {n_noise}/60")
    assert n_noise <= 15, "planted pure-noise block mostly dropped"
    # selection signature has no scored-row argument; permutation preserves yf multiset
    assert "Xs" not in inspect.signature(select_null).parameters
    sh = null_shares(Xf, yf, 1, 20, k=2)
    assert sh.shape == (2, X.shape[1]) and np.allclose(sh.sum(1), 1)
    # Xs guard: fit must not touch Xs before selection finished
    class Guard(pd.DataFrame):
        touched = []
        def __getitem__(self, k):
            Guard.touched.append(len(log_ref)); return super().__getitem__(k)
    log_ref = []
    Xs = Guard(X.iloc[16000:])
    meta = {"log": log_ref and {} or {}}
    orig = select_null
    def spy(*a, **k):
        r = orig(*a, **k); log_ref.append(1); return r
    global select_null_spy
    out = _fit(spy, "prune_null_check", Xf, yf, Xe, ye, Xs, 42, meta)
    assert Guard.touched and min(Guard.touched) == 1, "Xs touched before selection ended"
    assert len(out) == len(Xs) and np.isfinite(out).all()
    print("SELF-CHECK OK: noise dropped, signal kept, Xs untouched until step 4")


if __name__ == "__main__":
    _main()
