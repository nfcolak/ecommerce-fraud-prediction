"""Diversified-seed LightGBM fit for arm final_jitter (round p5, unit bagscale; model 'custom').

Exactly final's LightGBM (baseline_lgbm.PARAMS + scale_pos_weight 28, early stopping 100 on Xe, max 5000
rounds) except that each seed draws a small parameter jitter from np.random.default_rng(seed):
  num_leaves in {192,256,320}, feature_fraction ~ U[0.4,0.6], bagging_fraction ~ U[0.7,0.9],
  min_child_samples in {70,100,140}.
Decision variable: per-seed parameter jitter (more diversity inside a bag). No label is read except
yf/ye (fit-window training and early-stopping targets). Env (smoke only): EF_SMOKE_N_JOBS -> n_jobs,
EF_SMOKE_MAX_ROUNDS -> cap on rounds. Full runs set neither (n_jobs 12, 5000 rounds).
"""
from __future__ import annotations

import os

import numpy as np

SPW = 28.0
MAX_ROUNDS = 5000
EARLY_STOP = 100


def draw_params(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {"num_leaves": int(rng.choice([192, 256, 320])),
            "feature_fraction": float(rng.uniform(0.4, 0.6)),
            "bagging_fraction": float(rng.uniform(0.7, 0.9)),
            "min_child_samples": int(rng.choice([70, 100, 140]))}


def fit_jitter(Xf, yf, Xe, ye, Xs, seed, meta):
    import lightgbm as lgb
    from ecommerce_fraud.baseline_lgbm import PARAMS
    jit = draw_params(int(seed))
    params = {**PARAMS, "seed": int(seed), "scale_pos_weight": SPW, **jit,
              "n_jobs": int(os.environ.get("EF_SMOKE_N_JOBS") or 12)}
    rounds = MAX_ROUNDS
    if os.environ.get("EF_SMOKE_MAX_ROUNDS"):
        rounds = min(rounds, int(os.environ["EF_SMOKE_MAX_ROUNDS"]))
    print(f"[final_jitter] seed {seed} window {meta.get('window')} jitter {jit}", flush=True)
    dtrain = lgb.Dataset(Xf, yf)
    dvalid = lgb.Dataset(Xe, ye, reference=dtrain)
    booster = lgb.train(params, dtrain, num_boost_round=rounds, valid_sets=[dvalid], valid_names=["valid"],
                        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(500)])
    best = int(booster.best_iteration)
    print(f"[final_jitter] seed {seed} best_iteration {best} of cap {rounds}", flush=True)
    if isinstance(meta.get("log"), dict):
        meta["log"]["jitter"] = {**jit, "best_iteration": best}
    return booster.predict(Xs, num_iteration=best)


def _main():
    a, b, c = draw_params(42), draw_params(43), draw_params(44)
    assert draw_params(42) == a, "draw must be deterministic"
    assert len({str(a), str(b), str(c)}) >= 2, "seeds should differ"
    for s in range(40, 400):
        p = draw_params(s)
        assert p["num_leaves"] in (192, 256, 320) and p["min_child_samples"] in (70, 100, 140)
        assert 0.4 <= p["feature_fraction"] <= 0.6 and 0.7 <= p["bagging_fraction"] <= 0.9
    for s in (42, 43, 44, 45, 46, 47):
        print(s, draw_params(s))
    print("ok")


if __name__ == "__main__":
    _main()
