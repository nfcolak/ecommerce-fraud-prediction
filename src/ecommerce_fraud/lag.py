"""Label-lag robustness arms for round p5 (option A): final and refit when labels arrive 14 days later.

In a protocol window the labelled rows end at day S-8 (7-day gap): fit = days <= S-22, ES = S-21..S-8.
Here we pretend the last 14 labelled days (the ES rows) are not labelled yet, so labelled data end at S-22:
  new ES  = fit rows with day S-35..S-22 (14 days), new fit = fit rows with day <= S-36; Xe/ye are dropped.
  final_lag14 : final's LightGBM on (new fit, ES on new ES), scores Xs.
  refit_lag14 : the same stage 1, then refit on all rows with day <= S-22 for B rounds (refit.fit_refit).
Contrast refit_lag14 vs final_lag14 asks whether refit's gain survives a 21-day label delay.
w1 (S=40) leaves only days <= 4 for fitting, so the contrast uses w2-w6 (PREREGISTRATION amendment).
meta day_fit is used only to split rows, never as a feature. The valid window is refused.
"""
from __future__ import annotations

import numpy as np

from ecommerce_fraud.protocol import ES_DAYS, GAP_DAYS

LAG = 14


def split_lag(yf, meta, lag=LAG):
    if meta.get("window") == "valid" or meta.get("score_days") is None:
        raise ValueError("lag arms run on protocol windows only, not on the valid screen")
    S = int(meta["score_days"][0])
    d = np.asarray(meta["day_fit"], dtype=float)
    if len(d) != len(yf):
        raise ValueError("day_fit length differs from fit rows")
    if d.max() > S - GAP_DAYS - 1 - ES_DAYS:
        raise ValueError(f"fit rows reach day {d.max()} > S-{GAP_DAYS + 1 + ES_DAYS}")
    es_lo = S - GAP_DAYS - 1 - ES_DAYS - lag + 1          # S-35
    fit_m, es_m = d < es_lo, d >= es_lo
    if fit_m.sum() == 0 or es_m.sum() == 0 or yf[fit_m].sum() == 0 or yf[es_m].sum() == 0:
        raise ValueError("lag split leaves an empty fit or ES part")
    return fit_m, es_m


def _sub_meta(meta, fit_m, es_m):
    m = dict(meta)
    ids = np.asarray(meta["ids_fit"])
    m["ids_fit"], m["ids_es"] = ids[fit_m], ids[es_m]
    return m


def fit_final_lag(Xf, yf, Xe, ye, Xs, seed, meta):
    from ecommerce_fraud import protocol
    yf = np.asarray(yf)
    fit_m, es_m = split_lag(yf, meta)
    best, _, _, predict = protocol.fit_lgbm({}, Xf[fit_m], yf[fit_m], Xf[es_m], yf[es_m], seed)
    print(f"[final_lag14] B={best} fit {int(fit_m.sum())} es {int(es_m.sum())} (dropped ES rows {len(ye)})", flush=True)
    return np.asarray(predict(Xs), dtype=float)


def fit_refit_lag(Xf, yf, Xe, ye, Xs, seed, meta):
    from ecommerce_fraud.refit import fit_refit
    yf = np.asarray(yf)
    fit_m, es_m = split_lag(yf, meta)
    print(f"[refit_lag14] fit {int(fit_m.sum())} es {int(es_m.sum())} (dropped ES rows {len(ye)})", flush=True)
    return fit_refit(Xf[fit_m], yf[fit_m], Xf[es_m], yf[es_m], Xs, seed, _sub_meta(meta, fit_m, es_m), factor=1.0)
