"""Separate model for transactions without an identity record (arm model 'custom').

A1 (2026-10-16): rows whose TransactionID is not in train_identity.csv are ~80% of rows and 44% of
fraud, but `final` ranks them worse (PR-AUC 0.537 vs 0.695 overall; global top-1% recall there 3%).

fit_seg_noid(Xf, yf, Xe, ye, Xs, seed, meta) -> scores for Xs
  1. Global model: LightGBM as `final` (improve.fit_lgbm: PARAMS, spw 28, early stopping 100 on Xe).
  2. Segment model: same PARAMS and spw 28 (kept: the segment's fraud rate is lower, but a changed
     spw is a second untested knob), trained on no-identity fit rows only, early-stopped on the
     no-identity ES rows. Columns that are all-NaN in the segment's fit rows are dropped.
  3. One ranking: identity rows keep the raw global score. No-identity rows get the segment score
     mapped onto the global score scale:
        q  = Platt_seg(segment score)              Platt fitted on no-identity ES rows
        s' = Platt_all^-1(q)                       Platt_all: global score -> probability, fitted on ALL ES rows
     i.e. a segment row gets the global score that a row with the same estimated fraud probability
     would have. Platt (logistic on the logit of the score; monotone and invertible when slope > 0)
     is used instead of isotonic because isotonic has flat steps that create ties and would destroy
     the within-segment ranking; with 14 ES days the segment has only a few hundred frauds, so two
     parameters are also less noisy. Calibration labels are ES labels, which lie in the training
     range; no label of the score window is used. The ES set is also the early-stopping set, so
     both calibrations see slightly optimistic scores; this is the same for both models.
  Variant blend=True: no-identity rows get the average of the two calibrated probabilities,
     q = (Platt_global_noid(global score) + Platt_seg(segment score)) / 2, then mapped to the
     global scale by Platt_all^-1. (Platt_global_noid is fitted on no-identity ES rows.)
  Falls back to the plain global scores when the segment is too small (see MIN_FIT_FRAUD) or a
  Platt slope is not positive. Best iterations of both models are printed and stored in
  meta['log'] if meta has a dict under that key.
"""
from __future__ import annotations

import time

import numpy as np

from ecommerce_fraud.decision import _logit

MIN_FIT_FRAUD = 200   # segment fit rows need at least this many frauds
MIN_ES_FRAUD = 20     # and the no-identity ES rows this many, else fall back to the global model


def _platt_params(p, y):
    """(slope, intercept) of logit(P(y)) = a * logit(p) + b."""
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(_logit(np.asarray(p)).reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _fit_seg_noid(Xf, yf, Xe, ye, Xs, seed, meta, blend):
    from ecommerce_fraud.improve import fit_lgbm  # lazy: improve may import arms
    t0 = time.time()
    log = meta.get("log") if isinstance(meta.get("log"), dict) else {}
    yf, ye = np.asarray(yf), np.asarray(ye)
    nf = ~np.asarray(meta["has_identity_fit"], bool)
    ne = ~np.asarray(meta["has_identity_es"], bool)
    ns = ~np.asarray(meta["has_identity_score"], bool)

    _, best_g, pred_g = fit_lgbm(Xf, yf, Xe, ye, seed, log_every=500, verbose=False)
    sg_s, sg_e = np.asarray(pred_g(Xs)), np.asarray(pred_g(Xe))
    log["best_iter_global"] = best_g
    print(f"[seg_noid] global best_iter={best_g} fit rows={len(yf)} ({time.time()-t0:.0f}s)")

    if yf[nf].sum() < MIN_FIT_FRAUD or ye[ne].sum() < MIN_ES_FRAUD or not ns.any():
        print("[seg_noid] segment too small -> global scores")
        log["fallback"] = "segment too small"
        return sg_s

    keep = [c for c in Xf.columns if Xf.loc[nf, c].notna().any()]
    t1 = time.time()
    _, best_s, pred_s = fit_lgbm(Xf.loc[nf, keep], yf[nf], Xe.loc[ne, keep], ye[ne], seed,
                                 log_every=500, verbose=False)
    ss_s, ss_e = np.asarray(pred_s(Xs.loc[ns, keep])), np.asarray(pred_s(Xe.loc[ne, keep]))
    log["best_iter_segment"] = best_s
    print(f"[seg_noid] segment best_iter={best_s} fit rows={int(nf.sum())} fraud={int(yf[nf].sum())} "
          f"cols={len(keep)}/{Xf.shape[1]} ({time.time()-t1:.0f}s)")

    a_all, b_all = _platt_params(sg_e, ye)           # global scale: score -> probability
    a_seg, b_seg = _platt_params(ss_e, ye[ne])
    if a_all <= 0 or a_seg <= 0:
        print("[seg_noid] non-positive Platt slope -> global scores")
        log["fallback"] = "platt slope"
        return sg_s
    z_seg = a_seg * _logit(ss_s) + b_seg
    q = _sigmoid(z_seg)
    if blend:
        a_gn, b_gn = _platt_params(sg_e[ne], ye[ne])
        if a_gn <= 0:
            log["fallback"] = "platt slope"
            return sg_s
        q = 0.5 * (q + _sigmoid(a_gn * _logit(sg_s[ns]) + b_gn))
    z = _logit(q)                                    # invert Platt_all
    mapped = _sigmoid((z - b_all) / a_all)
    out = sg_s.copy()
    out[ns] = mapped
    log["platt_all"], log["platt_seg"] = (a_all, b_all), (a_seg, b_seg)
    print(f"[seg_noid] platt_all=({a_all:.3f},{b_all:.3f}) platt_seg=({a_seg:.3f},{b_seg:.3f}) "
          f"blend={blend} total {time.time()-t0:.0f}s")
    return out


def fit_seg_noid(Xf, yf, Xe, ye, Xs, seed, meta):
    return _fit_seg_noid(Xf, yf, Xe, ye, Xs, seed, meta, blend=False)


def fit_seg_noid_blend(Xf, yf, Xe, ye, Xs, seed, meta):
    return _fit_seg_noid(Xf, yf, Xe, ye, Xs, seed, meta, blend=True)
