"""Decision layer for the final model: calibration, capacity, cost and day-block bootstrap.

Pure functions on numpy arrays; no data loading and no model training here.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

EPS = 1e-12
DAY = 86400


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------- calibration
def fit_platt(p, y):
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(_logit(p).reshape(-1, 1), y)
    return lambda q: lr.predict_proba(_logit(np.asarray(q)).reshape(-1, 1))[:, 1]


def fit_isotonic(p, y):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p, y)
    return lambda q: iso.predict(np.asarray(q))


def ece(y, p, n_bins=10):
    """Expected calibration error, equal-width bins on [0, 1]."""
    idx = np.minimum((np.asarray(p) * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.sum() * abs(y[m].mean() - p[m].mean())
    return float(total / len(y))


def reliability(y, p, n_bins=10):
    idx = np.minimum((np.asarray(p) * n_bins).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            rows.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return rows


def calib_report(y, p):
    return {"brier": float(brier_score_loss(y, p)), "ece_10bins": ece(y, p),
            "mean_pred": float(np.mean(p)), "observed_rate": float(np.mean(y))}


# ------------------------------------------------------------------- capacity
def _caught(y, amt, sel):
    tp = float(y[sel].sum())
    return tp, float((amt * y)[sel].sum())


def capacity_row(y, amt, score, day, frac, per_day):
    n_days = len(np.unique(day))
    sel = np.zeros(len(y), dtype=bool)
    if per_day:
        for d in np.unique(day):
            idx = np.flatnonzero(day == d)
            k = max(1, int(np.ceil(len(idx) * frac)))
            sel[idx[np.argsort(-score[idx], kind="stable")[:k]]] = True
    else:
        k = max(1, int(np.ceil(len(y) * frac)))
        sel[np.argsort(-score, kind="stable")[:k]] = True
    n_alerts = int(sel.sum())
    tp, amt_tp = _caught(y, amt, sel)
    return {"k_pct": frac * 100, "n_alerts": n_alerts, "alerts_per_day": n_alerts / n_days,
            "precision": tp / n_alerts, "recall": tp / float(y.sum()),
            "fraud_amount_share": amt_tp / float((amt * y).sum())}


def capacity_table(y, amt, score, day, fracs=(0.005, 0.01, 0.02, 0.05)):
    return {"global": [capacity_row(y, amt, score, day, f, False) for f in fracs],
            "per_day": [capacity_row(y, amt, score, day, f, True) for f in fracs]}


# ----------------------------------------------------------------------- cost
def cost_curve(y, amt, rank, c, friction=2.0):
    """Hold the top-n orders by `rank`. loss(n) = c*n + friction*legit_held + missed fraud amount.
    Returns (order, loss_by_n) with loss_by_n[n] for n = 0..N."""
    order = np.argsort(-rank, kind="stable")
    yo, ao = y[order], amt[order]
    n = np.arange(1, len(y) + 1)
    fraud_held_amt = np.cumsum(ao * yo)
    legit_held = n - np.cumsum(yo)
    loss = np.empty(len(y) + 1)
    loss[0] = float((amt * y).sum())
    loss[1:] = c * n + friction * legit_held + (loss[0] - fraud_held_amt)
    return order, loss


def choose_threshold(y, amt, rank, c, friction=2.0):
    """Best cut on valid: returns the rank-value threshold (hold if rank >= thr) and its loss."""
    order, loss = cost_curve(y, amt, rank, c, friction)
    n = int(np.argmin(loss))
    if n == 0:
        return float("inf"), float(loss[0])
    return float(rank[order][n - 1]), float(loss[n])


def apply_threshold(y, amt, rank, thr, c, friction=2.0):
    hold = rank >= thr
    n_hold = int(hold.sum())
    legit_held = int((hold & (y == 0)).sum())
    missed = float((amt * y)[~hold].sum())
    loss_model = c * n_hold + friction * legit_held + missed
    loss_none = float((amt * y).sum())
    return {"threshold": thr, "n_held": n_hold, "share_held": n_hold / len(y),
            "precision": float(y[hold].mean()) if n_hold else float("nan"),
            "recall": float(y[hold].sum() / y.sum()),
            "loss_model": loss_model, "loss_no_model": loss_none,
            "savings": 1.0 - loss_model / loss_none}


# ------------------------------------------------------------------ bootstrap
class WeightedMetrics:
    """AP and precision@1% for one score vector under integer row weights (day multiplicities).
    The sort is done once; each resample is a gather plus cumsums. Equals sklearn's
    average_precision_score / baseline_lgbm.top_k on the materialised resample."""

    def __init__(self, y, score):
        self.order = np.argsort(-score, kind="stable")
        self.y = y[self.order].astype(float)
        s = score[self.order]
        last = np.ones(len(s), dtype=bool)
        last[:-1] = s[1:] != s[:-1]  # last row of each tie group
        self.ends = np.flatnonzero(last)

    def compute(self, w_rows):
        """w_rows: weight per original row."""
        w = w_rows[self.order]
        wy = w * self.y
        ctp, cn = np.cumsum(wy), np.cumsum(w)
        tp_g, n_g = ctp[self.ends], cn[self.ends]
        total_pos = ctp[-1]
        prec_g = np.divide(tp_g, n_g, out=np.zeros_like(tp_g), where=n_g > 0)
        rec_g = tp_g / total_pos
        ap = float(np.sum(np.diff(np.concatenate(([0.0], rec_g))) * prec_g))
        total_n = cn[-1]
        k = np.ceil(0.01 * total_n)
        pos = int(np.searchsorted(cn, k, side="left"))
        before_n = cn[pos - 1] if pos > 0 else 0.0
        before_tp = ctp[pos - 1] if pos > 0 else 0.0
        tp = before_tp + self.y[pos] * (k - before_n)
        return ap, float(tp / k)


def day_block_bootstrap(y, day, scores, n_boot=1000, seed=0):
    """scores: dict name -> score vector (aligned to y). Returns dict name -> array (n_boot, 2)
    of (pr_auc, precision@1%), plus the point estimates."""
    days = np.unique(day)
    d_idx = np.searchsorted(days, day)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(days), size=(n_boot, len(days)))
    wm = {k: WeightedMetrics(y, v) for k, v in scores.items()}
    out = {k: np.empty((n_boot, 2)) for k in scores}
    ones = np.ones(len(y))
    point = {k: wm[k].compute(ones) for k in scores}
    for b in range(n_boot):
        mult = np.bincount(draws[b], minlength=len(days)).astype(float)
        w = mult[d_idx]
        for k in scores:
            out[k][b] = wm[k].compute(w)
    return out, point


def quantiles(a):
    q = np.percentile(a, [2.5, 50, 97.5])
    return {"p2.5": float(q[0]), "p50": float(q[1]), "p97.5": float(q[2])}
