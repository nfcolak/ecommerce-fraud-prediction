"""Feature block time_cyclic: cyclic clock time plus per-uid / per-card1 timing history (no labels)."""
from __future__ import annotations

import math
import random
import resource
import time

import numpy as np
import pandas as pd

NAME = "time_cyclic"
DESCRIPTION = "hour/day-of-week cyclic encodings, uid past hour-of-day profile, gaps and 1h/24h counts"
USES_LABELS = False

HOUR, DAY = 3600, 86400
BINS = 8  # 3-hour bins


def _hour(t):
    return (t % DAY) / HOUR  # fractional hour of day


def _keys(df):
    from ecommerce_fraud.features_history import make_keys
    k = make_keys(df)
    return (pd.factorize(k["uid"])[0].astype(np.int64), pd.factorize(k["card1"])[0].astype(np.int64))


def _sorted_key(codes, t):
    """Return order, sorted codes/times, comb, group start, hi (= rows of key with DT < t)."""
    big = int(t.max()) + 1
    order = np.argsort(codes, kind="stable")
    c, ts = codes[order], t[order]
    comb = c * big + ts
    gstart = np.searchsorted(comb, c * big, side="left")
    hi = np.searchsorted(comb, comb, side="left")
    return order, c, ts, comb, gstart, hi, big


def build(df: pd.DataFrame, fit_end_dt: int) -> pd.DataFrame:
    t = df["TransactionDT"].to_numpy().astype(np.int64)
    n = len(t)
    h = _hour(t)
    dow = (t // DAY) % 7
    out = {
        "hour": np.floor(h).astype(np.int8),
        "hour_sin": np.sin(2 * np.pi * h / 24),
        "hour_cos": np.cos(2 * np.pi * h / 24),
        "dow": dow.astype(np.int8),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
    }
    uid, card1 = _keys(df)

    # uid history
    order, c, ts, comb, gs, hi, big = _sorted_key(uid, t)
    n_prev = hi - gs
    has = n_prev > 0
    res = {}

    def put(name, vals):
        a = np.empty(n, dtype=np.float32)
        a[order] = vals
        res[name] = a

    prev_t = ts[np.maximum(hi - 1, 0)]
    put("uid_gap_prev", np.where(has, ts - prev_t, np.nan))
    for name, w in (("uid_cnt_1h", HOUR), ("uid_cnt_24h", DAY)):
        lo = np.searchsorted(comb, c * big + np.maximum(ts - w, 0), side="left")
        put(name, hi - np.maximum(lo, gs))
    hs = _hour(ts)
    bin_ = (hs // 3).astype(np.int64)
    cur_share = np.zeros(n)
    for b in range(BINS):
        cs = np.concatenate([[0], np.cumsum(bin_ == b)])
        m = bin_ == b
        cur_share[m] = (cs[hi] - cs[gs])[m]
    put("uid_hour_bin_share", np.where(has, cur_share / np.maximum(n_prev, 1), np.nan))
    ang = 2 * np.pi * hs / 24
    css = np.concatenate([[0.0], np.cumsum(np.sin(ang))])
    csc = np.concatenate([[0.0], np.cumsum(np.cos(ang))])
    ssum, csum = css[hi] - css[gs], csc[hi] - csc[gs]
    mean_ang = np.arctan2(ssum, csum)
    mean_h = (mean_ang * 24 / (2 * np.pi)) % 24
    d = np.abs(hs - mean_h)
    d = np.minimum(d, 24 - d)
    put("uid_hour_mean", np.where(has, mean_h, np.nan))
    put("uid_hour_dist", np.where(has, d, np.nan))
    put("uid_hour_conc", np.where(has, np.hypot(ssum, csum) / np.maximum(n_prev, 1), np.nan))

    # card1 gap
    order, c, ts, comb, gs, hi, big = _sorted_key(card1, t)
    put("card1_gap_prev", np.where(hi > gs, ts - ts[np.maximum(hi - 1, 0)], np.nan))

    for k, v in res.items():
        out[k] = v
    r = pd.DataFrame({f"{NAME}__{k}": np.asarray(v, dtype=np.float32) for k, v in out.items()})
    r.insert(0, "TransactionID", df["TransactionID"].to_numpy())
    return r


def leakage_check(df: pd.DataFrame, feats: pd.DataFrame) -> str:
    uid, card1 = _keys(df)
    t = df["TransactionDT"].to_numpy().astype(np.int64)
    rng = random.Random(0)
    # prefer rows with history
    idx = [i for i in rng.sample(range(len(df)), 2000)]
    idx = [i for i in idx if (uid[:i] == uid[i]).any()][:5]
    for i in idx:
        prior = [j for j in np.flatnonzero(uid == uid[i]) if t[j] < t[i]]
        pc = [j for j in np.flatnonzero(card1 == card1[i]) if t[j] < t[i]]
        exp = {}
        exp["uid_gap_prev"] = t[i] - max(t[j] for j in prior) if prior else float("nan")
        exp["uid_cnt_1h"] = sum(1 for j in prior if t[j] >= t[i] - HOUR)
        exp["uid_cnt_24h"] = sum(1 for j in prior if t[j] >= t[i] - DAY)
        exp["card1_gap_prev"] = t[i] - max(t[j] for j in pc) if pc else float("nan")
        if prior:
            hh = [(t[j] % DAY) / HOUR for j in prior]
            exp["uid_hour_bin_share"] = sum(1 for x in hh if int(x // 3) == int(((t[i] % DAY) / HOUR) // 3)) / len(hh)
            s = sum(math.sin(2 * math.pi * x / 24) for x in hh)
            co = sum(math.cos(2 * math.pi * x / 24) for x in hh)
            m = (math.atan2(s, co) * 24 / (2 * math.pi)) % 24
            dd = abs((t[i] % DAY) / HOUR - m)
            exp["uid_hour_dist"] = min(dd, 24 - dd)
        for k, v in exp.items():
            got = float(feats[f"{NAME}__{k}"].iloc[i])
            ok = (math.isnan(v) and math.isnan(got)) or abs(got - v) <= 1e-3 * max(1.0, abs(v))
            assert ok, f"row {i} {k}: got {got}, expected {v}"
    return f"leakage_check passed on rows {idx} (uid/card1 features vs plain-Python recompute)"


def _main():
    from sklearn.metrics import average_precision_score
    from ecommerce_fraud import data

    df = data.load_joined().sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    train, valid, _test, _ = data.time_split(df)
    fit_end = int(train["TransactionDT"].max())
    t0 = time.time()
    f = build(df, fit_end)
    dt = time.time() - t0
    print("shape", f.shape)
    print("NaN share per column:")
    print(f.isna().mean().round(4).to_string())
    print(f"run time {dt:.1f}s, peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB")
    print(leakage_check(df, f))
    assert (f["TransactionID"].to_numpy() == df["TransactionID"].to_numpy()).all()
    vmask = df["TransactionID"].isin(valid["TransactionID"]).to_numpy()
    y = df["isFraud"].to_numpy()[vmask]
    for col in f.columns[1:]:
        s = f[col].to_numpy()[vmask].astype(float)
        s = np.nan_to_num(s, nan=np.nanmin(s) - 1 if np.isfinite(s).any() else 0)
        print(f"SIGNAL {col}: PR-AUC {average_precision_score(y, s):.4f} neg {average_precision_score(y, -s):.4f}")
    print(f"BLOCK {NAME} OK")


if __name__ == "__main__":
    _main()
