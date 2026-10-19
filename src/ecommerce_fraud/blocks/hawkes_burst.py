"""Feature block hawkes_burst: self-exciting burst intensity of the same card / client / device / email.

For row i and key K (uid, card1, DeviceInfo, P_emaildomain) the block computes the exponentially
decayed count of STRICTLY earlier events of the same key,
    lambda_i = sum_{j: t_j < t_i, same key} exp(-beta * (t_i - t_j)),
for half-lives of 1 hour, 1 day and 7 days (closed form Hawkes-style intensity, nothing fitted),
plus the same decayed sum weighted by TransactionAmt for uid. Rows with identical TransactionDT
never count each other (events are collapsed per (key, time) first). Rows with a NaN key part get
NaN. No labels are used; no fitted statistic, so `fit_end_dt` only documents the fit window.

The recursion x_g = a_g * (x_{g-1} + S_{g-1}) with a_g = exp(-beta * dt_g) is an affine scan; it is
solved exactly and stably with a log-step (Hillis-Steele) doubling scan in numpy, no row loop.
"""
from __future__ import annotations

import resource
import time

import numpy as np
import pandas as pd

from ecommerce_fraud.features_history import DAY, make_keys

NAME = "hawkes_burst"
DESCRIPTION = "Exponentially decayed count of earlier events (1h/1d/7d half-life) per uid/card1/DeviceInfo/P_emaildomain, plus decayed uid amount"
USES_LABELS = False

HOURS = {"1h": 3600.0, "1d": float(DAY), "7d": 7.0 * DAY}
BETAS = {k: np.log(2.0) / v for k, v in HOURS.items()}
KEYS = {"uid": ["card1", "addr1", "D1"], "card1": ["card1"], "dev": ["DeviceInfo"],
        "email": ["P_emaildomain"]}


def _decayed(codes: np.ndarray, t: np.ndarray, weights: dict) -> dict:
    """codes/t: rows with valid key, time-sorted. weights: {name: float array}.

    Returns {(name, half_life_label): decayed sum over strictly earlier same-key rows}.
    """
    n = len(codes)
    order = np.argsort(codes, kind="stable")  # stays time-sorted within a key
    c, ts = codes[order].astype(np.int64), t[order].astype(np.int64)
    # collapse ties: unique (key, time) events
    newg = np.ones(n, dtype=bool)
    if n > 1:
        newg[1:] = (c[1:] != c[:-1]) | (ts[1:] != ts[:-1])
    gid = np.cumsum(newg) - 1
    gstart_idx = np.flatnonzero(newg)
    ng = len(gstart_idx)
    gc, gt = c[gstart_idx], ts[gstart_idx]
    first = np.ones(ng, dtype=bool)
    if ng > 1:
        first[1:] = gc[1:] != gc[:-1]
    dt = np.zeros(ng)
    if ng > 1:
        dt[1:] = (gt[1:] - gt[:-1]).astype(np.float64)
    maxrun = np.diff(np.append(np.flatnonzero(first), ng)).max() if ng else 1
    out = {}
    for wname, w in weights.items():
        S = np.bincount(gid, weights=w[order], minlength=ng)
        Sprev = np.zeros(ng)
        Sprev[1:] = S[:-1]
        Sprev[first] = 0.0
        for hl, beta in BETAS.items():
            a = np.exp(-beta * dt)
            a[first] = 0.0  # first event of a key: lambda = 0
            A, B = a.copy(), a * Sprev
            k = 1
            while k < maxrun:  # affine doubling scan, A=0 at key starts cuts the chain
                B[k:] = A[k:] * B[:-k] + B[k:]
                A[k:] = A[k:] * A[:-k]
                k <<= 1
            res = np.empty(n)
            res[order] = B[gid]
            out[(wname, hl)] = res
    return out


def build(df: pd.DataFrame, fit_end_dt: int) -> pd.DataFrame:
    t_all = df["TransactionDT"].to_numpy(np.int64)
    if len(t_all) > 1 and (np.diff(t_all) < 0).any():
        raise ValueError("df must be sorted by TransactionDT")
    keys = make_keys(df)
    amt = df["TransactionAmt"].to_numpy(np.float64)
    out = {"TransactionID": df["TransactionID"].to_numpy()}
    for kname, parts in KEYS.items():
        ok = ~df[parts].isna().any(axis=1).to_numpy()
        idx = np.flatnonzero(ok)
        raw = keys["uid"].to_numpy()[idx] if kname == "uid" else df[parts[0]].to_numpy()[idx]
        codes = pd.factorize(raw)[0]
        weights = {"cnt": np.ones(len(idx))}
        if kname == "uid":
            weights["amt"] = np.nan_to_num(amt[idx], nan=0.0)
        res = _decayed(codes.astype(np.int64), t_all[idx], weights)
        for (wname, hl), v in res.items():
            arr = np.full(len(df), np.nan, dtype=np.float32)
            arr[idx] = v
            out[f"{NAME}__{kname}_{wname}_{hl}"] = arr
    return pd.DataFrame(out)


def leakage_check(df: pd.DataFrame, feats: pd.DataFrame) -> str:
    """Recompute 5 random rows by hand (plain Python, strictly earlier rows of the same key)."""
    rng = np.random.default_rng(11)
    keys = make_keys(df)
    t = df["TransactionDT"].tolist()
    amt = df["TransactionAmt"].tolist()
    lines, all_ok = [], True
    picks = [int(i) for i in rng.choice(np.flatnonzero(feats[f"{NAME}__uid_cnt_1d"].notna().to_numpy()
                                                      & (feats[f"{NAME}__uid_cnt_1d"].to_numpy() > 0)), 5, replace=False)]
    for i in picks:
        lines.append(f"row {i} TransactionID {df['TransactionID'].iat[i]} DT {t[i]}")
        for kname, parts in KEYS.items():
            if df[parts].iloc[i].isna().any():
                got = feats[[c for c in feats.columns if c.startswith(f"{NAME}__{kname}_")]].iloc[i]
                assert got.isna().all(), f"{kname}: NaN key must give NaN features"
                continue
            ks = (keys["uid"] if kname == "uid" else df[parts[0]]).to_numpy()
            same = [j for j in np.flatnonzero(ks == ks[i]).tolist() if t[j] < t[i]]
            for hl, beta in BETAS.items():
                for wname in (["cnt", "amt"] if kname == "uid" else ["cnt"]):
                    v = sum(np.exp(-beta * (t[i] - t[j])) * (1.0 if wname == "cnt" else amt[j]) for j in same)
                    got = float(feats[f"{NAME}__{kname}_{wname}_{hl}"].iat[i])
                    ok = bool(np.isclose(got, v, rtol=1e-5, atol=1e-6))
                    all_ok = all_ok and ok
                    lines.append(f"  {kname:6s} {wname} {hl} feature={got:.6f} by_hand={v:.6f} {'OK' if ok else 'MISMATCH'}")
    if not all_ok:
        raise AssertionError("leakage_check mismatch\n" + "\n".join(lines))
    lines.append("ALL MATCH")
    return "\n".join(lines)


def _main() -> None:
    from sklearn.metrics import average_precision_score

    from ecommerce_fraud.data import load_joined, time_split

    df = load_joined().sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    train, valid, _test, _ = time_split(df)
    fit_end_dt = int(train["TransactionDT"].max())
    t0 = time.time()
    feats = build(df, fit_end_dt)
    secs = time.time() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # bytes on macOS
    print(f"shape {feats.shape}  fit_end_dt {fit_end_dt}")
    print("NaN share per column:")
    for c in feats.columns[1:]:
        print(f"  {c:40s} {feats[c].isna().mean():.4f}")
    print(f"build time {secs:.1f}s  peak RSS {rss:.2f} GB")
    assert len(feats) == len(df) and (feats["TransactionID"].to_numpy() == df["TransactionID"].to_numpy()).all()
    print(leakage_check(df, feats))
    vmask = df["TransactionID"].isin(valid["TransactionID"]).to_numpy()
    yv = df["isFraud"].to_numpy()[vmask]
    print(f"SIGNAL valid PR-AUC per column alone (base rate {yv.mean():.4f}); score / negative score:")
    for c in feats.columns[1:]:
        s = feats[c].to_numpy(np.float64)[vmask]
        pos = np.where(np.isnan(s), np.nanmin(s) - 1 if (~np.isnan(s)).any() else 0, s)
        neg = np.where(np.isnan(s), -(np.nanmax(s) + 1) if (~np.isnan(s)).any() else 0, -s)
        print(f"SIGNAL {c:40s} {average_precision_score(yv, pos):.4f} / {average_precision_score(yv, neg):.4f}")
    print(f"BLOCK {NAME} OK")


if __name__ == "__main__":
    _main()
