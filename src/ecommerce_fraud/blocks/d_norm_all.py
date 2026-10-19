"""Feature block d_norm_all: all D columns normalized as day - Dk, plus uid-relative deviations.

Dk_norm = floor(TransactionDT / 86400) - Dk for k = 1..15 (the IEEE-CIS winners' trick: a D column
counts days since an event, so day - Dk is a roughly constant event date per client). The final
model already has D1n/D4n/D10n/D15n, so only the other 11 normalized Ds are emitted as values.
For every one of the 15 normalized Ds the block also emits the row's value minus the mean of the
same normalized value over STRICTLY earlier rows (TransactionDT smaller) of the same uid
(card1, addr1, D1n); NaN when there is no earlier non-NaN row, when Dk is NaN, or when the uid has
a NaN component. Finally `miss_mask` is the missing-pattern of D1-D15 as one integer (bit k-1 set
when Dk is NaN). No label, no raw TransactionDT, no day index; `fit_end_dt` is not used (no fit).
"""
from __future__ import annotations

import resource
import time

import numpy as np
import pandas as pd

from ecommerce_fraud.features_history import DAY, make_keys

NAME = "d_norm_all"
DESCRIPTION = "D1-D15 as day-Dk (11 new) + deviation from uid's strictly-earlier mean + missing bitmask"
USES_LABELS = False

D_COLS = [f"D{k}" for k in range(1, 16)]
ALREADY = {"D1", "D4", "D10", "D15"}  # D1n/D4n/D10n/D15n exist in the final model
UID_PARTS = ["card1", "addr1", "D1"]


def _past_mean_dev(codes: np.ndarray, t: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """vals - mean of non-NaN vals over earlier rows (DT strictly smaller) of the same code."""
    n = len(codes)
    big = int(t.max()) + 1 if n else 1
    order = np.argsort(codes, kind="stable")  # time-sorted inside a key
    c, ts, v = codes[order].astype(np.int64), t[order].astype(np.int64), vals[order]
    comb = c * big + ts
    gstart = np.searchsorted(comb, c * big, side="left")
    lo = np.searchsorted(comb, comb, side="left")  # earlier rows of the key: [gstart, lo)
    ok = ~np.isnan(v)
    cs = np.concatenate([[0.0], np.cumsum(np.where(ok, v, 0.0))])
    cn = np.concatenate([[0], np.cumsum(ok)])
    s, m = cs[lo] - cs[gstart], cn[lo] - cn[gstart]
    with np.errstate(invalid="ignore", divide="ignore"):
        dev = np.where(m > 0, v - s / np.maximum(m, 1), np.nan)
    out = np.empty(n, dtype=np.float64)
    out[order] = dev
    return out


def build(df: pd.DataFrame, fit_end_dt: int) -> pd.DataFrame:
    t_all = df["TransactionDT"].to_numpy(np.int64)
    if len(t_all) > 1 and (np.diff(t_all) < 0).any():
        raise ValueError("df must be sorted by TransactionDT")
    day = (t_all // DAY).astype(np.float64)
    dn = {d: day - df[d].to_numpy(np.float64) for d in D_COLS}
    out = {"TransactionID": df["TransactionID"].to_numpy()}
    for d in D_COLS:
        if d not in ALREADY:
            out[f"{NAME}__{d}n"] = dn[d].astype(np.float32)
    mask = np.zeros(len(df), dtype=np.int64)
    for k, d in enumerate(D_COLS):
        mask |= df[d].isna().to_numpy().astype(np.int64) << k
    out[f"{NAME}__miss_mask"] = mask.astype(np.float32)  # < 2**15, exact in float32

    keys = make_keys(df)["uid"].to_numpy()
    ok = ~df[UID_PARTS].isna().any(axis=1).to_numpy()
    idx = np.flatnonzero(ok)
    codes = pd.factorize(keys[idx])[0].astype(np.int64)
    for d in D_COLS:
        arr = np.full(len(df), np.nan, dtype=np.float32)
        arr[idx] = _past_mean_dev(codes, t_all[idx], dn[d][idx])
        out[f"{NAME}__{d}n_uiddev"] = arr
    return pd.DataFrame(out)


def leakage_check(df: pd.DataFrame, feats: pd.DataFrame) -> str:
    """Recompute 5 random rows by hand with plain Python over strictly earlier raw rows."""
    rng = np.random.default_rng(7)
    keys = make_keys(df)["uid"].to_numpy()
    t = df["TransactionDT"].tolist()
    D = {d: df[d].tolist() for d in D_COLS}
    dev_ok = feats[f"{NAME}__D2n_uiddev"].notna().to_numpy()
    cand = np.flatnonzero(dev_ok)
    other = np.flatnonzero(~df[UID_PARTS].isna().any(axis=1).to_numpy())
    picks = list(rng.choice(cand, 3, replace=False)) + list(rng.choice(other, 2, replace=False))
    lines, all_ok = [], True

    def cmp(name, got, want):
        nonlocal all_ok
        good = (np.isnan(got) and np.isnan(want)) or np.isclose(got, want, rtol=1e-5, atol=1e-4)
        all_ok = all_ok and bool(good)
        lines.append(f"  {name:22s} feature={got:.5f} by_hand={want:.5f} {'OK' if good else 'MISMATCH'}")

    for i in picks:
        i = int(i)
        lines.append(f"row {i} TransactionID {df['TransactionID'].iat[i]} DT {t[i]}")
        earlier = [j for j in np.flatnonzero(keys == keys[i]).tolist() if t[j] < t[i]]
        valid_key = not df[UID_PARTS].iloc[i].isna().any()
        mbits = sum((1 << k) for k, d in enumerate(D_COLS) if np.isnan(D[d][i]))
        cmp("miss_mask", float(feats[f"{NAME}__miss_mask"].iat[i]), float(mbits))
        for d in D_COLS:
            own = t[i] // DAY - D[d][i]
            if d not in ALREADY:
                cmp(f"{d}n", float(feats[f"{NAME}__{d}n"].iat[i]), own)
            past = [t[j] // DAY - D[d][j] for j in earlier if not np.isnan(D[d][j])]
            want = own - sum(past) / len(past) if (valid_key and past) else float("nan")
            cmp(f"{d}n_uiddev", float(feats[f"{NAME}__{d}n_uiddev"].iat[i]), want)
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
        print(f"  {c:45s} {feats[c].isna().mean():.4f}")
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
        print(f"SIGNAL {c:45s} {average_precision_score(yv, pos):.4f} / {average_precision_score(yv, neg):.4f}")
    print(f"BLOCK {NAME} OK")


if __name__ == "__main__":
    _main()
