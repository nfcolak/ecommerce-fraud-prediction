"""Past-only per-client (uid) aggregates of raw columns.

uid = card1 + addr1 + D1n, exactly as in features_history.make_keys. Every feature of a
row uses only rows of the same uid with a strictly earlier TransactionDT (ties at the same
TransactionDT are not earlier). No labels, no future rows, no full-data counts.
Input must be sorted by TransactionDT.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ecommerce_fraud.features_history import DAY, make_keys

AGG_RAW = [f"C{i}" for i in range(1, 15)] + ["D2", "D4n", "D10n", "D15n", "dist1",
                                              "V258", "V294", "V307"]
PREFIX = "uid_past_mean_"


def _source(df: pd.DataFrame, col: str) -> np.ndarray:
    if col in ("D4n", "D10n", "D15n"):  # same definition as features_history
        return ((df["TransactionDT"] // DAY) - df[col[:-1]]).to_numpy(np.float64)
    return df[col].to_numpy(np.float64)


def add_uid_aggregates(df: pd.DataFrame):
    """Return (df with new columns, list of added column names)."""
    t = df["TransactionDT"].to_numpy(np.int64)
    if len(t) > 1 and (np.diff(t) < 0).any():
        raise ValueError("df must be sorted by TransactionDT")
    n = len(df)
    big = int(t.max()) + 1
    codes = pd.factorize(make_keys(df)["uid"])[0].astype(np.int64)
    order = np.argsort(codes, kind="stable")  # time-sorted inside a uid
    c, ts = codes[order], t[order]
    comb = c * big + ts
    gstart = np.searchsorted(comb, c * big, side="left")
    hi = np.searchsorted(comb, comb, side="left")  # earlier rows of the uid: [gstart, hi)
    n_prev = hi - gstart
    has = n_prev > 0
    last = np.maximum(hi - 1, 0)

    def put(vals):
        arr = np.empty(n, dtype=np.float64)
        arr[order] = vals
        return arr

    new = {}
    for col in AGG_RAW:
        v = _source(df, col)[order]
        ok = ~np.isnan(v)
        s = pd.Series(np.where(ok, v, 0.0)).groupby(c).cumsum().to_numpy()
        k = pd.Series(ok.astype(np.int64)).groupby(c).cumsum().to_numpy()
        ssum = np.where(has, s[last], np.nan)
        scnt = np.where(has, k[last], 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = np.where(scnt > 0, ssum / np.maximum(scnt, 1), np.nan)
        new[PREFIX + col] = put(mean)

    new["uid_past_cnt"] = put(n_prev.astype(np.float64))

    prod = pd.factorize(df["ProductCD"].astype(str))[0].astype(np.int64)[order]
    pk = c * (int(prod.max()) + 1) + prod  # (uid, product) pair id
    porder = np.argsort(pk, kind="stable")  # time-sorted inside the pair
    pc, pts = pk[porder], ts[porder]
    pbig = big
    pcomb = pc * pbig + pts
    pstart = np.searchsorted(pcomb, pc * pbig, side="left")
    phi = np.searchsorted(pcomb, pcomb, side="left")
    same_sorted = np.empty(n, dtype=np.int64)
    same_sorted[porder] = phi - pstart  # earlier same-uid same-product rows, in uid order
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(has, same_sorted / np.maximum(n_prev, 1), np.nan)
    new["uid_past_same_product_share"] = put(share)

    added = list(new.keys())
    out = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    return out, added


def leakage_check(full: pd.DataFrame, rows, seed: int = 0, n: int = 3) -> str:
    """Recompute uid_past_cnt, uid_past_mean_C1 and uid_past_same_product_share with a plain
    filter on strictly earlier rows for n random rows of `rows` (index labels of `full`,
    which must be the output of add_uid_aggregates). Return a text report."""
    uid = make_keys(full)["uid"]
    rng = np.random.default_rng(seed)
    cand = [i for i in rows if full.at[i, "uid_past_cnt"] > 0]
    picks = list(rng.choice(np.array(cand), size=n, replace=False))
    lines, all_ok = [], True
    for i in picks:
        t = full.at[i, "TransactionDT"]
        past = full[(uid == uid[i]) & (full["TransactionDT"] < t)]
        hand = {
            "uid_past_cnt": float(len(past)),
            "uid_past_mean_C1": float(past["C1"].mean()) if past["C1"].notna().any() else np.nan,
            "uid_past_same_product_share":
                float((past["ProductCD"] == full.at[i, "ProductCD"]).mean()),
        }
        lines.append(f"row index {i} TransactionID {full.at[i, 'TransactionID']} "
                     f"DT {t}: {len(past)} earlier rows with the same uid")
        for k, v in hand.items():
            got = float(full.at[i, k])
            ok = bool(np.isclose(got, v, rtol=1e-9, atol=1e-9, equal_nan=True))
            all_ok = all_ok and ok
            lines.append(f"  {k:28s} feature={got:.6f} by_hand={v:.6f} {'OK' if ok else 'MISMATCH'}")
    lines.append("ALL MATCH" if all_ok else "MISMATCH FOUND")
    return "\n".join(lines)
