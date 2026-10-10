"""Past-only entity-history features.

Every feature of a row uses only that row's own fields and rows with a strictly
earlier TransactionDT (ties at the same TransactionDT count as "not earlier").
No labels, no full-data frequency counts. Input must be sorted by TransactionDT.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

DAY = 86400
WINDOWS = {"1d": 1 * DAY, "7d": 7 * DAY, "30d": 30 * DAY}  # window = [t - W, t)
KEY_NAMES = ["card1", "card1_addr1", "uid", "uid_email", "P_emaildomain"]
NUNIQUE_KEYS = ["uid", "card1"]
NUNIQUE_COLS = ["P_emaildomain", "DeviceInfo", "addr1"]


def _str(s: pd.Series) -> pd.Series:
    return s.astype(str)  # NaN -> "nan"


def make_keys(df: pd.DataFrame) -> dict:
    """Entity keys as strings (NaN kept as "nan"). Uses only row-level fields."""
    day = df["TransactionDT"] // DAY
    d1n = day - df["D1"]
    card1 = _str(df["card1"])
    c_a = card1 + "_" + _str(df["addr1"])
    uid = c_a + "_" + _str(d1n)
    return {
        "card1": card1,
        "card1_addr1": c_a,
        "uid": uid,
        "uid_email": uid + "_" + _str(df["P_emaildomain"]),
        "P_emaildomain": _str(df["P_emaildomain"]),
    }


def _key_features(codes, t, amt, prefix, out):
    """Past-only count/recency/amount stats of one entity key (codes = int ids)."""
    n = len(codes)
    big = int(t.max()) + 1
    order = np.argsort(codes, kind="stable")  # stays time-sorted inside a key
    c, ts, am = codes[order], t[order], amt[order]
    comb = c * big + ts
    gstart = np.searchsorted(comb, c * big, side="left")
    hi = np.searchsorted(comb, comb, side="left")  # rows of the key with DT < t: [gstart, hi)
    n_prev = hi - gstart
    has = n_prev > 0

    def put(name, vals):
        arr = np.empty(n, dtype=np.float64)
        arr[order] = vals
        out[f"{prefix}_{name}"] = arr

    for tag, w in WINDOWS.items():
        lo = np.searchsorted(comb, c * big + np.maximum(ts - w, 0), side="left")
        put(f"cnt_{tag}", hi - lo)
    put("cnt_all", n_prev)

    prev_t = np.where(has, ts[np.maximum(hi - 1, 0)], np.nan)
    put("since_prev", ts - prev_t)

    g = pd.Series(am)
    s1 = g.groupby(c).cumsum().to_numpy()  # per-key running sums (small, precise)
    s2 = (g * g).groupby(c).cumsum().to_numpy()
    last = np.maximum(hi - 1, 0)
    cnt = np.where(has, n_prev, np.nan)
    sum1 = np.where(has, s1[last], np.nan)
    sum2 = np.where(has, s2[last], np.nan)
    mean = sum1 / cnt
    var = np.where(n_prev >= 2, (sum2 - sum1 * sum1 / cnt) / np.maximum(n_prev - 1, 1), np.nan)
    std = np.sqrt(np.maximum(var, 0.0))
    put("amt_mean", mean)
    put("amt_std", std)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = am / mean
    put("amt_ratio", np.where(np.isfinite(ratio), ratio, np.nan))


def _nunique_features(codes, t, val_codes, name, out):
    """Number of distinct earlier non-NaN values of a column per key (own row excluded)."""
    big = int(t.max()) + 1
    ok = val_codes >= 0
    pairs = (pd.DataFrame({"k": codes[ok], "v": val_codes[ok], "t": t[ok]})
             .groupby(["k", "v"], sort=False)["t"].min().reset_index())
    pairs = pairs.sort_values(["k", "t"], kind="stable")
    comb = pairs["k"].to_numpy(np.int64) * big + pairs["t"].to_numpy(np.int64)
    lo = np.searchsorted(comb, codes * big, side="left")
    hi = np.searchsorted(comb, codes * big + t, side="left")  # first seen strictly before t
    out[name] = (hi - lo).astype(np.float64)


def add_history_features(df: pd.DataFrame):
    """Return (df with new columns, list of added column names)."""
    t0 = time.time()
    t = df["TransactionDT"].to_numpy(np.int64)
    if len(t) > 1 and (np.diff(t) < 0).any():
        raise ValueError("df must be sorted by TransactionDT")
    new = {}
    day = df["TransactionDT"] // DAY
    new["day"] = day
    for d in ("D1", "D4", "D10", "D15"):
        new[f"{d}n"] = day - df[d]
    new["amt_log"] = np.log1p(df["TransactionAmt"])
    new["amt_cents"] = df["TransactionAmt"] % 1

    amt = df["TransactionAmt"].to_numpy(np.float64)
    keys = make_keys(df)
    codes = {}
    for name in KEY_NAMES:
        codes[name] = pd.factorize(keys[name])[0].astype(np.int64)
        _key_features(codes[name], t, amt, name, new)
    for name in NUNIQUE_KEYS:
        for col in NUNIQUE_COLS:
            vc = pd.factorize(df[col])[0].astype(np.int64)  # NaN -> -1 (ignored)
            _nunique_features(codes[name], t, vc, f"{name}_nuniq_{col}", new)

    added = list(new.keys())
    out = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    print(f"history features: {len(added)} columns built in {time.time() - t0:.1f}s", flush=True)
    return out, added


def leakage_check(full: pd.DataFrame, rows, seed: int = 0, n: int = 3) -> str:
    """Recompute features by hand (plain filter on strictly earlier rows) for n random
    rows of `rows` (index labels of `full`) that have history; return a text report.
    `full` must be the output of add_history_features."""
    keys = make_keys(full)
    rng = np.random.default_rng(seed)
    cand = [i for i in rows if full.at[i, "uid_cnt_all"] > 0]
    picks = list(rng.choice(np.array(cand), size=n, replace=False))
    lines, all_ok = [], True
    for i in picks:
        t = full.at[i, "TransactionDT"]
        key = keys["uid"]
        past = full[(key == key[i]) & (full["TransactionDT"] < t)]
        win = past[past["TransactionDT"] >= t - WINDOWS["7d"]]
        hand = {
            "uid_cnt_7d": float(len(win)),
            "uid_amt_mean": float(past["TransactionAmt"].mean()),
            "uid_since_prev": float(t - past["TransactionDT"].max()),
            "uid_nuniq_P_emaildomain": float(past["P_emaildomain"].nunique()),
        }
        lines.append(f"row index {i} TransactionID {full.at[i, 'TransactionID']} "
                     f"DT {t}: {len(past)} earlier rows with the same uid")
        for k, v in hand.items():
            got = float(full.at[i, k])
            ok = bool(np.isclose(got, v, rtol=1e-9, atol=1e-9))
            all_ok = all_ok and ok
            lines.append(f"  {k:26s} feature={got:.6f} by_hand={v:.6f} {'OK' if ok else 'MISMATCH'}")
    lines.append("ALL MATCH" if all_ok else "MISMATCH FOUND")
    return "\n".join(lines)
