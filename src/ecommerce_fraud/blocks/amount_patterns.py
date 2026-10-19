"""Feature block amount_patterns: amount shape and amount behaviour versus the uid's past.

Uses no labels. Row i only looks at rows with TransactionDT strictly smaller than its own.
No fitted statistics, so fit_end_dt is accepted but unused.
"""
from __future__ import annotations

import bisect
import math

import numpy as np
import pandas as pd

NAME = "amount_patterns"
DESCRIPTION = "Amount cents/decimals/roundness, uid log-amount z-score and rank, repeat same-amount counts, ratio to card1 median"
USES_LABELS = False

DAY = 86400
WEEK = 7 * DAY


def _amt_int(amt: np.ndarray) -> np.ndarray:
    """Amount in thousandths as int64 (TransactionAmt has at most 3 decimals)."""
    return np.rint(np.nan_to_num(amt, nan=0.0) * 1000).astype(np.int64)


def _codes(s: pd.Series) -> np.ndarray:
    return pd.factorize(s, sort=False)[0].astype(np.int64)


def _uid(df: pd.DataFrame) -> pd.Series:
    from ecommerce_fraud.features_history import make_keys
    return make_keys(df)["uid"]


def _past_walk(codes: np.ndarray, t: np.ndarray, x: np.ndarray, want_rank: bool, want_median: bool):
    """Per group, walk rows in time order and compare each row with rows of strictly smaller DT.

    Returns n_past, mean, std (of x), rank fraction (past < x, ties half), median of past x.
    """
    n = len(codes)
    order = np.lexsort((np.arange(n), t, codes))  # by code, then time, stable
    c, ts, xs = codes[order], t[order], x[order]
    n_past = np.zeros(n)
    mean = np.full(n, np.nan)
    std = np.full(n, np.nan)
    rank = np.full(n, np.nan)
    med = np.full(n, np.nan)
    starts = np.flatnonzero(np.r_[True, c[1:] != c[:-1]])
    ends = np.r_[starts[1:], n]
    for s, e in zip(starts, ends):
        if e - s == 1:
            continue
        srt: list = []
        sm = sq = 0.0
        k = 0
        pend: list = []
        last_t = None
        for i in range(s, e):
            if ts[i] != last_t:
                for v in pend:
                    if want_rank or want_median:
                        bisect.insort(srt, v)
                    sm += v
                    sq += v * v
                    k += 1
                pend = []
                last_t = ts[i]
            if k:
                oi = order[i]
                v = xs[i]
                n_past[oi] = k
                m = sm / k
                mean[oi] = m
                if k >= 2:
                    var = max(sq / k - m * m, 0.0) * k / (k - 1)
                    std[oi] = math.sqrt(var)
                if want_rank:
                    lo = bisect.bisect_left(srt, v)
                    hi = bisect.bisect_right(srt, v)
                    rank[oi] = (lo + 0.5 * (hi - lo)) / k
                if want_median:
                    h = k // 2
                    med[oi] = srt[h] if k % 2 else 0.5 * (srt[h - 1] + srt[h])
            pend.append(xs[i])
    return n_past, mean, std, rank, med


def _window_counts(key: np.ndarray, t: np.ndarray, window: int) -> np.ndarray:
    """Count earlier rows with same key and t-window <= DT_j < DT_i."""
    big = int(t.max()) + window + 2
    comb = key * big + t
    order = np.argsort(comb, kind="stable")
    cs = comb[order]
    hi = np.searchsorted(cs, cs, side="left")
    lo = np.searchsorted(cs, key[order] * big + np.maximum(t[order] - window, 0), side="left")
    out = np.empty(len(key))
    out[order] = hi - lo
    return out


def build(df: pd.DataFrame, fit_end_dt: int) -> pd.DataFrame:
    amt = df["TransactionAmt"].to_numpy(dtype="float64")
    t = df["TransactionDT"].to_numpy(dtype="int64")
    a = _amt_int(amt)
    out = pd.DataFrame({"TransactionID": df["TransactionID"].to_numpy()})
    p = NAME + "__"

    out[p + "cents"] = ((a % 1000) / 10.0).astype("float32")
    out[p + "n_decimals"] = np.where(a % 10, 3, np.where(a % 100, 2, np.where(a % 1000, 1, 0))).astype("float32")
    for k in (1, 5, 10, 50, 100):
        out[p + f"round_{k}"] = (a % (1000 * k) == 0).astype("float32")

    la = np.log1p(amt)
    uid = _uid(df)
    ucode = _codes(uid)
    npast, mean, std, rank, _ = _past_walk(ucode, t, la, True, False)
    z = (la - mean) / np.maximum(std, 0.05)
    out[p + "uid_logamt_z"] = np.where(npast >= 2, z, np.nan).astype("float32")
    out[p + "uid_amt_rank"] = rank.astype("float32")

    prod = _codes(df["ProductCD"].astype("object").astype(str))
    amt_code = _codes(pd.Series(a))
    key = ucode * (int(amt_code.max()) + 1) + amt_code
    key = key * (int(prod.max()) + 1) + prod
    key = _codes(pd.Series(key))
    out[p + "uid_same_amt_prod_1d"] = _window_counts(key, t, DAY).astype("float32")
    out[p + "uid_same_amt_prod_7d"] = _window_counts(key, t, WEEK).astype("float32")

    ccode = _codes(df["card1"].astype("object").astype(str))
    npc, _, _, _, med = _past_walk(ccode, t, amt, False, True)
    out[p + "card1_amt_ratio_median"] = (amt / np.where(med > 0, med, np.nan)).astype("float32")
    return out


def leakage_check(df: pd.DataFrame, feats: pd.DataFrame) -> str:
    """Recompute 5 random rows in plain Python over raw rows and compare with the vectorized output."""
    rng = np.random.default_rng(0)
    uid = _uid(df).tolist()
    card1 = df["card1"].astype("object").astype(str).tolist()
    prod = df["ProductCD"].astype("object").astype(str).tolist()
    amt = df["TransactionAmt"].tolist()
    t = df["TransactionDT"].tolist()
    p = NAME + "__"
    n = len(df)
    # pick rows that have history so the check is not vacuous
    cand = rng.permutation(n)[:20000].tolist()
    uc = _codes(pd.Series(uid))
    done = 0
    msgs = []
    for i in cand:
        if done == 5:
            break
        past = [int(j) for j in np.flatnonzero(uc[:i] == uc[i]) if t[j] < t[i]]
        if len(past) < 3 and done < 3:
            continue
        a = int(round(amt[i] * 1000))
        exp = {}
        exp["cents"] = (a % 1000) / 10.0
        exp["n_decimals"] = 3 if a % 10 else 2 if a % 100 else 1 if a % 1000 else 0
        for k in (1, 5, 10, 50, 100):
            exp[f"round_{k}"] = float(a % (1000 * k) == 0)
        pl = [math.log1p(amt[j]) for j in past]
        lv = math.log1p(amt[i])
        if len(pl) >= 2:
            m = sum(pl) / len(pl)
            sd = math.sqrt(max(sum(v * v for v in pl) / len(pl) - m * m, 0) * len(pl) / (len(pl) - 1))
            exp["uid_logamt_z"] = (lv - m) / max(sd, 0.05)
        else:
            exp["uid_logamt_z"] = float("nan")
        if pl:
            exp["uid_amt_rank"] = (sum(v < lv for v in pl) + 0.5 * sum(v == lv for v in pl)) / len(pl)
        else:
            exp["uid_amt_rank"] = float("nan")
        same = [j for j in past if int(round(amt[j] * 1000)) == a and prod[j] == prod[i]]
        exp["uid_same_amt_prod_1d"] = float(sum(t[i] - t[j] <= 86400 for j in same))
        exp["uid_same_amt_prod_7d"] = float(sum(t[i] - t[j] <= 604800 for j in same))
        cp = sorted(amt[j] for j in range(i) if t[j] < t[i] and card1[j] == card1[i])
        if cp:
            h = len(cp) // 2
            md = cp[h] if len(cp) % 2 else 0.5 * (cp[h - 1] + cp[h])
            exp["card1_amt_ratio_median"] = amt[i] / md if md > 0 else float("nan")
        else:
            exp["card1_amt_ratio_median"] = float("nan")
        for name, v in exp.items():
            got = float(feats.iloc[i][p + name])
            if not (math.isnan(v) and math.isnan(got)) and not math.isclose(got, v, rel_tol=1e-4, abs_tol=1e-4):
                raise AssertionError(f"row {i} {name}: vectorized {got} != hand {v}")
        msgs.append(f"row {i} ok (past uid rows={len(past)})")
        done += 1
    assert done == 5, "could not find 5 rows to check"
    return "leakage_check passed: " + "; ".join(msgs)


def _main() -> None:
    import resource
    import time

    from sklearn.metrics import average_precision_score

    from ecommerce_fraud import data

    df = data.load_joined().sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    train, valid, _test, _ = data.time_split(df)
    fit_end = int(train["TransactionDT"].max())
    t0 = time.time()
    feats = build(df, fit_end)
    dt = time.time() - t0
    print("shape", feats.shape)
    cols = [c for c in feats.columns if c != "TransactionID"]
    print("NaN share:")
    for c in cols:
        print(f"  {c}: {feats[c].isna().mean():.4f}")
    print(f"run time {dt:.1f}s")
    print(f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB")
    print(leakage_check(df, feats))
    vm = df["TransactionID"].isin(valid["TransactionID"]).to_numpy()
    y = df["isFraud"].to_numpy()[vm]
    for c in cols:
        s = feats[c].to_numpy()[vm].astype("float64")
        s = np.nan_to_num(s, nan=np.nanmin(s) if np.isfinite(s).any() else 0.0)
        print(f"SIGNAL {c}: PR-AUC {average_precision_score(y, s):.4f}  neg {average_precision_score(y, -s):.4f}"
              f"  (base {y.mean():.4f})")
    print(f"BLOCK {NAME} OK")


if __name__ == "__main__":
    _main()
