"""Block identity_velocity: distinct-entity velocity on device / IP-like ids.

Every feature of row i uses only rows with TransactionDT strictly smaller than
its own. No labels, no fitted statistics (fit_end_dt is accepted but unused).
"""
from __future__ import annotations

import resource
import time

import numpy as np
import pandas as pd

NAME = "identity_velocity"
DESCRIPTION = ("Distinct card1 per DeviceInfo/id_19/id_20/email+addr1 (1d, 7d, all), "
               "distinct DeviceInfo/id_31 per uid, card1-DeviceInfo age, uid first-seen flags")
USES_LABELS = False

DAY = 86400
WINDOWS = {"1d": DAY, "7d": 7 * DAY, "all": None}


def _codes(s: pd.Series) -> np.ndarray:
    """Integer codes, -1 for missing."""
    return pd.factorize(s, use_na_sentinel=True)[0].astype(np.int64)


def _count_lt(p: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """For each query: #{j < b : p[j] < a}, vectorized over bit levels."""
    n = len(p)
    out = np.zeros(len(b), dtype=np.int64)
    if n == 0:
        return out
    m = n + 2
    idx = np.arange(n, dtype=np.int64)
    for lev in range(int(n).bit_length() + 1):
        sel = ((b >> lev) & 1).astype(bool)
        if not sel.any():
            continue
        key = (idx >> lev) * m + (p + 1)
        skey = np.sort(key)
        blk = ((b[sel] >> (lev + 1)) << 1)  # block id at this level (size 2^lev)
        q = blk * m + (a[sel] + 1)
        out[sel] += np.searchsorted(skey, q, side="left") - (blk << lev)
    return out


def distinct_before(key: np.ndarray, val: np.ndarray, t: np.ndarray, window) -> np.ndarray:
    """# distinct `val` among rows with same `key` and t' in [t-window, t) (window None = all).

    key/val int codes (-1 missing). Result float32, NaN where key is missing.
    """
    n = len(t)
    big = int(t.max()) + 2
    ev = np.flatnonzero((key >= 0) & (val >= 0))
    order = np.lexsort((t[ev], key[ev]))  # (key, t), stable on original order
    ev = ev[order]
    ek, ev_val, et = key[ev], val[ev], t[ev]
    m = len(ev)
    pos = np.arange(m, dtype=np.int64)
    # previous event of the same (key, val) in key-time order
    o2 = np.lexsort((pos, ev_val, ek))
    p = np.full(m, -1, dtype=np.int64)
    same = (ek[o2][1:] == ek[o2][:-1]) & (ev_val[o2][1:] == ev_val[o2][:-1])
    p[o2[1:][same]] = o2[:-1][same]
    comb = ek * big + et
    q = np.flatnonzero(key >= 0)
    qk, qt = key[q], t[q]
    lo_t = np.zeros_like(qt) if window is None else np.maximum(qt - window, 0)
    a = np.searchsorted(comb, qk * big + lo_t, side="left").astype(np.int64)
    b = np.searchsorted(comb, qk * big + qt, side="left").astype(np.int64)
    cnt = _count_lt(p, b, a) - a
    res = np.full(n, np.nan, dtype=np.float32)
    res[q] = cnt
    return res


def first_before(key: np.ndarray, t: np.ndarray):
    """(has_earlier, first_time) for rows with key>=0: any row with same key and t' < t."""
    k = np.where(key >= 0, key, 0)
    first = np.full(k.max() + 1 if len(k) else 1, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(first, k[key >= 0], t[key >= 0])
    ft = first[k]
    has = (key >= 0) & (ft < t)
    return has, ft


def build(df: pd.DataFrame, fit_end_dt: int) -> pd.DataFrame:
    t = df["TransactionDT"].to_numpy(dtype=np.int64)
    card1 = _codes(df["card1"])
    dev = _codes(df["DeviceInfo"])
    id19, id20 = _codes(df["id_19"]), _codes(df["id_20"])
    id31 = _codes(df["id_31"])
    em_addr = _codes(df["P_emaildomain"].astype(object).where(
        df["P_emaildomain"].notna() & df["addr1"].notna(), None).astype(str)
        + "|" + df["addr1"].astype(str))
    em_addr[(df["P_emaildomain"].isna() | df["addr1"].isna()).to_numpy()] = -1
    em = _codes(df["P_emaildomain"])
    addr = _codes(df["addr1"])
    day = df["TransactionDT"] // DAY
    uid_s = (df["card1"].astype(str) + "_" + df["addr1"].astype(str) + "_"
             + (day - df["D1"]).astype(str))
    uid = _codes(uid_s)

    out = {"TransactionID": df["TransactionID"].to_numpy()}
    for kname, kcode in (("DeviceInfo", dev), ("id_19", id19), ("id_20", id20),
                         ("email_addr1", em_addr)):
        for wname, w in WINDOWS.items():
            out[f"{NAME}__nuniq_card1_by_{kname}_{wname}"] = distinct_before(kcode, card1, t, w)
    out[f"{NAME}__nuniq_DeviceInfo_by_uid"] = distinct_before(uid, dev, t, None)
    out[f"{NAME}__nuniq_id_31_by_uid"] = distinct_before(uid, id31, t, None)

    # time since this card1 first appeared with this DeviceInfo
    pair = _codes(pd.Series(card1 * (dev.max() + 2) + dev))
    pair[(card1 < 0) | (dev < 0)] = -1
    has, ft = first_before(pair, t)
    age = np.where(has, t - ft, np.nan).astype(np.float32)
    out[f"{NAME}__card1_device_age"] = age

    # first-seen flags: 1 if no earlier row of this uid with the same value
    for fname, col in (("DeviceInfo", dev), ("email", em), ("addr1", addr)):
        pr = _codes(pd.Series(uid * (col.max() + 2) + col))
        pr[(uid < 0) | (col < 0)] = -1
        h, _ = first_before(pr, t)
        flag = np.where(pr >= 0, (~h).astype(np.float32), np.nan).astype(np.float32)
        out[f"{NAME}__uid_first_{fname}"] = flag
    return pd.DataFrame(out)


def leakage_check(df: pd.DataFrame, feats: pd.DataFrame) -> str:
    rng = np.random.default_rng(7)
    t = df["TransactionDT"].to_numpy()
    uid_s = (df["card1"].astype(str) + "_" + df["addr1"].astype(str) + "_"
             + (df["TransactionDT"] // DAY - df["D1"]).astype(str)).to_numpy()
    col = {c: df[c].to_numpy() for c in ("card1", "DeviceInfo", "id_19", "id_20", "id_31",
                                         "P_emaildomain", "addr1")}
    nn = lambda v: v == v and v is not None  # noqa: E731
    notes = []
    # prefer rows that actually have a DeviceInfo so the check is not trivial
    cand = np.flatnonzero(df["DeviceInfo"].notna().to_numpy() & (np.arange(len(df)) > 1000))
    rows = list(rng.choice(cand, 3, replace=False)) + list(rng.choice(len(df), 2, replace=False))
    for i in rows:
        ti = t[i]
        prior = [j for j in np.flatnonzero(t < ti)]
        exp = {}
        for kname, kf in (("DeviceInfo", lambda j: col["DeviceInfo"][j]),
                          ("id_19", lambda j: col["id_19"][j]),
                          ("id_20", lambda j: col["id_20"][j]),
                          ("email_addr1", lambda j: (col["P_emaildomain"][j], col["addr1"][j])
                           if nn(col["P_emaildomain"][j]) and nn(col["addr1"][j]) else np.nan)):
            ki = kf(i)
            valid = (ki == ki) if not isinstance(ki, tuple) else True
            for wname, w in WINDOWS.items():
                if not valid:
                    exp[f"nuniq_card1_by_{kname}_{wname}"] = np.nan
                    continue
                s = set()
                for j in prior:
                    if w is not None and t[j] < ti - w:
                        continue
                    kj = kf(j)
                    if kj == kj and kj == ki and nn(col["card1"][j]):
                        s.add(col["card1"][j])
                exp[f"nuniq_card1_by_{kname}_{wname}"] = len(s)
        sd, s31 = set(), set()
        seen = {"d": False, "e": False, "a": False}
        for j in prior:
            if uid_s[j] == uid_s[i]:
                if nn(col["DeviceInfo"][j]):
                    sd.add(col["DeviceInfo"][j])
                if nn(col["id_31"][j]):
                    s31.add(col["id_31"][j])
                seen["d"] |= nn(col["DeviceInfo"][i]) and col["DeviceInfo"][j] == col["DeviceInfo"][i]
                seen["e"] |= nn(col["P_emaildomain"][i]) and col["P_emaildomain"][j] == col["P_emaildomain"][i]
                seen["a"] |= nn(col["addr1"][i]) and col["addr1"][j] == col["addr1"][i]
        exp["nuniq_DeviceInfo_by_uid"] = len(sd)
        exp["nuniq_id_31_by_uid"] = len(s31)
        for nm, k, c in (("DeviceInfo", "d", "DeviceInfo"), ("email", "e", "P_emaildomain"),
                         ("addr1", "a", "addr1")):
            exp[f"uid_first_{nm}"] = np.nan if not nn(col[c][i]) else float(not seen[k])
        if nn(col["DeviceInfo"][i]):
            ts = [t[j] for j in prior if col["DeviceInfo"][j] == col["DeviceInfo"][i]
                  and col["card1"][j] == col["card1"][i]]
            exp["card1_device_age"] = float(ti - min(ts)) if ts else np.nan
        else:
            exp["card1_device_age"] = np.nan
        for f, e in exp.items():
            g = feats[f"{NAME}__{f}"].iloc[i]
            ok = (np.isnan(e) and np.isnan(g)) or (not np.isnan(e) and not np.isnan(g) and abs(e - g) < 1e-3)
            assert ok, f"row {i} {f}: hand={e} vectorized={g}"
        notes.append(f"row {i} (DT {ti}): {len(exp)} features match")
    return "leakage_check PASS: " + "; ".join(notes)


def _signal(df, feats, fit_end_dt):
    from sklearn.metrics import average_precision_score

    from ecommerce_fraud import data
    _, valid, _, _ = data.time_split(df)
    msk = df["TransactionID"].isin(valid["TransactionID"]).to_numpy()
    y = df["isFraud"].to_numpy()[msk]
    for c in feats.columns[1:]:
        x = feats[c].to_numpy(dtype=np.float64)[msk]
        x = np.where(np.isnan(x), -1.0, x)
        print(f"SIGNAL {c}: pr_auc={average_precision_score(y, x):.4f} "
              f"neg={average_precision_score(y, -x):.4f}")


def main():
    from ecommerce_fraud import data
    df = data.load_joined().sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    train, _, _, _ = data.time_split(df)
    fit_end_dt = int(train["TransactionDT"].iloc[-1])
    t0 = time.time()
    feats = build(df, fit_end_dt)
    dt = time.time() - t0
    print("shape", feats.shape)
    print("NaN share:")
    print(feats.iloc[:, 1:].isna().mean().round(4).to_string())
    print(f"run time {dt:.1f}s, peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.2f} GB")
    print(leakage_check(df, feats))
    _signal(df, feats, fit_end_dt)
    print(f"BLOCK {NAME} OK")


if __name__ == "__main__":
    main()
