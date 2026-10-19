"""Block uid2: past-only behaviour features on refined client keys (32 columns).

What final already has (features_history / features_uid_agg), NOT repeated here:
  uid = card1+addr1+D1n and uid_email: cnt 1d/7d/30d/all, seconds since previous, amount mean/std/ratio
  (to the mean), nunique of P_emaildomain/DeviceInfo/addr1, past means of C/D/V columns, same-product share.
New here:
  * three other client keys:  kfull = card1..card6+addr1+D1n (strict card identity),
                              k15   = card1+addr1+D15n (other D normalisation),
                              k1    = card1+D1n (no address)
    A key is NaN (all its features NaN) if any key part is missing.
  * per key: short windows (1h, 1d), gap between the previous two transactions (prev_gap),
    amount z-score of log1p(amount) against the key's past mean/std (restart-at-key cumsums),
    log amount ratio to the previous amount, "current DeviceInfo / P_emaildomain / id_31 seen earlier for
    this key" flags, and the share of distinct DeviceInfo / id_31 / P_emaildomain among the key's past rows.
Every value of row i uses only rows with TransactionDT strictly smaller than row i's (ties are not earlier).
Cumulative quantities live on high-cardinality keys only; no labels, no absolute time.
"""
from __future__ import annotations

import argparse
import resource
import time

import numpy as np
import pandas as pd

NAME = "uid2"
DESCRIPTION = ("Past-only short-window counts, gaps, amount z-score and device/email/browser novelty on "
               "card1..6+addr1+D1n, card1+addr1+D15n, card1+D1n and uid keys (32 cols)")
USES_LABELS = False

DAY = 86400
HOUR = 3600
VAL_COLS = {"dev": "DeviceInfo", "email": "P_emaildomain", "id31": "id_31"}
# which features each key gets
KEY_FEATS = {
    "uid": ["cnt_1h", "prev_gap", "amt_z", "amt_dprev", "seen_dev", "seen_email", "seen_id31",
            "share_dev", "share_email", "share_id31"],
    "kfull": ["cnt_1h", "cnt_1d", "since_prev", "prev_gap", "amt_z", "amt_dprev", "seen_dev", "seen_email",
              "seen_id31", "share_dev", "share_email", "share_id31"],
    "k15": ["cnt_1h", "since_prev", "amt_z", "seen_email", "seen_id31"],
    "k1": ["cnt_1h", "since_prev", "amt_z", "seen_email", "seen_id31"],
}
MIN_STD = 1e-4


def _s(s: pd.Series) -> pd.Series:
    return s.astype(str)


def make_keys(df: pd.DataFrame) -> dict:
    """String keys; None-like NaN when any key part is missing."""
    d1n = (df["TransactionDT"] // DAY) - df["D1"]
    d15n = (df["TransactionDT"] // DAY) - df["D15"]
    card = _s(df["card1"])
    for c in ("card2", "card3", "card4", "card5", "card6"):
        card = card + "_" + _s(df[c])
    ca = _s(df["card1"]) + "_" + _s(df["addr1"])
    ok_c1 = df["card1"].notna()
    ok_all = ok_c1 & df[["card2", "card3", "card4", "card5", "card6"]].notna().all(axis=1)
    return {
        "uid": (ca + "_" + _s(d1n)).where(ok_c1 & df["addr1"].notna() & df["D1"].notna()),
        "kfull": (card + "_" + _s(df["addr1"]) + "_" + _s(d1n)).where(
            ok_all & df["addr1"].notna() & df["D1"].notna()),
        "k15": (ca + "_" + _s(d15n)).where(ok_c1 & df["addr1"].notna() & df["D15"].notna()),
        "k1": (_s(df["card1"]) + "_" + _s(d1n)).where(ok_c1 & df["D1"].notna()),
    }


def _key_feats(codes, t, la, vals, want, prefix, out):
    """codes: int key id (-1 = invalid key); t: DT; la: log1p amount; vals: {tag: int codes (-1 = NaN)}."""
    n = len(codes)
    valid = codes >= 0
    big = int(t.max()) + 1
    order = np.argsort(codes, kind="stable")  # time order is kept inside a key
    c, ts = codes[order], t[order]
    comb = c * big + ts
    gstart = np.searchsorted(comb, c * big, side="left")
    hi = np.searchsorted(comb, comb, side="left")  # earlier rows of the key: [gstart, hi)
    n_prev = hi - gstart
    has = n_prev > 0
    has2 = n_prev > 1
    last = np.maximum(hi - 1, 0)
    last2 = np.maximum(hi - 2, 0)

    def put(name, arr):
        res = np.full(n, np.nan, dtype=np.float64)
        res[order] = arr
        res[~valid] = np.nan
        out[f"{NAME}__{prefix}_{name}"] = res.astype(np.float32)

    if "cnt_1h" in want:
        put("cnt_1h", hi - np.searchsorted(comb, c * big + np.maximum(ts - HOUR, 0), side="left"))
    if "cnt_1d" in want:
        put("cnt_1d", hi - np.searchsorted(comb, c * big + np.maximum(ts - DAY, 0), side="left"))
    if "since_prev" in want:
        put("since_prev", np.where(has, ts - ts[last], np.nan))
    if "prev_gap" in want:
        put("prev_gap", np.where(has2, ts[last] - ts[last2], np.nan))
    if "amt_z" in want or "amt_dprev" in want:
        a = la[order]
        if "amt_z" in want:
            a0 = a - a.mean()
            s1 = pd.Series(a0).groupby(c).cumsum().to_numpy()  # cumsums restart at each key
            s2 = pd.Series(a0 * a0).groupby(c).cumsum().to_numpy()
            cnt = np.where(has, n_prev, 1).astype(np.float64)
            m1 = s1[last] / cnt
            var = (s2[last] - s1[last] ** 2 / cnt) / np.maximum(cnt - 1, 1)
            sd = np.sqrt(np.maximum(var, 0.0))
            z = np.where(has2 & (sd > MIN_STD), (a0 - m1) / np.where(sd > MIN_STD, sd, 1.0), np.nan)
            put("amt_z", z)
        if "amt_dprev" in want:
            put("amt_dprev", np.where(has, a - a[last], np.nan))
    for tag, v in vals.items():
        if f"seen_{tag}" not in want and f"share_{tag}" not in want:
            continue
        vv = v[order]
        ok = (vv >= 0) & (c >= 0)
        # first time each (key, value) pair appears (a pair "was seen before t" iff first_t < t)
        pid = pd.factorize(c[ok] * (int(vv.max()) + 2) + vv[ok])[0]
        first_t = pd.Series(ts[ok]).groupby(pid).transform("min").to_numpy()
        if f"seen_{tag}" in want:
            seen = np.full(len(c), np.nan)
            seen[ok] = (first_t < ts[ok]).astype(np.float64)
            put(f"seen_{tag}", seen)
        if f"share_{tag}" in want:
            pairs = (pd.DataFrame({"k": c[ok], "p": pid, "t": ts[ok]}).groupby("p", sort=False)
                     .agg(k=("k", "first"), t=("t", "min")).sort_values(["k", "t"], kind="stable"))
            pc = pairs["k"].to_numpy(np.int64) * big + pairs["t"].to_numpy(np.int64)
            lo_p = np.searchsorted(pc, c * big, side="left")
            hi_p = np.searchsorted(pc, c * big + ts, side="left")  # pairs first seen strictly before t
            put(f"share_{tag}", np.where(has, (hi_p - lo_p) / np.maximum(n_prev, 1), np.nan))


def build(df: pd.DataFrame, fit_end_dt: int) -> pd.DataFrame:
    t = df["TransactionDT"].to_numpy(np.int64)
    if len(t) > 1 and (np.diff(t) < 0).any():
        raise ValueError("df must be sorted by TransactionDT")
    la = np.log1p(df["TransactionAmt"].to_numpy(np.float64))
    keys = make_keys(df)
    vals = {tag: pd.factorize(df[col])[0].astype(np.int64) for tag, col in VAL_COLS.items()}
    out = {"TransactionID": df["TransactionID"].to_numpy()}
    for kname, want in KEY_FEATS.items():
        codes = pd.factorize(keys[kname])[0].astype(np.int64)  # NaN key -> -1
        _key_feats(codes, t, la, vals, set(want), kname, out)
    return pd.DataFrame(out)


# ----------------------------------------------------------------- brute-force check
def brute_row(i, df, keys, kname, dts, la, vals_raw):
    """Plain-loop recompute of every feature of one row from strictly earlier rows (dict name->value)."""
    res = {}
    key = keys[kname]
    ki = key.iloc[i]
    want = KEY_FEATS[kname]
    nan = float("nan")
    if not isinstance(ki, str):
        return {f"{NAME}__{kname}_{f}": nan for f in want}
    ti = dts[i]
    idx = [j for j in key_index[kname].get(ki, []) if dts[j] < ti]  # all rows of this key, earlier DT
    # (key_index is built once from the key column, rows in input order)
    idx.sort(key=lambda j: (dts[j], j))
    n_prev = len(idx)
    for f in want:
        name = f"{NAME}__{kname}_{f}"
        if f == "cnt_1h":
            v = float(sum(1 for j in idx if dts[j] >= ti - HOUR))
        elif f == "cnt_1d":
            v = float(sum(1 for j in idx if dts[j] >= ti - DAY))
        elif f == "since_prev":
            v = float(ti - max(dts[j] for j in idx)) if idx else nan
        elif f == "prev_gap":
            v = float(dts[idx[-1]] - dts[idx[-2]]) if n_prev >= 2 else nan
        elif f == "amt_z":
            if n_prev >= 2:
                a = np.array([la[j] for j in idx])
                sd = a.std(ddof=1)
                v = float((la[i] - a.mean()) / sd) if sd > MIN_STD else nan
            else:
                v = nan
        elif f == "amt_dprev":
            v = float(la[i] - la[idx[-1]]) if idx else nan
        elif f.startswith("seen_"):
            cur = vals_raw[f[5:]][i]
            v = nan if cur is None else float(any(vals_raw[f[5:]][j] == cur for j in idx))
        elif f.startswith("share_"):
            col = vals_raw[f[6:]]
            v = len({col[j] for j in idx if col[j] is not None}) / n_prev if n_prev else nan
        res[name] = v
    return res


key_index: dict = {}


def leakage_check(df: pd.DataFrame, feats: pd.DataFrame, n_rows: int = 200, seed: int = 0) -> str:
    """Recompute all 32 features for ~n_rows rows by a plain loop and compare. Includes tie rows."""
    keys = make_keys(df)
    dts = df["TransactionDT"].to_numpy(np.int64).tolist()
    la = np.log1p(df["TransactionAmt"].to_numpy(np.float64)).tolist()
    vals_raw = {tag: [None if pd.isna(x) else x for x in df[col].tolist()] for tag, col in VAL_COLS.items()}
    key_index.clear()
    for kname in KEY_FEATS:
        d: dict = {}
        for j, k in enumerate(keys[kname].tolist()):
            if isinstance(k, str):
                d.setdefault(k, []).append(j)
        key_index[kname] = d
    rng = np.random.default_rng(seed)
    n = len(df)
    # rows with same-DT same-key neighbours (tie test) + rows with history + random rows
    ties = []
    for d in key_index.values():
        for rows in d.values():
            if len(rows) > 1:
                tt = [dts[j] for j in rows]
                if len(set(tt)) < len(tt):
                    ties += [j for j in rows if tt.count(dts[j]) > 1]
    ties = sorted(set(ties))
    pick = set(rng.choice(n, size=min(n_rows - min(len(ties), 40), n), replace=False).tolist())
    pick |= set(rng.choice(ties, size=min(40, len(ties)), replace=False).tolist()) if ties else set()
    rows = sorted(pick)[:n_rows + 40]
    cols = [c for c in feats.columns if c != "TransactionID"]
    bad, ncmp, nonnan = 0, 0, 0
    for i in rows:
        exp = {}
        for kname in KEY_FEATS:
            exp.update(brute_row(i, df, keys, kname, dts, la, vals_raw))
        for c in cols:
            got, e = float(feats[c].iloc[i]), exp[c]
            ncmp += 1
            if e == e:
                nonnan += 1
            same = (got != got and e != e) or (got == got and e == e and abs(got - e) <= 2e-4 * max(1.0, abs(e)))
            if not same:
                bad += 1
                if bad <= 10:
                    print(f"MISMATCH row {i} {c}: feature={got} brute={e}")
    return (f"leakage recompute: {len(rows)} rows ({len(ties)} tie-rows available), {ncmp} values compared, "
            f"{nonnan} non-NaN, mismatches {bad} -> {'ALL MATCH' if bad == 0 else 'MISMATCH FOUND'}")


def drift_check(df: pd.DataFrame, feats: pd.DataFrame, fit_end_dt: int) -> list:
    """Early vs late half of the fit rows (DT <= fit_end_dt), AUC per column (NaN -> -1e9)."""
    from sklearn.metrics import roc_auc_score
    m = (df["TransactionDT"] <= fit_end_dt).to_numpy()
    t = df["TransactionDT"].to_numpy()[m]
    late = (t > np.median(t)).astype(int)
    res = []
    for c in [c for c in feats.columns if c != "TransactionID"]:
        x = feats[c].to_numpy(np.float64)[m]
        x = np.where(np.isnan(x), -1e9, x)
        a = roc_auc_score(late, x)
        res.append((c, a, max(a, 1 - a)))
    return res


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, required=True)
    ap.add_argument("--drift-end-day", type=float, default=0.0, help="if >0, run drift check on rows <= this day")
    ap.add_argument("--no-brute", action="store_true")
    a = ap.parse_args()
    from ecommerce_fraud.blocks import discover, sorted_full_frame, check_frame

    df = sorted_full_frame().iloc[: a.max_rows].reset_index(drop=True)
    print("discover():", sorted(discover()))
    assert USES_LABELS is False
    t0 = time.time()
    feats = build(df, int(df["TransactionDT"].max()))
    secs = time.time() - t0
    check_frame(NAME, feats, df)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    nf = feats.shape[1] - 1
    print(f"rows {len(df)} cols {nf} build {secs:.2f}s ({secs / len(df) * 590540:.0f}s est. at 590540 rows) "
          f"peak RSS {rss:.0f} MB")
    print("NaN share per column:")
    for c, v in feats.drop(columns="TransactionID").isna().mean().items():
        print(f"  {c}: {v:.3f}")
    if not a.no_brute:
        t0 = time.time()
        print(leakage_check(df, feats), f"({time.time() - t0:.0f}s)")
    if a.drift_end_day > 0:
        fe = int(a.drift_end_day * DAY)
        print(f"drift check (fit rows DT <= {fe}, early vs late half), flag > 0.70:")
        for c, auc, mx in sorted(drift_check(df, feats, fe), key=lambda r: -r[2]):
            print(f"  DRIFT {c}: AUC {auc:.3f} {'FLAG' if mx > 0.70 else ''}")


if __name__ == "__main__":
    _main()
