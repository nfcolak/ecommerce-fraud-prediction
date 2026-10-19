"""Block freq_enc: causal frequency encodings fitted on the fit window.

For each key, the encoding is count(value) / n_rows over rows with
TransactionDT <= fit_end_dt. Choice for causality:
  * rows AFTER the fit window (valid/test scoring rows) use the full fit-window
    count of their value divided by the number of fit-window rows; unseen -> 0.
  * rows INSIDE the fit window use only strictly earlier fit-window rows
    (expanding count of the same value / number of strictly earlier rows), so
    they never see their own future. The first fit rows (no earlier row) are NaN.
Train and scoring rows are therefore both computed from the past only.
No labels are used.
"""
from __future__ import annotations

import random
import resource
import time

import numpy as np
import pandas as pd

from ecommerce_fraud.features_history import make_keys

NAME = "freq_enc"
DESCRIPTION = "Causal frequency encodings (fit-window counts; expanding for in-window rows)"
USES_LABELS = False

COLS = ["card1", "card2", "card3", "card5", "addr1", "addr2", "P_emaildomain",
        "R_emaildomain", "DeviceInfo", "id_30", "id_31", "id_33"]
PAIRS = {"card1_P_emaildomain": ("card1", "P_emaildomain"),
         "card1_DeviceInfo": ("card1", "DeviceInfo")}


def _key_strings(df: pd.DataFrame) -> dict:
    keys = {c: df[c].astype(str).fillna("nan") for c in COLS}
    mk = make_keys(df)
    keys["uid"] = mk["uid"]
    keys["card1_addr1"] = mk["card1_addr1"]
    for name, (a, b) in PAIRS.items():
        keys[name] = keys[a] + "_" + keys[b]
    return {k: v.fillna("nan").astype(str) for k, v in keys.items()}


def _encode(codes: np.ndarray, t: np.ndarray, fit_end_dt: int) -> np.ndarray:
    n = len(codes)
    out = np.zeros(n, dtype=np.float64)
    nfit = int(np.searchsorted(t, fit_end_dt, side="right"))  # t is sorted
    if nfit == 0:
        return out.astype(np.float32)
    fc, ft = codes[:nfit], t[:nfit]
    # scoring rows: full fit-window share
    cnt = np.bincount(fc, minlength=int(codes.max()) + 1)
    out[nfit:] = cnt[codes[nfit:]] / nfit
    # fit rows: strictly earlier same-key rows / strictly earlier rows
    big = int(ft.max()) + 1
    order = np.argsort(fc, kind="stable")
    sc = fc[order].astype(np.int64)
    comb = sc * big + ft[order]
    gstart = np.searchsorted(comb, sc * big, side="left")
    n_prev_key = np.searchsorted(comb, comb, side="left") - gstart
    n_prev = np.searchsorted(ft, ft, side="left")[order]
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(n_prev > 0, n_prev_key / np.maximum(n_prev, 1), np.nan)
    fit_out = np.empty(nfit, dtype=np.float64)
    fit_out[order] = v
    out[:nfit] = fit_out
    return out.astype(np.float32)


def build(df: pd.DataFrame, fit_end_dt: int) -> pd.DataFrame:
    t = df["TransactionDT"].to_numpy(np.int64)
    res = {"TransactionID": df["TransactionID"].to_numpy()}
    for name, s in _key_strings(df).items():
        codes = pd.factorize(s, use_na_sentinel=False)[0].astype(np.int64)
        res[f"{NAME}__{name}"] = _encode(codes, t, fit_end_dt)
    return pd.DataFrame(res)


def leakage_check(df: pd.DataFrame, feats: pd.DataFrame, fit_end_dt: int = None) -> str:
    keys = _key_strings(df)
    t = df["TransactionDT"].to_numpy()
    if fit_end_dt is None:
        fit_end_dt = int(t[int(len(t) * 0.6)])
    rng = random.Random(0)
    n = len(df)
    nfit = int(np.searchsorted(t, fit_end_dt, side="right"))
    idx = [rng.randrange(nfit) for _ in range(3)] + [rng.randrange(nfit, n) for _ in range(2)]
    lines = []
    for i in idx:
        for name, s in keys.items():
            vals = s.to_numpy()
            if t[i] > fit_end_dt:
                rows = list(range(nfit))
            else:
                rows = [j for j in range(nfit) if t[j] < t[i]]
            exp = (sum(1 for j in rows if vals[j] == vals[i]) / len(rows)) if rows else np.nan
            got = feats[f"{NAME}__{name}"].iloc[i]
            ok = (np.isnan(exp) and np.isnan(got)) or abs(exp - got) < 1e-6
            assert ok, (i, name, exp, got)
        lines.append(f"row {i} dt={t[i]} {'scoring' if t[i] > fit_end_dt else 'fit'} ok")
    return "leakage_check passed: " + "; ".join(lines)


def _signal(feats, df, valid_ids):
    from sklearn.metrics import average_precision_score
    m = df["TransactionID"].isin(valid_ids).to_numpy()
    y = df["isFraud"].to_numpy()[m]
    for c in feats.columns[1:]:
        x = np.nan_to_num(feats[c].to_numpy()[m], nan=0.0)
        print(f"SIGNAL {c}: pr_auc={average_precision_score(y, x):.4f} neg={average_precision_score(y, -x):.4f}")


if __name__ == "__main__":
    from ecommerce_fraud import data
    df = data.load_joined().sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    train, valid, _, _ = data.time_split(df)
    fit_end = int(train["TransactionDT"].max())
    t0 = time.time()
    feats = build(df, fit_end)
    el = time.time() - t0
    print("shape", feats.shape)
    print(feats.isna().mean().round(4).to_string())
    print(f"run time {el:.1f}s peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.2f} GB")
    print(leakage_check(df, feats, fit_end))
    _signal(feats, df, set(valid["TransactionID"]))
    print(f"BLOCK {NAME} OK")
