"""Block anomaly_score: unsupervised anomaly scores fitted on the fit window only (no labels)."""
from __future__ import annotations

import math
import random
import resource
import time

import numpy as np
import pandas as pd

NAME = "anomaly_score"
DESCRIPTION = "IsolationForest score and per-ProductCD robust (median/MAD) distance on 30 numeric columns, fit window only"
USES_LABELS = False

NUM_COLS = (["TransactionAmt"] + [f"C{i}" for i in range(1, 15)] + ["dist1"]
            + [f"D{i}" for i in range(1, 16) if i != 9])  # 30 columns; TransactionAmt -> log1p
Z_CLIP = 10.0
_STATE: dict = {}


def _matrix(df: pd.DataFrame) -> np.ndarray:
    x = df[NUM_COLS].astype("float64").copy()
    x["TransactionAmt"] = np.log1p(x["TransactionAmt"])
    return x.to_numpy()


def _robust_stats(x: np.ndarray):
    med = np.nanmedian(x, axis=0)
    mad = np.nanmedian(np.abs(x - med), axis=0) * 1.4826
    mad = np.where(mad > 1e-9, mad, 1.0)  # constant columns -> scale 1
    return med, mad


def build(df: pd.DataFrame, fit_end_dt: int) -> pd.DataFrame:
    from sklearn.ensemble import IsolationForest

    x = _matrix(df)
    fit_mask = (df["TransactionDT"] <= fit_end_dt).to_numpy()
    fill = np.nanmedian(x[fit_mask], axis=0)
    fill = np.where(np.isnan(fill), 0.0, fill)
    xf = np.where(np.isnan(x), fill, x)

    iso = IsolationForest(n_estimators=200, max_samples=4096, random_state=42, n_jobs=4)
    iso.fit(xf[fit_mask])
    iso_score = -iso.score_samples(xf)  # higher = more anomalous

    pcd = df["ProductCD"].astype(str).to_numpy()
    gmed, gmad = _robust_stats(xf[fit_mask])
    stats = {"__global__": (gmed, gmad)}
    for p in np.unique(pcd[fit_mask]):
        m = fit_mask & (pcd == p)
        if m.sum() >= 100:
            stats[p] = _robust_stats(xf[m])
    maha = np.zeros(len(df))
    zmax = np.zeros(len(df))
    for p in np.unique(pcd):
        med, mad = stats.get(p, stats["__global__"])
        idx = pcd == p
        z = np.clip(np.abs(xf[idx] - med) / mad, 0, Z_CLIP)
        maha[idx] = np.sqrt((z ** 2).sum(axis=1))
        zmax[idx] = z.max(axis=1)
    _STATE.update(iso=iso, fill=fill, stats=stats, fit_mask=fit_mask)
    return pd.DataFrame({
        "TransactionID": df["TransactionID"].to_numpy(),
        f"{NAME}__iforest": iso_score.astype("float32"),
        f"{NAME}__robust_maha": maha.astype("float32"),
        f"{NAME}__robust_zmax": zmax.astype("float32"),
    })


def leakage_check(df: pd.DataFrame, feats: pd.DataFrame) -> str:
    """Recompute 5 random rows in plain Python from raw values and the fit-window stats."""
    assert "isFraud" not in NUM_COLS
    rng = random.Random(7)
    fit_mask = _STATE["fit_mask"]
    stats, fill, iso = _STATE["stats"], _STATE["fill"], _STATE["iso"]
    # independently recompute one group's median/MAD by plain sorting on raw fit rows
    notes = []
    for i in rng.sample(range(len(df)), 5):
        row = df.iloc[i]
        vals = []
        for j, c in enumerate(NUM_COLS):
            v = row[c]
            v = float("nan") if v is None else float(v)
            if c == "TransactionAmt":
                v = math.log1p(v)
            vals.append(fill[j] if math.isnan(v) else v)
        p = str(row["ProductCD"])
        med, mad = stats.get(p, stats["__global__"])
        zs = [min(abs(vals[j] - med[j]) / mad[j], Z_CLIP) for j in range(len(vals))]
        maha = math.sqrt(sum(z * z for z in zs))
        got = feats.iloc[i]
        assert abs(maha - got[f"{NAME}__robust_maha"]) < 1e-3 * max(1.0, maha), (i, maha)
        assert abs(max(zs) - got[f"{NAME}__robust_zmax"]) < 1e-3 * max(1.0, max(zs)), i
        s = -iso.score_samples(np.array([vals]))[0]
        assert abs(s - got[f"{NAME}__iforest"]) < 1e-5, (i, s)
        assert feats["TransactionID"].iloc[i] == row["TransactionID"]
        notes.append(i)
    # plain-Python MAD check of the global stats on a fit-window column
    col = [float(v) for v in df.loc[fit_mask, "C1"].tolist() if v == v]
    col.sort()
    n = len(col)
    m = (col[n // 2] + col[(n - 1) // 2]) / 2
    dev = sorted(abs(v - m) for v in col)
    mad = (dev[n // 2] + dev[(n - 1) // 2]) / 2 * 1.4826
    gmed, gmad = stats["__global__"]
    j = NUM_COLS.index("C1")
    # global stats are on NaN-filled data, so only check the median loosely
    assert abs(gmed[j] - m) <= max(1.0, abs(m)) * 0.5, (gmed[j], m)
    return f"leakage_check passed on rows {notes}: no labels used, stats from fit window only (global C1 median {gmed[j]:.2f} vs raw {m:.2f}, MAD {gmad[j]:.2f} vs raw {mad:.2f})"


def _signal(df, feats, fit_end_dt):
    from sklearn.metrics import average_precision_score
    from ecommerce_fraud.data import time_split
    _, valid, _, _ = time_split(df)
    f = feats.set_index("TransactionID").loc[valid["TransactionID"]]
    y = valid["isFraud"].to_numpy()
    for c in f.columns:
        s = np.nan_to_num(f[c].to_numpy(dtype="float64"))
        print(f"SIGNAL {c}: valid PR-AUC {average_precision_score(y, s):.4f} | negative {average_precision_score(y, -s):.4f} | base rate {y.mean():.4f}")


if __name__ == "__main__":
    from ecommerce_fraud.data import load_joined, time_split

    t0 = time.time()
    raw = load_joined()
    df = raw.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    train, _, _, _ = time_split(df)
    fit_end_dt = int(train["TransactionDT"].max())
    feats = build(df, fit_end_dt)
    dt = time.time() - t0
    print("shape", feats.shape)
    print("NaN share per column:")
    print(feats.isna().mean().to_string())
    print(f"run time {dt:.1f}s")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # bytes on macOS
    print(f"peak RSS {rss:.2f} GB")
    print(leakage_check(df, feats))
    _signal(df, feats, fit_end_dt)
    print(f"BLOCK {NAME} OK")
