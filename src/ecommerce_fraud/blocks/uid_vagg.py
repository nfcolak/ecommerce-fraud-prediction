"""Block uid_vagg: past-only per-uid expanding mean/std of reduced V columns, C and D values.

V1-V339 are reduced by NaN-pattern grouping (fit window only): columns with the identical
NaN mask form a group, the column with most unique values represents it (plus second-ranked
columns of the largest groups until ~30 columns). For each kept column, each row gets the
mean/std over strictly earlier rows of the same uid (non-NaN values only) and its own
deviation from that past mean. No labels are used.
"""
from __future__ import annotations

import random
import time

import numpy as np
import pandas as pd

NAME = "uid_vagg"
DESCRIPTION = "Past-only per-uid expanding mean/std/deviation of reduced V, C1/2/5/13/14, D4n/D10n/D15n"
USES_LABELS = False

DAY = 86400
TARGET_V = 30
C_COLS = ["C1", "C2", "C5", "C13", "C14"]
D_COLS = ["D4", "D10", "D15"]


def _uid(df: pd.DataFrame) -> pd.Series:
    """Same uid as features_history.make_keys; NaN when card1 or D1 is missing."""
    day = df["TransactionDT"] // DAY
    d1n = day - df["D1"]
    s = df["card1"].astype(str) + "_" + df["addr1"].astype(str) + "_" + d1n.astype(str)
    return s.where(df["card1"].notna() & df["D1"].notna())


def select_v(df: pd.DataFrame, fit_end_dt: int, target: int = TARGET_V) -> list[str]:
    fit = df[df["TransactionDT"] <= fit_end_dt]
    vcols = [c for c in df.columns if c.startswith("V") and c[1:].isdigit()]
    groups: dict[bytes, list[tuple[int, str]]] = {}
    for c in vcols:
        col = fit[c]
        groups.setdefault(col.isna().to_numpy().tobytes(), []).append((col.nunique(), c))
    ranked = [sorted(g, key=lambda x: (-x[0], x[1])) for g in groups.values()]
    ranked.sort(key=lambda g: -len(g))
    keep = [g[0][1] for g in ranked]
    rank = 1
    while len(keep) < target:
        added = False
        for g in ranked:
            if len(g) > rank and len(keep) < target:
                keep.append(g[rank][1])
                added = True
        if not added:
            break
        rank += 1
    return sorted(keep, key=lambda c: int(c[1:]) if c[0] == "V" else 0)


def _source_columns(df: pd.DataFrame, fit_end_dt: int) -> dict[str, np.ndarray]:
    day = df["TransactionDT"] // DAY
    src = {}
    for c in select_v(df, fit_end_dt) + C_COLS:
        src[c] = df[c].to_numpy(np.float64)
    for d in D_COLS:
        src[f"{d}n"] = (day - df[d]).to_numpy(np.float64)
    return src


def build(df: pd.DataFrame, fit_end_dt: int) -> pd.DataFrame:
    t = df["TransactionDT"].to_numpy(np.int64)
    uid = _uid(df)
    codes = pd.factorize(uid)[0].astype(np.int64)  # NaN uid -> -1
    valid_uid = codes >= 0
    n = len(df)
    big = int(t.max()) + 1
    order = np.argsort(codes, kind="stable")  # time order kept inside a uid
    c, ts = codes[order], t[order]
    comb = c * big + ts
    gstart = np.searchsorted(comb, c * big, side="left")
    hi = np.searchsorted(comb, comb, side="left")  # earlier rows of the uid: [gstart, hi)
    last = np.maximum(hi - 1, 0)
    has = (hi > gstart) & (c >= 0)

    out = {"TransactionID": df["TransactionID"].to_numpy()}
    for name, vals in _source_columns(df, fit_end_dt).items():
        v = vals[order]
        ok = ~np.isnan(v)
        v0 = np.where(ok, v, 0.0)
        # center to keep the running sums of squares precise
        shift = np.nanmean(v) if ok.any() else 0.0
        v0 = np.where(ok, v - shift, 0.0)
        g_cnt = pd.Series(ok.astype(np.float64)).groupby(c).cumsum().to_numpy()
        g_s1 = pd.Series(v0).groupby(c).cumsum().to_numpy()
        g_s2 = pd.Series(v0 * v0).groupby(c).cumsum().to_numpy()
        cnt = np.where(has, g_cnt[last], 0.0)
        s1 = np.where(has, g_s1[last], 0.0)
        s2 = np.where(has, g_s2[last], 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_c = s1 / cnt
            var = (s2 - s1 * s1 / cnt) / (cnt - 1)
        mean = np.where(cnt >= 1, mean_c + shift, np.nan)
        std = np.where(cnt >= 2, np.sqrt(np.maximum(var, 0.0)), np.nan)
        dev = v - mean
        for fname, arr in (("mean", mean), ("std", std), ("dev", dev)):
            res = np.empty(n, dtype=np.float32)
            res[order] = arr
            res[~valid_uid] = np.nan
            out[f"{NAME}__{name}_{fname}"] = res
    return pd.DataFrame(out)


def leakage_check(df: pd.DataFrame, feats: pd.DataFrame, seed: int = 0, n: int = 5) -> str:
    """Recompute rows by hand (plain Python over raw rows) and compare to build() output."""
    rng = random.Random(seed)
    uid = _uid(df).tolist()
    t = df["TransactionDT"].tolist()
    cols = [c[len(NAME) + 2:-5] for c in feats.columns if c.endswith("_mean")]
    by_uid: dict[str, list[int]] = {}
    for i, u in enumerate(uid):
        if isinstance(u, str):
            by_uid.setdefault(u, []).append(i)
    cand = [i for i, u in enumerate(uid) if isinstance(u, str) and any(t[j] < t[i] for j in by_uid[u])]
    picks = rng.sample(cand, n)
    day = (df["TransactionDT"] // DAY)
    src = {}
    for name in cols:
        src[name] = ((day - df[name[:-1]]) if name.endswith("n") and name[:-1] in D_COLS else df[name]).to_numpy(np.float64)
    lines = []
    for i in picks:
        past = [j for j in by_uid[uid[i]] if t[j] < t[i]]
        for name in cols:
            vals = [src[name][j] for j in past if src[name][j] == src[name][j]]
            row = feats.iloc[i]
            m = sum(vals) / len(vals) if vals else float("nan")
            sd = (sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5 if len(vals) >= 2 else float("nan")
            cur = src[name][i]
            dv = cur - m
            for suffix, exp in (("mean", m), ("std", sd), ("dev", dv)):
                got = float(row[f"{NAME}__{name}_{suffix}"])
                if exp != exp:
                    assert got != got, f"row {i} {name}_{suffix}: expected NaN got {got}"
                else:
                    assert abs(got - exp) <= 1e-3 * max(1.0, abs(exp)), \
                        f"row {i} {name}_{suffix}: got {got} expected {exp}"
        lines.append(f"row {i} (DT {t[i]}): {len(past)} earlier rows of uid, {len(cols) * 3} features match")
    return "leakage_check OK\n" + "\n".join(lines)


def _main() -> None:
    import resource
    from sklearn.metrics import average_precision_score
    from ecommerce_fraud import data

    t0 = time.time()
    df = data.load_joined().sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    train, valid, _test, _ = data.time_split(df)
    fit_end = int(train["TransactionDT"].max())
    feats = build(df, fit_end)
    secs = time.time() - t0
    print(f"shape {feats.shape}")
    print("V kept:", select_v(df, fit_end))
    nan = feats.drop(columns="TransactionID").isna().mean()
    for col, share in nan.items():
        print(f"NaN {col}: {share:.3f}")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # bytes on macOS -> MB
    print(f"run time {secs:.1f}s, peak RSS {rss:.0f} MB")
    print(leakage_check(df, feats))
    vm = df["TransactionID"].isin(valid["TransactionID"]).to_numpy()
    y = df.loc[vm, "isFraud"].to_numpy()
    res = []
    for col in feats.columns[1:]:
        x = feats.loc[vm, col].to_numpy(np.float64)
        if np.isnan(x).all():
            continue
        x = np.where(np.isnan(x), np.nanmin(x) - 1.0, x)
        res.append((average_precision_score(y, x), average_precision_score(y, -x), col))
    res.sort(reverse=True)
    print(f"SIGNAL base rate {y.mean():.4f}; top 10 by PR-AUC (col, pos, neg):")
    for a, b, col in res[:10]:
        print(f"SIGNAL {col}: pos {a:.4f} neg {b:.4f}")
    print(f"SIGNAL best neg: {max(res, key=lambda r: r[1])[2]} {max(r[1] for r in res):.4f}")
    print(f"BLOCK {NAME} OK")


if __name__ == "__main__":
    _main()
