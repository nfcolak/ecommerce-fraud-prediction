"""Past-only transaction-entity index (temporal bipartite graph) for the graph feature blocks.

Entity types: uid (card1+addr1+D1), card1, addr1, P_emaildomain, R_emaildomain, fp (device
fingerprint). A NaN entity value never links (code -1). Input frame must be sorted by
TransactionDT (stable). For row i and entity type e, "earlier rows" are the rows of the same entity
with TransactionDT STRICTLY smaller than t_i (ties excluded). Nothing here reads isFraud.

Per type the rows with a valid code are stored entity-major, time-sorted inside an entity, so the
earlier rows of row i form one contiguous slice [lo_i, hi_i) of that array (hi_i by searchsorted).
Windows, counts, amount sums and distinct counts are prefix-sum / searchsorted lookups on it.

Hub rule (causal): entity e is a hub for row i when more than HUB_CAP (50) distinct uids occur among
the STRICTLY earlier rows of e at time t_i. Multi-hop expansion does not pass through a hub.

`selfcheck_main(mod, brute_fn)` is the shared `_main()` body of the graph blocks.
"""
from __future__ import annotations

import argparse
import resource
import time

import numpy as np
import pandas as pd

from ecommerce_fraud.features_graph import make_fp
from ecommerce_fraud.features_history import DAY, make_keys

ENTITY_TYPES = ["uid", "card1", "addr1", "P_emaildomain", "R_emaildomain", "fp"]
HUB_CAP = 50
UID_PARTS = ["card1", "addr1", "D1"]


def entity_codes(df: pd.DataFrame) -> dict:
    """{entity type: int64 code per row, -1 = NaN (never links)}."""
    out = {}
    uid_ok = ~df[UID_PARTS].isna().any(axis=1).to_numpy()
    uid = make_keys(df)["uid"].astype(object).to_numpy()
    c = np.full(len(df), -1, dtype=np.int64)
    c[uid_ok] = pd.factorize(uid[uid_ok])[0]
    out["uid"] = c
    for e in ("card1", "addr1", "P_emaildomain", "R_emaildomain"):
        out[e] = pd.factorize(df[e])[0].astype(np.int64)
    out["fp"] = pd.factorize(make_fp(df))[0].astype(np.int64)
    return out


class TypeIndex:
    """Entity-major, time-sorted rows of one entity type."""

    def __init__(self, codes: np.ndarray, t: np.ndarray):
        n = len(codes)
        self.n = n
        self.valid = codes >= 0
        idx = np.flatnonzero(self.valid)
        order = np.argsort(codes[idx], kind="stable")  # stays time-sorted inside an entity
        self.rows = idx[order]                          # original row index per sorted position
        self.cs = codes[self.rows]
        self.ts = t[self.rows]
        self.big = int(t.max()) + 1 if n else 1
        self.comb = self.cs * self.big + self.ts
        self.gstart = np.searchsorted(self.cs, self.cs, side="left")
        self.hi = np.searchsorted(self.comb, self.comb, side="left")  # strictly earlier end
        self.pos = np.full(n, -1, dtype=np.int64)
        self.pos[self.rows] = np.arange(len(self.rows))

    def lo_hi(self):
        """(valid mask, lo, hi) per ORIGINAL row; lo/hi index the sorted arrays (garbage if invalid)."""
        p = np.maximum(self.pos, 0)
        return self.valid, self.gstart[p], self.hi[p]

    def window_lo(self, window: int) -> np.ndarray:
        """Per ORIGINAL row: first sorted position with t_j >= t_i - window (same entity)."""
        p = np.maximum(self.pos, 0)
        q = self.cs[p] * self.big + np.maximum(self.ts[p] - window, 0)
        return np.maximum(np.searchsorted(self.comb, q, side="left"), self.gstart[p])


class EntityGraph:
    def __init__(self, df: pd.DataFrame):
        t = df["TransactionDT"].to_numpy(np.int64)
        if len(t) > 1 and (np.diff(t) < 0).any():
            raise ValueError("df must be sorted by TransactionDT")
        self.n = len(df)
        self.t = t
        self.amt = df["TransactionAmt"].to_numpy(np.float64)
        self.codes = entity_codes(df)
        self.ix = {e: TypeIndex(self.codes[e], t) for e in ENTITY_TYPES}
        self._hub = {}

    # ---- helpers -------------------------------------------------------------------------
    def earlier_rows(self, e: str, i: int) -> np.ndarray:
        """Original row indices of entity type e's earlier rows (t_j < t_i, same entity), time order."""
        ix = self.ix[e]
        if not ix.valid[i]:
            return np.empty(0, dtype=np.int64)
        p = ix.pos[i]
        return ix.rows[ix.gstart[p]:ix.hi[p]]

    def degree(self, e: str) -> np.ndarray:
        """Number of earlier rows of the entity (NaN where the entity is NaN)."""
        v, lo, hi = self.ix[e].lo_hi()
        return np.where(v, hi - lo, np.nan)

    def window_count(self, e: str, window: int) -> np.ndarray:
        ix = self.ix[e]
        v, _, hi = ix.lo_hi()
        wl = ix.window_lo(window)
        return np.where(v, hi - wl, np.nan)

    def age_days(self, e: str) -> np.ndarray:
        ix = self.ix[e]
        v, lo, hi = ix.lo_hi()
        age = (self.t - ix.ts[np.minimum(lo, max(len(ix.ts) - 1, 0))]) / DAY if len(ix.ts) else np.zeros(self.n)
        return np.where(v & (hi > lo), age, np.nan)

    def amount_stats(self, e: str):
        """(mean, population std) of earlier-row TransactionAmt; NaN if entity NaN or no earlier row."""
        ix = self.ix[e]
        v, lo, hi = ix.lo_hi()
        a = self.amt[ix.rows]
        # Per-entity cumulative sums that restart at each entity (lo is always the entity start), on
        # amounts shifted by the entity's first amount: global running sums of squares lost ~1e-3
        # relative precision on the std (full-data self-check mismatch, 2026-10-16).
        d = a - a[ix.gstart] if len(a) else a
        g1 = pd.Series(d).groupby(ix.cs).cumsum().to_numpy()
        g2 = pd.Series(d * d).groupby(ix.cs).cumsum().to_numpy()
        k = hi - lo
        last = np.maximum(hi - 1, 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            s1 = np.where(k > 0, g1[last] if len(a) else 0.0, 0.0)
            s2 = np.where(k > 0, g2[last] if len(a) else 0.0, 0.0)
            md = s1 / k
            var = np.maximum(s2 / k - md * md, 0.0)
            m = md + (a[lo] if len(a) else 0.0)
        ok = v & (k > 0)
        return np.where(ok, m, np.nan), np.where(ok, np.sqrt(var), np.nan)

    def distinct_before(self, e: str, other: np.ndarray) -> np.ndarray:
        """Number of distinct non-NaN `other` codes among earlier rows of the entity (NaN where entity NaN)."""
        ix = self.ix[e]
        v, lo, hi = ix.lo_hi()
        o = other[ix.rows]
        m = int(other.max()) + 2 if len(other) else 1
        key = ix.cs * m + (o + 1)
        first = ~pd.Series(key).duplicated(keep="first").to_numpy() & (o >= 0)
        cf = np.concatenate([[0], np.cumsum(first)])
        return np.where(v, cf[hi] - cf[lo], np.nan)

    def hub(self, e: str) -> np.ndarray:
        """bool per row: entity of row i has > HUB_CAP distinct uids among rows strictly before t_i."""
        if e not in self._hub:
            d = self.distinct_before(e, self.codes["uid"])
            self._hub[e] = np.nan_to_num(d, nan=0.0) > HUB_CAP
        return self._hub[e]


def _auc_line(c, s, yv, ap):
    pos = np.where(np.isnan(s), np.nanmin(s) - 1 if (~np.isnan(s)).any() else 0, s)
    neg = np.where(np.isnan(s), -(np.nanmax(s) + 1) if (~np.isnan(s)).any() else 0, -s)
    return f"SIGNAL {c:50s} {ap(yv, pos):.4f} / {ap(yv, neg):.4f}"


def selfcheck_main(mod, brute_fn, argv=None) -> None:
    """Shared _main() for graph blocks. brute_fn(df, feats, i) -> list of (col, got, hand)."""
    from sklearn.metrics import average_precision_score

    from ecommerce_fraud.blocks import check_frame, sorted_full_frame
    from ecommerce_fraud.data import time_split

    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--max-rows", type=int, default=0)
    args = ap_.parse_args(argv)
    df = sorted_full_frame()
    if args.max_rows:
        df = df.iloc[: args.max_rows].reset_index(drop=True)
        fit_end_dt = int(df["TransactionDT"].iloc[int(len(df) * 0.75)])
        vmask = (df["TransactionDT"] > fit_end_dt).to_numpy()
    else:
        train, valid, _test, _ = time_split(df)
        fit_end_dt = int(train["TransactionDT"].max())
        vmask = df["TransactionID"].isin(valid["TransactionID"]).to_numpy()
    t0 = time.time()
    feats = mod.build(df, fit_end_dt)
    secs = time.time() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    check_frame(mod.NAME, feats, df)
    print(f"shape {feats.shape}  rows {len(df)}  fit_end_dt {fit_end_dt}")
    print("NaN share per column:")
    for c in feats.columns[1:]:
        print(f"  {c:50s} {feats[c].isna().mean():.4f}")
    print(f"build time {secs:.1f}s  peak RSS {rss:.2f} GB")
    # leakage check: 5 random rows by brute force (plain Python over strictly earlier rows)
    rng = np.random.default_rng(7)
    lo = len(df) // 2
    picks = [int(x) for x in rng.choice(np.arange(lo, len(df)), 5, replace=False)]
    bad = []
    for i in picks:
        print(f"row {i} TransactionID {df['TransactionID'].iat[i]} DT {df['TransactionDT'].iat[i]}")
        cmp = brute_fn(df, feats, i)
        for col, got, hand in cmp:
            ok = (np.isnan(got) and np.isnan(hand)) or abs(got - hand) <= 1e-6 * max(1.0, abs(hand))
            if not ok:
                bad.append((i, col, got, hand))
        print(f"  {len(cmp)} columns compared, nonzero/non-NaN hand values: "
              f"{sum(1 for _, _, h in cmp if not np.isnan(h) and h != 0)}")
    if bad:
        raise AssertionError(f"leakage_check mismatch: {bad[:10]}")
    print("ALL MATCH (brute force, 1e-6)")
    # label-permutation check on a 20k subset
    sub = df.iloc[:20000].reset_index(drop=True)
    a = mod.build(sub, fit_end_dt)
    sub2 = sub.copy()
    sub2["isFraud"] = np.random.default_rng(3).permutation(sub2["isFraud"].to_numpy())
    b = mod.build(sub2, fit_end_dt)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    print("label-permutation check: identical frames")
    yv = df["isFraud"].to_numpy()[vmask]
    print(f"SIGNAL valid PR-AUC per column alone (base rate {yv.mean():.4f}); score / negative score:")
    for c in feats.columns[1:]:
        print(_auc_line(c, feats[c].to_numpy(np.float64)[vmask], yv, average_precision_score))
    print(f"BLOCK {mod.NAME} OK")
