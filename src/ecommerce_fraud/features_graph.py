"""Past-only graph / device-sharing features (no labels, no all-time counts).

Input must be sorted by TransactionDT. Every feature of a row uses only rows with a strictly
earlier TransactionDT (windowed distinct counts) or strictly earlier days (component size).

fp  = DeviceInfo|id_30|id_31|id_33 as one string, NaN when DeviceInfo is NaN (NaN never groups).
uid = features_history.make_keys(df)["uid"]. Under pandas 3 that key is NaN whenever card1, addr1
or D1 is NaN (astype(str) keeps NaN); such rows are not graph nodes and get NaN graph features.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ecommerce_fraud.features_history import DAY, make_keys

HUB_CAP = 50
WIN = {"7d": 7 * DAY, "30d": 30 * DAY}


def make_fp(df: pd.DataFrame) -> pd.Series:
    # pandas 3: astype(str) keeps NaN, so fill missing parts explicitly
    p = [df[c].astype(object).where(df[c].notna(), "nan").astype(str)
         for c in ("DeviceInfo", "id_30", "id_31", "id_33")]
    fp = p[0] + "|" + p[1] + "|" + p[2] + "|" + p[3]
    return fp.where(df["DeviceInfo"].notna())


def _codes(s: pd.Series) -> np.ndarray:
    return pd.factorize(s)[0].astype(np.int64)  # NaN -> -1


def _distinct_in_window(key, val, t, window) -> np.ndarray:
    """For each row: number of distinct `val` among rows with the same `key` and
    DT in [t - window, t). key/val are int codes, -1 = NaN (NaN key -> NaN result,
    NaN val is never counted). Rows with equal DT are queried before being inserted."""
    n = len(t)
    out = np.full(n, np.nan)
    state = {}  # key -> (deque[(t, val)], dict val -> count)
    i = 0
    while i < n:
        j = i
        while j < n and t[j] == t[i]:
            j += 1
        tt = int(t[i])
        lo = tt - window
        for r in range(i, j):  # queries first
            k = key[r]
            if k < 0:
                continue
            st = state.get(k)
            if st is None:
                out[r] = 0.0
                continue
            dq, cnt = st
            while dq and dq[0][0] < lo:
                _, v = dq.popleft()
                c = cnt[v] - 1
                if c:
                    cnt[v] = c
                else:
                    del cnt[v]
            out[r] = len(cnt)
        for r in range(i, j):  # then inserts
            k, v = key[r], val[r]
            if k < 0 or v < 0:
                continue
            st = state.get(k)
            if st is None:
                st = state[k] = (deque(), defaultdict(int))
            st[0].append((tt, v))
            st[1][v] += 1
        i = j
    return out


def _component_sizes(uid, fp, day) -> np.ndarray:
    """uid_fp_component_size: for a row on day d, the number of distinct uids in the connected
    component of its uid in the uid--fp graph built from edges of days < d, where fp nodes
    linking more than HUB_CAP distinct uids in those edges are removed. 1 if no usable edge."""
    n_uid, n_fp = int(uid.max()) + 1, int(fp.max()) + 1
    ok = (fp >= 0) & (uid >= 0)
    pairs = pd.DataFrame({"u": uid[ok], "f": fp[ok], "d": day[ok]})
    pairs = pairs.groupby(["u", "f"], sort=False)["d"].min().reset_index().sort_values("d")
    pu, pf, pd_ = pairs["u"].to_numpy(), pairs["f"].to_numpy(), pairs["d"].to_numpy()
    out = np.where(uid >= 0, 1.0, np.nan)  # NaN uid (a key part is NaN) never forms a node
    for d in np.unique(day):
        m = int(np.searchsorted(pd_, d, side="left"))  # edges with first day < d
        rows = np.flatnonzero(day == d)
        if m == 0:
            continue
        eu, ef = pu[:m], pf[:m]
        deg = np.bincount(ef, minlength=n_fp)
        keep = deg[ef] <= HUB_CAP
        eu, ef = eu[keep], ef[keep]
        if len(eu) == 0:
            continue
        g = coo_matrix((np.ones(len(eu), dtype=np.int8), (eu, n_uid + ef)),
                       shape=(n_uid + n_fp, n_uid + n_fp))
        _, lab = connected_components(g, directed=False)
        size = np.bincount(lab[:n_uid], minlength=int(lab.max()) + 1)
        rows = rows[uid[rows] >= 0]
        out[rows] = size[lab[uid[rows]]]
    return out


def add_graph_features(df: pd.DataFrame):
    """Return (df with new columns, list of added column names). fp is not added."""
    t0 = time.time()
    t = df["TransactionDT"].to_numpy(np.int64)
    if len(t) > 1 and (np.diff(t) < 0).any():
        raise ValueError("df must be sorted by TransactionDT")
    keys = make_keys(df)
    uid = _codes(keys["uid"])
    fp = _codes(make_fp(df))
    c_a = _codes(keys["card1_addr1"].where(df["addr1"].notna()))
    a1 = _codes(df["addr1"])
    new = {}
    for tag, w in WIN.items():
        new[f"fp_nuniq_uid_{tag}"] = _distinct_in_window(fp, uid, t, w)
        new[f"card1_addr1_nuniq_uid_{tag}"] = _distinct_in_window(c_a, uid, t, w)
        new[f"addr1_nuniq_uid_{tag}"] = _distinct_in_window(a1, uid, t, w)
    new["uid_nuniq_fp_30d"] = _distinct_in_window(uid, fp, t, WIN["30d"])
    new["uid_fp_component_size"] = _component_sizes(uid, fp, t // DAY)
    added = list(new.keys())
    out = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    print(f"graph features: {len(added)} columns built in {time.time() - t0:.1f}s", flush=True)
    return out, added


def graph_leakage_check(full: pd.DataFrame, rows, seed: int = 0, n: int = 3) -> str:
    """Recompute, for n random rows of `rows` (index labels of `full`), the 7-day fp distinct-uid
    count and the component size by hand (plain filter / BFS on earlier-day edges)."""
    keys = make_keys(full)
    uid_s, fp_s = keys["uid"], make_fp(full)
    day = full["TransactionDT"] // DAY
    rng = np.random.default_rng(seed)
    cand = [i for i in rows if full.at[i, "fp_nuniq_uid_7d"] > 0
            and full.at[i, "uid_fp_component_size"] > 1]
    picks = list(rng.choice(np.array(cand), size=n, replace=False))
    lines, all_ok = [], True
    for i in picks:
        t, d, u, f = full.at[i, "TransactionDT"], day[i], uid_s[i], fp_s[i]
        past = full[(fp_s == f) & (full["TransactionDT"] < t) & (full["TransactionDT"] >= t - 7 * DAY)]
        hand_cnt = float(uid_s[past.index].nunique())
        # BFS on edges of days < d with hubs (> HUB_CAP distinct uids) removed
        e = pd.DataFrame({"u": uid_s[day < d], "f": fp_s[day < d]}).dropna().drop_duplicates()
        hubs = e.groupby("f")["u"].nunique()
        e = e[~e["f"].isin(hubs[hubs > HUB_CAP].index)]
        u2f, f2u = defaultdict(set), defaultdict(set)
        for uu, ff in zip(e["u"], e["f"]):
            u2f[uu].add(ff)
            f2u[ff].add(uu)
        seen_u, seen_f, queue = {u}, set(), deque([("u", u)])
        while queue:
            kind, x = queue.popleft()
            if kind == "u":
                for ff in u2f.get(x, ()):
                    if ff not in seen_f:
                        seen_f.add(ff)
                        queue.append(("f", ff))
            else:
                for uu in f2u.get(x, ()):
                    if uu not in seen_u:
                        seen_u.add(uu)
                        queue.append(("u", uu))
        hand_comp = float(len(seen_u))
        lines.append(f"row index {i} TransactionID {full.at[i, 'TransactionID']} DT {t} day {d}: "
                     f"{len(past)} earlier same-fp rows in 7d, {len(e)} edges on days < {d}")
        for k, h in (("fp_nuniq_uid_7d", hand_cnt), ("uid_fp_component_size", hand_comp)):
            got = float(full.at[i, k])
            ok = bool(np.isclose(got, h, rtol=1e-9, atol=1e-9))
            all_ok = all_ok and ok
            lines.append(f"  {k:26s} feature={got:.6f} by_hand={h:.6f} {'OK' if ok else 'MISMATCH'}")
    lines.append("ALL MATCH" if all_ok else "MISMATCH FOUND")
    return "\n".join(lines)
