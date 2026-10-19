"""Feature block graph_hop2: 2-hop reach in the past-only transaction-entity graph.

Row i -> its entities (uid, card1, addr1, P_emaildomain, R_emaildomain, fp; NaN never links) that are
NOT hubs at t_i (hub = more than 50 distinct uids among strictly earlier rows of the entity) ->
earlier rows j (TransactionDT_j < t_i, ties excluded) of those entities -> the entities of j.
The neighbour set S_i is the union (deduplicated rows) over i's non-hub entities. Features:
  reach_uid / reach_fp / reach_card : distinct uid / fp / card1 values among S_i other than i's own
  cnt_7d, cnt_30d, amt_mean_7d, amt_mean_30d : |S_i| and mean TransactionAmt of S_i rows within 7d / 30d
  reach_uid_1d / _7d / _30d : growth of the 2-hop uid reach (distinct other uids within 1d / 7d / 30d)
  n_hub : how many of i's valid entities are hubs at t_i (excluded from expansion)
Counts are 0 (not NaN) when S_i is empty; means are NaN then.

APPROXIMATION (cost cap): each entity contributes at most its LAST 50 strictly-earlier rows to S_i.
Everything is still strictly past-only; counts for very busy entities (and the 30d counts through them)
are therefore lower bounds. No labels, no fitted statistic (`fit_end_dt` only documents the fit window).
Vectorised: pairs (i, j) are expanded with repeat/arange per row chunk (bounded memory), deduplicated
with np.unique, and reduced with bincount.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ecommerce_fraud.entity_graph import ENTITY_TYPES, HUB_CAP, EntityGraph, selfcheck_main
from ecommerce_fraud.features_history import DAY

NAME = "graph_hop2"
DESCRIPTION = ("2-hop past-only entity graph reach via non-hub entities (hub >50 uids): distinct uids/fps/cards "
               "reached, 7d/30d neighbour count + amount mean, 1d/7d/30d uid-reach growth; APPROX: each entity "
               "contributes its last 50 earlier rows only")
USES_LABELS = False

PAIR_CAP = 50
CHUNK = 4_000_000  # max (i, j) pairs expanded per chunk
WIN = {"1d": DAY, "7d": 7 * DAY, "30d": 30 * DAY}
REACH = {"uid": "reach_uid", "fp": "reach_fp", "card1": "reach_card"}


def _expand(g: EntityGraph, a: int, b: int, plan: dict) -> tuple:
    """Deduplicated (i, j) pairs for original rows [a, b)."""
    ii, jj = [], []
    for e, (rows_ok, lo_c, cnt) in plan.items():
        r = rows_ok[(rows_ok >= a) & (rows_ok < b)]
        if not len(r):
            continue
        c = cnt[r]
        tot = int(c.sum())
        starts = np.cumsum(c) - c
        off = np.arange(tot) - np.repeat(starts, c)
        pos = np.repeat(lo_c[r], c) + off
        ii.append(np.repeat(r, c))
        jj.append(g.ix[e].rows[pos])
    if not ii:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    key = np.unique(np.concatenate(ii) * g.n + np.concatenate(jj))
    return key // g.n, key % g.n


def _distinct(i, val, a, m, size):
    """#distinct (i, val) per row of [a, a+size); val >= 0."""
    if not len(i):
        return np.zeros(size)
    k = np.unique((i - a) * m + val)
    return np.bincount(k // m, minlength=size).astype(np.float64)


def build(df: pd.DataFrame, fit_end_dt: int) -> pd.DataFrame:
    g = EntityGraph(df)
    n = g.n
    plan, n_hub = {}, np.zeros(n)
    per_row = np.zeros(n, dtype=np.int64)
    for e in ENTITY_TYPES:
        ix = g.ix[e]
        valid, lo, hi = ix.lo_hi()
        hub = g.hub(e)
        n_hub += (valid & hub)
        ok = valid & ~hub & (hi > lo)
        lo_c = np.maximum(lo, hi - PAIR_CAP)
        cnt = np.where(ok, hi - lo_c, 0)
        plan[e] = (np.flatnonzero(ok), lo_c, cnt)
        per_row += cnt
    cum = np.cumsum(per_row)
    bounds = [0]
    while bounds[-1] < n:
        nxt = int(np.searchsorted(cum, (cum[bounds[-1] - 1] if bounds[-1] else 0) + CHUNK, side="right"))
        bounds.append(min(max(nxt, bounds[-1] + 1), n))
    names = (["reach_uid", "reach_fp", "reach_card", "cnt_7d", "cnt_30d", "amt_mean_7d", "amt_mean_30d",
              "reach_uid_1d", "reach_uid_7d", "reach_uid_30d"])
    res = {k: np.zeros(n) for k in names}
    res["amt_mean_7d"][:] = np.nan
    res["amt_mean_30d"][:] = np.nan
    m = {e: int(g.codes[e].max()) + 2 for e in REACH}
    for a, b in zip(bounds[:-1], bounds[1:]):
        size = b - a
        i, j = _expand(g, a, b, plan)
        if not len(i):
            continue
        dt = g.t[i] - g.t[j]
        assert (dt > 0).all()  # strictly earlier only
        for e, name in REACH.items():
            cj, ci = g.codes[e][j], g.codes[e][i]
            v = (cj >= 0) & (cj != ci)
            res[name][a:b] = _distinct(i[v], cj[v] + 1, a, m[e], size)
        for tag, w in (("7d", 7 * DAY), ("30d", 30 * DAY)):
            wm = dt <= w
            cnt = np.bincount(i[wm] - a, minlength=size).astype(np.float64)
            sm = np.bincount(i[wm] - a, weights=g.amt[j[wm]], minlength=size)
            res[f"cnt_{tag}"][a:b] = cnt
            with np.errstate(invalid="ignore", divide="ignore"):
                res[f"amt_mean_{tag}"][a:b] = np.where(cnt > 0, sm / cnt, np.nan)
        cj, ci = g.codes["uid"][j], g.codes["uid"][i]
        for tag, w in WIN.items():
            v = (cj >= 0) & (cj != ci) & (dt <= w)
            res[f"reach_uid_{tag}"][a:b] = _distinct(i[v], cj[v] + 1, a, m["uid"], size)
    out = {"TransactionID": df["TransactionID"].to_numpy(), f"{NAME}__n_hub": n_hub}
    for k in names:
        out[f"{NAME}__{k}"] = res[k]
    return pd.DataFrame(out)


def brute(df: pd.DataFrame, feats: pd.DataFrame, i: int) -> list:
    """Plain-Python recomputation for row i over strictly earlier rows (hub rule + 50-row cap applied)."""
    from ecommerce_fraud.entity_graph import entity_codes

    codes = entity_codes(df)
    t = df["TransactionDT"].tolist()
    amt = df["TransactionAmt"].tolist()
    uidc = codes["uid"]
    S, hubs = set(), 0
    for e in ENTITY_TYPES:
        c = codes[e]
        if c[i] < 0:
            continue
        prev = [j for j in range(len(df)) if c[j] == c[i] and t[j] < t[i]]
        if len({uidc[j] for j in prev if uidc[j] >= 0}) > HUB_CAP:
            hubs += 1
            continue
        S.update(prev[-PAIR_CAP:])  # df sorted by time: list order is time order
    nan = float("nan")
    hand = {"n_hub": float(hubs)}
    for e, name in REACH.items():
        hand[name] = float(len({codes[e][j] for j in S if codes[e][j] >= 0 and codes[e][j] != codes[e][i]}))
    for tag, w in (("7d", 7 * DAY), ("30d", 30 * DAY)):
        s = [j for j in S if t[i] - t[j] <= w]
        hand[f"cnt_{tag}"] = float(len(s))
        hand[f"amt_mean_{tag}"] = sum(amt[j] for j in s) / len(s) if s else nan
    for tag, w in WIN.items():
        hand[f"reach_uid_{tag}"] = float(len({uidc[j] for j in S
                                             if t[i] - t[j] <= w and uidc[j] >= 0 and uidc[j] != uidc[i]}))
    return [(f"{NAME}__{k}", float(feats[f"{NAME}__{k}"].iat[i]), float(v)) for k, v in hand.items()]


def _main() -> None:
    selfcheck_main(__import__(__name__, fromlist=["x"]), brute)


if __name__ == "__main__":
    _main()
