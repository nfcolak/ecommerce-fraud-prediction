"""Feature block graph_hop1: 1-hop entity statistics of the past-only transaction-entity graph.

For each entity type e in (uid, card1, addr1, P_emaildomain, R_emaildomain, fp) and row i, over the
rows of the same entity with TransactionDT STRICTLY earlier than t_i (ties excluded; exact, no cap):
degree, age (days since the entity's first earlier row), distinct linked uids (not for uid) and
distinct linked fps (not for fp), earlier-row TransactionAmt mean / population std, the row's amount
over that mean, and earlier-row counts in the last 1 and 7 days. A NaN entity gives NaN for all its
columns; an entity without earlier rows gives degree/counts 0 and NaN age/amount statistics.
No labels, no fitted statistic (`fit_end_dt` only documents the fit window). Hubs (> 50 distinct
uids before t_i) are not expanded at 1 hop either side, so their degree/uid-count are still given.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ecommerce_fraud.entity_graph import ENTITY_TYPES, EntityGraph, selfcheck_main
from ecommerce_fraud.features_history import DAY

NAME = "graph_hop1"
DESCRIPTION = ("1-hop past-only entity graph stats (uid/card1/addr1/P_email/R_email/fp): degree, age, "
               "(cumulative ones only for uid/card1/fp: no trend proxies) "
               "distinct uids/fps, earlier amount mean/std/ratio, 1d/7d counts; exact, strictly earlier rows")
USES_LABELS = False

FEATS = ["deg", "age_d", "nuid", "nfp", "amt_mean", "amt_std", "amt_ratio", "cnt_1d", "cnt_7d"]
# Low-cardinality entities: all-time cumulative counts (degree, age, distinct uids/fps) grow with
# calendar time and act as a time proxy under the time split (ADR-002 forbids trend features), so
# only amount statistics and windowed counts are emitted for them.
LOW_CARD = {"addr1", "P_emaildomain", "R_emaildomain"}
CUMULATIVE = {"deg", "age_d", "nuid", "nfp"}


def _keep(e: str, f: str) -> bool:
    return not (e in LOW_CARD and f in CUMULATIVE)


def _features(g: EntityGraph, e: str) -> dict:
    out = {"deg": g.degree(e), "age_d": g.age_days(e)}
    if e != "uid":
        out["nuid"] = g.distinct_before(e, g.codes["uid"])
    if e != "fp":
        out["nfp"] = g.distinct_before(e, g.codes["fp"])
    m, s = g.amount_stats(e)
    out["amt_mean"], out["amt_std"] = m, s
    with np.errstate(invalid="ignore", divide="ignore"):
        r = g.amt / m
    out["amt_ratio"] = np.where(np.isfinite(r), r, np.nan)
    out["cnt_1d"] = g.window_count(e, 1 * DAY)
    out["cnt_7d"] = g.window_count(e, 7 * DAY)
    return out


def build(df: pd.DataFrame, fit_end_dt: int) -> pd.DataFrame:
    g = EntityGraph(df)
    out = {"TransactionID": df["TransactionID"].to_numpy()}
    for e in ENTITY_TYPES:
        for f, v in _features(g, e).items():
            if not _keep(e, f):
                continue
            out[f"{NAME}__{e}_{f}"] = np.asarray(v, dtype=np.float64)
    return pd.DataFrame(out)


def brute(df: pd.DataFrame, feats: pd.DataFrame, i: int) -> list:
    """Plain-Python recomputation for row i over strictly earlier rows: [(col, got, hand)]."""
    from ecommerce_fraud.entity_graph import entity_codes

    codes = entity_codes(df)
    t = df["TransactionDT"].tolist()
    amt = df["TransactionAmt"].tolist()
    res = []
    nan = float("nan")
    for e in ENTITY_TYPES:
        c = codes[e]
        hand = {}
        if c[i] < 0:
            hand = {f: nan for f in FEATS if not (f == "nuid" and e == "uid") and not (f == "nfp" and e == "fp")}
        else:
            prev = [j for j in range(len(df)) if c[j] == c[i] and t[j] < t[i]]
            n = len(prev)
            a = [amt[j] for j in prev]
            mean = sum(a) / n if n else nan
            std = (sum((x - mean) ** 2 for x in a) / n) ** 0.5 if n else nan
            hand["deg"] = float(n)
            hand["age_d"] = (t[i] - min(t[j] for j in prev)) / DAY if n else nan
            if e != "uid":
                hand["nuid"] = float(len({codes["uid"][j] for j in prev if codes["uid"][j] >= 0}))
            if e != "fp":
                hand["nfp"] = float(len({codes["fp"][j] for j in prev if codes["fp"][j] >= 0}))
            hand["amt_mean"], hand["amt_std"] = mean, std
            hand["amt_ratio"] = amt[i] / mean if n and mean != 0 else nan
            hand["cnt_1d"] = float(sum(1 for j in prev if t[j] >= t[i] - DAY))
            hand["cnt_7d"] = float(sum(1 for j in prev if t[j] >= t[i] - 7 * DAY))
        for f, v in hand.items():
            if not _keep(e, f):
                continue
            res.append((f"{NAME}__{e}_{f}", float(feats[f"{NAME}__{e}_{f}"].iat[i]), float(v)))
    return res


def _main() -> None:
    selfcheck_main(__import__(__name__, fromlist=["x"]), brute)


if __name__ == "__main__":
    _main()
