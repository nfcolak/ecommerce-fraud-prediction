"""Feature block fastrp_past: strictly past-only, label-free FastRP-style entity embedding (2 x 16 dims).

Entities (all high-cardinality, NaN never links):
  ID side  : uid (card1+addr1+D1n), card1, addr1|card1, P_emaildomain|card1
  DEV side : DeviceInfo, fp (DeviceInfo|id_30|id_31|id_33), id_30|id_31|id_33
Every entity gets a FIXED random N(0,1) vector (seeded by type; nothing is fitted). A transaction j has
an ID vector v_id(j) = mean of its ID-entity vectors and a DEV vector v_dev(j) likewise.
For row i with TransactionDT t_i, over rows j with t_j STRICTLY smaller (ties at the same
TransactionDT excluded; streaming in time order, query of a tie group before its update):
  a-dims (16): for each own ID entity e: decay-weighted mean of v_dev(j) over earlier rows j of e;
               mean over e, L2-normalised  ("which devices did my card/uid use before")
  b-dims (16): the same through own DEV entities e collecting v_id(j) ("which cards used this device before")
This is a 2-step FastRP (row -> shared entity -> earlier rows -> their entities of the OTHER side, so
the shared entity's own vector never enters). Decay weight of row j: exp(-(t_i - t_j) / TAU_DAYS).
Hub rule (causal, as in entity_graph): an entity with > HUB_CAP distinct uids among earlier rows is not
used as link and not used as vector source. NaN output where no earlier information exists.
No labels, no counts/ages (the output is a direction, L2-normalised), no absolute time.
"""
from __future__ import annotations

import argparse
import math
import resource
import time

import numpy as np
import pandas as pd

from ecommerce_fraud.entity_graph import HUB_CAP, entity_codes
from ecommerce_fraud.features_history import DAY

NAME = "fastrp_past"
DESCRIPTION = ("past-only 2-step FastRP of uid/card/addr+card/email+card <-> DeviceInfo/fingerprint entities: "
               "2x16 L2-normalised dims, random fixed entity vectors, exp decay 30d, hubs (>50 uids) removed")
USES_LABELS = False

DIM = 16
SEED = 20261017
TAU_DAYS = 30.0
ID_TYPES = ["uid", "card1", "addr_card", "pe_card"]
DEV_TYPES = ["dev", "fp", "osbr"]


def _combo(df: pd.DataFrame, cols: list) -> np.ndarray:
    ok = ~df[cols].isna().any(axis=1).to_numpy()
    s = df[cols[0]].astype(object).astype(str)
    for c in cols[1:]:
        s = s + "|" + df[c].astype(object).astype(str)
    out = np.full(len(df), -1, dtype=np.int64)
    out[ok] = pd.factorize(s.to_numpy()[ok])[0]
    return out


def entity_table(df: pd.DataFrame):
    """(codes {type: global int code or -1}, n_global, uid code per row, type order)."""
    base = entity_codes(df)
    codes = {
        "uid": base["uid"], "card1": base["card1"],
        "addr_card": _combo(df, ["addr1", "card1"]),
        "pe_card": _combo(df, ["P_emaildomain", "card1"]),
        "dev": pd.factorize(df["DeviceInfo"])[0].astype(np.int64),
        "fp": base["fp"],
        "osbr": _combo(df, ["id_30", "id_31", "id_33"]),
    }
    off = 0
    out = {}
    for t in ID_TYPES + DEV_TYPES:
        c = codes[t]
        out[t] = np.where(c >= 0, c + off, -1)
        off += int(c.max()) + 1 if len(c) and c.max() >= 0 else 0
    return out, off, base["uid"]


def random_vectors(codes: dict, n_global: int) -> np.ndarray:
    """Fixed N(0,1) vector per entity; one seeded stream per entity type (content-independent of rows)."""
    R = np.zeros((max(n_global, 1), DIM))
    for k, t in enumerate(ID_TYPES + DEV_TYPES):
        c = codes[t]
        v = c[c >= 0]
        if len(v):
            lo, hi = int(v.min()), int(v.max())
            R[lo:hi + 1] = np.random.default_rng([SEED, k]).standard_normal((hi - lo + 1, DIM))
    return R


def _unit(m: np.ndarray):
    n = float(np.sqrt((m * m).sum()))
    return m / n if n > 1e-12 else None


def build_arrays(df: pd.DataFrame, tau_days: float = TAU_DAYS) -> np.ndarray:
    t = df["TransactionDT"].to_numpy(np.int64)
    if len(t) > 1 and (np.diff(t) < 0).any():
        raise ValueError("df must be sorted by TransactionDT")
    n = len(df)
    codes, ng, uidc = entity_table(df)
    R = random_vectors(codes, ng)
    idc = np.stack([codes[k] for k in ID_TYPES], axis=1)
    dvc = np.stack([codes[k] for k in DEV_TYPES], axis=1)
    S = np.zeros((max(ng, 1), DIM))
    C = np.zeros(max(ng, 1))
    last = np.zeros(max(ng, 1), dtype=np.int64)
    hub = np.zeros(max(ng, 1), dtype=bool)
    uids: dict = {}
    out = np.full((n, 2 * DIM), np.nan)
    tau = tau_days * DAY
    bounds = np.flatnonzero(np.diff(t)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [n]])
    for s, e in zip(starts, ends):
        # 1) query every row of the tie group against the state of strictly earlier rows
        for i in range(s, e):
            for half, via in ((0, idc[i]), (DIM, dvc[i])):
                acc = None
                k = 0
                for g in via:
                    if g < 0 or hub[g] or C[g] <= 0:
                        continue
                    m = S[g] / C[g]
                    acc = m if acc is None else acc + m
                    k += 1
                if k:
                    u = _unit(acc / k)
                    if u is not None:
                        out[i, half:half + DIM] = u
        # 2) row vectors from the pre-group hub state
        upd = []
        for i in range(s, e):
            vi = [g for g in idc[i] if g >= 0 and not hub[g]]
            vd = [g for g in dvc[i] if g >= 0 and not hub[g]]
            vid = R[vi].mean(axis=0) if vi else None
            vdv = R[vd].mean(axis=0) if vd else None
            upd.append((i, vid, vdv))
        # 3) state update (via entities that are not hubs yet): ID entities collect DEV vector and vice versa
        for i, vid, vdv in upd:
            ti = t[i]
            for via, v in ((idc[i], vdv), (dvc[i], vid)):
                if v is None:
                    continue
                for g in via:
                    if g < 0 or hub[g]:
                        continue
                    f = math.exp(-(ti - last[g]) / tau) if C[g] > 0 else 0.0
                    S[g] = S[g] * f + v
                    C[g] = C[g] * f + 1.0
                    last[g] = ti
        # 4) distinct-uid sets and hub flags (after the group, so ties never see each other)
        for i in range(s, e):
            u = uidc[i]
            if u < 0:
                continue
            for g in list(idc[i]) + list(dvc[i]):
                if g < 0 or hub[g]:
                    continue
                st = uids.get(g)
                if st is None:
                    uids[g] = {u}
                else:
                    st.add(u)
                    if len(st) > HUB_CAP:
                        hub[g] = True
                        del uids[g]
    return out


def build(df: pd.DataFrame, fit_end_dt: int) -> pd.DataFrame:
    arr = build_arrays(df)
    cols = {"TransactionID": df["TransactionID"].to_numpy()}
    for h, p in ((0, "a"), (DIM, "b")):
        for d in range(DIM):
            cols[f"{NAME}__{p}{d}"] = arr[:, h + d]
    return pd.DataFrame(cols)


def brute(df: pd.DataFrame, feats: pd.DataFrame, i: int, tau_days: float = TAU_DAYS) -> list:
    """Plain recomputation for row i from the raw rows with t_j < t_i: [(col, got, hand)]."""
    t = df["TransactionDT"].to_numpy(np.int64)
    codes, ng, uidc = entity_table(df)
    R = random_vectors(codes, ng)
    prior = np.flatnonzero(t < t[i])

    allc = np.stack([codes[k] for k in ID_TYPES + DEV_TYPES], axis=1)

    def distinct_uids(g, before):  # entity g, rows with t < before
        m = (t < before) & (allc == g).any(axis=1) & (uidc >= 0)
        return len(set(uidc[m].tolist()))

    cache = {}

    def is_hub(g, before):
        key = (g, before)
        if key not in cache:
            cache[key] = distinct_uids(g, before) > HUB_CAP
        return cache[key]

    def rowvec(j, types):
        gs = [codes[k][j] for k in types if codes[k][j] >= 0 and not is_hub(codes[k][j], t[j])]
        return R[gs].mean(axis=0) if gs else None

    res = []
    for half, via_types, src_types in ((0, ID_TYPES, DEV_TYPES), (DIM, DEV_TYPES, ID_TYPES)):
        acc, k = None, 0
        for vt in via_types:
            g = codes[vt][i]
            if g < 0 or is_hub(g, t[i]):
                continue
            num, den = np.zeros(DIM), 0.0
            for j in prior:
                if codes[vt][j] != g:
                    continue
                v = rowvec(j, src_types)
                if v is None:
                    continue
                w = math.exp(-(t[i] - t[j]) / (tau_days * DAY))
                num += w * v
                den += w
            if den > 0:
                acc = num / den if acc is None else acc + num / den
                k += 1
        u = _unit(acc / k) if k else None
        for d in range(DIM):
            col = f"{NAME}__{'a' if half == 0 else 'b'}{d}"
            res.append((col, float(feats[col].iat[i]), float("nan") if u is None else float(u[d])))
    return res


def _drift(df: pd.DataFrame, feats: pd.DataFrame, fit_end_dt: int) -> None:
    """Adversarial AUC per new column (train rows only, first vs second half in time, 2-fold LightGBM)."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    tr = (df["TransactionDT"] <= fit_end_dt).to_numpy()
    X = feats.drop(columns="TransactionID")[tr].reset_index(drop=True)
    y = (np.arange(len(X)) >= len(X) // 2).astype(int)
    rng = np.random.default_rng(0)
    fold = rng.integers(0, 2, len(X))

    def auc(cols):
        sc = np.zeros(len(X))
        for f in (0, 1):
            m = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31, verbose=-1, n_jobs=4)
            m.fit(X.loc[fold != f, cols], y[fold != f])
            sc[fold == f] = m.predict_proba(X.loc[fold == f, cols])[:, 1]
        return roc_auc_score(y, sc)

    worst = 0.0
    for c in X.columns:
        a = auc([c])
        worst = max(worst, a)
        print(f"  drift AUC {c:26s} {a:.3f}{'  FLAG >0.70' if a > 0.70 else ''}")
    joint = auc(list(X.columns))
    print(f"drift AUC joint (all {X.shape[1]} cols) {joint:.3f}; worst single {worst:.3f}")
    # clock diagnostics: is the joint drift just the cold start (NaN share falls over time)?
    X["nan_a"] = X["fastrp_past__a0"].isna().astype(float)
    X["nan_b"] = X["fastrp_past__b0"].isna().astype(float)
    print(f"  NaN-indicator-only drift AUC (a,b) {auc(['nan_a', 'nan_b']):.3f}; NaN share first/second half "
          f"{X['nan_a'][:len(X)//2].mean():.3f}/{X['nan_a'][len(X)//2:].mean():.3f} (a), "
          f"{X['nan_b'][:len(X)//2].mean():.3f}/{X['nan_b'][len(X)//2:].mean():.3f} (b)")
    for nm, cs in (("a", [c for c in X.columns if c.startswith("fastrp_past__a")]),
                   ("b", [c for c in X.columns if c.startswith("fastrp_past__b")])):
        keep = X[cs[0]].notna().to_numpy()
        Xk, yk, fk = X.loc[keep, cs], y[keep], fold[keep]
        sc = np.zeros(len(Xk))
        for f in (0, 1):
            m = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31, verbose=-1, n_jobs=4)
            m.fit(Xk[fk != f], yk[fk != f])
            sc[fk == f] = m.predict_proba(Xk[fk == f])[:, 1]
        print(f"  joint drift AUC of the 16 {nm}-dims on non-NaN rows only: {roc_auc_score(yk, sc):.3f} (n={len(Xk)})")


def _main() -> None:
    from ecommerce_fraud.blocks import check_frame, sorted_full_frame

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--n-check", type=int, default=100)
    ap.add_argument("--drift", action="store_true")
    args = ap.parse_args()
    df = sorted_full_frame()
    if args.max_rows:
        df = df.iloc[: args.max_rows].reset_index(drop=True)
    fit_end_dt = int(df["TransactionDT"].iloc[int(len(df) * 0.75)])
    t0 = time.time()
    feats = build(df, fit_end_dt)
    secs = time.time() - t0
    check_frame(NAME, feats, df)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    print(f"shape {feats.shape} rows {len(df)} build {secs:.1f}s ({1e6 * secs / len(df):.1f} us/row) peak RSS {rss:.2f} GB")
    nanshare = feats.iloc[:, 1:].isna().mean()
    print(f"NaN share min/mean/max {nanshare.min():.3f}/{nanshare.mean():.3f}/{nanshare.max():.3f}")
    # brute force on a sample of rows: half random from the late part, half from rows with a tie at their DT
    rng = np.random.default_rng(11)
    t = df["TransactionDT"].to_numpy()
    tie = np.flatnonzero(np.r_[False, np.diff(t) == 0])
    picks = list(rng.choice(np.arange(len(df) // 4, len(df)), args.n_check - 20, replace=False))
    picks += list(rng.choice(tie, min(20, len(tie)), replace=False)) if len(tie) else []
    bad, informative = [], 0
    for i in picks:
        cmp = brute(df, feats, int(i))
        informative += any(not np.isnan(h) for _, _, h in cmp)
        for col, got, hand in cmp:
            if not ((np.isnan(got) and np.isnan(hand)) or abs(got - hand) <= 1e-9 * max(1.0, abs(hand))):
                bad.append((int(i), col, got, hand))
    print(f"brute-force: {len(picks)} rows ({informative} with non-NaN hand values) x {2 * DIM} cols")
    if bad:
        raise AssertionError(f"brute-force mismatch: {bad[:10]}")
    print("ALL MATCH (brute force, 1e-9)")
    sub = df.iloc[:20000].reset_index(drop=True)
    sub2 = sub.copy()
    sub2["isFraud"] = np.random.default_rng(3).permutation(sub2["isFraud"].to_numpy())
    pd.testing.assert_frame_equal(build(sub, fit_end_dt), build(sub2, fit_end_dt), check_exact=True)
    print("label-permutation check: identical frames")
    # prefix invariance: features of the first 30k rows must not change when later rows are removed/added
    pre = build(df.iloc[:30000].reset_index(drop=True), fit_end_dt)
    pd.testing.assert_frame_equal(pre, feats.iloc[:30000].reset_index(drop=True), check_exact=True)
    print("prefix-invariance check: first 30000 rows identical when later rows are absent")
    if args.drift:
        _drift(df, feats, fit_end_dt)
    print(f"BLOCK {NAME} OK")


if __name__ == "__main__":
    _main()
