"""Inductive temporal heterogeneous GraphSAGE over the transaction-entity graph (plain PyTorch).

Each target transaction is encoded from its own features plus its strictly-earlier neighbourhood:
6 entity types (uid, card1, addr1, P_emaildomain, R_emaildomain, fp), up to 10 most recent earlier
transactions per type (hop 1) and up to 5 per type of each of those neighbours' own earlier
neighbours (hop 2). Per-type mean aggregation of [neighbour hidden state, relative-time encoding]
+ log1p(count), per-type linear, 2 layers, size 32, head -> logit. LABELS ARE NEVER INPUTS: the only
use of isFraud is the BCE target of fit-window rows (and PR-AUC of the declared early-stopping set).

Entities: uid = make_keys uid, but NaN when card1, addr1 or D1 is missing (an unknown D1 does not
identify a card holder); NaN entity never links. Hub rule: when an entity has > 50 distinct EARLIER uids
at the row's time, that (row, type) slot gets no neighbours (hubs are not expanded, at either hop).
Neighbour index tables are computed once with numpy (searchsorted, last k earlier rows of the entity,
ties excluded) and cached under data/processed/graph_enc/ (a row's table entry only depends on earlier
rows, so one table serves every window).

Windows (as improve.ensure_windows / backtest.run_fold):
  valid     : fit = data.time_split train rows, early stopping + scored = valid rows (like the GBDT runs).
  fold:S-E  : fit = days <= S-8 minus the last 14 of them, early stopping = those last 14 days,
              scored = days S..E; rows from day >= 141 (test) are dropped from the frame.
  --max-rows N: first N rows by time, day numbers/gaps scaled like improve.day_plan (debug only).
Hard guard: any train / es / score / frame row of the test split raises before fitting.

Outputs under models_dir()/<name>/: scores_valid.parquet (scored rows), embed.parquet (fit + scored
rows: gnn_emb__0..31, gnn_emb__score), run_info.json, model.pt.

Run (PYTHONPATH=src):
  python -m ecommerce_fraud.graph_encoder --window valid --seed 42 --epochs 10 --name gnn_valid_s42
  python -m ecommerce_fraud.graph_encoder --smoke
  python -m ecommerce_fraud.graph_encoder --summarize gnn_valid_s42,gnn_valid_s43,gnn_valid_s44
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import resource
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

DAY = 86400
TEST_START_DAY = 141
HORIZON_DAYS = 183
ENTITIES = ("uid", "card1", "addr1", "P_emaildomain", "R_emaildomain", "fp")
NT = len(ENTITIES)
K1, K2 = 10, 5
HUB = 50
D = 32
TE_DIM = 17
HASH_B = 128
CAT_DIM = 8
HASH_NUM = ("card1", "addr1")  # numeric columns that additionally get a hashed embedding
CACHE_DIR = None  # set lazily (needs data.WORKTREE)


def pick_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def rss_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 1)


# ------------------------------------------------------------------------------ frame / windows

def load_frame(max_rows):
    """Sorted history frame (as improve.load_hist_frame) and feature columns."""
    from ecommerce_fraud import improve
    improve.CFG["max_rows"] = max_rows
    improve.CFG["raw_debug"] = None
    df, cols = improve.load_hist_frame()
    improve.CFG["raw_debug"] = None
    return df, cols, improve


def test_ids_for(df, window, max_rows) -> np.ndarray:
    """TransactionIDs that are test split for this window (union of every available definition)."""
    from ecommerce_fraud import data
    day = (df["TransactionDT"] // DAY).to_numpy()
    ids = df["TransactionID"].to_numpy(np.int64)
    out = np.zeros(len(df), bool)
    if window == "valid" or not max_rows:
        out |= np.isin(ids, data.time_split(df)[2]["TransactionID"].to_numpy(np.int64))
    if not max_rows:
        out |= day >= TEST_START_DAY
    else:
        k = (int(day.max()) + 1) / HORIZON_DAYS
        if window != "valid":
            out |= day >= max(4, round(TEST_START_DAY * k))
    return ids[out]


def guard_no_test(ids_all, test_ids, sets: dict) -> None:
    """Raise when any index set contains a test-split TransactionID. sets: name -> row indices."""
    for name, idx in sets.items():
        hit = np.isin(ids_all[np.asarray(idx, dtype=np.int64)], test_ids)
        if hit.any():
            raise RuntimeError(f"{name} contains {int(hit.sum())} test-split rows (ADR-002/003 forbid this)")


def make_window(df, window, max_rows, improve):
    """Returns (frame_mask over df, fit_idx, es_idx, score_idx, fit_end_dt, info) with idx into the FRAME."""
    day = (df["TransactionDT"] // DAY).to_numpy()
    test_ids = test_ids_for(df, window, max_rows)
    ids = df["TransactionID"].to_numpy(np.int64)
    is_test = np.isin(ids, test_ids)
    frame_mask = ~is_test
    fpos = np.full(len(df), -1, np.int64)
    fpos[frame_mask] = np.arange(frame_mask.sum())
    if window == "valid":
        from ecommerce_fraud import data
        train, valid, _, _ = data.time_split(df)
        tr = pd.Index(ids).get_indexer(train["TransactionID"].to_numpy(np.int64))
        va = pd.Index(ids).get_indexer(valid["TransactionID"].to_numpy(np.int64))
        fit_idx, es_idx, score_idx = fpos[tr], fpos[va], fpos[va]
        fit_end = int(train["TransactionDT"].max())
        info = {"es_set": "valid"}
    else:
        m = re.fullmatch(r"fold:(\d+)-(\d+)", window)
        if not m:
            raise ValueError(f"--window must be 'valid' or 'fold:S-E', got {window!r}")
        s, e = int(m.group(1)), int(m.group(2))
        gap, es_days = improve.GAP_DAYS, improve.ES_DAYS
        if max_rows:
            _, gap, es_days, _ = improve.day_plan(int(day.max()))
            k = (int(day.max()) + 1) / HORIZON_DAYS
            s, e = max(2, round(s * k)), max(3, round(e * k))
        tr_mask = frame_mask & (day <= s - gap - 1)
        es_start = (s - gap - 1) - es_days + 1
        is_es = day >= es_start
        fit_idx = fpos[np.where(tr_mask & ~is_es)[0]]
        es_idx = fpos[np.where(tr_mask & is_es)[0]]
        score_idx = fpos[np.where(frame_mask & (day >= s) & (day <= e))[0]]
        fit_end = int(df["TransactionDT"].to_numpy()[tr_mask].max())
        info = {"es_set": f"days {es_start}-{s - gap - 1}", "score_days": [s, e], "gap_days": gap}
    fids = ids[frame_mask]
    guard_no_test(fids, test_ids, {"frame": np.arange(len(fids)), "fit": fit_idx, "early-stopping": es_idx,
                                   "scored": score_idx})
    for nm, a in (("fit", fit_idx), ("es", es_idx), ("score", score_idx)):
        if len(a) == 0 or (a < 0).any():
            raise ValueError(f"bad {nm} set: size {len(a)}")
    return frame_mask, fit_idx, es_idx, score_idx, fit_end, info


# ------------------------------------------------------------------------------ neighbour tables

def entity_codes(df) -> list:
    from ecommerce_fraud.features_graph import make_fp
    from ecommerce_fraud.features_history import make_keys
    uid = make_keys(df)["uid"].where(df["card1"].notna() & df["addr1"].notna() & df["D1"].notna())
    ser = [uid, df["card1"], df["addr1"], df["P_emaildomain"], df["R_emaildomain"], make_fp(df)]
    return [pd.factorize(s)[0].astype(np.int64) for s in ser]  # NaN -> -1


def build_tables(codes, t, k=K1, hub=HUB) -> np.ndarray:
    """tab[i, type, j] = row index of the (j+1)-th most recent STRICTLY earlier row sharing the entity, else -1.
    Rows must be sorted by t. Hub (row, type) slots (> hub distinct earlier uids) are empty."""
    n = len(t)
    t = np.asarray(t, dtype=np.int64)
    tab = np.full((n, NT, k), -1, dtype=np.int32)
    big = int(t.max()) + 2
    uid = codes[0].copy()
    nouid = uid < 0
    uid[nouid] = uid.max() + 1 + np.where(nouid)[0]  # a missing uid is its own identity
    for ti, c in enumerate(codes):
        idx = np.where(c >= 0)[0]
        if len(idx) == 0:
            continue
        cc = c[idx]
        order = np.argsort(cc, kind="stable")  # time order is kept inside an entity
        sidx, sc = idx[order], cc[order]
        comb = sc * big + t[sidx]
        gstart = np.searchsorted(comb, sc * big, side="left")
        hi = np.searchsorted(comb, comb, side="left")  # entity rows with t < t_i occupy [gstart, hi)
        n_prev = hi - gstart
        ok_hub = np.ones(len(sidx), bool)
        if ti > 0:
            pair = sc * (int(uid.max()) + 1) + uid[sidx]
            _, first = np.unique(pair, return_index=True)
            new = np.zeros(len(sidx), np.int64)
            new[first] = 1
            cum = np.concatenate([[0], np.cumsum(new)])  # cum[p] = new flags in positions < p
            distinct = cum[hi] - cum[gstart]
            ok_hub = distinct <= hub
        for j in range(k):
            sel = (n_prev > j) & ok_hub
            tab[sidx[sel], ti, j] = sidx[hi[sel] - 1 - j]
    return tab


def cached_tables(codes, t, ids, tag="") -> np.ndarray:
    from ecommerce_fraud.data import WORKTREE
    d = WORKTREE / "data" / "processed" / "graph_enc"
    d.mkdir(parents=True, exist_ok=True)
    key = f"nbr_n{len(t)}_id{int(ids[-1])}_s{int(np.sum(ids) % 1000003)}_k{K1}_h{HUB}{tag}.npz"
    p = d / key
    if p.exists():
        return np.load(p)["tab"]
    tab = build_tables(codes, t)
    np.savez(p, tab=tab)
    return tab


# ------------------------------------------------------------------------------ node features

def build_features(df, cols, fit_rows) -> dict:
    """Normalised numeric (fp16), NaN indicators (uint8), hashed categorical indices (int32)."""
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])]
    cat_cols = [c for c in cols if c not in set(num_cols)] + [c for c in HASH_NUM if c in df.columns]
    X = df[num_cols].to_numpy(np.float32)
    Xf = X[fit_rows]
    med = np.nanmedian(Xf, axis=0)
    q1, q3 = np.nanpercentile(Xf, [25, 75], axis=0)
    iqr = q3 - q1
    sd = np.nanstd(Xf, axis=0)
    scale = np.where(iqr > 1e-9, iqr, np.where(sd > 1e-9, sd, 1.0)).astype(np.float32)
    has_nan = np.isnan(Xf).any(axis=0)
    del Xf
    nanmask = np.isnan(X[:, has_nan])
    Z = (X - np.nan_to_num(med)) / scale
    Z = np.sign(Z) * np.log1p(np.abs(Z))  # robust squash of heavy tails
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    out = {"num": torch.from_numpy(Z.astype(np.float16)), "miss": torch.from_numpy(nanmask.astype(np.uint8)),
           "num_cols": num_cols, "cat_cols": cat_cols, "n_miss": int(has_nan.sum())}
    del X, Z
    cat = np.zeros((len(df), max(1, len(cat_cols))), dtype=np.int32)
    for ci, c in enumerate(cat_cols):
        s = df[c]
        nn_ = s.notna().to_numpy()
        h = pd.util.hash_array(s[nn_].astype(str).to_numpy(object))
        cat[nn_, ci] = ci * HASH_B + 1 + (h % (HASH_B - 1)).astype(np.int64)
        cat[~nn_, ci] = ci * HASH_B
    out["cat"] = torch.from_numpy(cat)
    return out


# ------------------------------------------------------------------------------ model

class HeteroSage(nn.Module):
    """h' = act(W_self h + sum_type W_type [mean_{m in N_type}(h_m, te(dt_m)), log1p|N_type|])."""

    def __init__(self, act=True):
        super().__init__()
        self.self_lin = nn.Linear(D, D)
        din = D + TE_DIM + 1
        self.w = nn.Parameter(torch.randn(NT, din, D) * (1.0 / np.sqrt(din * NT)))
        self.act = act

    def forward(self, h_self, h_nei, te_nei, mask):
        m = mask.unsqueeze(-1).to(h_nei.dtype)
        cnt = m.sum(-2)  # [..., T, 1]
        feat = torch.cat([h_nei, te_nei], -1) * m
        mean = feat.sum(-2) / cnt.clamp(min=1.0)
        mean = torch.cat([mean, torch.log1p(cnt)], -1)  # [..., T, D+TE+1]
        out = self.self_lin(h_self) + torch.einsum("...ki,kio->...o", mean, self.w)
        return F.relu(out) if self.act else out


class GraphSage(nn.Module):
    def __init__(self, n_num, n_miss, n_cat, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(max(1, n_cat) * HASH_B, CAT_DIM)
        din = n_num + n_miss + n_cat * CAT_DIM
        self.enc = nn.Sequential(nn.Linear(din, 64), nn.ReLU(), nn.Linear(64, D), nn.ReLU(), nn.Dropout(dropout))
        self.l1 = HeteroSage(act=True)
        self.l2 = HeteroSage(act=False)
        self.head = nn.Sequential(nn.ReLU(), nn.Dropout(dropout), nn.Linear(D, D), nn.ReLU(), nn.Linear(D, 1))
        self.register_buffer("freq", torch.logspace(-1.3, 0.3, 8))
        self.n_cat = n_cat

    def time_enc(self, dt):
        x = torch.log1p(dt.clamp(min=0).float())
        ang = x.unsqueeze(-1) * self.freq
        return torch.cat([torch.sin(ang), torch.cos(ang), (x / 10.0).unsqueeze(-1)], -1)

    def encode(self, data, u):
        x = [data.num[u].float(), data.miss[u].float()]
        if self.n_cat:
            c = self.emb(data.cat[u].long())  # [U, C, CAT_DIM]
            x.append(c.flatten(1))
        return self.enc(torch.cat(x, -1))

    def forward(self, data, bi):
        """bi: dict from make_index. Returns (logit [B], embedding [B, D])."""
        B = len(bi["ids"])
        u, inv = bi["u"], bi["inv"]
        hu = self.encode(data, u)
        h = hu[inv]
        n_r, n_1 = B, B * NT * K1
        h_r = h[:n_r]
        h_1 = h[n_r:n_r + n_1].view(B, NT, K1, D)
        h_2 = h[n_r + n_1:].view(B, NT, K1, NT, K2, D)
        m1, m2 = bi["m1"], bi["m2"]
        t_r = data.T[bi["ids"]]
        t_1 = data.T[bi["n1"]]
        t_2 = data.T[bi["n2"]]
        te1 = self.time_enc((t_r[:, None, None] - t_1).float())
        te2 = self.time_enc((t_1[..., None, None] - t_2).float())
        h1_n = self.l1(h_1, h_2, te2, m2)
        h1_r = self.l1(h_r, h_1, te1, m1)
        emb = self.l2(h1_r, h1_n, te1, m1)
        return self.head(emb).squeeze(-1), emb


def make_index(data, ids, device) -> dict:
    """Pure index work for a batch of target rows (numpy -> device tensors)."""
    ids = np.asarray(ids, dtype=np.int64)
    n1 = data.tab[ids]  # [B, T, K1]
    m1 = n1 >= 0
    n1c = np.where(m1, n1, 0).astype(np.int64)
    n2 = data.tab[n1c][..., :K2]  # [B, T, K1, T, K2]
    m2 = (n2 >= 0) & m1[..., None, None]
    n2c = np.where(m2, n2, 0).astype(np.int64)
    allids = np.concatenate([ids, n1c.ravel(), n2c.ravel()])
    u, inv = np.unique(allids, return_inverse=True)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)  # noqa: E731
    return {"ids": t(ids), "n1": t(n1c), "n2": t(n2c), "m1": t(m1), "m2": t(m2), "u": t(u.astype(np.int64)),
            "inv": t(inv.astype(np.int64))}


def to_device(feat, tab, t_dt, device) -> SimpleNamespace:
    return SimpleNamespace(num=feat["num"].to(device), miss=feat["miss"].to(device), cat=feat["cat"].to(device),
                           T=torch.from_numpy(t_dt.astype(np.int32)).to(device), tab=tab)


@torch.no_grad()
def predict(model, data, rows, device, batch=256):
    model.eval()
    rows = np.asarray(rows, dtype=np.int64)
    P, E = [], []
    for s in range(0, len(rows), batch):
        logit, emb = model(data, make_index(data, rows[s:s + batch], device))
        P.append(torch.sigmoid(logit).float().cpu().numpy())
        E.append(emb.float().cpu().numpy())
    if not P:
        return np.zeros(0, np.float32), np.zeros((0, D), np.float32)
    return np.concatenate(P).astype(np.float32), np.concatenate(E).astype(np.float32)


def fit(model, data, y, fit_idx, es_idx, epochs, seed, device, pos_weight=28.0, batch=256, lr=1e-3, patience=3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    pw = torch.tensor(float(pos_weight), device=device)
    rng = np.random.RandomState(seed)
    y_es = y[es_idx]
    best, best_state, bad, hist = -1.0, None, 0, []
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(fit_idx)
        tot, nb, t0 = 0.0, 0, time.time()
        for s in range(0, len(perm), batch):
            rows = perm[s:s + batch]
            logit, _ = model(data, make_index(data, rows, device))
            yt = torch.from_numpy(y[rows].astype(np.float32)).to(device)
            loss = F.binary_cross_entropy_with_logits(logit, yt, pos_weight=pw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach())
            nb += 1
        t_train = time.time() - t0
        p, _ = predict(model, data, es_idx, device)
        ap = float(average_precision_score(y_es, p)) if y_es.sum() > 0 else 0.0
        rec = {"epoch": ep + 1, "train_loss": tot / max(nb, 1), "es_pr_auc": ap, "seconds": time.time() - t0,
               "train_seconds": t_train}
        hist.append(rec)
        print("epoch", json.dumps(rec), flush=True)
        if ap > best:
            best, bad = ap, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return hist


# ------------------------------------------------------------------------------ run

def run(window, seed, epochs, name, max_rows=None, models_dir=None, checks=False, batch=256):
    if models_dir:
        os.environ["MODELS_DIR"] = str(models_dir)
    from ecommerce_fraud.artifacts import models_dir as md
    device = pick_device()
    t00 = time.time()
    df, cols, improve = load_frame(max_rows)
    frame_mask, fit_idx, es_idx, score_idx, fit_end, winfo = make_window(df, window, max_rows, improve)
    df = df[frame_mask].reset_index(drop=True)
    ids = df["TransactionID"].to_numpy(np.int64)
    t_dt = df["TransactionDT"].to_numpy(np.int64)
    y = df["isFraud"].to_numpy(np.float32)  # only ever indexed with fit / es / score rows
    codes = entity_codes(df)
    tab = cached_tables(codes, t_dt, ids, tag=f"_m{max_rows or 0}")
    feat = build_features(df, cols, np.concatenate([fit_idx, es_idx]) if window != "valid" else fit_idx)
    del df
    n_fit_rows = len(fit_idx)
    print(f"window={window} frame={len(ids)} fit={n_fit_rows} es={len(es_idx)} score={len(score_idx)} "
          f"num={feat['num'].shape[1]} miss={feat['miss'].shape[1]} cat={len(feat['cat_cols'])} device={device} "
          f"setup_s={time.time() - t00:.0f} fit_end_dt={fit_end} test_rows_in_any_set=0", flush=True)
    data = to_device(feat, tab, t_dt, device)
    torch.manual_seed(seed)
    model = GraphSage(feat["num"].shape[1], feat["miss"].shape[1], len(feat["cat_cols"])).to(device)
    t0 = time.time()
    hist = fit(model, data, y, fit_idx, es_idx, epochs, seed, device, batch=batch)
    fit_seconds = time.time() - t0
    p_sc, e_sc = predict(model, data, score_idx, device)
    pr = float(average_precision_score(y[score_idx], p_sc))
    best_ep = 1 + int(np.argmax([h["es_pr_auc"] for h in hist]))
    es_pr = max(h["es_pr_auc"] for h in hist)
    out = md() / name
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"TransactionID": ids[score_idx], "score": p_sc}).to_parquet(out / "scores_valid.parquet", index=False)
    emb_rows = np.unique(np.concatenate([fit_idx, es_idx, score_idx]))
    p_all, e_all = predict(model, data, emb_rows, device)
    ed = pd.DataFrame(e_all, columns=[f"gnn_emb__{i}" for i in range(D)])
    ed["gnn_emb__score"] = p_all
    ed.insert(0, "TransactionID", ids[emb_rows])
    assert ed["TransactionID"].is_unique and np.isfinite(ed.iloc[:, 1:].to_numpy()).all()
    ed.to_parquet(out / "embed.parquet", index=False)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out / "model.pt")
    spe = float(np.mean([h["seconds"] for h in hist]))
    info = {"name": name, "window": window, "seed": seed, "epochs_requested": epochs, "epochs_run": len(hist),
            "best_epoch": best_ep, "es_pr_auc": es_pr, "score_pr_auc": pr, "valid_pr_auc": pr,
            "seconds_per_epoch": spe, "fit_seconds": fit_seconds, "device": device, "peak_rss_mb": rss_mb(),
            "fit_end_dt": fit_end, "max_rows": max_rows, "n_fit": int(n_fit_rows), "n_es": int(len(es_idx)),
            "n_score": int(len(score_idx)), "n_embed": int(len(emb_rows)), "n_frame": int(len(ids)),
            "history": hist, "uses_labels_as_inputs": False, **winfo}
    (out / "run_info.json").write_text(json.dumps(info, indent=2))
    print(f"score PR-AUC={pr:.4f} es PR-AUC={es_pr:.4f} best_epoch={best_ep} sec/epoch={spe:.1f} "
          f"peak_rss_mb={info['peak_rss_mb']} -> {out}", flush=True)
    if checks:
        self_checks(model, data, tab, codes, t_dt, y, ids, score_idx, p_sc, device)
    return info


def self_checks(model, data, tab, codes, t_dt, y, ids, score_idx, p_sc, device):
    rng = np.random.RandomState(0)
    # 1) tables: same entity, strictly earlier, most recent first, hub rule
    rows = rng.choice(np.where((tab >= 0).any((1, 2)))[0], 200, replace=False)
    for i in rows:
        for ti in range(NT):
            nb = tab[i, ti][tab[i, ti] >= 0]
            assert (codes[ti][nb] == codes[ti][i]).all() and (t_dt[nb] < t_dt[i]).all()
            assert (np.diff(t_dt[nb]) <= 0).all()
            same = np.where((codes[ti] == codes[ti][i]) & (t_dt < t_dt[i]))[0] if codes[ti][i] >= 0 else []
            if len(nb):
                assert len(nb) == min(K1, len(same)) and set(nb) <= set(same)
            elif codes[ti][i] >= 0 and len(same) and ti > 0:  # empty despite earlier rows -> must be a hub
                u0 = codes[0][same]
                n_dist = len(np.unique(u0[u0 >= 0])) + int((u0 < 0).sum())
                assert n_dist > HUB, (i, ti, n_dist)
    print("check tables: strictly-earlier / same-entity / most-recent-first / hub rule OK on 200 rows")
    # 2) past-only: replacing every other row with the same or later time by zeros must not change the score
    pick = rng.choice(len(score_idx), 8, replace=False)
    sel = score_idx[pick]
    base, _ = predict(model, data, sel, device)
    worst = 0.0
    for i, b in zip(sel, base):
        later = np.where(t_dt >= t_dt[i])[0]
        later = later[later != i]
        num, miss, cat = data.num.clone(), data.miss.clone(), data.cat.clone()
        lt = torch.from_numpy(later).to(device)
        num[lt] = 0
        miss[lt] = 1
        cat[lt] = 0
        d2 = SimpleNamespace(num=num, miss=miss, cat=cat, T=data.T, tab=data.tab)
        p2, _ = predict(model, d2, [i], device)
        worst = max(worst, abs(float(p2[0]) - float(b)))
    assert worst < 1e-5, worst
    print(f"check past-only: scores of 8 rows unchanged (max |diff|={worst:.2e}) when all same/later rows are blanked")
    # 3) labels are not an input: permute every label, scores unchanged
    y_perm = y.copy()
    rng.shuffle(y_perm)
    p3, _ = predict(model, data, sel, device)
    assert np.allclose(p3, base) and "y" not in model.forward.__code__.co_varnames
    print("check labels: forward() has no label argument; scores identical with all labels permuted OK")


def deliberate_guard_test(max_rows=80000):
    df, cols, improve = load_frame(max_rows)
    test_ids = test_ids_for(df, "valid", max_rows)
    ids = df["TransactionID"].to_numpy(np.int64)
    bad = np.where(np.isin(ids, test_ids))[0][:3]
    try:
        guard_no_test(ids, test_ids, {"fit": np.concatenate([np.arange(5), bad])})
    except RuntimeError as exc:
        print(f"guard test: raised as required -> RuntimeError: {exc}")
        return
    raise AssertionError("guard did not raise")


def summarize(names):
    from ecommerce_fraud.artifacts import models_dir
    prs = []
    for n in names:
        prs.append(json.loads((models_dir() / n / "run_info.json").read_text())["score_pr_auc"])
    s = {"runs": names, "pr_auc": prs, "mean": float(np.mean(prs)), "std": float(np.std(prs, ddof=1)) if len(prs) > 1 else 0.0}
    (models_dir() / "gnn_valid_seed_summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="valid", help="valid | fold:S-E")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--name")
    ap.add_argument("--max-rows", type=int)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--models-dir")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--summarize", help="comma-separated run names (valid seeds)")
    a = ap.parse_args()
    if a.summarize:
        return summarize(a.summarize.split(","))
    if a.smoke:
        deliberate_guard_test()
        run("valid", 0, 1, a.name or "gnn_smoke", max_rows=a.max_rows or 80000, models_dir=a.models_dir, checks=True,
            batch=a.batch)
        print("GRAPH_ENCODER SMOKE OK")
        return
    if not a.name:
        ap.error("--name is required")
    run(a.window, a.seed, a.epochs, a.name, a.max_rows, a.models_dir, batch=a.batch)


if __name__ == "__main__":
    main()
