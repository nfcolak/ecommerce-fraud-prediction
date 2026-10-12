"""Arm lgbm_final_emb: the final LightGBM (A_hist_nanfix, spw 28) plus 16 PCA components of the
sequence-model embedding (SSL-pretrained model, models/seq_ssl_<window>_s<seed>/embed.npz).

Run (PYTHONPATH=src), one heavy LightGBM at a time:
  python -m ecommerce_fraud.model_emb --window valid --seed {0,1,2}
  python -m ecommerce_fraud.model_emb --window fold:S-E          (seed 0)
Features, parameters, scale_pos_weight and early stopping are those of final.py (valid window:
fit on train, early stopping on valid) and backtest.py (fold window: train days 1..S-8, early
stopping on the last 14 of them, scored days S..E). PCA is fitted on the fit-window rows only;
rows without an embedding get NaN. The test split is never scored or early-stopped on here.
LightGBM seed = 42 + seed (model_cw SEEDS 42, 43, 44), fold windows 42 (backtest).
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from ecommerce_fraud.artifacts import models_dir, save_artifacts
from ecommerce_fraud.backtest import ES_DAYS, GAP_DAYS
from ecommerce_fraud.baseline_lgbm import EARLY_STOP, PARAMS, git_sha, metrics, prepare
from ecommerce_fraud.data import WORKTREE
from ecommerce_fraud.final import MAX_ROUNDS, SEED, SPW, build, rss_mb, versions

ARM = "lgbm_final_emb"
N_PCA = 16
DAY = 86400
EMB_COLS = [f"emb_pca_{i:02d}" for i in range(N_PCA)]


def parse_window(window: str, seed: int):
    """Returns (kind, S, E, model name, seq_ssl model name, lightgbm seed)."""
    if window == "valid":
        if seed not in (0, 1, 2):
            raise ValueError("--seed must be 0, 1 or 2 for --window valid")
        return "valid", None, None, f"{ARM}_valid_s{seed}", f"seq_ssl_valid_s{seed}", SEED + seed
    m = re.fullmatch(r"fold:(\d+)-(\d+)", window)
    if not m:
        raise ValueError(f"--window must be 'valid' or 'fold:S-E', got {window!r}")
    if seed != 0:
        raise ValueError("fold windows use seed 0 only")
    s, e = int(m.group(1)), int(m.group(2))
    if s - GAP_DAYS - 1 < 1 or e < s:
        raise ValueError(f"bad fold {window}")
    return "fold", s, e, f"{ARM}_fold{s}_s0", f"seq_ssl_fold{s}_s0", SEED


def load_embed(path: Path):
    z = np.load(path)
    ids, emb = z["ids"].astype(np.int64), z["emb"]
    assert emb.ndim == 2 and len(ids) == len(emb), (ids.shape, emb.shape)
    assert len(np.unique(ids)) == len(ids), "duplicate ids in the embedding file"
    return ids, emb


def emb_frames(embed, fit_ids, frames, test_ids=None):
    """PCA(16) fitted on the embeddings of fit_ids only; one NaN-padded DataFrame per frame.

    test_ids is given for every window except the final test view: any embedded test row aborts.
    """
    ids, emb = embed
    if test_ids is not None and np.isin(ids, test_ids).any():
        raise RuntimeError("embedding file contains test-split rows (ADR-002/003 forbid this)")
    pos = pd.Series(np.arange(len(ids)), index=ids)

    def lookup(tid):
        return pos.reindex(np.asarray(tid)).to_numpy()

    p_fit = lookup(fit_ids)
    ok = ~np.isnan(p_fit)
    if ok.sum() < N_PCA:
        raise ValueError(f"only {int(ok.sum())} fit-window rows have an embedding")
    pca = PCA(n_components=N_PCA, svd_solver="full", random_state=0)
    pca.fit(emb[p_fit[ok].astype(np.int64)].astype(np.float64))
    out, cover = [], []
    for f in frames:
        p = lookup(f["TransactionID"].to_numpy())
        h = ~np.isnan(p)
        z = np.full((len(f), N_PCA), np.nan)
        z[h] = pca.transform(emb[p[h].astype(np.int64)].astype(np.float64))
        out.append(pd.DataFrame(z, columns=EMB_COLS, index=f.index))
        cover.append(float(h.mean()))
    info = {"fit_rows": int(len(fit_ids)), "fit_rows_with_embedding": int(ok.sum()),
            "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
            "coverage_per_frame": cover}
    return out, info


def fit(X_fit, y_fit, X_es, y_es, lgb_seed, max_rounds, log_every):
    params = {**PARAMS, "seed": lgb_seed, "scale_pos_weight": SPW}
    dtrain = lgb.Dataset(X_fit, y_fit)
    dval = lgb.Dataset(X_es, y_es, reference=dtrain)
    booster = lgb.train(params, dtrain, num_boost_round=max_rounds, valid_sets=[dval],
                        valid_names=["es"],
                        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                                   lgb.log_evaluation(log_every)])
    return booster, int(booster.best_iteration), params


def run(window, seed, embed_root, results_root, max_rounds):
    t0 = time.time()
    kind, s, e, name, seq_name, lgb_seed = parse_window(window, seed)
    embed_path = Path(embed_root) / seq_name / "embed.npz"
    embed = load_embed(embed_path)
    train, valid, test, cols, stats = build()
    test_ids = test["TransactionID"].to_numpy()
    del test
    gc.collect()
    all_cols = cols + EMB_COLS
    if kind == "valid":
        (emb_fit, emb_val), einfo = emb_frames(embed, train["TransactionID"].to_numpy(),
                                              [train, valid], test_ids)
        fit_df = pd.concat([train[cols], emb_fit], axis=1)
        sc_df = pd.concat([valid[cols], emb_val], axis=1)
        y_fit, y_sc = train["isFraud"].to_numpy(), valid["isFraud"].to_numpy()
        ids_sc = valid["TransactionID"]
        X, _, _ = prepare(fit_df, sc_df, sc_df.iloc[:0])
        booster, best, params = fit(X["train"], y_fit, X["valid"], y_sc, lgb_seed, max_rounds, 100)
        fit_info = {"early_stopping": "full valid (as final.py)", "n_fit": len(y_fit)}
    else:
        df = pd.concat([train, valid], ignore_index=True)
        del train, valid
        gc.collect()
        day = df["TransactionDT"] // DAY
        tr_mask = (day <= s - GAP_DAYS - 1).to_numpy()
        sc_mask = ((day >= s) & (day <= e)).to_numpy()
        tr_all, sc = df[tr_mask], df[sc_mask]
        d_tr = day[tr_mask].to_numpy()
        es_start = (s - GAP_DAYS - 1) - ES_DAYS + 1
        is_es = d_tr >= es_start
        (emb_tr, emb_sc), einfo = emb_frames(embed, tr_all["TransactionID"].to_numpy(),
                                            [tr_all, sc], test_ids)
        tr_x = pd.concat([tr_all[cols], emb_tr], axis=1)
        sc_x = pd.concat([sc[cols], emb_sc], axis=1)
        X, _, _ = prepare(tr_x, sc_x, sc_x.iloc[:0])
        y_tr, y_sc = tr_all["isFraud"].to_numpy(), sc["isFraud"].to_numpy()
        ids_sc = sc["TransactionID"]
        booster, best, params = fit(X["train"][~is_es], y_tr[~is_es], X["train"][is_es], y_tr[is_es],
                                    lgb_seed, max_rounds, 500)
        fit_info = {"early_stopping": f"last {ES_DAYS} train days (as backtest.py)",
                    "n_fit": int((~is_es).sum()), "n_es": int(is_es.sum()),
                    "train_days": [1, s - GAP_DAYS - 1], "score_days": [s, e]}
    s_sc = booster.predict(X["valid"], num_iteration=best)
    m = metrics(y_sc, s_sc)
    save_artifacts(name, booster, ids_sc, s_sc, None, None)
    out = Path(results_root) / name
    out.mkdir(parents=True, exist_ok=True)
    result = {"model": name, "window": window, "seed": seed, "lightgbm_seed": lgb_seed,
              "valid": m, "best_iteration": best, "n_features": len(all_cols),
              "n_base_features": len(cols), "embedding": {"source": str(embed_path), **einfo},
              "fit": fit_info, "params": {**params, "max_rounds": max_rounds,
                                          "early_stopping_rounds": EARLY_STOP},
              "git_sha": git_sha(), "versions": versions(),
              "runtime_seconds": round(time.time() - t0, 1), "peak_rss_mb": rss_mb()}
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def train_test_view(embed_root):
    """ADR-003 test view of lgbm_final_emb, used only by seq_select --test-view: fit on train,
    early stopping on valid, score test once (as final.py). Needs models/seq_ssl_test_s0/embed.npz
    that covers train, valid and test rows. Returns (test ids, test scores)."""
    train, valid, test, cols, _ = build()
    embed = load_embed(Path(embed_root) / "seq_ssl_test_s0" / "embed.npz")
    (e_tr, e_va, e_te), _ = emb_frames(embed, train["TransactionID"].to_numpy(), [train, valid, test])
    tr, va, te = (pd.concat([d[cols], z], axis=1) for d, z in ((train, e_tr), (valid, e_va), (test, e_te)))
    X, _, _ = prepare(tr, va, te)
    booster, best, _ = fit(X["train"], train["isFraud"].to_numpy(), X["valid"],
                           valid["isFraud"].to_numpy(), SEED, MAX_ROUNDS, 100)
    s_va = booster.predict(X["valid"], num_iteration=best)
    s_te = booster.predict(X["test"], num_iteration=best)  # the single test scoring
    save_artifacts(f"{ARM}_test_s0", booster, valid["TransactionID"], s_va, test["TransactionID"], s_te)
    return test["TransactionID"].to_numpy(), s_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", required=True, help="valid | fold:S-E")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embed-root", default=None, help="default: models dir (MODELS_DIR)")
    ap.add_argument("--results-root", default=str(WORKTREE / "results"))
    ap.add_argument("--max-rounds", type=int, default=MAX_ROUNDS, help="only lowered for plumbing checks")
    a = ap.parse_args()
    run(a.window, a.seed, a.embed_root or models_dir(), a.results_root, a.max_rounds)


if __name__ == "__main__":
    main()
