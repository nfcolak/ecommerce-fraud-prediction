"""Improvement round: screen feature blocks, combine them, confirm the set, view test once.

One heavy run at a time; every stage is resumable (a result file that exists is not recomputed).
PYTHONPATH=src, results under results/improve/ (override with --out-dir):

  python -m ecommerce_fraud.improve --screen  [--only a,b]
      final (reference, same harness) and final + <block> for every discovered block, LightGBM exactly
      as final.py (PARAMS, spw 28, seed 42, MAX_ROUNDS 5000, early stopping on valid), valid window.
      -> screen/<name>.json, per-row valid scores models/imp_screen_<name>/
  python -m ecommerce_fraud.improve --combine
      greedy forward selection over blocks with screen delta > +0.002, stop below +0.002.
      -> combine.json
  python -m ecommerce_fraud.improve --confirm --blocks a,b,c [--members] [--stack]
      valid seeds 42/43/44 and the 3 forward folds (as backtest.run_fold; block frames built with each
      fold's fit_end_dt); --members adds XGBoost/CatBoost (members.py) and their rank blend, plus the
      early meta fold META_FOLD (days 39-59; never used in selection) for every member model.
      An optional 'gnn' member is picked up from models_dir() (gnn_valid_s<seed>, gnn_fold{0,1,2,E}_s42).
      --stack (implies --members) adds 'stack_blocks': per scored window an L2 logistic regression on
      rank-normalised member scores, fitted on the predictions of all folds that end before the window.
      ADR-003 rule vs final -> confirm.json (selected = candidate name or "none")
  python -m ecommerce_fraud.improve --test-view
      refuses unless confirm.json selected a candidate; trains it once as final.py, scores test once,
      paired day-block bootstrap vs final -> test_view.json.   (NOT run in the improvement phase)

Debug: --max-rows N (first N rows by time, folds/gap scaled) --max-rounds R --out-dir DIR.
Peak RSS values are process-wide maxima so far (ru_maxrss), not per run.
"""
from __future__ import annotations

import argparse
import gc
import json
import resource
import time
from pathlib import Path
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd

from ecommerce_fraud import blocks as blk
from ecommerce_fraud import decision as dc
from ecommerce_fraud.artifacts import models_dir, save_artifacts
from ecommerce_fraud.baseline_lgbm import DROP, EARLY_STOP, PARAMS, git_sha, metrics, prepare
from ecommerce_fraud.data import WORKTREE, load_joined, time_split
from ecommerce_fraud.features_history import add_history_features

OUT_DIR = WORKTREE / "results" / "improve"
SEED = 42
SEEDS = (42, 43, 44)
SPW = 28.0
MAX_ROUNDS = 5000
MIN_GAIN = 0.002          # screen / combine threshold (valid PR-AUC)
MARGIN_SD = 2.0           # ADR-003: valid delta > 2 x max(sd cand, sd final)
MIN_WINS = 2
DAY = 86400
FOLDS = [(63, 83), (87, 107), (111, 131)]
META_FOLD = (39, 59)      # window "foldE": meta-learner training data only, never selection
GAP_DAYS = 7
ES_DAYS = 14
TEST_START_DAY = 141
HORIZON_DAYS = 183
CANDIDATES = ("lgbm_blocks", "xgb_blocks", "cat_blocks", "blend_blocks", "stack_blocks")
STACK_C = 1.0

CFG = {"max_rows": None, "max_rounds": MAX_ROUNDS, "out_dir": OUT_DIR, "raw_debug": None}
T0 = time.time()


# ------------------------------------------------------------------ small helpers
def rss_mb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 1)


def log(*a):
    print(f"[{time.time() - T0:7.0f}s rss {rss_mb():.0f}MB]", *a, flush=True)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def jwrite(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), indent=2) + "\n")


def jread(path: Path):
    return json.loads(path.read_text())


def out_dir() -> Path:
    return Path(CFG["out_dir"])


def slug(names) -> str:
    return "+".join(sorted(names)) or "none"


def parse_names(text) -> list:
    names = [n for n in (text or "").split(",") if n]
    known = blk.discover()
    bad = [n for n in names if n not in known]
    if bad:
        raise SystemExit(f"unknown block(s) {bad}; discovered: {sorted(known)}")
    return names


# ------------------------------------------------------------------ data
def day_plan(max_day: int):
    """(folds, meta_fold, gap_days, es_days, test_start_day); real values unless --max-rows scales them."""
    if CFG["max_rows"] is None:
        return FOLDS, META_FOLD, GAP_DAYS, ES_DAYS, TEST_START_DAY
    k = (max_day + 1) / HORIZON_DAYS
    gap, es_days = max(1, round(GAP_DAYS * k)), max(1, round(ES_DAYS * k))
    folds = [(max(2, round(s * k)), max(3, round(e * k))) for s, e in FOLDS]
    for i in range(1, len(folds)):  # rounding must not make scored windows touch (stack date rule)
        s, e = folds[i]
        s = max(s, folds[i - 1][1] + 1)
        folds[i] = (s, max(e, s + 1))
    # meta fold: room for a non-empty fit part (gap + early-stopping days + 1 fit day) and ends before fold0
    ms = max(round(META_FOLD[0] * k), gap + es_days + 2)
    meta = (ms, max(ms + 1, min(round(META_FOLD[1] * k), folds[0][0] - 1)))
    return (folds, meta, gap, es_days, max(4, round(TEST_START_DAY * k)))


def load_hist_frame():
    """Sorted frame with the A_hist_nanfix history features (as final.build) and its feature list."""
    raw = load_joined().sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    if CFG["max_rows"]:
        raw = raw.iloc[: CFG["max_rows"]].reset_index(drop=True)
        CFG["raw_debug"] = raw
    base_cols = [c for c in raw.columns if c not in DROP]
    df, hist_added = add_history_features(raw, nan_keys="nan")
    if CFG["raw_debug"] is None:
        del raw
    cols = base_cols + [c for c in hist_added if c != "day"]
    assert not {"day", "TransactionDT", "isFraud", "TransactionID"} & set(cols)
    return df, cols


def block_input():
    """df argument for blk.load: None (full cached data) or the debug subset."""
    return CFG["raw_debug"]


def make_base(with_test=False):
    """Prepared train/valid(/test) matrices of the final feature set plus small label/id arrays."""
    df, cols = load_hist_frame()
    train, valid, test, stats = time_split(df)
    del df
    gc.collect()
    empty = valid[cols].iloc[:0]
    X, _, _ = prepare(train[cols], valid[cols], test[cols] if with_test else empty)
    b = SimpleNamespace(
        X=X, cols=cols, stats=stats, fit_end_dt=int(train["TransactionDT"].max()),
        y_tr=train["isFraud"].to_numpy(), y_va=valid["isFraud"].to_numpy(),
        ids_tr=train["TransactionID"].to_numpy(), ids_va=valid["TransactionID"].to_numpy(),
        y_te=test["isFraud"].to_numpy() if with_test else None,
        ids_te=test["TransactionID"].to_numpy() if with_test else None,
        dt_te=test["TransactionDT"].to_numpy() if with_test else None)
    del train, valid, test
    gc.collect()
    return b


def block_columns(names, fit_end_dt, ids, index):
    """Block feature frames aligned to `ids` (row order of the model matrix), index = `index`."""
    frames, secs = [], {}
    for n in names:
        t = time.time()
        bf = blk.load(n, fit_end_dt, df=block_input())
        if not np.isin(ids, bf["TransactionID"].to_numpy()).all():
            raise ValueError(f"block {n}: TransactionIDs of the matrix are not all in the block frame")
        aligned = bf.set_index("TransactionID").reindex(ids)
        aligned.index = index
        non_num = [c for c in aligned.columns if not pd.api.types.is_numeric_dtype(aligned[c])]
        if non_num:
            raise ValueError(f"block {n}: non-numeric columns {non_num[:5]}")
        frames.append(aligned)
        secs[n] = {"load_seconds": round(time.time() - t, 1),
                   "build_seconds": blk.build_seconds(n, fit_end_dt)}
        del bf
    return frames, secs


def with_blocks(X_part, names, fit_end_dt, ids):
    if not names:
        return X_part, {}
    frames, secs = block_columns(names, fit_end_dt, ids, X_part.index)
    out = pd.concat([X_part] + frames, axis=1)
    dup = out.columns[out.columns.duplicated()].tolist()
    if dup:
        raise ValueError(f"duplicated feature names after adding blocks: {dup[:5]}")
    return out, secs


# ------------------------------------------------------------------ models
def fit_lgbm(Xf, yf, Xe, ye, seed, log_every=100, verbose=True):
    """LightGBM exactly as final.py: PARAMS + seed + spw 28, early stopping on the (Xe, ye) set."""
    params = {**PARAMS, "seed": seed, "scale_pos_weight": SPW}
    dtrain = lgb.Dataset(Xf, yf)
    dvalid = lgb.Dataset(Xe, ye, reference=dtrain)
    booster = lgb.train(params, dtrain, num_boost_round=CFG["max_rounds"], valid_sets=[dvalid],
                        valid_names=["valid"],
                        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=verbose),
                                   lgb.log_evaluation(log_every)])
    best = int(booster.best_iteration)
    return booster, best, (lambda X: booster.predict(X, num_iteration=best))


def fit_member(model, Xf, yf, Xe, ye, seed):
    from ecommerce_fraud import members  # lazy: written by unit m12
    fn = members.fit_xgb if model == "xgb" else members.fit_cat
    return fn(Xf, yf, Xe, ye, seed)


def blend(score_dict):
    from ecommerce_fraud import members
    return np.asarray(members.rank_blend(score_dict))


def read_scores(art_name, which="valid") -> pd.DataFrame:
    return pd.read_parquet(models_dir() / art_name / f"scores_{which}.parquet")


def labels_for(ids) -> np.ndarray:
    load_joined()  # makes sure the parquet cache exists
    from ecommerce_fraud.data import CACHE_PATH
    lab = pd.read_parquet(CACHE_PATH, columns=["TransactionID", "isFraud"]).set_index("TransactionID")["isFraud"]
    return lab.reindex(ids).to_numpy()


def days_for(ids) -> np.ndarray:
    """Day index (TransactionDT // 86400) of each id; only used for the time-order assertion."""
    load_joined()
    from ecommerce_fraud.data import CACHE_PATH
    dt = pd.read_parquet(CACHE_PATH, columns=["TransactionID", "TransactionDT"]).set_index("TransactionID")["TransactionDT"]
    return (dt.reindex(ids).to_numpy() // DAY).astype(int)


# ------------------------------------------------------------------ screen
def valid_run(b, names, seed, art_name=None, save_booster=False):
    """final + blocks on the valid window (train fit, valid early stopping). Returns (result, scores)."""
    t = time.time()
    Xtr, secs = with_blocks(b.X["train"], names, b.fit_end_dt, b.ids_tr)
    Xva, _ = with_blocks(b.X["valid"], names, b.fit_end_dt, b.ids_va)
    load_s = round(time.time() - t, 1)
    n_block = Xtr.shape[1] - len(b.cols)
    t1 = time.time()
    booster, best, predict = fit_lgbm(Xtr, b.y_tr, Xva, b.y_va, seed)
    s_va = predict(Xva)
    fit_s = round(time.time() - t1, 1)
    res = {"seed": seed, "valid": metrics(b.y_va, s_va), "best_iteration": best,
           "n_features": int(Xtr.shape[1]), "n_block_features": int(n_block),
           "fit_seconds": fit_s, "block_load_seconds": load_s, "blocks": secs,
           "hit_max_rounds": bool(best >= CFG["max_rounds"] - 1), "peak_rss_mb": rss_mb()}
    if art_name:
        save_artifacts(art_name, booster if save_booster else None, b.ids_va, s_va, None, None)
    del Xtr, Xva, booster
    gc.collect()
    return res, s_va


def cmd_screen(only):
    d = out_dir() / "screen"
    mods = blk.discover()
    names = [n for n in sorted(mods) if not only or n in only]
    todo = []
    for n in ["final"] + names:
        p = d / f"{n}.json"
        if not (p.exists() and "error" not in jread(p)):
            todo.append(n)
    if not todo:
        log("screen: all results exist")
        return
    b = make_base()
    log("base ready:", b.stats["train"]["n_rows"], "train rows,", len(b.cols), "features, fit_end_dt", b.fit_end_dt)
    for n in todo:
        names_n = [] if n == "final" else [n]
        try:
            res, _ = valid_run(b, names_n, SEED, art_name=f"imp_screen_{n}")
            err = None
        except Exception as exc:  # one broken block must not stop the screen
            res, err = {}, f"{type(exc).__name__}: {exc}"
            log(f"screen {n}: ERROR {err}")
        out = {"name": n, "description": getattr(mods.get(n), "DESCRIPTION", "reference"),
               "uses_labels": getattr(mods.get(n), "USES_LABELS", None),
               "fit_end_dt": b.fit_end_dt, "git_sha": git_sha(), "max_rounds": CFG["max_rounds"],
               "max_rows": CFG["max_rows"], **res}
        if err:
            out["error"] = err
        else:
            out["valid_pr_auc"] = res["valid"]["pr_auc"]
            ref_p = d / "final.json"
            if n != "final" and ref_p.exists():
                ref = jread(ref_p)["valid"]["pr_auc"]
                out["reference_pr_auc"] = ref
                out["delta_vs_final"] = res["valid"]["pr_auc"] - ref
            out["block_build_seconds"] = {k: v["build_seconds"] for k, v in res["blocks"].items()}
            log(f"screen {n}: valid PR-AUC {res['valid']['pr_auc']:.5f} best_it {res['best_iteration']} "
                f"delta {out.get('delta_vs_final', float('nan')):+.5f}")
        jwrite(d / f"{n}.json", out)
        gc.collect()


# ------------------------------------------------------------------ combine
def cmd_combine():
    d = out_dir()
    ref_p = d / "screen" / "final.json"
    if not ref_p.exists():
        raise SystemExit("combine needs screen/final.json: run --screen first")
    ref = jread(ref_p)["valid"]["pr_auc"]
    shots = {}
    for p in sorted((d / "screen").glob("*.json")):
        j = jread(p)
        if p.stem != "final" and "error" not in j and j.get("delta_vs_final") is not None:
            shots[p.stem] = j["delta_vs_final"]
    eligible = sorted([n for n, v in shots.items() if v > MIN_GAIN], key=lambda n: -shots[n])
    cpath = d / "combine.json"
    trials = jread(cpath).get("trials", {}) if cpath.exists() else {}
    out = {"reference_valid_pr_auc": ref, "min_gain": MIN_GAIN, "seed": SEED, "screen_delta": shots,
           "eligible": eligible, "trials": trials, "steps": [], "selected_blocks": [],
           "valid_pr_auc": ref, "complete": False}
    if not eligible:
        out["complete"] = True
        jwrite(cpath, out)
        log("combine: no block passes the screen threshold; selected = []")
        return
    b = None
    chosen, cur = [], ref
    remaining = list(eligible)
    while remaining:
        scored = {}
        for c in remaining:
            key = slug(chosen + [c])
            if not chosen:  # a single block: the screen run is the trial
                scored[c] = ref + shots[c]
                trials.setdefault(key, {"blocks": [c], "valid_pr_auc": scored[c], "source": "screen"})
                continue
            if key not in trials:
                if b is None:
                    b = make_base()
                res, _ = valid_run(b, chosen + [c], SEED, art_name=f"imp_combine_{key}")
                trials[key] = {"blocks": sorted(chosen + [c]), "valid_pr_auc": res["valid"]["pr_auc"],
                               "best_iteration": res["best_iteration"], "n_features": res["n_features"],
                               "fit_seconds": res["fit_seconds"], "source": "combine"}
                jwrite(cpath, out)
                log(f"combine trial {key}: {res['valid']['pr_auc']:.5f}")
            scored[c] = trials[key]["valid_pr_auc"]
        best = max(scored, key=scored.get)
        gain = scored[best] - cur
        out["steps"].append({"added": best, "valid_pr_auc": scored[best], "gain": gain,
                             "candidates": scored, "accepted": bool(gain > MIN_GAIN)})
        if gain <= MIN_GAIN:
            break
        chosen.append(best)
        remaining.remove(best)
        cur = scored[best]
    out.update(selected_blocks=chosen, valid_pr_auc=cur, complete=True)
    jwrite(cpath, out)
    log("combine selected", chosen, f"valid PR-AUC {cur:.5f} (reference {ref:.5f})")


# ------------------------------------------------------------------ confirm
def task_paths(cdir, window, model, seed):
    return cdir / f"{window}_{model}_s{seed}.json", f"imp_confirm_{cdir.name}_{window}_{model}_s{seed}"


def task_done(cdir, window, model, seed):
    p, art = task_paths(cdir, window, model, seed)
    return p.exists() and (models_dir() / art / "scores_valid.parquet").exists()


def run_task(cdir, window, model, seed, Xf, yf, Xe, ye, Xev, yev, ids_ev, extra=None):
    p, art = task_paths(cdir, window, model, seed)
    t1 = time.time()
    if model == "lgbm":
        booster, best, predict = fit_lgbm(Xf, yf, Xe, ye, seed, log_every=500, verbose=False)
        s = predict(Xev)
        save_artifacts(art, booster, ids_ev, s, None, None)
        del booster
    else:
        mdl, best, predict = fit_member(model, Xf, yf, Xe, ye, seed)
        s = np.asarray(predict(Xev))
        save_artifacts(art, None, ids_ev, s, None, None)
        del mdl
    m = metrics(yev, s)
    res = {"window": window, "model": model, "seed": seed, "metrics": m, "best_iteration": int(best),
           "n_features": int(Xf.shape[1]), "fit_seconds": round(time.time() - t1, 1),
           "peak_rss_mb": rss_mb(), **(extra or {})}
    jwrite(p, res)
    log(f"{cdir.name} {window} {model} s{seed}: PR-AUC {m['pr_auc']:.5f} best_it {best} ({res['fit_seconds']}s)")
    gc.collect()
    return res


def ensure_windows(names, models, seeds=SEEDS, fold_seed=SEED, with_meta=False):
    """Run (resumably) valid seeds and the 3 folds (+ early meta fold "foldE" if with_meta) for
    final + blocks `names`; returns the task results."""
    cdir = out_dir() / "confirm" / slug(names)
    valid_tasks = [(m, s) for m in models for s in seeds]
    if any(not task_done(cdir, "valid", m, s) for m, s in valid_tasks):
        b = make_base()
        Xtr, secs = with_blocks(b.X["train"], names, b.fit_end_dt, b.ids_tr)
        Xva, _ = with_blocks(b.X["valid"], names, b.fit_end_dt, b.ids_va)
        for m, s in valid_tasks:
            if not task_done(cdir, "valid", m, s):
                run_task(cdir, "valid", m, s, Xtr, b.y_tr, Xva, b.y_va, Xva, b.y_va, b.ids_va,
                         {"blocks": secs, "fit_end_dt": b.fit_end_dt})
        del b, Xtr, Xva
        gc.collect()

    df = None
    wins = [(f"fold{i}", i) for i in range(len(FOLDS))] + ([("foldE", "E")] if with_meta else [])
    for wname, wi in wins:
        pending = [m for m in models if not task_done(cdir, wname, m, fold_seed)]
        if not pending:
            continue
        if df is None:  # same construction as backtest.main: history on all rows, test days dropped
            df, cols = load_hist_frame()
            day = df["TransactionDT"] // DAY
            folds, meta_fold, gap, es_days, test_start = day_plan(int(day.max()))
            df = df[(day < test_start).to_numpy()].reset_index(drop=True)
            day = df["TransactionDT"] // DAY
            gc.collect()
        s, e = meta_fold if wi == "E" else folds[wi]
        # the lines below mirror backtest.run_fold
        tr_mask = (day <= s - gap - 1).to_numpy()
        sc_mask = ((day >= s) & (day <= e)).to_numpy()
        tr_all, sc = df[tr_mask], df[sc_mask]
        d_tr = day[tr_mask].to_numpy()
        es_start = (s - gap - 1) - es_days + 1
        is_es = d_tr >= es_start
        X, _, _ = prepare(tr_all[cols], sc[cols], sc[cols].iloc[:0])
        y_tr, y_sc = tr_all["isFraud"].to_numpy(), sc["isFraud"].to_numpy()
        fit_end = int(tr_all["TransactionDT"].max())
        ids_tr, ids_sc = tr_all["TransactionID"].to_numpy(), sc["TransactionID"].to_numpy()
        del tr_all, sc
        Xt, secs = with_blocks(X["train"], names, fit_end, ids_tr)
        Xs, _ = with_blocks(X["valid"], names, fit_end, ids_sc)
        del X
        extra = {"score_start_day": s, "score_end_day": e, "train_days": [1, s - gap - 1],
                 "es_days": [es_start, s - gap - 1], "fit_end_dt": fit_end, "blocks": secs,
                 "n_fit": int((~is_es).sum()), "n_es": int(is_es.sum()), "n_score": int(len(y_sc)),
                 "n_fraud_score": int(y_sc.sum())}
        for m in pending:
            run_task(cdir, wname, m, fold_seed, Xt[~is_es], y_tr[~is_es], Xt[is_es], y_tr[is_es],
                     Xs, y_sc, ids_sc, extra)
        del Xt, Xs
        gc.collect()
    del df
    gc.collect()

    out = {"valid": {}, "folds": {i: {} for i in range(len(FOLDS))}, "dir": cdir}
    if with_meta:
        out["meta"] = {m: jread(task_paths(cdir, "foldE", m, fold_seed)[0]) for m in models}
    for m, s in valid_tasks:
        out["valid"][(m, s)] = jread(task_paths(cdir, "valid", m, s)[0])
    for i in range(len(FOLDS)):
        for m in models:
            out["folds"][i][m] = jread(task_paths(cdir, f"fold{i}", m, fold_seed)[0])
    return out


def seed_stats(values):
    v = np.asarray(values, dtype=float)
    return {"per_seed": v.tolist(), "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else float("nan")}


def reference_final(real):
    """final's valid seeds and fold PR-AUCs: committed files (full run) or recomputed (debug)."""
    if real:
        cw = jread(WORKTREE / "results" / "lgbm_cw" / "metrics.json")["selection"]["candidates"]["spw28"]
        bt = jread(WORKTREE / "results" / "backtest" / "metrics.json")["folds"]
        return ({"per_seed": cw["valid_pr_auc_per_seed"], "mean": cw["valid_pr_auc_mean"],
                 "sd": cw["valid_pr_auc_sd"], "source": "results/lgbm_cw spw28"},
                [f["final"]["pr_auc"] for f in sorted(bt, key=lambda f: f["score_start_day"])],
                "results/backtest/metrics.json")
    r = ensure_windows([], ["lgbm"])
    v = seed_stats([r["valid"][("lgbm", s)]["metrics"]["pr_auc"] for s in SEEDS])
    return ({**v, "source": "recomputed on the debug subset"},
            [r["folds"][i]["lgbm"]["metrics"]["pr_auc"] for i in range(len(FOLDS))],
            "recomputed on the debug subset")


def apply_rule(cand_seeds, cand_folds, fin_valid, fin_folds):
    st = seed_stats(cand_seeds)
    delta = st["mean"] - fin_valid["mean"]
    sd = max(st["sd"], fin_valid["sd"]) if np.isfinite(st["sd"]) else fin_valid["sd"]
    thr = MARGIN_SD * sd
    wins = [bool(c > f) for c, f in zip(cand_folds, fin_folds)]
    pv, pf = bool(delta > thr), bool(sum(wins) >= MIN_WINS)
    return {"valid": st, "valid_delta": float(delta), "threshold": float(thr), "folds_pr_auc": cand_folds,
            "fold_wins": wins, "n_wins": int(sum(wins)), "passes_valid": pv, "passes_folds": pf,
            "eligible": bool(pv and pf)}


# ------------------------------------------------------------------ gnn member + stack
WINDOWS = ("foldE", "fold0", "fold1", "fold2", "valid")
STACK_TRAIN = {"fold0": ("foldE",), "fold1": ("foldE", "fold0"), "fold2": ("foldE", "fold0", "fold1"),
               "valid": ("foldE", "fold0", "fold1")}   # fold2 and valid overlap in days: never each other's


def rank01(a):
    return pd.Series(np.asarray(a, dtype=float)).rank(method="average").to_numpy() / len(a)


def gnn_art(window, seed):
    return f"gnn_{window}_s{seed}"


def member_frame(cdir, window, seed, members):
    """ids (row order of the tree-model score file) and a DataFrame of raw member scores for one window."""
    first = read_scores(task_paths(cdir, window, "lgbm", seed)[1])
    ids = first["TransactionID"].to_numpy()
    cols = {}
    for m in members:
        if m == "gnn":
            g = read_scores(gnn_art(window, seed)).set_index("TransactionID")["score"].reindex(ids)
            if g.isna().any():
                raise ValueError(f"gnn scores for {window} s{seed} miss {int(g.isna().sum())} of {len(ids)} rows")
            cols[m] = g.to_numpy()
        else:
            f = read_scores(task_paths(cdir, window, m, seed)[1])
            assert (f["TransactionID"].to_numpy() == ids).all(), f"{m} {window}: id order differs"
            cols[m] = f["score"].to_numpy()
    return ids, pd.DataFrame(cols)


def score_matrix(cdir, window, seed, members):
    ids, raw = member_frame(cdir, window, seed, members)
    return {"ids": ids, "raw": raw, "X": np.column_stack([rank01(raw[m]) for m in members])}


def check_gnn(cdir, with_meta):
    """True if every gnn score file exists and covers its window; else log the reason and return False."""
    try:
        for s in SEEDS:
            member_frame(cdir, "valid", s, ["lgbm", "gnn"])
        for w in ["fold0", "fold1", "fold2"] + (["foldE"] if with_meta else []):
            member_frame(cdir, w, SEED, ["lgbm", "gnn"])
        return True
    except (FileNotFoundError, ValueError, OSError) as exc:
        log(f"gnn member skipped: {type(exc).__name__}: {exc}")
        return False


def build_stack(cdir, members):
    """Time-forward stack. Returns (pr_auc per window dict, report). valid -> one value per seed."""
    from sklearn.linear_model import LogisticRegression
    W = {w: score_matrix(cdir, w, SEED, members) for w in WINDOWS}
    for w in WINDOWS:
        W[w]["y"] = labels_for(W[w]["ids"])
        W[w]["day"] = days_for(W[w]["ids"])
    pr, report = {}, {}
    for w, train_ws in STACK_TRAIN.items():
        dmin = int(W[w]["day"].min())
        if CFG["max_rows"] is not None and w == "valid":
            # debug subset: time_split's valid window is not day-scaled and can start before the scaled folds;
            # keep only folds that end earlier (real run: all of foldE+fold0+fold1, asserted below)
            keep = tuple(t for t in train_ws if int(W[t]["day"].max()) < dmin)
            if keep != train_ws:
                log(f"stack valid (debug): valid starts day {dmin}; meta-training restricted {train_ws} -> {keep}")
            train_ws = keep
            if not train_ws:
                log("stack valid (debug): no earlier fold, skipped")
                continue
        Xm = np.vstack([W[t]["X"] for t in train_ws])
        ym = np.concatenate([W[t]["y"] for t in train_ws])
        dmax = max(int(W[t]["day"].max()) for t in train_ws)
        assert dmax < dmin, f"stack {w}: meta-training days reach {dmax}, window starts {dmin}"
        assert "valid" not in train_ws and (w == "fold2" or "fold2" not in train_ws)
        assert len(set(ym.tolist())) == 2, f"stack {w}: meta-training labels have one class"
        lr = LogisticRegression(C=STACK_C, class_weight=None, max_iter=1000).fit(Xm, ym)
        coef = dict(zip(members, [float(c) for c in lr.coef_[0]]))
        entry = {"trained_on": list(train_ws), "n_rows": int(len(ym)), "n_fraud": int(ym.sum()),
                 "max_train_day": dmax, "min_score_day": dmin, "date_rule_ok": True,
                 "intercept": float(lr.intercept_[0]), "coef": coef,
                 "coef_share": {m: float(abs(c) / sum(abs(v) for v in coef.values()))
                                for m, c in coef.items()}}
        if w == "valid":
            vals = []
            for s in SEEDS:
                M = W["valid"] if s == SEED else score_matrix(cdir, "valid", s, members)
                y = W["valid"]["y"] if s == SEED else labels_for(M["ids"])
                vals.append(metrics(y, lr.predict_proba(M["X"])[:, 1])["pr_auc"])
            pr[w] = vals
        else:
            pr[w] = metrics(W[w]["y"], lr.predict_proba(W[w]["X"])[:, 1])["pr_auc"]
        entry["pr_auc"] = pr[w]
        report[w] = entry
        log(f"stack {w}: trained on {train_ws} ({len(ym)} rows, {int(ym.sum())} fraud, days<= {dmax} < {dmin}) "
            f"coef {coef} PR-AUC {pr[w]}")
    return pr, report


def cmd_confirm(names, use_members, use_stack=False):
    use_members = use_members or use_stack
    models = ["lgbm"] + (["xgb", "cat"] if use_members else [])
    r = ensure_windows(names, models, with_meta=use_members)
    fin_valid, fin_folds, fin_src = reference_final(CFG["max_rows"] is None)
    cdir = r["dir"]

    def vscores(m, s):
        return read_scores(task_paths(cdir, "valid", m, s)[1])

    def fscores(m, i):
        return read_scores(task_paths(cdir, f"fold{i}", m, SEED)[1])

    cands = {"lgbm_blocks": ([r["valid"][("lgbm", s)]["metrics"]["pr_auc"] for s in SEEDS],
                             [r["folds"][i]["lgbm"]["metrics"]["pr_auc"] for i in range(len(FOLDS))])}
    if use_members:
        for m in ("xgb", "cat"):
            cands[f"{m}_blocks"] = ([r["valid"][(m, s)]["metrics"]["pr_auc"] for s in SEEDS],
                                    [r["folds"][i][m]["metrics"]["pr_auc"] for i in range(len(FOLDS))])

        def blend_pr(frames):
            ids = frames["lgbm"]["TransactionID"].to_numpy()
            for m in ("xgb", "cat"):
                assert (frames[m]["TransactionID"].to_numpy() == ids).all()
            sc = blend({m: frames[m]["score"].to_numpy() for m in frames})
            return metrics(labels_for(ids), sc)["pr_auc"]

        cands["blend_blocks"] = (
            [blend_pr({m: vscores(m, s) for m in models}) for s in SEEDS],
            [blend_pr({m: fscores(m, i) for m in models}) for i in range(len(FOLDS))])
    members, stack_report = list(models), None
    if use_members and check_gnn(r["dir"], True):
        members.append("gnn")
        g_pr = lambda ids, sc: metrics(labels_for(ids), sc)["pr_auc"]
        gv, gf = [], []
        for s in SEEDS:
            ids, fr = member_frame(r["dir"], "valid", s, members)
            gv.append(g_pr(ids, fr["gnn"].to_numpy()))
        for i in range(len(FOLDS)):
            ids, fr = member_frame(r["dir"], f"fold{i}", SEED, members)
            gf.append(g_pr(ids, fr["gnn"].to_numpy()))
        cands["gnn"] = (gv, gf)

        def blend_g(window, seed):
            ids, fr = member_frame(r["dir"], window, seed, members)
            return g_pr(ids, blend({m: fr[m].to_numpy() for m in members}))

        cands["blend_blocks"] = ([blend_g("valid", s) for s in SEEDS],
                                 [blend_g(f"fold{i}", SEED) for i in range(len(FOLDS))])
    if use_stack:
        pr, stack_report = build_stack(r["dir"], members)
        cands["stack_blocks"] = (pr["valid"], [pr[f"fold{i}"] for i in range(len(FOLDS))])
    outcome = {k: apply_rule(v[0], v[1], fin_valid, fin_folds) for k, v in cands.items()}
    ok = [k for k, v in outcome.items() if v["eligible"]]
    selected = max(ok, key=lambda k: outcome[k]["valid"]["mean"]) if ok else "none"
    res = {"blocks": names, "members": use_members, "stack": use_stack, "members_used": members,
           "meta_fold": {"window": "foldE", "days": list(META_FOLD), "role": "meta-learner training only, never in selection"},
           "stack_meta": stack_report, "max_rows": CFG["max_rows"],
           "max_rounds": CFG["max_rounds"], "git_sha": git_sha(),
           "rule": (f"valid mean PR-AUC delta > {MARGIN_SD:g} x max(sd cand, sd final) (3 seeds, ddof=1) "
                    f"AND PR-AUC above final on >= {MIN_WINS} of 3 forward folds"),
           "final": {"valid": fin_valid, "folds_pr_auc": fin_folds, "folds_source": fin_src},
           "candidates": outcome, "selected": selected, "peak_rss_mb": rss_mb()}
    jwrite(out_dir() / "confirm.json", res)
    for k, v in outcome.items():
        log(f"{k}: valid {v['valid']['mean']:.5f} (final {fin_valid['mean']:.5f}, delta {v['valid_delta']:+.5f}, "
            f"need > {v['threshold']:.5f}) fold wins {v['fold_wins']} -> eligible {v['eligible']}")
    log("selected:", selected)


# ------------------------------------------------------------------ test view
def final_test_scores(ids):
    for root in (models_dir(), WORKTREE.parents[1] / "models"):
        p = root / "final" / "scores_test.parquet"
        if p.exists():
            s = pd.read_parquet(p).set_index("TransactionID")["score"].reindex(ids)
            if s.isna().any():
                raise SystemExit("final test scores do not cover all test rows")
            return s.to_numpy()
    raise SystemExit("models/final/scores_test.parquet not found")


def cmd_test_view():
    cpath = out_dir() / "confirm.json"
    if not cpath.exists():
        raise SystemExit("refusing: confirm.json missing")
    conf = jread(cpath)
    sel = conf.get("selected")
    if sel not in CANDIDATES:
        raise SystemExit(f"refusing: confirm.json selected {sel!r}, nothing to view on test")
    if (out_dir() / "test_view.json").exists():
        raise SystemExit("refusing: test_view.json exists; the test split is scored once per candidate")
    names = conf["blocks"]
    need = {"lgbm_blocks": ["lgbm"], "xgb_blocks": ["xgb"], "cat_blocks": ["cat"],
            "blend_blocks": ["lgbm", "xgb", "cat"], "stack_blocks": ["lgbm", "xgb", "cat"]}[sel]
    use_gnn = "gnn" in conf.get("members_used", []) and sel in ("blend_blocks", "stack_blocks")
    b = make_base(with_test=True)
    Xtr, _ = with_blocks(b.X["train"], names, b.fit_end_dt, b.ids_tr)
    Xva, _ = with_blocks(b.X["valid"], names, b.fit_end_dt, b.ids_va)
    Xte, _ = with_blocks(b.X["test"], names, b.fit_end_dt, b.ids_te)
    s_va, s_te, info = {}, {}, {}
    for m in need:
        t1 = time.time()
        if m == "lgbm":
            booster, best, predict = fit_lgbm(Xtr, b.y_tr, Xva, b.y_va, SEED)
            save_name = f"imp_test_view_{slug(names)}"
        else:
            booster, best, predict = fit_member(m, Xtr, b.y_tr, Xva, b.y_va, SEED)
            save_name = f"imp_test_view_{slug(names)}_{m}"
        s_va[m], s_te[m] = np.asarray(predict(Xva)), np.asarray(predict(Xte))  # the single test scoring
        save_artifacts(save_name, booster if m == "lgbm" else None, b.ids_va, s_va[m], b.ids_te, s_te[m])
        info[m] = {"best_iteration": int(best), "fit_seconds": round(time.time() - t1, 1)}
        log(f"test view {m}: best_it {best}")
    if use_gnn:  # gnn trained once on the valid-window setup (seed 42); its test scores ride in the same artefact
        gp = models_dir() / gnn_art("valid", SEED) / "scores_test.parquet"
        if not gp.exists():
            raise SystemExit(f"refusing: gnn is a member but {gp} is missing")
        gt = pd.read_parquet(gp).set_index("TransactionID")["score"].reindex(b.ids_te)
        gv = read_scores(gnn_art("valid", SEED)).set_index("TransactionID")["score"].reindex(b.ids_va)
        if gt.isna().any() or gv.isna().any():
            raise SystemExit("refusing: gnn scores do not cover valid/test rows")
        s_va["gnn"], s_te["gnn"] = gv.to_numpy(), gt.to_numpy()
    if sel == "stack_blocks":  # meta-learner = the one fitted for valid on foldE+fold0+fold1 (confirm.json)
        meta = conf["stack_meta"]["valid"]
        ms = list(meta["coef"])
        def stack_apply(sd):
            z = meta["intercept"] + sum(meta["coef"][m] * rank01(sd[m]) for m in ms)
            return 1.0 / (1.0 + np.exp(-z))
        sv, st = stack_apply(s_va), stack_apply(s_te)
    elif sel == "blend_blocks":
        sv, st = blend(s_va), blend(s_te)
    else:
        sv, st = s_va[need[0]], s_te[need[0]]
    y_te = b.y_te
    m_valid, m_test = metrics(b.y_va, sv), metrics(y_te, st)
    s_fin = final_test_scores(b.ids_te)
    day_te = (b.dt_te // DAY).astype(int)
    boot, point = dc.day_block_bootstrap(y_te, day_te, {"final": s_fin, "candidate": st},
                                         n_boot=1000, seed=0)
    d = boot["candidate"] - boot["final"]
    paired = {}
    for j, mname in enumerate(("pr_auc", "precision_at_top_1pct")):
        paired[mname] = {"point": point["candidate"][j] - point["final"][j], **dc.quantiles(d[:, j]),
                         "share_resamples_positive": float((d[:, j] > 0).mean())}
    res = {"candidate": sel, "blocks": names, "git_sha": git_sha(), "models": info,
           "n_features": int(Xtr.shape[1]), "split": b.stats, "fit_end_dt": b.fit_end_dt,
           "valid": m_valid, "test": m_test, "final_test": metrics(y_te, s_fin),
           "bootstrap": {"n_resamples": 1000, "seed": 0, "unit": f"{len(set(day_te.tolist()))} test days",
                         "per_model": {k: {"point": {"pr_auc": point[k][0],
                                                     "precision_at_top_1pct": point[k][1]},
                                           "pr_auc": dc.quantiles(boot[k][:, 0]),
                                           "precision_at_top_1pct": dc.quantiles(boot[k][:, 1])}
                                       for k in boot},
                         "paired_delta_candidate_minus_final": paired},
           "note": "single test scoring of the confirm-selected candidate; valid metrics use full-valid early stopping",
           "peak_rss_mb": rss_mb(), "runtime_seconds": round(time.time() - T0, 1)}
    jwrite(out_dir() / "test_view.json", res)
    log(f"test PR-AUC {m_test['pr_auc']:.4f} (final {res['final_test']['pr_auc']:.4f}); "
        f"paired delta {paired['pr_auc']['point']:+.4f} "
        f"[{paired['pr_auc']['p2.5']:+.4f}, {paired['pr_auc']['p97.5']:+.4f}]")


# ------------------------------------------------------------------ cli
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--screen", action="store_true")
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--test-view", action="store_true")
    ap.add_argument("--blocks", default="", help="comma list for --confirm")
    ap.add_argument("--members", action="store_true", help="--confirm: add XGBoost, CatBoost and their rank blend")
    ap.add_argument("--stack", action="store_true", help="--confirm: time-forward L2-logistic stack (implies --members)")
    ap.add_argument("--only", default="", help="--screen: restrict to these blocks")
    ap.add_argument("--max-rows", type=int, default=None, help="debug: first N rows by time")
    ap.add_argument("--max-rounds", type=int, default=MAX_ROUNDS, help="debug: LightGBM rounds")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    a = ap.parse_args()
    CFG.update(max_rows=a.max_rows, max_rounds=a.max_rounds, out_dir=Path(a.out_dir))
    if not any((a.screen, a.combine, a.confirm, a.test_view)):
        ap.error("pick at least one of --screen --combine --confirm --test-view")
    if a.test_view and (a.max_rows or a.max_rounds != MAX_ROUNDS):
        raise SystemExit("--test-view refuses debug options")
    if a.screen:
        cmd_screen(parse_names(a.only))
    if a.combine:
        cmd_combine()
    if a.confirm:
        names = [] if a.blocks == "none" else parse_names(a.blocks)
        if not names and a.blocks != "none":
            ap.error("--confirm needs --blocks a,b,c (or 'none': final's features, members/stack/gnn only)")
        cmd_confirm(names, a.members, a.stack)
    if a.test_view:
        cmd_test_view()
    log("done")


if __name__ == "__main__":
    main()
