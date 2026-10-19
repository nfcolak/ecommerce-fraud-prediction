"""Selection protocol v2 (round p4, ADR-005): arm registry runner, screen, confirm, analyze.

PYTHONPATH=src, results under results/p4 (override with --out-dir):

  python -m ecommerce_fraud.protocol --screen  --arms a,b
      Valid window (main split valid, early stopping on valid, as improve), seed 42; `final` is run
      first if missing.  -> <arm>/valid_s42/{result.json,scores.parquet}, screen/<arm>.json
         (valid_pr_auc, delta_vs_final)
  python -m ecommerce_fraud.protocol --confirm --arms a,b --seeds 42,43,44,45,46
      Six windows w1..w6 = (40,55) (56,71) (72,87) (88,103) (104,119) (120,134): train days 1..S-8,
      early stopping on the last 14 of those days, score [S,E]; days >= 141 are dropped before any
      fold is built.  `final` is always included.  Resumable: a task whose result.json exists is skipped.
  python -m ecommerce_fraud.protocol --analyze --arms a,b [--power]
      Seed-averaged PR-AUC per window, Delta_w, pooled delta, paired day-block bootstrap (7-day blocks
      inside each window, same resample for arm and final, 2000 reps), one-sided p, Holm across the
      listed arms, fold wins, pass flag and winner (PREREGISTRATION Part 2) -> confirm.json.
      --power adds the sd of the pooled delta and the minimum detectable gain at 80% power.

Debug: --max-rows N (first N rows by time; windows without full day coverage are reported as skipped),
--max-rounds R, --windows w1,w2.  Features of `final` are exactly improve's A_hist_nanfix (483 columns).
"""
from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd

from ecommerce_fraud import arms as arms_mod
from ecommerce_fraud import improve as imp
from ecommerce_fraud.baseline_lgbm import PARAMS, git_sha, metrics, prepare
from ecommerce_fraud.data import CACHE_PATH, RAW_DIR, WORKTREE, time_split

OUT_DIR = WORKTREE / "results" / "p4"
DAY = 86400
SPW = 28.0
SEED = 42
DEFAULT_SEEDS = (42, 43, 44, 45, 46)
WINDOWS = {"w1": (40, 55), "w2": (56, 71), "w3": (72, 87), "w4": (88, 103), "w5": (104, 119), "w6": (120, 134)}
GAP_DAYS, ES_DAYS, TEST_START_DAY = 7, 14, 141
DEFAULT_ES, DEFAULT_MAX_ROUNDS = 100, 5000
BLOCK_DAYS, N_BOOT, BOOT_SEED = 7, 2000, 20261016
MIN_WINS, ALPHA, TEST_PROPOSAL_DELTA = 4, 0.05, 0.004
CFG = {"out_dir": OUT_DIR, "max_rounds": None, "max_rows": None}
T0 = time.time()

log, clean, jwrite, jread, rss_mb = imp.log, imp.clean, imp.jwrite, imp.jread, imp.rss_mb


def out_dir() -> Path:
    return Path(CFG["out_dir"])


def task_dir(arm, window, seed) -> Path:
    return out_dir() / arm / f"{window}_s{seed}"


def task_done(arm, window, seed) -> bool:
    return (task_dir(arm, window, seed) / "result.json").exists()


def arm_record(arm) -> dict:
    return {k: ("<callable %s>" % getattr(v, "__name__", "fit") if callable(v) else v) for k, v in arm.items()}


# ------------------------------------------------------------------ data
class Frame:
    """The sorted frame with final's history features (built once per process)."""

    def __init__(self):
        imp.CFG["max_rows"] = CFG["max_rows"]
        self.df, self.cols = imp.load_hist_frame()
        self.windows_only = False

    def valid_data(self):
        """Mirror of improve.make_base: time_split, prepare on (train, valid, empty)."""
        assert not self.windows_only
        df, cols = self.df, self.cols
        train, valid, test, stats = time_split(df)
        del test
        gc.collect()
        assert int((valid["TransactionDT"].max() // DAY)) < TEST_START_DAY, "valid reaches test days"
        assert int((train["TransactionDT"].max() // DAY)) < TEST_START_DAY
        wd = SimpleNamespace(
            window="valid", S=None, E=None, cols=cols, stats=stats,
            tr=train, sc=valid, es_mask=None, is_valid=True,
            fit_end_dt=int(train["TransactionDT"].max()))
        wd.ids_tr, wd.ids_sc = train["TransactionID"].to_numpy(), valid["TransactionID"].to_numpy()
        wd.y_tr, wd.y_sc = train["isFraud"].to_numpy(), valid["isFraud"].to_numpy()
        wd.n_fit, wd.n_es, wd.n_score = len(train), len(valid), len(valid)
        wd.day_fit = (train["TransactionDT"].to_numpy() // DAY)
        wd.day_es = wd.day_sc = (valid["TransactionDT"].to_numpy() // DAY)
        return wd

    def drop_test_days(self):
        """Windows path: test days >= 141 are removed before anything is built (as ensure_windows)."""
        if not self.windows_only:
            day = self.df["TransactionDT"] // DAY
            self.df = self.df[(day < TEST_START_DAY).to_numpy()].reset_index(drop=True)
            self.windows_only = True
            gc.collect()
        assert int(self.df["TransactionDT"].max() // DAY) < TEST_START_DAY

    def window_data(self, name):
        """Mirror of backtest.run_fold / improve.ensure_windows for one protocol window; None if the data
        do not cover it (debug subsets)."""
        self.drop_test_days()
        s, e = WINDOWS[name]
        df, cols = self.df, self.cols
        day = df["TransactionDT"] // DAY
        if int(day.max()) < e:
            return None, f"data end at day {int(day.max())} < window end {e}"
        tr_mask = (day <= s - GAP_DAYS - 1).to_numpy()
        sc_mask = ((day >= s) & (day <= e)).to_numpy()
        tr_all, sc = df[tr_mask], df[sc_mask]
        d_tr = day[tr_mask].to_numpy()
        es_start = (s - GAP_DAYS - 1) - ES_DAYS + 1
        is_es = d_tr >= es_start
        y_tr, y_sc = tr_all["isFraud"].to_numpy(), sc["isFraud"].to_numpy()
        if is_es.sum() == 0 or (~is_es).sum() == 0 or y_sc.sum() == 0 or y_tr[is_es].sum() == 0 \
                or y_tr[~is_es].sum() == 0:
            return None, "too few rows/fraud in fit, early-stopping or scoring part"
        wd = SimpleNamespace(
            window=name, S=s, E=e, cols=cols, stats=None, tr=tr_all, sc=sc, es_mask=is_es, is_valid=False,
            fit_end_dt=int(tr_all["TransactionDT"].max()),
            ids_tr=tr_all["TransactionID"].to_numpy(), ids_sc=sc["TransactionID"].to_numpy(),
            y_tr=y_tr, y_sc=y_sc, n_fit=int((~is_es).sum()), n_es=int(is_es.sum()), n_score=int(len(sc)),
            day_fit=d_tr[~is_es], day_es=d_tr[is_es], day_sc=(sc["TransactionDT"].to_numpy() // DAY))
        return wd, None


def wd_guard(wd):
    """No fitted, early-stopping or scored row may be on a test day."""
    for nm in ("day_fit", "day_es", "day_sc"):
        d = getattr(wd, nm)
        assert len(d) == 0 or int(d.max()) < TEST_START_DAY, f"{wd.window}: {nm} reaches day {int(d.max())}"
    if not wd.is_valid:
        assert int(wd.day_fit.max()) <= wd.S - GAP_DAYS - 1 and int(wd.day_es.max()) <= wd.S - GAP_DAYS - 1
        assert int(wd.day_sc.min()) >= wd.S and int(wd.day_sc.max()) <= wd.E


def base_matrices(wd):
    """prepare() exactly as make_base (valid) / run_fold (windows)."""
    if getattr(wd, "Xtr", None) is None:
        cols = wd.cols
        X, _, _ = prepare(wd.tr[cols], wd.sc[cols], wd.sc[cols].iloc[:0])
        wd.Xtr, wd.Xsc = X["train"], X["valid"]
        assert list(wd.Xtr.columns) == list(cols), "final feature list differs from improve's"
        wd.ident_ids = None
        wd.amt_sc = wd.sc["TransactionAmt"].to_numpy()
        wd.amt_tr = wd.tr["TransactionAmt"].to_numpy()
        # raw frames are not needed any more except the id/label arrays already extracted
        wd.tr = wd.sc = None
        gc.collect()
    return wd


def arm_feature_key(arm):
    return json.dumps([arm.get("blocks", []), arm.get("block_cols", {}), arm.get("drop_cols", [])], sort_keys=True)


def apply_arm_features(wd, arm):
    """Final's matrices + the arm's blocks (whitelisted columns) - dropped columns. Cached per feature key."""
    key = arm_feature_key(arm)
    if getattr(wd, "feat_key", None) == key:
        return wd.Xt_arm, wd.Xs_arm
    wd.feat_key, wd.Xt_arm, wd.Xs_arm = None, None, None
    gc.collect()
    names = list(arm.get("blocks", []))
    Xt, Xs = wd.Xtr, wd.Xsc
    if names:
        known = imp.blk.discover()
        bad = [n for n in names if n not in known]
        if bad:
            raise ValueError(f"unknown block(s) {bad}")
        Xt, _ = imp.with_blocks(Xt, names, wd.fit_end_dt, wd.ids_tr)
        Xs, _ = imp.with_blocks(Xs, names, wd.fit_end_dt, wd.ids_sc)
    drop = []
    bc = arm.get("block_cols", {})
    for b in names:
        if b in bc:
            pre = f"{b}__"
            allc = [c for c in Xt.columns if c.startswith(pre)]
            keep = {c if c.startswith(pre) else pre + c for c in bc[b]}
            missing = sorted(keep - set(allc))
            if missing:
                raise ValueError(f"block_cols: block {b} has no column(s) {missing[:5]}")
            drop += [c for c in allc if c not in keep]
    import re
    for d in arm.get("drop_cols", []):
        if d.startswith("re:"):
            rx = re.compile(d[3:])
            hit = [c for c in Xt.columns if rx.search(c)]
            if not hit:
                log(f"WARNING drop_cols {d!r} matches no column")
            drop += hit
        else:
            if d not in Xt.columns:
                raise ValueError(f"drop_cols: column {d!r} not among the features")
            drop.append(d)
    drop = sorted(set(drop))
    if drop:
        Xt, Xs = Xt.drop(columns=drop), Xs.drop(columns=drop)
    wd.feat_key, wd.Xt_arm, wd.Xs_arm = key, Xt, Xs
    return Xt, Xs


def split_fit_es(wd, Xt, Xs):
    if wd.is_valid:
        return Xt, wd.y_tr, Xs, wd.y_sc          # ES on valid, as improve's valid runs
    m = wd.es_mask
    return Xt[~m], wd.y_tr[~m], Xt[m], wd.y_tr[m]


# ------------------------------------------------------------------ models
def member_fit_fn(model):
    from ecommerce_fraud import members
    cfg = getattr(members, "CONFIGS", {}).get(model) if isinstance(getattr(members, "CONFIGS", None), dict) else None
    if callable(cfg):
        return cfg
    if isinstance(cfg, dict) and callable(cfg.get("fit")):
        return cfg["fit"]
    fn = getattr(members, f"fit_{model}", None)
    if callable(fn):
        return fn
    raise SystemExit(f"model {model!r}: no members.py entry (looked for CONFIGS[{model!r}] and fit_{model}); "
                     f"the 'members' unit must be merged first")


def fit_lgbm(arm, Xf, yf, Xe, ye, seed):
    params = {**PARAMS, "seed": seed, "scale_pos_weight": SPW, **arm.get("params", {})}
    if os.environ.get("EF_SMOKE_N_JOBS"):          # p5: subset smokes only; full runs never set it
        params["n_jobs"] = int(os.environ["EF_SMOKE_N_JOBS"])
    rounds = arm.get("max_rounds", DEFAULT_MAX_ROUNDS)
    if CFG["max_rounds"]:
        rounds = min(rounds, CFG["max_rounds"])
    dtrain = lgb.Dataset(Xf, yf)
    dvalid = lgb.Dataset(Xe, ye, reference=dtrain)
    booster = lgb.train(params, dtrain, num_boost_round=rounds, valid_sets=[dvalid], valid_names=["valid"],
                        callbacks=[lgb.early_stopping(arm.get("early_stop", DEFAULT_ES), verbose=False),
                                   lgb.log_evaluation(500)])
    best = int(booster.best_iteration)
    return best, rounds, booster.predict, (lambda X, b=best: booster.predict(X, num_iteration=b))


def identity_ids():
    p = RAW_DIR / "train_identity.csv"
    return pd.read_csv(p, usecols=["TransactionID"])["TransactionID"].to_numpy()


def build_meta(wd):
    if wd.ident_ids is None:
        wd.ident_ids = identity_ids()
    ids_tr = wd.ids_tr
    if wd.is_valid:
        ids_fit, ids_es = ids_tr, wd.ids_sc
    else:
        ids_fit, ids_es = ids_tr[~wd.es_mask], ids_tr[wd.es_mask]
    f = lambda ids: np.isin(ids, wd.ident_ids)
    amt_fit, amt_es = (wd.amt_tr, wd.amt_sc) if wd.is_valid else (wd.amt_tr[~wd.es_mask], wd.amt_tr[wd.es_mask])
    # p5 additions (additive keys; earlier arms ignore them): row ids, days and amounts of every part,
    # the window name and score-window bounds (None on the valid screen).
    return {"has_identity_fit": f(ids_fit), "has_identity_es": f(ids_es), "has_identity_score": f(wd.ids_sc),
            "amount_score": wd.amt_sc.astype(float), "day_score": wd.day_sc.astype(float),
            "ids_fit": ids_fit, "ids_es": ids_es, "ids_score": wd.ids_sc,
            "day_fit": np.asarray(wd.day_fit, dtype=float), "day_es": np.asarray(wd.day_es, dtype=float),
            "amount_fit": amt_fit.astype(float), "amount_es": amt_es.astype(float),
            "window": wd.window, "score_days": None if wd.is_valid else (wd.S, wd.E)}


def blend_scores(arm, wd, seed):
    from ecommerce_fraud import members
    parts = {}
    for comp in arm["blend_of"]:
        p = task_dir(comp, wd.window, seed) / "scores.parquet"
        if not p.exists():
            raise FileNotFoundError(f"blend_of {comp!r}: {p} missing (run that arm first, same window and seed)")
        s = pd.read_parquet(p).set_index("TransactionID")["score"].reindex(wd.ids_sc)
        if s.isna().any():
            raise ValueError(f"blend_of {comp!r}: scores miss {int(s.isna().sum())} rows of {wd.window} s{seed}")
        parts[comp] = s.to_numpy()
    return np.asarray(members.rank_blend(parts))


def run_one(wd, arm, seed):
    """Fit/score one arm x seed on the prepared window; writes scores.parquet then result.json."""
    wd_guard(wd)
    name, d = arm["name"], task_dir(arm["name"], wd.window, seed)
    t0 = time.time()
    best, n_feat, hit_cap = None, None, None
    if "blend_of" in arm:
        s_sc = blend_scores(arm, wd, seed)
    else:
        base_matrices(wd)
        Xt, Xs = apply_arm_features(wd, arm)
        n_feat = int(Xt.shape[1])
        if not arm.get("blocks") and not arm.get("drop_cols"):
            assert n_feat == len(wd.cols), "arm without blocks/drop_cols must keep final's features"
        Xf, yf, Xe, ye = split_fit_es(wd, Xt, Xs)
        assert len(Xf) == wd.n_fit and len(Xe) == wd.n_es
        model = arm.get("model", "lgbm")
        rounds = arm.get("max_rounds", DEFAULT_MAX_ROUNDS)
        if model == "lgbm":
            best, rounds, _, predict = fit_lgbm(arm, Xf, yf, Xe, ye, seed)
            s_sc = predict(Xs)
        elif model == "custom":
            s_sc = np.asarray(arm["fit"](Xf, yf, Xe, ye, Xs, seed, build_meta(wd)), dtype=float)
        else:
            fn = member_fit_fn(model)
            kw = {k: arm[k] for k in ("params", "early_stop", "max_rounds") if k in arm}
            accepted = set(inspect.signature(fn).parameters)
            bad = [k for k in kw if k not in accepted]
            if bad:
                raise ValueError(f"model {model}: fit function does not accept arm key(s) {bad}")
            from ecommerce_fraud import members
            old = members.MAX_ROUNDS
            if CFG["max_rounds"]:
                members.MAX_ROUNDS = CFG["max_rounds"]
            try:
                _, best, predict = fn(Xf, yf, Xe, ye, seed, **kw)
            finally:
                members.MAX_ROUNDS = old
            s_sc = np.asarray(predict(Xs))
            rounds = CFG["max_rounds"] or kw.get("max_rounds", members.MAX_ROUNDS)
        hit_cap = None if best is None else bool(best >= rounds - 1)
        del Xf, Xe
    s_sc = np.asarray(s_sc, dtype=float)
    assert s_sc.shape == (wd.n_score,) and np.isfinite(s_sc).all(), "scores must be finite, one per scored row"
    m = metrics(wd.y_sc, s_sc)
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"TransactionID": wd.ids_sc, "score": s_sc}).to_parquet(d / "scores.parquet", index=False)
    res = {"arm": name, "window": wd.window, "seed": int(seed), "pr_auc": m["pr_auc"], "metrics": m,
           "best_iteration": best, "hit_max_rounds": hit_cap, "seconds": round(time.time() - t0, 1),
           "rss_mb": rss_mb(), "n_features": n_feat, "n_fit": wd.n_fit, "n_es": wd.n_es,
           "n_score": wd.n_score, "n_fraud_score": int(wd.y_sc.sum()),
           "score_days": [wd.S, wd.E] if not wd.is_valid else None, "fit_end_dt": wd.fit_end_dt,
           "arm_spec": arm_record(arm), "git_sha": git_sha(), "max_rows": CFG["max_rows"],
           "max_rounds_override": CFG["max_rounds"]}
    jwrite(d / "result.json", res)
    log(f"{name} {wd.window} s{seed}: PR-AUC {m['pr_auc']:.5f} best_it {best} ({res['seconds']}s)")
    gc.collect()
    return res


# ------------------------------------------------------------------ task orchestration
def resolve_arms(names, registry):
    out = []
    for n in names:
        if n not in registry:
            raise SystemExit(f"unknown arm {n!r}; discovered: {sorted(registry)}")
        arms_mod.validate(registry[n], known=set(registry))
        out.append(n)
    return out


def depth(name, registry, seen=()):
    arm = registry[name]
    if name in seen:
        raise SystemExit(f"blend_of cycle at {name}")
    return 1 + max((depth(c, registry, seen + (name,)) for c in arm.get("blend_of", [])), default=-1)


def order_arms(names, registry):
    return sorted(names, key=lambda n: (depth(n, registry), arm_feature_key(registry[n]), n != "final", n))


def run_tasks(window, names, seeds, registry, frame):
    """Run all pending arm x seed tasks of one window on one built frame. Returns (n_run, errors, skipped)."""
    pending = [(n, s) for n in order_arms(names, registry) for s in seeds if not task_done(n, window, s)]
    if not pending:
        log(f"{window}: all {len(names) * len(seeds)} tasks exist")
        return 0, {}, None
    if window == "valid":
        wd, why = frame.valid_data(), None
    else:
        wd, why = frame.window_data(window)
    if wd is None:
        log(f"{window}: SKIPPED ({why})")
        return 0, {}, why
    log(f"{window}: {len(pending)} pending tasks; fit {wd.n_fit} es {wd.n_es} score {wd.n_score} "
        f"({int(wd.y_sc.sum())} fraud)")
    errors, n = {}, 0
    for arm_name, seed in pending:
        try:
            run_one(wd, registry[arm_name], seed)
            n += 1
        except Exception as exc:
            errors[f"{arm_name}/{window}_s{seed}"] = f"{type(exc).__name__}: {exc}"
            log(f"ERROR {arm_name} {window} s{seed}: {errors[f'{arm_name}/{window}_s{seed}']}")
    wd.Xtr = wd.Xsc = wd.Xt_arm = wd.Xs_arm = None
    del wd
    gc.collect()
    return n, errors, None


def cmd_screen(names, registry):
    names = [n for n in dict.fromkeys(["final"] + names)]
    todo = [n for n in names if not task_done(n, "valid", SEED)]
    errors = {}
    if todo:
        frame = Frame()
        _, errors, _ = run_tasks("valid", names, [SEED], registry, frame)
        del frame
        gc.collect()
    fin_p = task_dir("final", "valid", SEED) / "result.json"
    ref = jread(fin_p)["pr_auc"] if fin_p.exists() else None
    for n in names:
        p = task_dir(n, "valid", SEED) / "result.json"
        out = {"arm": n, "description": registry[n].get("description", ""), "seed": SEED,
               "max_rows": CFG["max_rows"], "max_rounds": CFG["max_rounds"]}
        if p.exists():
            r = jread(p)
            out.update(valid_pr_auc=r["pr_auc"], best_iteration=r["best_iteration"],
                       hit_max_rounds=r["hit_max_rounds"], n_features=r["n_features"],
                       seconds=r["seconds"], rss_mb=r["rss_mb"], git_sha=r["git_sha"],
                       arm_spec=r["arm_spec"], reference_pr_auc=ref,
                       delta_vs_final=None if ref is None or n == "final" else r["pr_auc"] - ref)
            log(f"screen {n}: valid PR-AUC {r['pr_auc']:.5f}" +
                ("" if out["delta_vs_final"] is None else f" delta {out['delta_vs_final']:+.5f}"))
        else:
            out["error"] = errors.get(f"{n}/valid_s{SEED}", "no result")
        jwrite(out_dir() / "screen" / f"{n}.json", out)
    return errors


def cmd_confirm(names, seeds, registry, windows):
    names = [n for n in dict.fromkeys(["final"] + names)]
    frame, errors, skipped = None, {}, {}
    for w in windows:
        if all(task_done(n, w, s) for n in names for s in seeds):
            log(f"{w}: all tasks exist")
            continue
        frame = frame or Frame()
        _, err, why = run_tasks(w, names, seeds, registry, frame)
        errors.update(err)
        if why:
            skipped[w] = why
    if skipped:
        log("skipped windows:", skipped)
    if errors:
        log("ERRORS:", errors)
    return errors


# ------------------------------------------------------------------ analysis
def weighted_ap(y_sorted, w, group_end):
    tp = np.cumsum(w * y_sorted)[group_end]
    cnt = np.cumsum(w)[group_end]
    total = tp[-1]
    if total <= 0:
        return np.nan
    prec = np.divide(tp, cnt, out=np.zeros_like(tp), where=cnt > 0)
    rec = tp / total
    return float(np.sum(np.diff(rec, prepend=0.0) * prec))


def sort_info(score, y):
    order = np.argsort(-score, kind="stable")
    s = score[order]
    ge = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s) - 1]
    return order, y[order].astype(float), ge


def plain_ap(score, y):
    _, ys, ge = sort_info(score, y)
    return weighted_ap(ys, np.ones(len(ys)), ge)


def load_labels_days():
    t = pd.read_parquet(CACHE_PATH, columns=["TransactionID", "isFraud", "TransactionDT"]).set_index("TransactionID")
    return t["isFraud"], (t["TransactionDT"] // DAY)


def seed_scores(arm, window, seeds, ids):
    """Seed-averaged raw score per id (ids = reference order)."""
    acc = np.zeros(len(ids))
    for s in seeds:
        sc = pd.read_parquet(task_dir(arm, window, s) / "scores.parquet").set_index("TransactionID")["score"]
        v = sc.reindex(ids)
        if v.isna().any():
            raise ValueError(f"{arm} {window} s{s}: scores do not cover the rows of final's window")
        acc += v.to_numpy()
    return acc / len(seeds)


def bootstrap_pooled(per_window, arm_scores, fin_scores, n_boot=N_BOOT):
    """per_window[w] = dict(y, day, S, counts (n_boot x n_blocks), block_of_row). Returns array of pooled deltas."""
    deltas = np.zeros((len(per_window), n_boot))
    for i, (w, pw) in enumerate(per_window.items()):
        o_a, ya, ge_a = sort_info(arm_scores[w], pw["y"])
        o_f, yf_, ge_f = sort_info(fin_scores[w], pw["y"])
        blk_a, blk_f = pw["block_of_row"][o_a], pw["block_of_row"][o_f]
        for r in range(n_boot):
            cnt = pw["counts"][r]
            deltas[i, r] = weighted_ap(ya, cnt[blk_a], ge_a) - weighted_ap(yf_, cnt[blk_f], ge_f)
    return deltas.mean(0)


def holm(pvals):
    names = sorted(pvals, key=lambda n: pvals[n])
    m, adj, run = len(names), {}, 0.0
    for i, n in enumerate(names):
        run = max(run, min(1.0, (m - i) * pvals[n]))
        adj[n] = run
    return adj


def cmd_analyze(names, seeds, windows, power):
    from scipy.stats import norm
    names = [n for n in dict.fromkeys(names) if n != "final"]
    # windows available: final has all seeds
    wins = [w for w in windows if all(task_done("final", w, s) for s in seeds)]
    if not wins:
        raise SystemExit("no window has all seeds for final")
    for n in names:
        miss = [f"{n}/{w}_s{s}" for w in wins for s in seeds if not task_done(n, w, s)]
        if miss:
            raise SystemExit(f"arm {n}: missing tasks {miss[:6]}{' ...' if len(miss) > 6 else ''}")
    y_all, day_all = load_labels_days()
    rng_parent = np.random.SeedSequence(BOOT_SEED)
    rngs = rng_parent.spawn(len(WINDOWS))
    ref_ids, per_window, fin_scores = {}, {}, {}
    for w in wins:
        ids = pd.read_parquet(task_dir("final", w, seeds[0]) / "scores.parquet")["TransactionID"].to_numpy()
        S = WINDOWS[w][0]
        day = day_all.reindex(ids).to_numpy().astype(int)
        y = y_all.reindex(ids).to_numpy()
        assert day.min() >= S and day.max() <= WINDOWS[w][1] and day.max() < TEST_START_DAY
        blk = (day - S) // BLOCK_DAYS
        nb = int(blk.max()) + 1
        rng = np.random.default_rng(rngs[list(WINDOWS).index(w)])
        draws = rng.integers(0, nb, size=(N_BOOT, nb))
        counts = (draws[:, :, None] == np.arange(nb)).sum(1).astype(float)
        per_window[w] = {"y": y, "block_of_row": blk, "counts": counts, "n_blocks": nb}
        ref_ids[w] = ids
        fin_scores[w] = seed_scores("final", w, seeds, ids)
    # sanity: weighted AP with unit weights equals sklearn
    from sklearn.metrics import average_precision_score
    w0 = wins[0]
    o, ys, ge = sort_info(fin_scores[w0], per_window[w0]["y"])
    assert abs(weighted_ap(ys, np.ones(len(ys)), ge) - average_precision_score(per_window[w0]["y"], fin_scores[w0])) < 1e-9

    def pr(arm, w, s):
        return jread(task_dir(arm, w, s) / "result.json")["pr_auc"]

    fin = {w: {"per_seed": [pr("final", w, s) for s in seeds]} for w in wins}
    for w in wins:
        fin[w]["mean"] = float(np.mean(fin[w]["per_seed"]))
    res_arms, pvals = {}, {}
    for n in names:
        scores = {w: seed_scores(n, w, seeds, ref_ids[w]) for w in wins}
        pw = {}
        for w in wins:
            ps = [pr(n, w, s) for s in seeds]
            pw[w] = {"per_seed": ps, "mean": float(np.mean(ps)), "delta": float(np.mean(ps) - fin[w]["mean"])}
        deltas = [pw[w]["delta"] for w in wins]
        boot = bootstrap_pooled(per_window, scores, fin_scores)
        p = float((boot <= 0).mean())
        pvals[n] = p
        res_arms[n] = {"per_window": pw, "pooled_delta": float(np.mean(deltas)), "fold_wins": [bool(d > 0) for d in deltas],
                       "n_wins": int(sum(d > 0 for d in deltas)),
                       "bootstrap": {"n_reps": N_BOOT, "block_days": BLOCK_DAYS, "seed": BOOT_SEED,
                                     "pooled_delta_on_seed_averaged_scores": float(np.mean([
                                         plain_ap(scores[w], per_window[w]["y"])
                                         - plain_ap(fin_scores[w], per_window[w]["y"]) for w in wins])),
                                     "sd": float(boot.std(ddof=1)),
                                     "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
                                     "p_one_sided": p}}
        if power:
            seed_pool = [float(np.mean([pr(n, w, s) - pr("final", w, s) for w in wins])) for s in seeds]
            res_arms[n]["seed_sd_of_pooled_delta"] = float(np.std(seed_pool, ddof=1)) if len(seeds) > 1 else None
            res_arms[n]["_boot"] = boot
    adj = holm(pvals) if pvals else {}
    passing = []
    for n in names:
        a = res_arms[n]
        a["holm_p"] = adj[n]
        a["passes"] = bool(adj[n] < ALPHA and a["n_wins"] >= MIN_WINS)
        if a["passes"]:
            passing.append(n)
    winner = max(passing, key=lambda n: res_arms[n]["pooled_delta"]) if passing else "final"
    best = max(names, key=lambda n: res_arms[n]["pooled_delta"]) if names else None
    out = {"windows": wins, "seeds": list(seeds), "n_windows_expected": len(WINDOWS), "git_sha": git_sha(),
           "rule": f"Holm-adjusted p < {ALPHA} across {len(names)} arm(s) AND Delta_w > 0 in >= {MIN_WINS} windows",
           "final": fin, "arms": res_arms, "passing": passing, "winner": winner,
           "test_view_proposal": [n for n in passing if res_arms[n]["pooled_delta"] >= TEST_PROPOSAL_DELTA]}
    if not passing and best:
        out["negative_finding"] = (
            "Under the stronger protocol (6 windows x 5 seeds, paired day-block bootstrap, Holm), no candidate "
            f"beat the single LightGBM; final stays (best pooled delta = {res_arms[best]['pooled_delta']:+.5f}, "
            f"windows won = {res_arms[best]['n_wins']}/{len(wins)}).")
    if power:
        k = float(norm.ppf(1 - ALPHA) + norm.ppf(0.8))
        kh = float(norm.ppf(1 - ALPHA / max(1, len(names))) + norm.ppf(0.8))
        pw_out = {"mde_formula": f"(z_(1-alpha) + z_0.8) x sd(pooled delta); one-sided alpha {ALPHA}: {k:.3f}, "
                                 f"Holm worst case alpha/{max(1, len(names))}: {kh:.3f}", "per_arm": {}}
        for n in names:
            sd = res_arms[n]["bootstrap"]["sd"]
            pw_out["per_arm"][n] = {"sd_boot": sd, "sd_seed_pooled": res_arms[n].get("seed_sd_of_pooled_delta"),
                                    "mde_80": k * sd, "mde_80_holm_worst": kh * sd}
        if len(seeds) >= 2:  # final vs itself: seeds split into two halves, same bootstrap
            a_s, b_s = list(seeds[0::2]), list(seeds[1::2])
            sa = {w: seed_scores("final", w, a_s, ref_ids[w]) for w in wins}
            sb = {w: seed_scores("final", w, b_s, ref_ids[w]) for w in wins}
            boot = bootstrap_pooled(per_window, sa, sb)
            pw_out["final_vs_final_split"] = {
                "seeds_a": a_s, "seeds_b": b_s, "sd_boot": float(boot.std(ddof=1)),
                "mean_delta": float(boot.mean()), "mde_80": k * float(boot.std(ddof=1)),
                "note": "null comparison (same model, disjoint seed halves); sd includes day sampling and seed noise "
                        "of a 2-3 seed average, so it is conservative for a 5-seed arm"}
        out["power"] = pw_out
    for a in res_arms.values():
        a.pop("_boot", None)
    jwrite(out_dir() / "confirm.json", out)
    for n in names:
        a = res_arms[n]
        log(f"analyze {n}: pooled delta {a['pooled_delta']:+.5f} wins {a['n_wins']}/{len(wins)} "
            f"p {a['bootstrap']['p_one_sided']:.4f} holm {a['holm_p']:.4f} pass {a['passes']}")
    log("winner:", winner)
    return out


# ------------------------------------------------------------------ cli
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--screen", action="store_true")
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--power", action="store_true", help="--analyze: sd of the pooled delta and MDE at 80% power")
    ap.add_argument("--arms", default="", help="comma list of arm names")
    ap.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    ap.add_argument("--windows", default=",".join(WINDOWS), help="--confirm/--analyze: subset of w1..w6")
    ap.add_argument("--max-rows", type=int, default=None, help="debug: first N rows by time")
    ap.add_argument("--max-rounds", type=int, default=None, help="debug: cap on boosting rounds")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    a = ap.parse_args()
    if not (a.screen or a.confirm or a.analyze):
        ap.error("pick one of --screen --confirm --analyze")
    CFG.update(out_dir=Path(a.out_dir), max_rows=a.max_rows, max_rounds=a.max_rounds)
    registry = arms_mod.discover()
    names = [n for n in a.arms.split(",") if n]
    resolve_arms(["final"] + names, registry)
    seeds = [int(s) for s in a.seeds.split(",") if s]
    windows = [w for w in a.windows.split(",") if w]
    bad = [w for w in windows if w not in WINDOWS]
    if bad:
        ap.error(f"unknown window(s) {bad}")
    rc = 0
    if a.screen:
        rc |= bool(cmd_screen(names, registry))
    if a.confirm:
        rc |= bool(cmd_confirm(names, seeds, registry, windows))
    if a.analyze:
        cmd_analyze(names, seeds, windows, a.power)
    log("done")
    raise SystemExit(int(rc))


if __name__ == "__main__":
    main()
