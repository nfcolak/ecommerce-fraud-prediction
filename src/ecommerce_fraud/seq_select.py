"""ADR-003 selection for the sequence + SSL round: valid metrics, fold wins, the rule, the table.

  python -m ecommerce_fraud.seq_select                       # collect, apply the rule, write summary
  python -m ecommerce_fraud.seq_select --test-view <cand>    # only if summary.json selected <cand>

Candidates: seq_sup, seq_ssl, lgbm_final_emb, blend (fixed 50/50 rank average of final and
seq_ssl scores, nothing tuned). Reference: final.
Rule per candidate: valid mean PR-AUC - final valid mean > 2 x max(candidate sd, final sd) over 3
seeds (ddof=1) AND PR-AUC higher than final on >= 2 of the 3 forward folds (63-83, 87-107,
111-131). A candidate with any missing number does not pass (reason in summary.json).
Sources:
  final valid, 3 seeds : results/lgbm_cw/metrics.json -> selection.candidates.spw28
                         (A_hist_nanfix, spw 28, seeds 42/43/44: valid_pr_auc_per_seed/_mean/_sd);
                         top-1%/5% from results/final/metrics.json valid (seed 42 only)
  final folds          : results/backtest/metrics.json folds[*].final
  candidates           : results/<model>/metrics.json -> valid (seq_train / model_emb write it)
  blend                : models/final + models/seq_ssl_* per-row scores (backtest_d<S>_final on folds)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ecommerce_fraud.artifacts import models_dir
from ecommerce_fraud.baseline_lgbm import metrics
from ecommerce_fraud.data import RAW_DIR, WORKTREE

CANDIDATES = ("seq_sup", "seq_ssl", "lgbm_final_emb", "blend")
FOLD_STARTS = (63, 87, 111)  # backtest.FOLDS: (63, 83), (87, 107), (111, 131)
SEEDS = (0, 1, 2)
KEYS = ("pr_auc", "precision_at_top_1pct", "recall_at_top_1pct",
        "precision_at_top_5pct", "recall_at_top_5pct")
MARGIN_SD = 2.0
MIN_WINS = 2
MAIN_MODELS = WORKTREE.parents[1] / "models"  # main checkout; worktrees live in <main>/.worktrees
NA = "not available"


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def valid_of(results_root: Path, model: str):
    d = read_json(results_root / model / "metrics.json")
    if d is None:
        return None
    v = d["valid"] if "valid" in d else d  # model_emb nests under "valid"; seq_train is flat
    return {k: v[k] for k in KEYS}


def model_roots(models_root):
    """An explicit --models-root is used alone; the default is models_dir() then the main checkout."""
    return [Path(models_root)] if models_root else [models_dir(), MAIN_MODELS]


def scores_path(models_roots, name, fname="scores_valid.parquet"):
    for r in models_roots:
        p = Path(r) / name / fname
        if p.exists():
            return p
    return None


def load_labels(path=None) -> pd.Series:
    if path:
        df = pd.read_parquet(path)
    else:
        cands = [WORKTREE / "data" / "processed" / "ieee_cis_train_joined.parquet",
                 WORKTREE.parents[1] / "data" / "processed" / "ieee_cis_train_joined.parquet"]
        hit = next((c for c in cands if c.exists()), None)
        df = (pd.read_parquet(hit, columns=["TransactionID", "isFraud"]) if hit else
              pd.read_csv(RAW_DIR / "train_transaction.csv", usecols=["TransactionID", "isFraud"]))
    return df.set_index("TransactionID")["isFraud"]


def rank01(x):
    return pd.Series(x).rank(method="average").to_numpy() / len(x)


def rank_blend(sa: pd.DataFrame, sb: pd.DataFrame, w=0.5):
    """Fixed-weight rank average on the TransactionIDs both frames have; returns (ids, score)."""
    j = sa.merge(sb, on="TransactionID", suffixes=("_a", "_b"))
    return j["TransactionID"].to_numpy(), w * rank01(j["score_a"].to_numpy()) + (1 - w) * rank01(j["score_b"].to_numpy())


def blend_metrics(path_a, path_b, labels):
    ids, s = rank_blend(pd.read_parquet(path_a), pd.read_parquet(path_b))
    if len(ids) == 0:
        raise ValueError(f"no common TransactionIDs between {path_a} and {path_b}")
    m = metrics(labels.reindex(ids).to_numpy(), s)
    return {k: m[k] for k in KEYS}, int(len(ids))


def summarize_seeds(per_seed):
    if len(per_seed) != len(SEEDS) or any(p is None for p in per_seed):
        return None
    out = {"pr_auc_per_seed": [p["pr_auc"] for p in per_seed], "n_seeds": len(per_seed)}
    for k in KEYS:
        v = np.array([p[k] for p in per_seed])
        out[k + "_mean"], out[k + "_sd"] = float(v.mean()), float(v.std(ddof=1))
    return out


def collect_final(results_root: Path):
    cw = read_json(results_root / "lgbm_cw" / "metrics.json")
    fin = read_json(results_root / "final" / "metrics.json")
    bt = read_json(results_root / "backtest" / "metrics.json")
    spw28 = cw["selection"]["candidates"]["spw28"]
    valid = {"pr_auc_per_seed": spw28["valid_pr_auc_per_seed"], "pr_auc_mean": spw28["valid_pr_auc_mean"],
             "pr_auc_sd": spw28["valid_pr_auc_sd"], "n_seeds": len(spw28["valid_pr_auc_per_seed"]),
             "source": "results/lgbm_cw/metrics.json selection.candidates.spw28 (A_hist_nanfix, spw 28)"}
    if fin is not None:
        for k in KEYS[1:]:
            valid[k + "_mean"] = fin["valid"][k]
        valid["topk_note"] = "precision/recall top-k from results/final (seed 42 only, no sd)"
    folds = {}
    for f in (bt or {}).get("folds", []):
        folds[str(f["score_start_day"])] = {k: f["final"][k] for k in KEYS}
    return valid, folds


def collect_candidate(cand, results_root, models_roots, labels):
    valid_seeds, folds, notes = [], {}, []
    if cand == "blend":
        n_rows = {}
        for s in SEEDS:
            pa, pb = scores_path(models_roots, "final"), scores_path(models_roots, f"seq_ssl_valid_s{s}")
            if pa is None or pb is None:
                valid_seeds.append(None)
                notes.append(f"valid s{s}: scores missing")
                continue
            m, n = blend_metrics(pa, pb, labels)
            valid_seeds.append(m)
            n_rows[f"valid_s{s}"] = n
        for S in FOLD_STARTS:
            pa, pb = scores_path(models_roots, f"backtest_d{S}_final"), scores_path(models_roots, f"seq_ssl_fold{S}_s0")
            if pa is None or pb is None:
                folds[str(S)] = NA
                notes.append(f"fold {S}: final fold scores or seq_ssl fold scores missing")
                continue
            folds[str(S)], n_rows[f"fold{S}"] = blend_metrics(pa, pb, labels)
        extra = {"weights": {"final": 0.5, "seq_ssl": 0.5}, "n_rows": n_rows,
                 "final_valid_scores": "single final model (seed 42), paired with ssl seed s"}
    else:
        valid_seeds = [valid_of(results_root, f"{cand}_valid_s{s}") for s in SEEDS]
        for s, v in zip(SEEDS, valid_seeds):
            if v is None:
                notes.append(f"valid s{s}: results/{cand}_valid_s{s}/metrics.json missing")
        for S in FOLD_STARTS:
            v = valid_of(results_root, f"{cand}_fold{S}_s0")
            folds[str(S)] = v if v is not None else NA
            if v is None:
                notes.append(f"fold {S}: results/{cand}_fold{S}_s0/metrics.json missing")
        extra = {}
    return {"valid": summarize_seeds(valid_seeds) or NA, "folds": folds, "notes": notes, **extra}


def apply_rule(cand, final_valid, final_folds):
    v = cand["valid"]
    out = {"rule": f"valid mean PR-AUC delta > {MARGIN_SD:g} x max(sd cand, sd final) AND wins >= {MIN_WINS} of 3 folds"}
    wins, missing = [], []
    for S in FOLD_STARTS:
        f, g = cand["folds"].get(str(S), NA), final_folds.get(str(S))
        if f == NA or g is None:
            wins.append(None)
            missing.append(S)
        else:
            wins.append(bool(f["pr_auc"] > g["pr_auc"]))
    out["fold_wins"] = dict(zip(map(str, FOLD_STARTS), wins))
    out["n_wins"] = sum(w for w in wins if w)
    if v == NA:
        out.update(valid_delta=None, threshold=None, passes_valid=False, passes_folds=False, selected_eligible=False,
                   reason="valid seeds incomplete")
        return out
    delta = v["pr_auc_mean"] - final_valid["pr_auc_mean"]
    thr = MARGIN_SD * max(v["pr_auc_sd"], final_valid["pr_auc_sd"])
    pv = bool(delta > thr)
    pf = bool(not missing and out["n_wins"] >= MIN_WINS)
    reason = []
    if not pv:
        reason.append(f"valid delta {delta:+.4f} <= threshold {thr:.4f}")
    if missing:
        reason.append(f"fold result not available for {missing}: not evaluable")
    elif not pf:
        reason.append(f"wins {out['n_wins']} < {MIN_WINS}")
    out.update(valid_delta=float(delta), threshold=float(thr), passes_valid=pv, passes_folds=pf,
               selected_eligible=bool(pv and pf), reason="; ".join(reason) or "passes")
    return out


def table(summary) -> str:
    fv = summary["final"]["valid"]
    rows = [("final", fv, {S: summary["final"]["folds"].get(S) for S in map(str, FOLD_STARTS)}, None)]
    for c in CANDIDATES:
        d = summary["candidates"][c]
        rows.append((c, d["valid"], d["folds"], d["rule_outcome"]))

    def f(x):
        return "   n/a " if x is None else f"{x:7.4f}"
    head = (f"{'model':16s} {'PR-AUC mean':>11s} {'sd':>7s} {'P@1%':>7s} {'R@1%':>7s} {'P@5%':>7s} {'R@5%':>7s}"
            f" | {'f63':>7s} {'f87':>7s} {'f111':>7s} | {'delta':>8s} {'thr':>7s} {'wins':>4s}  rule")
    lines = [head, "-" * len(head)]
    for name, v, fo, r in rows:
        ok = v != NA
        pr = lambda S: f(fo[S]["pr_auc"]) if isinstance(fo.get(S), dict) else "   n/a "  # noqa: E731
        lines.append(
            f"{name:16s} {f(v['pr_auc_mean']) if ok else '   n/a ':>11s} {f(v['pr_auc_sd']) if ok else '   n/a ':>7s}"
            f" {f(v.get('precision_at_top_1pct_mean')) if ok else '   n/a '} {f(v.get('recall_at_top_1pct_mean')) if ok else '   n/a '}"
            f" {f(v.get('precision_at_top_5pct_mean')) if ok else '   n/a '} {f(v.get('recall_at_top_5pct_mean')) if ok else '   n/a '}"
            f" | {pr('63')} {pr('87')} {pr('111')} |"
            + (f" {f(r['valid_delta']) if r['valid_delta'] is not None else '   n/a ':>8s} "
               f"{f(r['threshold']) if r['threshold'] is not None else '   n/a ':>7s} {r['n_wins']:>4d}  "
               f"{'PASS' if r['selected_eligible'] else 'fail'}: {r['reason']}" if r else "  (reference)"))
    lines.append(f"selected: {summary['selected']}")
    return "\n".join(lines)


def select(args) -> dict:
    results_root = Path(args.results_root)
    models_roots = model_roots(args.models_root)
    labels = load_labels(args.labels_file)
    fvalid, ffolds = collect_final(results_root)
    summary = {"reference": "final", "rule": {"margin_sd_multiple": MARGIN_SD, "min_fold_wins": MIN_WINS,
                                              "sd": "std over 3 seeds, ddof=1", "windows": "valid + folds 63-83, 87-107, 111-131"},
               "final": {"valid": fvalid, "folds": ffolds}, "candidates": {}}
    for c in CANDIDATES:
        d = collect_candidate(c, results_root, models_roots, labels)
        d["rule_outcome"] = apply_rule(d, fvalid, ffolds)
        summary["candidates"][c] = d
    ok = [(summary["candidates"][c]["valid"]["pr_auc_mean"], c) for c in CANDIDATES
          if summary["candidates"][c]["rule_outcome"]["selected_eligible"]]
    summary["selected"] = max(ok)[1] if ok else "none"
    summary["selection_note"] = "among passing candidates the highest valid mean PR-AUC; none if no candidate passes"
    out = results_root / "seq_round" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(table(summary))
    print(f"wrote {out}")
    return summary


def test_view(args) -> None:
    from ecommerce_fraud import decision as dc
    from ecommerce_fraud.data import load_joined, time_split
    results_root = Path(args.results_root)
    summ = read_json(results_root / "seq_round" / "summary.json")
    cand = args.test_view
    if summ is None or summ.get("selected") != cand:
        sys.exit(f"refused: summary.json selected {None if summ is None else summ.get('selected')!r}, not {cand!r}")
    models_roots = model_roots(args.models_root)
    _, _, test, _ = time_split(load_joined())
    ids = test["TransactionID"].to_numpy()
    y = test["isFraud"].to_numpy()
    day = (test["TransactionDT"].to_numpy() // 86400).astype(int)
    p_final = scores_path(models_roots, "final", "scores_test.parquet")
    s_final = pd.read_parquet(p_final).set_index("TransactionID")["score"].reindex(ids).to_numpy()
    ref = read_json(results_root / "final" / "metrics.json")["test"]["pr_auc"]
    assert abs(metrics(y, s_final)["pr_auc"] - ref) < 1e-9, "final test scores do not match results/final"

    def test_scores(name):
        p = scores_path(models_roots, name, "scores_test.parquet")
        if p is None:
            sys.exit(f"models/{name}/scores_test.parquet missing: the test-window run of {name} "
                     f"(train/early-stop as in the valid window, score test once) is not done")
        return pd.read_parquet(p)

    if cand == "lgbm_final_emb":
        from ecommerce_fraud import model_emb
        t_ids, t_s = model_emb.train_test_view(args.models_root or models_dir())
        s_c = pd.Series(t_s, index=t_ids).reindex(ids).to_numpy()
    elif cand == "blend":
        b_ids, b_s = rank_blend(pd.DataFrame({"TransactionID": ids, "score": s_final}),
                                test_scores("seq_ssl_test_s0"))
        s_c = pd.Series(b_s, index=b_ids).reindex(ids).to_numpy()
    else:
        s_c = test_scores(f"{cand}_test_s0").set_index("TransactionID")["score"].reindex(ids).to_numpy()
    assert not np.isnan(s_c).any() and not np.isnan(s_final).any()
    boot, point = dc.day_block_bootstrap(y, day, {"final": s_final, cand: s_c}, n_boot=1000, seed=0)
    d = boot[cand] - boot["final"]
    out = {"candidate": cand, "reference": "final", "n_resamples": 1000, "seed": 0,
           "unit": f"{len(np.unique(day))} test days", "test_metrics": {"final": metrics(y, s_final), cand: metrics(y, s_c)},
           "paired_delta_candidate_minus_final": {
               m: {"point": point[cand][j] - point["final"][j], **dc.quantiles(d[:, j]),
                   "share_resamples_positive": float((d[:, j] > 0).mean())}
               for j, m in enumerate(("pr_auc", "precision_at_top_1pct"))}}
    p = results_root / "seq_round" / "test_view.json"
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["paired_delta_candidate_minus_final"], indent=2))
    print(f"wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default=str(WORKTREE / "results"))
    ap.add_argument("--models-root", default=None, help="default: models dir, then the main checkout's models/")
    ap.add_argument("--labels-file", default=None, help="parquet with TransactionID, isFraud (default: raw data)")
    ap.add_argument("--test-view", metavar="CANDIDATE", choices=CANDIDATES)
    args = ap.parse_args()
    test_view(args) if args.test_view else select(args)


if __name__ == "__main__":
    main()
