"""Amount-aware alert ranking evaluation (decision layer, no training).

Policies: rank by p, or by p_cal * amount^a (a in 0.25, 0.5, 0.75, 1.0), alerting the top-K rows per
day (K = 15, 30, 60). p_cal = isotonic fit on the PREVIOUS window's scores and labels (labels are only
the calibration target). Paired day-block bootstrap of each policy minus rank-by-p.
Choice of a on the earlier half of the calibrated windows, reported on the later half (PREREGISTRATION
Part 1). Test rows (day >= 141) are never read into the evaluation: asserted.

  python -m ecommerce_fraud.alert_rank --source existing
  python -m ecommerce_fraud.alert_rank --source p4 --p4-dir <dir> [--seed 42]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from ecommerce_fraud.artifacts import models_dir


TEST_START_DAY = 141
A_GRID = [0.25, 0.5, 0.75, 1.0]
KS = [15, 30, 60]
HEADLINE_K = 30
RECALL_FLOOR = 0.8
BLOCK = 7
N_BOOT = 2000
SEED = 0
POLICIES = ["p"] + [f"a{a}" for a in A_GRID]
METRICS = ["dollar_share", "count_recall", "precision", "dollars_per_alert"]

ROOT = Path(__file__).resolve().parents[2]
# true out-of-time scores of `final`: backtest folds (train days 1..S-8, scored S..S+20). models/final and
# models/imp_screen_final scores_valid are identical (max abs diff 0) and cover days 111-134, which overlap
# the d111 fold with a different model: reported as a supplementary window, not chained or pooled.
EXISTING_CHAIN = [("d63", "backtest_d63_final"), ("d87", "backtest_d87_final"), ("d111", "backtest_d111_final")]
EXISTING_EXTRA = ("valid", "final", "d87")  # name, folder, calibrator window


def joined_path() -> Path:
    p = ROOT / "data" / "processed" / "ieee_cis_train_joined.parquet"
    return p


def load_truth() -> pd.DataFrame:
    m = pd.read_parquet(joined_path(), columns=["TransactionID", "isFraud", "TransactionDT", "TransactionAmt"])
    m["day"] = (m["TransactionDT"] // 86400).astype(int)
    return m.drop(columns="TransactionDT")


def build_window(name: str, score_file: Path, truth: pd.DataFrame) -> dict:
    s = pd.read_parquet(score_file, columns=["TransactionID", "score"])
    d = s.merge(truth, on="TransactionID", how="left", validate="one_to_one")
    assert d["isFraud"].notna().all(), f"{name}: scores without truth rows"
    assert int(d["day"].max()) < TEST_START_DAY, f"{name}: test rows (day >= {TEST_START_DAY}) present"
    return {"name": name, "file": str(score_file), "y": d["isFraud"].to_numpy(float), "amt": d["TransactionAmt"].to_numpy(float),
            "day": d["day"].to_numpy(int), "score": d["score"].to_numpy(float)}


def calibrate(prev: dict, cur: dict) -> np.ndarray:
    # y_min > 0 keeps p_cal * amount from collapsing to ties at 0
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1.0).fit(prev["score"], prev["y"])
    return iso.predict(cur["score"])


def policy_ranks(w: dict) -> dict:
    base = w["pcal"] if w["pcal"] is not None else w["score"]
    r = {"p": w["score"]}
    for a in A_GRID:
        r[f"a{a}"] = base * np.power(w["amt"], a)
    return r


def day_tables(w: dict) -> dict:
    """Per day, policy, K: (fraud dollars caught, fraud caught, alerts). Plus per-day totals."""
    ranks = policy_ranks(w)
    days = np.unique(w["day"])
    y, amt = w["y"], w["amt"]
    tot = np.zeros((len(days), 3))  # fraud dollars, fraud count, rows
    caught = np.zeros((len(POLICIES), len(KS), len(days), 3))
    for j, d in enumerate(days):
        idx = np.flatnonzero(w["day"] == d)
        tot[j] = [(y[idx] * amt[idx]).sum(), y[idx].sum(), len(idx)]
        for pi, pol in enumerate(POLICIES):
            order = idx[np.argsort(-ranks[pol][idx], kind="stable")]
            for ki, k in enumerate(KS):
                sel = order[:k]
                caught[pi, ki, j] = [(y[sel] * amt[sel]).sum(), y[sel].sum(), len(sel)]
    return {"days": days, "tot": tot, "caught": caught}


def metrics_from(c: np.ndarray, t: np.ndarray) -> dict:
    """c: (..., 3) summed caught; t: (3,) summed totals."""
    return {"dollar_share": c[..., 0] / t[0], "count_recall": c[..., 1] / t[1],
            "precision": c[..., 1] / np.maximum(c[..., 2], 1), "dollars_per_alert": c[..., 0] / np.maximum(c[..., 2], 1)}


def pooled(tabs: list[dict]):
    c = sum(t["caught"].sum(axis=2) for t in tabs)  # (P,K,3)
    t = sum(t["tot"].sum(axis=0) for t in tabs)
    return c, t


def table_for(tabs: list[dict]) -> dict:
    c, t = pooled(tabs)
    m = metrics_from(c, t)
    return {pol: {str(k): {mn: float(m[mn][pi, ki]) for mn in METRICS} for ki, k in enumerate(KS)} for pi, pol in enumerate(POLICIES)}


def block_ids(days: np.ndarray) -> list[np.ndarray]:
    first = days.min()
    b = (days - first) // BLOCK
    return [np.flatnonzero(b == i) for i in range(b.max() + 1)]


def bootstrap(tabs: list[dict], n_boot=N_BOOT, seed=SEED) -> dict:
    """Paired 7-day-block bootstrap (resampled within each window, same resample for every policy).
    Returns CIs of metric(policy) - metric(p) and of metric(p) itself, pooled over the given windows."""
    rng = np.random.default_rng(seed)
    blocks = [block_ids(t["days"]) for t in tabs]
    P, K = len(POLICIES), len(KS)
    diffs = {mn: np.zeros((n_boot, P, K)) for mn in METRICS}
    levels = {mn: np.zeros((n_boot, P, K)) for mn in METRICS}
    for r in range(n_boot):
        c = np.zeros((P, K, 3))
        t = np.zeros(3)
        for tab, bl in zip(tabs, blocks):
            pick = rng.integers(0, len(bl), len(bl))
            idx = np.concatenate([bl[i] for i in pick])
            c += tab["caught"][:, :, idx].sum(axis=2)
            t += tab["tot"][idx].sum(axis=0)
        m = metrics_from(c, t)
        for mn in METRICS:
            levels[mn][r] = m[mn]
            diffs[mn][r] = m[mn] - m[mn][0:1]
    out = {}
    for pi, pol in enumerate(POLICIES):
        out[pol] = {}
        for ki, k in enumerate(KS):
            out[pol][str(k)] = {mn: {"level_ci95": [float(x) for x in np.percentile(levels[mn][:, pi, ki], [2.5, 97.5])],
                                     "diff_vs_p_ci95": [float(x) for x in np.percentile(diffs[mn][:, pi, ki], [2.5, 97.5])]}
                                for mn in METRICS}
    return out


def choose_a(tabs: list[dict]) -> dict:
    c, t = pooled(tabs)
    m = metrics_from(c, t)
    ki = KS.index(HEADLINE_K)
    floor = RECALL_FLOOR * m["count_recall"][0, ki]
    cand = [(m["dollar_share"][pi, ki], A_GRID[pi - 1]) for pi in range(1, len(POLICIES)) if m["count_recall"][pi, ki] >= floor]
    flagged = not cand
    if flagged:
        cand = [(m["dollar_share"][pi, ki], A_GRID[pi - 1]) for pi in range(1, len(POLICIES))]
    a = max(cand)[1]
    return {"a": a, "flagged_no_a_meets_recall_floor": flagged, "count_recall_floor": float(floor),
            "p_count_recall_k30": float(m["count_recall"][0, ki]),
            "selection_table_k30": {POLICIES[pi]: {"dollar_share": float(m["dollar_share"][pi, ki]),
                                                   "count_recall": float(m["count_recall"][pi, ki])} for pi in range(len(POLICIES))}}


def csv_rows(label: str, tabs: list[dict], boot: dict) -> list[list]:
    tb = table_for(tabs)
    rows = []
    for pol in POLICIES:
        for k in KS:
            v, b = tb[pol][str(k)], boot[pol][str(k)]
            rows.append([label, pol, k] + [round(v[mn], 4) for mn in METRICS] +
                        [round(b["dollar_share"]["diff_vs_p_ci95"][i], 4) for i in (0, 1)] +
                        [round(b["count_recall"]["diff_vs_p_ci95"][i], 4) for i in (0, 1)])
    return rows


def get_windows(args) -> tuple[list[dict], list[dict], dict]:
    truth = load_truth()
    models = Path(args.models_dir)
    chain, extra, info = [], [], {}
    if args.source == "existing":
        for name, folder in EXISTING_CHAIN:
            chain.append(build_window(name, models / folder / "scores_valid.parquet", truth))
        n, folder, calib = EXISTING_EXTRA
        w = build_window(n, models / folder / "scores_valid.parquet", truth)
        w["calib_from"] = calib
        extra.append(w)
        info["note"] = ("chain = backtest folds d63/d87/d111 of `final` (true out-of-time); `valid` (models/final scores_valid, "
                        "identical to imp_screen_final; days 111-134) overlaps fold d111 with a different model, so it is a "
                        "supplementary window calibrated on d87 and excluded from pooling and from the choice of a")
    else:
        for i in range(1, 7):
            f = Path(args.p4_dir) / "final" / f"w{i}_s{args.seed}" / "scores.parquet"
            chain.append(build_window(f"w{i}", f, truth))
    del truth
    return chain, extra, info


def run(args) -> dict:
    chain, extra, info = get_windows(args)
    for i, w in enumerate(chain):
        w["pcal"] = None if i == 0 else calibrate(chain[i - 1], w)
        w["calibrated"] = i > 0
    for w in extra:
        prev = next(c for c in chain if c["name"] == w["calib_from"])
        w["pcal"] = calibrate(prev, w)
        w["calibrated"] = True
    for w in chain + extra:
        w["tab"] = day_tables(w)
    cal = [w for w in chain if w["calibrated"]]
    assert len(cal) >= 2, "need >= 2 calibrated windows to choose a on an earlier half and report on a later half"
    h = len(cal) // 2
    early, late = cal[:h], cal[h:]
    sel = choose_a([w["tab"] for w in early])
    res = {"source": args.source, "ks": KS, "a_grid": A_GRID, "n_boot": N_BOOT, "block_days": BLOCK, "info": info,
           "windows": {}, "pooled": {}, "selection": {"selected_on": [w["name"] for w in early], "reported_on": [w["name"] for w in late], **sel}}
    for w in chain + extra:
        t = w["tab"]
        res["windows"][w["name"]] = {"file": w["file"], "n_rows": int(len(w["y"])), "n_fraud": int(w["y"].sum()),
                                     "days": [int(t["days"].min()), int(t["days"].max())], "calibrated": w["calibrated"],
                                     "table": table_for([t])}
    sets = {"all_calibrated": cal, "earlier_half": early, "later_half": late}
    rows = []
    for label, ws in sets.items():
        tabs = [w["tab"] for w in ws]
        boot = bootstrap(tabs)
        res["pooled"][label] = {"windows": [w["name"] for w in ws], "table": table_for(tabs), "bootstrap": boot}
        rows += csv_rows(label, tabs, boot)
    res["selection"]["later_half_chosen_vs_p_k30"] = {
        pol: res["pooled"]["later_half"]["table"][pol][str(HEADLINE_K)] for pol in ("p", f"a{sel['a']}")}
    res["selection"]["later_half_ci_k30"] = res["pooled"]["later_half"]["bootstrap"][f"a{sel['a']}"][str(HEADLINE_K)]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.source}.json").write_text(json.dumps(res, indent=2))
    with open(out_dir / f"{args.source}.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["set", "policy", "K", "dollar_share", "count_recall", "precision", "dollars_per_alert",
                     "dollar_diff_lo", "dollar_diff_hi", "count_diff_lo", "count_diff_hi"])
        wr.writerows(rows)
    return res


def summarize(res: dict) -> None:
    print(f"source={res['source']}  K={res['ks']}  bootstrap {res['n_boot']} reps, {res['block_days']}-day blocks")
    if res["info"].get("note"):
        print("note:", res["info"]["note"])
    for name, w in res["windows"].items():
        print(f"window {name}: days {w['days']}, rows {w['n_rows']}, fraud {w['n_fraud']}, calibrated={w['calibrated']}")
    s = res["selection"]
    print(f"\nchoice of a on {s['selected_on']} (K=30, count recall >= {RECALL_FLOOR} x p's {s['p_count_recall_k30']:.3f}): "
          f"a={s['a']}" + ("  [FLAG: no a met the recall floor, best dollar share reported]" if s["flagged_no_a_meets_recall_floor"] else ""))
    for label in ("earlier_half", "later_half", "all_calibrated"):
        P = res["pooled"][label]
        print(f"\n== pooled {label} {P['windows']}")
        print(f"{'policy':8s} {'K':>3s} {'$share':>7s} {'$diff CI95':>17s} {'cnt rec':>8s} {'cnt diff CI95':>17s} {'prec':>6s} {'$/alert':>8s}")
        for pol in POLICIES:
            for k in KS:
                v, b = P["table"][pol][str(k)], P["bootstrap"][pol][str(k)]
                d1, d2 = b["dollar_share"]["diff_vs_p_ci95"], b["count_recall"]["diff_vs_p_ci95"]
                dd = "" if pol == "p" else f"[{d1[0]:+.3f},{d1[1]:+.3f}]"
                cc = "" if pol == "p" else f"[{d2[0]:+.3f},{d2[1]:+.3f}]"
                print(f"{pol:8s} {k:3d} {v['dollar_share']:7.3f} {dd:>17s} {v['count_recall']:8.3f} {cc:>17s} {v['precision']:6.3f} {v['dollars_per_alert']:8.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["existing", "p4"], required=True)
    ap.add_argument("--p4-dir")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models-dir", default=str(models_dir()))
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "alert_rank"))
    args = ap.parse_args()
    if args.source == "p4" and not args.p4_dir:
        ap.error("--p4-dir required for --source p4")
    summarize(run(args))


if __name__ == "__main__":
    main()
