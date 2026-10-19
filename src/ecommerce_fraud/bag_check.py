"""Round p4 seed-bagging check (PREREGISTRATION amendment 2026-10-17). No training.

python -m ecommerce_fraud.bag_check   -> results/p4/bag_check.json
Candidate final_bag5 = rank average of final's 5 seeds per window. Primary: final_bag5 vs the
single-seed final (delta = AP(bag) - mean_s AP(seed s)); secondary: bagged seg_noid_blend vs
final_bag5. Paired 7-day-block bootstrap shared with protocol.cmd_analyze, Holm over both contrasts.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ecommerce_fraud.protocol import (BLOCK_DAYS, BOOT_SEED, N_BOOT, TEST_START_DAY, WINDOWS, holm,
                                      load_labels_days, out_dir, sort_info, task_dir, weighted_ap)

SEEDS = (42, 43, 44, 45, 46)


def ranks(arm, w, ids):
    out = []
    for s in SEEDS:
        sc = pd.read_parquet(task_dir(arm, w, s) / "scores.parquet").set_index("TransactionID")["score"].reindex(ids)
        assert not sc.isna().any(), f"{arm} {w} s{s} misses rows"
        out.append(sc.rank(pct=True).to_numpy())
    return out


def main():
    y_all, day_all = load_labels_days()
    rngs = np.random.SeedSequence(BOOT_SEED).spawn(len(WINDOWS))
    boot = {"bag_vs_single": np.zeros((len(WINDOWS), N_BOOT)), "blendbag_vs_bag": np.zeros((len(WINDOWS), N_BOOT))}
    point = {k: [] for k in boot}
    for i, (w, (S, E)) in enumerate(WINDOWS.items()):
        ids = pd.read_parquet(task_dir("final", w, SEEDS[0]) / "scores.parquet")["TransactionID"].to_numpy()
        day = day_all.reindex(ids).to_numpy().astype(int)
        y = y_all.reindex(ids).to_numpy()
        assert day.min() >= S and day.max() <= E and day.max() < TEST_START_DAY
        blk = (day - S) // BLOCK_DAYS
        nb = int(blk.max()) + 1
        rng = np.random.default_rng(rngs[i])
        draws = rng.integers(0, nb, size=(N_BOOT, nb))
        counts = (draws[:, :, None] == np.arange(nb)).sum(1).astype(float)
        fr = ranks("final", w, ids)
        bag = np.mean(fr, axis=0)
        bbag = np.mean(ranks("seg_noid_blend", w, ids), axis=0)
        infos = {"bag": sort_info(bag, y), "bbag": sort_info(bbag, y)}
        infos.update({f"s{k}": sort_info(r, y) for k, r in enumerate(fr)})
        ones = np.ones(len(y))
        ap1 = {k: weighted_ap(ys, ones[o], ge) for k, (o, ys, ge) in infos.items()}
        single = np.mean([ap1[f"s{k}"] for k in range(len(SEEDS))])
        point["bag_vs_single"].append(ap1["bag"] - single)
        point["blendbag_vs_bag"].append(ap1["bbag"] - ap1["bag"])
        for r in range(N_BOOT):
            c = counts[r]
            ap = {k: weighted_ap(ys, c[blk[o]], ge) for k, (o, ys, ge) in infos.items()}
            sg = np.mean([ap[f"s{k}"] for k in range(len(SEEDS))])
            boot["bag_vs_single"][i, r] = ap["bag"] - sg
            boot["blendbag_vs_bag"][i, r] = ap["bbag"] - ap["bag"]
        print(f"{w}: bag-single {point['bag_vs_single'][-1]:+.5f}  blendbag-bag {point['blendbag_vs_bag'][-1]:+.5f}", flush=True)
    pvals, res = {}, {}
    for k in boot:
        pooled = boot[k].mean(0)
        pvals[k] = float((pooled <= 0).mean())
        d = point[k]
        res[k] = {"per_window": [float(x) for x in d], "pooled_delta": float(np.mean(d)),
                  "n_wins": int(sum(x > 0 for x in d)), "ci95": [float(x) for x in np.percentile(pooled, [2.5, 97.5])],
                  "sd": float(pooled.std()), "p_one_sided": pvals[k]}
    adj = holm(pvals)
    for k in res:
        res[k]["holm_p"] = adj[k]
        res[k]["passes"] = bool(adj[k] < 0.05 and res[k]["n_wins"] >= 4)
    res["test_view_proposal"] = bool(res["bag_vs_single"]["passes"] and res["bag_vs_single"]["pooled_delta"] >= 0.004)
    res["meta"] = {"seeds": SEEDS, "n_boot": N_BOOT, "block_days": BLOCK_DAYS, "boot_seed": BOOT_SEED,
                   "candidate": "final_bag5 = rank average of final seeds 42-46"}
    p = out_dir() / "bag_check.json"
    p.write_text(json.dumps(res, indent=2))
    for k in ("bag_vs_single", "blendbag_vs_bag"):
        r = res[k]
        print(f"{k}: pooled {r['pooled_delta']:+.5f} wins {r['n_wins']}/6 CI {r['ci95']} holm {r['holm_p']:.4f} pass {r['passes']}")
    print("test view proposal:", res["test_view_proposal"])


if __name__ == "__main__":
    main()
