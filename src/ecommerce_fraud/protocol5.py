"""Round p5 runner and analyzer (PREREGISTRATION p5: Stage A / Stage B). Builds on protocol.py (unchanged).

PYTHONPATH=src; results under <worktree>/results/p5 (--out-dir D).  Reference scores of arm `final`
(seeds 42-46, w1-w6) are read from --ref-dir RD (default R/.worktrees/integ-p4/results/p4, READ-ONLY).

Lookup rule used by every reader and by --run: (arm, window, seed) -> D/<arm>/<w>_s<seed> if it exists,
else RD/final/... for arm `final` only, else "missing".  With --max-rows set (subset debug runs) the RD
lookup is disabled: subset scores never mix with full-data reference scores.

  --run --arms a,b --seeds 42[,..] [--windows w1,..] [--max-rows N] [--max-rounds R]
        Runs pending tasks into D (protocol.run_tasks).  `final` is run only if listed in --arms (and only
        for seeds not present in RD); with --max-rows it is added automatically (RD is not usable then).
  --stage-a --arms a,b [--arm-windows tuned5=w3,w4,w5,w6;...] [--alias name=arm:seed_offset]
        Solo / mix deltas on seed 42 -> D/stage_a.json.
  --make-spec --from stage_a.json --out spec.json      pre-registered Stage B contrasts.
  --contrasts spec.json [--boot-seed S]                paired day-block bootstrap, Holm -> D/stage_b.json
  --power --spec spec.json                             sd of pooled delta and MDE at 80% power -> D/power.json

Spec: list of {name, cand: {arm: [seeds]}, ref: {arm: [seeds]} | {mean_single: arm, seeds: [..]}, windows: [..]}.
A side is the equal-weight mean of per-window percentile ranks of all listed (arm, seed) scores.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ecommerce_fraud import arms as arms_mod
from ecommerce_fraud import protocol as P
from ecommerce_fraud.baseline_lgbm import git_sha
from ecommerce_fraud.data import WORKTREE
from ecommerce_fraud.protocol import (BLOCK_DAYS, TEST_START_DAY, WINDOWS, holm, jread, jwrite, load_labels_days,
                                      log, sort_info, weighted_ap)

R_ROOT = WORKTREE.parent.parent
DEFAULT_OUT = WORKTREE / "results" / "p5"
DEFAULT_REF = R_ROOT / ".worktrees" / "integ-p4" / "results" / "p4"
N_BOOT, BOOT_SEED, ALPHA = 2000, 20261018, 0.05
SEED_A, SEED_REF2 = 42, 43
MAX_ADVANCE, SOLO_BROKEN_GAP, SOLO_BROKEN_WINDOWS = 6, 0.03, 3
B_SEEDS, BIG_SEEDS = list(range(42, 47)), list(range(42, 52))
ST = {"D": DEFAULT_OUT, "RD": DEFAULT_REF, "max_rows": None, "max_rounds": None, "aliases": {}}
SOURCES: dict = {}          # task key -> "D" | "RD"
RESULT_META: dict = {}      # task key -> (max_rows, max_rounds_override)


class Missing(Exception):
    pass


# ------------------------------------------------------------------ lookup
def _complete(d: Path) -> bool:
    return (d / "result.json").exists() and (d / "scores.parquet").exists()


def locate(arm, window, seed, record=False) -> Path:
    """D first, then RD for `final` only (never when max_rows is set). Returns D's path if nothing exists."""
    if arm in ST["aliases"]:
        base, off = ST["aliases"][arm]
        return locate(base, window, seed + off, record)
    d = ST["D"] / arm / f"{window}_s{seed}"
    if _complete(d):
        if record:
            SOURCES[f"{arm}/{window}_s{seed}"] = "D"
        return d
    if arm == "final" and ST["RD"] is not None and ST["max_rows"] is None:
        r = ST["RD"] / arm / f"{window}_s{seed}"
        if _complete(r):
            if record:
                SOURCES[f"{arm}/{window}_s{seed}"] = "RD"
            return r
    return d


def check_dirs():
    D, RD = ST["D"].resolve(), ST["RD"].resolve() if ST["RD"] else None
    if RD is not None and (D == RD or RD in D.parents or D in RD.parents):
        raise SystemExit(f"--out-dir {D} overlaps the read-only reference dir {RD}")
    if D.name == "p4" and D.parent.name == "results":
        raise SystemExit(f"--out-dir {D} is a p4 results folder; refusing")


# ------------------------------------------------------------------ --run
def cmd_run(names, seeds, windows):
    registry = arms_mod.discover()
    names = list(dict.fromkeys(names))
    if ST["max_rows"] and "final" not in names:
        names = ["final"] + names
        log("subset run (--max-rows): RD is disabled; arm `final` added and run into D")
    P.resolve_arms(names, registry)
    P.task_dir = lambda arm, w, s: locate(arm, w, s)      # protocol.task_done / blend_scores see RD finals
    orig = P.run_one

    def guarded(wd, arm, seed):
        d = locate(arm["name"], wd.window, seed)
        D = ST["D"].resolve()
        assert D in d.resolve().parents, f"refusing to write outside the out dir: {d}"
        assert not _complete(d), f"task {d} already exists"
        return orig(wd, arm, seed)
    P.run_one = guarded
    Dres = ST["D"].resolve()
    reused = sorted(f"final/{w}_s{sd}" for w in windows for sd in seeds if "final" in names
                    and _complete(locate("final", w, sd)) and Dres not in locate("final", w, sd).resolve().parents)
    if reused:
        log(f"reusing {len(reused)} final tasks from RD (read-only): {reused[:4]}{' ...' if len(reused) > 4 else ''}")
    frame, errors, skipped = None, {}, {}
    for w in windows:
        if all(P.task_done(n, w, s) for n in names for s in seeds):
            log(f"{w}: all tasks exist")
            continue
        frame = frame or P.Frame()
        _, err, why = P.run_tasks(w, names, seeds, registry, frame)
        errors.update(err)
        if why:
            skipped[w] = why
    P.run_one = orig
    rec = {"time": time.strftime("%F %T"), "git_sha": git_sha(), "arms": names, "seeds": seeds, "windows": windows,
           "max_rows": ST["max_rows"], "max_rounds": ST["max_rounds"], "rd_reused": reused,
           "errors": errors, "skipped": skipped}
    ST["D"].mkdir(parents=True, exist_ok=True)
    with open(ST["D"] / "runs.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    if errors:
        log("ERRORS:", errors)
    return errors


# ------------------------------------------------------------------ readers
_SC: dict = {}


def load_score(arm, w, seed) -> pd.Series:
    d = locate(arm, w, seed, record=True)
    if not _complete(d):
        raise Missing(f"{arm}/{w}_s{seed}")
    key = str(d)
    if key not in _SC:
        _SC[key] = pd.read_parquet(d / "scores.parquet").set_index("TransactionID")["score"]
        r = jread(d / "result.json")
        RESULT_META[f"{arm}/{w}_s{seed}"] = (r.get("max_rows"), r.get("max_rounds_override"))
    return _SC[key]


def load_result(arm, w, seed) -> dict:
    d = locate(arm, w, seed)
    if not (d / "result.json").exists():
        raise Missing(f"{arm}/{w}_s{seed}")
    return jread(d / "result.json")


class Ctx:
    """Labels, days and per-window id universe; ranks are cached per (file, window)."""

    def __init__(self):
        self.y_all, self.day_all = load_labels_days()
        self.ids, self.y, self.blk, self.nb = {}, {}, {}, {}
        self._rk, self._counts = {}, {}
        self.id_checks = {}

    def set_ids(self, w, ids):
        ids = np.asarray(ids)
        if w not in self.ids:
            S, E = WINDOWS[w]
            day = self.day_all.reindex(ids).to_numpy().astype(int)
            y = self.y_all.reindex(ids).to_numpy()
            assert len(set(ids.tolist())) == len(ids), f"{w}: duplicate TransactionIDs"
            if ST["max_rows"] is None:
                assert day.min() >= S and day.max() <= E, f"{w}: score ids outside window days"
            assert day.max() < TEST_START_DAY
            self.ids[w], self.y[w] = ids, y
            self.blk[w] = (day - S) // BLOCK_DAYS
            self.nb[w] = int(self.blk[w].max()) + 1
        return self.ids[w]

    def ranks(self, arm, w, seed):
        """Percentile rank (ties averaged, as bag_check) of one score file, ordered on the window universe."""
        s = load_score(arm, w, seed)
        ids = self.set_ids(w, s.index.to_numpy()) if w not in self.ids else self.ids[w]
        key = (arm, w, seed, str(locate(arm, w, seed)))
        if key not in self._rk:
            assert len(s) == len(ids) and np.array_equal(np.sort(s.index.to_numpy()), np.sort(ids)), \
                f"{arm} {w} s{seed}: TransactionID set differs from the window universe"
            v = s.reindex(ids)
            assert not v.isna().any()
            self._rk[key] = v.rank(pct=True).to_numpy()
        return self._rk[key]

    def counts(self, w, boot_seed, n_boot):
        """Block resample counts per rep; stream identical to bag_check / protocol (SeedSequence.spawn by window)."""
        k = (w, boot_seed, n_boot)
        if k not in self._counts:
            rng = np.random.default_rng(np.random.SeedSequence(boot_seed).spawn(len(WINDOWS))[list(WINDOWS).index(w)])
            nb = self.nb[w]
            draws = rng.integers(0, nb, size=(n_boot, nb))
            self._counts[k] = (draws[:, :, None] == np.arange(nb)).sum(1).astype(float)
        return self._counts[k]


# ------------------------------------------------------------------ Stage A
def parse_arm_windows(s):
    out = {}
    for part in filter(None, (s or "").split(";")):
        a, ws = part.split("=")
        out[a.strip()] = [w for w in ws.split(",") if w]
        bad = [w for w in out[a.strip()] if w not in WINDOWS]
        if bad:
            raise SystemExit(f"--arm-windows: unknown window(s) {bad}")
    return out


def plain_ap(score, y):
    o, ys, ge = sort_info(score, y)
    return weighted_ap(ys, np.ones(len(ys)), ge)


def cmd_stage_a(names, arm_windows, windows_default):
    ctx = Ctx()
    try:
        registry = arms_mod.discover()
    except Exception:
        registry = {}
    out_arms, order_names = {}, [n for n in dict.fromkeys(names) if n != "final"]
    # reference per window: final s42, final s43 (needed once)
    fin = {}
    for n in order_names:
        ws = arm_windows.get(n, windows_default)
        rows, secs, hit, err = {}, 0.0, [], None
        for w in ws:
            try:
                f42, f43 = ctx.ranks("final", w, SEED_A), ctx.ranks("final", w, SEED_REF2)
                a42 = ctx.ranks(n, w, SEED_A)
                res = load_result(n, w, SEED_A)
            except Missing as e:
                err = f"missing {e}"
                break
            y = ctx.y[w]
            if w not in fin:
                fin[w] = (plain_ap(f42, y), plain_ap((f42 + f43) / 2, y))
            ap_f, ap_ff = fin[w]
            ap_a, ap_mix = plain_ap(a42, y), plain_ap((f42 + a42) / 2, y)
            rows[w] = {"ap_final_s42": ap_f, "ap_arm_s42": ap_a, "solo_delta": ap_a - ap_f,
                       "ap_mix": ap_mix, "ap_final_s42_s43": ap_ff, "mix_delta": ap_mix - ap_ff,
                       "best_iteration": res.get("best_iteration"), "hit_max_rounds": res.get("hit_max_rounds"),
                       "seconds": res.get("seconds"), "n_rows": int(len(y)), "n_fraud": int(y.sum())}
            secs += res.get("seconds") or 0.0
            if res.get("hit_max_rounds"):
                hit.append(w)
        if err:
            out_arms[n] = {"error": err, "windows": ws}
            log(f"stage A {n}: {err}")
            continue
        solo = float(np.mean([r["solo_delta"] for r in rows.values()]))
        mix = float(np.mean([r["mix_delta"] for r in rows.values()]))
        far = [w for w, r in rows.items() if r["ap_arm_s42"] < r["ap_final_s42"] - SOLO_BROKEN_GAP]
        broken = []
        if hit:
            broken.append(f"best_iteration hit the round cap in {hit}")
        if len(far) >= SOLO_BROKEN_WINDOWS:
            broken.append(f"solo AP more than {SOLO_BROKEN_GAP} below final in {far}")
        out_arms[n] = {"windows": ws, "solo_delta": solo, "mix_delta": mix, "n_solo_wins": sum(
            r["solo_delta"] > 0 for r in rows.values()), "n_mix_wins": sum(r["mix_delta"] > 0 for r in rows.values()),
            "total_seconds": round(secs, 1), "broken": broken, "per_window": rows,
            "positive": bool(solo > 0 or mix > 0),
            "spec": ({"alias": ST["aliases"][n]} if n in ST["aliases"] else
                     (P.arm_record(registry[n]) if n in registry else None))}
        log(f"stage A {n}: solo {solo:+.5f} mix {mix:+.5f} broken {bool(broken)} secs {secs:.0f}")
    cand = [n for n, a in out_arms.items() if "error" not in a and a["positive"] and not a["broken"]]
    cand.sort(key=lambda n: (-max(out_arms[n]["solo_delta"], out_arms[n]["mix_delta"]), out_arms[n]["total_seconds"]))
    advance = cand[:MAX_ADVANCE]
    all_w = sorted({w for a in out_arms.values() for w in a.get("windows", [])}, key=list(WINDOWS).index)
    out = {"stage": "A", "git_sha": git_sha(), "seed": SEED_A, "arms": out_arms, "advance": advance,
           "not_advanced_positive_over_cap": cand[MAX_ADVANCE:],
           "broken": [n for n, a in out_arms.items() if a.get("broken")],
           "rule": "advance if solo>0 or mix>0, not broken; <=6 by max(solo,mix), ties -> smaller total seconds",
           "seeds": [SEED_A, SEED_REF2], "windows": all_w, "arm_windows": arm_windows,
           "out_dir": str(ST["D"]), "ref_dir": str(ST["RD"]), "max_rows": ST["max_rows"],
           "aliases": {k: list(v) for k, v in ST["aliases"].items()},
           "sources": {"rd_tasks": sorted(k for k, v in SOURCES.items() if v == "RD"),
                       "d_tasks_n": sum(v == "D" for v in SOURCES.values())},
           "n_rows_fraud_per_window": {w: {"n": int(len(ctx.y[w])), "fraud": int(ctx.y[w].sum())} for w in all_w
                                       if w in ctx.y},
           "ids_identical_across_arms": True,    # enforced by assertion in Ctx.ranks
           "result_meta_consistent": check_meta()}
    jwrite(ST["D"] / "stage_a.json", out)
    log("advance:", advance)
    return out


def check_meta():
    metas = {tuple(v) for v in RESULT_META.values()}
    assert len(metas) <= 1, f"result.json files mix max_rows/max_rounds settings: {metas}"
    return True


# ------------------------------------------------------------------ Stage B
def side_items(side):
    if "mean_single" in side:
        return "mean_single", [(side["mean_single"], int(s)) for s in side["seeds"]]
    return "bag", [(a, int(s)) for a, ss in side.items() for s in ss]


def eval_side(ctx, kind, items, w, boot_seed, n_boot, cache):
    key = (kind, tuple(sorted(items)), w, boot_seed, n_boot)
    if key in cache:
        return cache[key]
    y, blk, counts = ctx.y[w], ctx.blk[w], ctx.counts(w, boot_seed, n_boot)
    scores = [np.mean([ctx.ranks(a, w, s) for a, s in items], axis=0)] if kind == "bag" else \
        [ctx.ranks(a, w, s) for a, s in items]
    pts, boots = [], []
    for sc in scores:
        o, ys, ge = sort_info(sc, y)
        bo = blk[o]
        ones = np.ones(len(ys))
        pts.append(weighted_ap(ys, ones, ge))
        boots.append(np.array([weighted_ap(ys, counts[r][bo], ge) for r in range(n_boot)]))
    cache[key] = (float(np.mean(pts)), np.mean(boots, axis=0))
    return cache[key]


def need_wins(n):
    return math.ceil(2 * n / 3)          # 6 -> 4, 4 -> 3


def run_contrasts(spec, boot_seed, n_boot, strict=True):
    ctx, cache, res = Ctx(), {}, {}
    arms_used = set()
    for c in spec:
        kc, ic = side_items(c["cand"])
        kr, ir = side_items(c["ref"])
        assert kc == "bag"
        arms_used |= {a for a, _ in ic + ir}
        miss = []
        for w in c["windows"]:
            assert w in WINDOWS, w
            for a, s in ic + ir:
                if not _complete(locate(a, w, s)):
                    miss.append(f"{a}/{w}_s{s}")
        if miss:
            if strict:
                raise SystemExit(f"contrast {c['name']}: missing tasks {miss[:8]}{' ...' if len(miss) > 8 else ''}")
            res[c["name"]] = {"status": "missing", "missing": miss}
            continue
        # window universe from the reference's first score file (deterministic row order)
        for w in c["windows"]:
            a0, s0 = ir[0]
            ctx.set_ids(w, load_score(a0, w, s0).index.to_numpy())
        per_w, boot_w = {}, []
        for w in c["windows"]:
            pc, bc = eval_side(ctx, "bag", ic, w, boot_seed, n_boot, cache)
            pr_, br = eval_side(ctx, kr, ir, w, boot_seed, n_boot, cache)
            per_w[w] = {"ap_cand": pc, "ap_ref": pr_, "delta": pc - pr_, "n_rows": int(len(ctx.y[w])),
                        "n_fraud": int(ctx.y[w].sum())}
            boot_w.append(bc - br)
        pooled_boot = np.mean(boot_w, axis=0)
        deltas = [v["delta"] for v in per_w.values()]
        n = len(deltas)
        res[c["name"]] = {
            "status": "ok", "cand": c["cand"], "ref": c["ref"], "windows": c["windows"], "per_window": per_w,
            "pooled_delta": float(np.mean(deltas)), "n_wins": int(sum(d > 0 for d in deltas)), "n_windows": n,
            "wins_needed": need_wins(n), "ci95": [float(x) for x in np.percentile(pooled_boot, [2.5, 97.5])],
            "sd": float(pooled_boot.std(ddof=1)), "boot_mean": float(pooled_boot.mean()),
            "p_one_sided": float((pooled_boot <= 0).mean())}
        log(f"contrast {c['name']}: pooled {res[c['name']]['pooled_delta']:+.5f} wins {res[c['name']]['n_wins']}/{n} "
            f"p {res[c['name']]['p_one_sided']:.4f}")
    return ctx, res, arms_used


def arm_specs(arms_used):
    try:
        registry = arms_mod.discover()
    except Exception:
        registry = {}
    out = {}
    for a in sorted(arms_used):
        if a in ST["aliases"]:
            out[a] = {"alias": list(ST["aliases"][a])}
        elif a in registry:
            out[a] = P.arm_record(registry[a])
        else:
            out[a] = None
    return out


def cmd_contrasts(spec_path, boot_seed, n_boot):
    from scipy.stats import norm
    spec = json.loads(Path(spec_path).read_text())
    names = [c["name"] for c in spec]
    assert len(set(names)) == len(names), "duplicate contrast names"
    ctx, res, arms_used = run_contrasts(spec, boot_seed, n_boot)
    adj = holm({n: r["p_one_sided"] for n, r in res.items()})
    m = len(res)
    kz = float(norm.ppf(1 - ALPHA / max(1, m)) + norm.ppf(0.8))
    for n, r in res.items():
        r["holm_p"] = adj[n]
        r["passes"] = bool(adj[n] < ALPHA and r["n_wins"] >= r["wins_needed"])
        r["mde_80_holm_worst"] = kz * r["sd"]
    passing = [n for n in res if res[n]["passes"]]
    winner = max(passing, key=lambda n: res[n]["pooled_delta"]) if passing else None
    best = max(res, key=lambda n: res[n]["pooled_delta"])
    out = {"stage": "B", "git_sha": git_sha(), "spec": spec, "contrasts": res, "passing": passing, "winner": winner,
           "winner_rule": "highest pooled delta among passing contrasts; none -> final_bag5 stays",
           "pass_rule": f"Holm p < {ALPHA} over {m} contrasts AND windows won >= ceil(2n/3) (6 of 6-window: 4; 4-window: 3)",
           "bootstrap": {"n_reps": n_boot, "block_days": BLOCK_DAYS, "seed": boot_seed, "one_sided": True},
           "test_view_proposal": bool(winner and res[winner]["pooled_delta"] >= P.TEST_PROPOSAL_DELTA),
           "arm_specs": arm_specs(arms_used), "seeds": sorted({sd for c in spec for side in (c["cand"], c["ref"]) for _, sd in side_items(side)[1]}),
           "windows": sorted({w for c in spec for w in c["windows"]}, key=list(WINDOWS).index),
           "out_dir": str(ST["D"]), "ref_dir": str(ST["RD"]), "max_rows": ST["max_rows"],
           "aliases": {k: list(v) for k, v in ST["aliases"].items()},
           "sources": {"rd_tasks": sorted(k for k, v in SOURCES.items() if v == "RD"),
                       "d_tasks_n": sum(v == "D" for v in SOURCES.values())},
           "n_rows_fraud_per_window": {w: {"n": int(len(ctx.y[w])), "fraud": int(ctx.y[w].sum())} for w in ctx.y},
           "ids_identical_cand_ref": True,       # enforced by assertion in Ctx.ranks for every score file used
           "result_meta_consistent": check_meta()}
    if not passing:
        out["negative_finding"] = (
            "Under ADR-005 extended to bags (6 windows x 5 seeds, paired day-block bootstrap, Holm), no candidate "
            f"beat the 5-seed bagged LightGBM; final_bag5 stays (best pooled delta = {res[best]['pooled_delta']:+.5f}, "
            f"windows won = {res[best]['n_wins']}/{res[best]['n_windows']}).")
    jwrite(ST["D"] / "stage_b.json", out)
    log("winner:", winner)
    return out


def cmd_power(spec_path, boot_seed, n_boot):
    from scipy.stats import norm
    spec = json.loads(Path(spec_path).read_text())
    ctx, res, arms_used = run_contrasts(spec, boot_seed, n_boot, strict=False)
    m = len(spec)
    k1 = float(norm.ppf(1 - ALPHA) + norm.ppf(0.8))
    kh = float(norm.ppf(1 - ALPHA / max(1, m)) + norm.ppf(0.8))
    per = {}
    for n, r in res.items():
        if r["status"] != "ok":
            per[n] = r
            continue
        per[n] = {"status": "ok", "windows": r["windows"], "sd_pooled_delta": r["sd"], "point_delta": r["pooled_delta"],
                  "mde_80_alpha_0.05": k1 * r["sd"], "mde_80_holm_worst": kh * r["sd"]}
    out = {"git_sha": git_sha(), "n_contrasts": m,
           "formula": f"MDE = (z_(1-alpha) + z_0.8) x sd(pooled delta), one-sided; alpha 0.05 -> {k1:.3f}, "
                      f"Holm worst case alpha/{m} -> {kh:.3f}",
           "bootstrap": {"n_reps": n_boot, "seed": boot_seed, "block_days": BLOCK_DAYS},
           "per_contrast": per, "arm_specs": arm_specs(arms_used), "out_dir": str(ST["D"]), "ref_dir": str(ST["RD"]),
           "max_rows": ST["max_rows"], "aliases": {k: list(v) for k, v in ST["aliases"].items()},
           "n_rows_fraud_per_window": {w: {"n": int(len(ctx.y[w])), "fraud": int(ctx.y[w].sum())} for w in ctx.y}}
    jwrite(ST["D"] / "power.json", out)
    for n, r in per.items():
        if r["status"] == "ok":
            log(f"power {n}: sd {r['sd_pooled_delta']:.5f} MDE80 {r['mde_80_alpha_0.05']:.5f} "
                f"(Holm {r['mde_80_holm_worst']:.5f})")
        else:
            log(f"power {n}: missing {r['missing'][:3]}...")
    return out


def cmd_make_spec(stage_a_path, out_path):
    sa = json.loads(Path(stage_a_path).read_text())
    spec = []
    for a in sa["advance"]:
        ws = sa["arms"][a]["windows"]
        spec.append({"name": f"solo_{a}", "cand": {a: B_SEEDS}, "ref": {"final": B_SEEDS}, "windows": ws})
        spec.append({"name": f"mix_{a}", "cand": {"final": B_SEEDS, a: B_SEEDS}, "ref": {"final": BIG_SEEDS},
                     "windows": ws})
    spec.append({"name": "bagscale_final10_vs_final5", "cand": {"final": BIG_SEEDS}, "ref": {"final": B_SEEDS},
                 "windows": list(WINDOWS)})
    try:
        present = "final_jitter" in arms_mod.discover()
    except Exception:
        present = False
    present = present or "final_jitter" in sa.get("arms", {})
    if present:
        spec.append({"name": "bagscale_jitter5_vs_final5", "cand": {"final_jitter": B_SEEDS},
                     "ref": {"final": B_SEEDS}, "windows": list(WINDOWS)})
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(spec, indent=2) + "\n")
    log(f"wrote {out_path}: {len(spec)} contrasts")
    return spec


# ------------------------------------------------------------------ cli
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in ("--run", "--stage-a", "--make-spec", "--power"):
        ap.add_argument(f, action="store_true")
    ap.add_argument("--contrasts", default=None, help="spec.json")
    ap.add_argument("--spec", default=None, help="--power: spec.json")
    ap.add_argument("--from", dest="from_", default=None, help="--make-spec: stage_a.json")
    ap.add_argument("--out", default=None, help="--make-spec: output spec.json")
    ap.add_argument("--arms", default="")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--windows", default=",".join(WINDOWS))
    ap.add_argument("--arm-windows", default="", help="e.g. 'tuned5=w3,w4,w5,w6;other=w1,w2'")
    ap.add_argument("--alias", action="append", default=[], help="debug: name=arm:seed_offset (reads arm at seed+offset)")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--ref-dir", default=str(DEFAULT_REF))
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--max-rounds", type=int, default=None)
    ap.add_argument("--boot-seed", type=int, default=BOOT_SEED)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    ST.update(D=Path(a.out_dir), RD=Path(a.ref_dir) if a.ref_dir else None, max_rows=a.max_rows,
              max_rounds=a.max_rounds)
    for al in a.alias:
        nm, rest = al.split("=")
        base, off = rest.split(":")
        ST["aliases"][nm] = (base, int(off))
    check_dirs()
    P.CFG.update(out_dir=ST["D"], max_rows=a.max_rows, max_rounds=a.max_rounds)
    names = [n for n in a.arms.split(",") if n]
    windows = [w for w in a.windows.split(",") if w]
    bad = [w for w in windows if w not in WINDOWS]
    if bad:
        ap.error(f"unknown window(s) {bad}")
    rc = 0
    if a.run:
        rc |= bool(cmd_run(names, [int(s) for s in a.seeds.split(",") if s], windows))
    if a.stage_a:
        cmd_stage_a(names, parse_arm_windows(a.arm_windows), windows)
    if a.make_spec:
        if not (a.from_ and a.out):
            ap.error("--make-spec needs --from and --out")
        cmd_make_spec(a.from_, a.out)
    if a.contrasts:
        cmd_contrasts(a.contrasts, a.boot_seed, a.n_boot)
    if a.power:
        if not a.spec:
            ap.error("--power needs --spec")
        cmd_power(a.spec, a.boot_seed, a.n_boot)
    if not (a.run or a.stage_a or a.make_spec or a.contrasts or a.power):
        ap.error("pick one of --run --stage-a --make-spec --contrasts --power")
    log("done")
    raise SystemExit(int(rc))


if __name__ == "__main__":
    main()
