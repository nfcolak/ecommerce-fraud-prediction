"""Per-transaction sequence arrays for the transformer round (history of the same card).

Protocol (ADR-002 / ADR-003): no labels and no absolute-time field in the inputs, history rows
are strictly earlier (dt < own dt), codes and normalization are fitted on rows with
dt <= fit_end_dt only. The producer here writes one .npz that `seq_train.py` reads.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

from ecommerce_fraud import data as data_mod
from ecommerce_fraud.features_history import DAY, make_keys

L = 31
CLIP = 10.0

C_COLS = [f"C{i}" for i in range(1, 15)]
D_COLS = [f"D{i}" for i in range(1, 16)]
D_RELATIVE = ("D1", "D4", "D10", "D15")  # converted like D1n/D4n/D10n/D15n: day - D

# categorical field -> cap on distinct fit-window values (None = keep every fit-window value)
CAT_SPEC = {
    "ProductCD": None, "card1": 2000, "card2": None, "card3": None, "card4": None,
    "card5": None, "card6": None, "addr1": None, "addr2": None,
    "P_emaildomain": None, "R_emaildomain": None,
    **{f"M{i}": None for i in range(1, 10)},
    "DeviceType": None, "DeviceInfo": 500, "id_30": 300, "id_31": 300, "id_33": 300,
}
FORBIDDEN_INPUT_NAMES = {"isFraud", "TransactionDT", "day"}


@dataclass
class SeqArrays:
    ids: np.ndarray
    dt: np.ndarray
    y: np.ndarray
    labeled: np.ndarray
    num: np.ndarray
    num_missing: np.ndarray
    cat: np.ndarray
    cat_cardinality: np.ndarray
    hist: np.ndarray
    num_names: np.ndarray
    cat_names: np.ndarray
    meta: str


def load_unlabeled(raw_dir: Path = data_mod.RAW_DIR) -> pd.DataFrame:
    """Kaggle test_transaction + test_identity; identity columns id-NN renamed to id_NN."""
    trans = pd.read_csv(raw_dir / "test_transaction.csv")
    ident = pd.read_csv(raw_dir / "test_identity.csv")
    ident = ident.rename(columns={c: c.replace("-", "_") for c in ident.columns if c.startswith("id-")})
    df = trans.merge(ident, on="TransactionID", how="left")
    df["isFraud"] = np.nan
    return df.copy()


# ---------------------------------------------------------------------------------- numerics
def _numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Raw (un-normalized) numeric fields. D1/D4/D10/D15 become day - D (as D1n etc.)."""
    day = df["TransactionDT"] // DAY  # only used to build the relative D fields, never emitted
    out = {
        "amt_log": np.log1p(df["TransactionAmt"]),
        "amt_cents": df["TransactionAmt"] % 1,
    }
    for c in C_COLS:
        out[c] = df[c]
    for d in D_COLS:
        out[f"{d}n" if d in D_RELATIVE else d] = (day - df[d]) if d in D_RELATIVE else df[d]
    return pd.DataFrame(out, index=df.index).astype(np.float64)


def _fit_numeric(frame: pd.DataFrame, fit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sub = frame[fit]
    med = sub.median().to_numpy(np.float64)
    q1, q3 = sub.quantile(0.25).to_numpy(np.float64), sub.quantile(0.75).to_numpy(np.float64)
    scale = q3 - q1
    std = sub.std().to_numpy(np.float64)
    scale = np.where(scale > 0, scale, np.where(std > 0, std, 1.0))
    med = np.where(np.isfinite(med), med, 0.0)
    scale = np.where(np.isfinite(scale), scale, 1.0)
    return med, scale


# -------------------------------------------------------------------------------- categoricals
def _fit_codes(col: pd.Series, fit: np.ndarray, cap: int | None) -> pd.Index:
    """Values (most frequent first) that get codes 1..len; everything else is 0."""
    vc = col[fit].value_counts(dropna=True)  # sorted by count desc, ties in first-seen order
    if cap is not None:
        vc = vc.iloc[:cap]
    return vc.index


def _apply_codes(col: pd.Series, vocab: pd.Index) -> np.ndarray:
    idx = vocab.get_indexer(col)  # -1 for NaN / unseen
    return (idx + 1).astype(np.int32)


# --------------------------------------------------------------------------------- history
def _uid_codes(df: pd.DataFrame) -> np.ndarray:
    """Factorized make_keys()['uid']; -1 when any uid component (card1, addr1, D1) is NaN."""
    uid = make_keys(df)["uid"]
    codes = pd.factorize(uid)[0].astype(np.int64)
    miss = (df["card1"].isna() | df["addr1"].isna() | df["D1"].isna()).to_numpy()
    codes[miss] = -1
    return codes


def build_history(df: pd.DataFrame, length: int = L) -> np.ndarray:
    """[N, length] int32 row indices of earlier same-uid rows (dt strictly smaller),
    most recent last, left-padded with -1. df must be sorted by TransactionDT."""
    n = len(df)
    t = df["TransactionDT"].to_numpy(np.int64)
    if n > 1 and (np.diff(t) < 0).any():
        raise ValueError("df must be sorted by TransactionDT")
    codes = _uid_codes(df)
    hist = np.full((n, length), -1, dtype=np.int32)
    sel = np.flatnonzero(codes >= 0)
    if len(sel) == 0:
        return hist
    c, ts = codes[sel], t[sel]
    order = np.argsort(c, kind="stable")  # time order is kept inside a uid
    c, ts, rows = c[order], ts[order], sel[order]
    big = int(t.max()) + 1
    comb = c * big + ts
    gstart = np.searchsorted(comb, c * big, side="left")
    hi = np.searchsorted(comb, comb, side="left")  # group rows with dt < own dt: [gstart, hi)
    for k in range(1, length + 1):
        pos = hi - k
        ok = pos >= gstart
        hist[rows[ok], length - k] = rows[pos[ok]]
    return hist


# ----------------------------------------------------------------------------------- build
def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def build_arrays(df: pd.DataFrame, fit_end_dt: int, corpus: str) -> SeqArrays:
    """df: rows sorted by TransactionDT (isFraud NaN for unlabeled rows)."""
    if corpus not in ("labeled_only", "with_unlabeled"):
        raise ValueError(f"corpus must be labeled_only or with_unlabeled, got {corpus!r}")
    t0 = time.time()
    if corpus == "labeled_only":
        df = df[df["isFraud"].notna()]
    df = df.reset_index(drop=True)
    dt = df["TransactionDT"].to_numpy(np.int64)
    if len(dt) > 1 and (np.diff(dt) < 0).any():
        raise ValueError("df must be sorted by TransactionDT")
    fit = dt <= fit_end_dt
    if not fit.any():
        raise ValueError("no rows in the fit window")

    frame = _numeric_frame(df)
    med, scale = _fit_numeric(frame, fit)
    raw = frame.to_numpy(np.float64)
    missing = np.isnan(raw)
    z = np.clip((raw - med) / scale, -CLIP, CLIP)
    num = np.where(missing, 0.0, z).astype(np.float32)

    cat_names = list(CAT_SPEC)
    cat = np.zeros((len(df), len(cat_names)), dtype=np.int32)
    card = np.zeros(len(cat_names), dtype=np.int32)
    for k, (name, cap) in enumerate(CAT_SPEC.items()):
        vocab = _fit_codes(df[name], fit, cap)
        cat[:, k] = _apply_codes(df[name], vocab)
        card[k] = len(vocab) + 1

    hist = build_history(df)
    y = df["isFraud"].to_numpy(np.float64).astype(np.float32)
    meta = {
        "fit_end_dt": int(fit_end_dt), "corpus": corpus, "L": L,
        "n_rows": int(len(df)), "n_fit_rows": int(fit.sum()),
        "num_names": list(frame.columns), "cat_names": cat_names,
        "cat_caps": {k: v for k, v in CAT_SPEC.items() if v is not None},
        "d_relative": list(D_RELATIVE), "normalization": "median/IQR on fit window, clip +-10",
        "history": "uid=make_keys()['uid'], NaN uid component -> no history, dt strictly earlier",
        "git_sha": _git_sha(),
    }
    print(f"build_arrays: {len(df):,} rows in {time.time() - t0:.1f}s", flush=True)
    return SeqArrays(
        ids=df["TransactionID"].to_numpy(np.int64), dt=dt, y=y, labeled=~np.isnan(y),
        num=num, num_missing=missing.astype(np.uint8), cat=cat, cat_cardinality=card,
        hist=hist, num_names=np.array(frame.columns, dtype=str),
        cat_names=np.array(cat_names, dtype=str), meta=json.dumps(meta),
    )


def save_arrays(path, arrays: SeqArrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {f.name: getattr(arrays, f.name) for f in fields(SeqArrays)}
    payload["meta"] = np.array(arrays.meta)
    np.savez(path, **payload)


def load_arrays(path) -> SeqArrays:
    with np.load(Path(path), allow_pickle=False) as z:
        kw = {f.name: z[f.name] for f in fields(SeqArrays) if f.name != "meta"}
        kw["meta"] = str(z["meta"])
    return SeqArrays(**kw)


# ------------------------------------------------------------------------------ leakage check
def leakage_check(arrays: SeqArrays, n: int = 20, seed: int = 0, raw: pd.DataFrame | None = None) -> str:
    """Check n random rows with history. `raw` = the frame the arrays were built from (matched
    on TransactionID); with it the uid is recomputed and the history is rebuilt by hand."""
    names = set(map(str, arrays.num_names)) | set(map(str, arrays.cat_names))
    bad = names & FORBIDDEN_INPUT_NAMES
    assert not bad, f"forbidden input columns: {bad}"
    has = np.flatnonzero((arrays.hist >= 0).any(axis=1))
    assert len(has) > 0, "no row has history"
    rng = np.random.default_rng(seed)
    picks = rng.choice(has, size=min(n, len(has)), replace=False)
    uid_str = nan_key = None
    if raw is not None:
        r = raw.drop_duplicates("TransactionID").set_index("TransactionID").loc[arrays.ids]
        r = r.reset_index().copy()
        uid_str = make_keys(r)["uid"].to_numpy()
        nan_key = (r["card1"].isna() | r["addr1"].isna() | r["D1"].isna()).to_numpy()
    n_hist_total = 0
    for i in picks:
        h = arrays.hist[i]
        real = h[h >= 0]
        n_hist_total += len(real)
        assert (arrays.dt[real] < arrays.dt[i]).all(), f"row {i}: history not strictly earlier"
        assert (np.diff(arrays.dt[real]) >= 0).all(), f"row {i}: history not in time order"
        k = len(real)
        assert (h[L - k:] >= 0).all() and (h[: L - k] == -1).all(), f"row {i}: padding not left"
        if raw is not None:
            assert not nan_key[i], f"row {i}: NaN uid component but has history"
            assert (uid_str[real] == uid_str[i]).all(), f"row {i}: history row with other uid"
            same = np.flatnonzero((uid_str == uid_str[i]) & ~nan_key & (arrays.dt < arrays.dt[i]))
            expect = same[-L:]  # rows are dt-sorted, so the last L are the most recent
            assert np.array_equal(expect, real), f"row {i}: history differs from hand-built"
    if raw is not None:
        nk = np.flatnonzero(nan_key)
        assert (arrays.hist[nk] == -1).all(), "NaN-key rows must have no history"
    return (f"leakage_check OK: {len(picks)} rows with history checked ({n_hist_total} history rows), "
            f"dt strictly earlier, left-padded, most recent last, "
            f"{'uid recomputed from raw and history rebuilt by hand, ' if raw is not None else 'uid NOT recomputed (no raw frame), '}"
            f"no input column named isFraud/TransactionDT/day")


# ------------------------------------------------------------------------------------- CLI
def _report(arrays: SeqArrays, raw: pd.DataFrame) -> None:
    for f in fields(SeqArrays):
        v = getattr(arrays, f.name)
        print(f"  {f.name:16s} {getattr(v, 'shape', '')} {getattr(v, 'dtype', '')}")
    share = float((arrays.hist >= 0).any(axis=1).mean())
    print(f"share of rows with >=1 history row: {share:.4f}")
    print(f"labeled rows: {int(arrays.labeled.sum())} / {len(arrays.ids)}; "
          f"cat_cardinality max {int(arrays.cat_cardinality.max())}")
    print(leakage_check(arrays, raw=raw), flush=True)


def _smoke() -> None:
    lab = data_mod.load_joined(use_cache=False).sort_values("TransactionDT", kind="stable").iloc[:20000]
    unl = load_unlabeled().sort_values("TransactionDT", kind="stable").iloc[:5000]
    fit_end_dt = int(lab["TransactionDT"].iloc[14999])  # dt of the 15,000th labeled row
    df = (pd.concat([lab, unl], ignore_index=True)
          .sort_values("TransactionDT", kind="stable").reset_index(drop=True))
    arrays = build_arrays(df, fit_end_dt, "with_unlabeled")
    out = data_mod.WORKTREE / "data" / "processed" / "seq" / "smoke.npz"
    save_arrays(out, arrays)
    back = load_arrays(out)
    assert np.array_equal(back.hist, arrays.hist) and back.meta == arrays.meta
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB), fit_end_dt={fit_end_dt}")
    _report(back, df)
    print("SEQ_DATA SMOKE OK")


def _full(fit_end_day: int, corpus: str, out: str) -> None:
    lab = data_mod.load_joined(use_cache=False)
    parts = [lab] + ([load_unlabeled()] if corpus == "with_unlabeled" else [])
    df = (pd.concat(parts, ignore_index=True)
          .sort_values("TransactionDT", kind="stable").reset_index(drop=True))
    fit_end_dt = (fit_end_day + 1) * DAY - 1  # whole fit_end_day belongs to the fit window
    arrays = build_arrays(df, fit_end_dt, corpus)
    save_arrays(out, arrays)
    print(f"saved {out}")
    _report(load_arrays(out), df)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--fit-end-day", type=int)
    ap.add_argument("--corpus", choices=["labeled_only", "with_unlabeled"])
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.smoke:
        _smoke()
    elif a.full:
        if a.fit_end_day is None or a.corpus is None or a.out is None:
            ap.error("--full needs --fit-end-day, --corpus and --out")
        _full(a.fit_end_day, a.corpus, a.out)
    else:
        ap.error("use --smoke or --full")


if __name__ == "__main__":
    main()
