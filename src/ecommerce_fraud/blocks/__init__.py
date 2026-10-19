"""Feature-block registry.

A block is a module in this folder (or in a folder listed in env IMP_EXTRA_BLOCK_DIR, os.pathsep
separated) exposing NAME, DESCRIPTION, USES_LABELS and build(df, fit_end_dt) -> DataFrame with
TransactionID + columns named "<NAME>__<feature>", one row per input row.

  discover()                          -> {name: module}
  load(name, fit_end_dt, df=None)     -> block frame, parquet-cached under
                                         data/processed/blocks/<name>_<fit_end_dt>.parquet
  build_seconds(name, fit_end_dt)     -> build time recorded next to the cache file (or None)

df=None means the full data.load_joined() sorted by TransactionDT (stable) with reset index, and the
cache is used. A df passed in (debug subsets) is never cached.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import resource
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

from ecommerce_fraud.data import WORKTREE, load_joined

BLOCK_DIR = Path(__file__).resolve().parent
CACHE_DIR = WORKTREE / "data" / "processed" / "blocks"
REQUIRED = ("NAME", "build")


def sorted_full_frame() -> pd.DataFrame:
    return load_joined().sort_values("TransactionDT", kind="stable").reset_index(drop=True)


def _module_files():
    for p in sorted(BLOCK_DIR.glob("*.py")):
        if not p.name.startswith("_"):
            yield p, f"ecommerce_fraud.blocks.{p.stem}"
    for d in filter(None, os.environ.get("IMP_EXTRA_BLOCK_DIR", "").split(os.pathsep)):
        for p in sorted(Path(d).glob("*.py")):
            if not p.name.startswith("_"):
                yield p, f"ecommerce_fraud.blocks._extra_{p.stem}"


def discover() -> dict:
    """Import every block module found; modules without NAME/build are ignored, import errors warn."""
    found = {}
    for path, modname in _module_files():
        try:
            if modname in sys.modules:
                mod = sys.modules[modname]
            elif path.parent == BLOCK_DIR:
                mod = importlib.import_module(modname)
            else:
                spec = importlib.util.spec_from_file_location(modname, path)
                mod = importlib.util.module_from_spec(spec)
                sys.modules[modname] = mod
                spec.loader.exec_module(mod)
        except Exception as exc:  # a broken block must not hide the others
            warnings.warn(f"block module {path} failed to import: {exc!r}", stacklevel=2)
            continue
        if not all(hasattr(mod, a) for a in REQUIRED):
            continue
        name = mod.NAME
        if name in found:
            raise RuntimeError(f"duplicate block name {name!r}: {found[name].__file__} and {path}")
        found[name] = mod
    return found


def cache_path(name: str, fit_end_dt: int) -> Path:
    return CACHE_DIR / f"{name}_{int(fit_end_dt)}.parquet"


def build_seconds(name: str, fit_end_dt: int):
    side = cache_path(name, fit_end_dt).with_suffix(".json")
    if side.exists():
        return json.loads(side.read_text()).get("build_seconds")
    return None


def check_frame(name: str, out: pd.DataFrame, df: pd.DataFrame | None = None) -> None:
    """Column-prefix, identity and size checks; raises ValueError."""
    if "TransactionID" not in out.columns:
        raise ValueError(f"block {name}: no TransactionID column")
    feats = [c for c in out.columns if c != "TransactionID"]
    if not feats:
        raise ValueError(f"block {name}: no feature columns")
    bad = [c for c in feats if not c.startswith(f"{name}__")]
    if bad:
        raise ValueError(f"block {name}: columns without prefix '{name}__': {bad[:5]}")
    if out["TransactionID"].duplicated().any():
        raise ValueError(f"block {name}: duplicated TransactionID")
    if df is not None:
        if len(out) != len(df):
            raise ValueError(f"block {name}: {len(out)} rows for {len(df)} input rows")
        if not (out["TransactionID"].to_numpy() == df["TransactionID"].to_numpy()).all():
            raise ValueError(f"block {name}: TransactionID order differs from the input")


def load(name: str, fit_end_dt: int, df: pd.DataFrame | None = None,
         use_cache: bool | None = None) -> pd.DataFrame:
    """Block frame for fit_end_dt. use_cache defaults to (df is None)."""
    mods = discover()
    if name not in mods:
        raise KeyError(f"unknown block {name!r}; discovered: {sorted(mods)}")
    mod = mods[name]
    if use_cache is None:
        use_cache = df is None
    path = cache_path(name, fit_end_dt)
    src = Path(mod.__file__)
    if use_cache and path.exists() and path.stat().st_mtime >= src.stat().st_mtime:
        out = pd.read_parquet(path)
        check_frame(name, out)
        return out
    t0 = time.time()
    frame = sorted_full_frame() if df is None else df
    out = mod.build(frame, int(fit_end_dt))
    seconds = time.time() - t0
    check_frame(name, out, frame)
    out = out.reset_index(drop=True)
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        out.to_parquet(path, index=False)
        path.with_suffix(".json").write_text(json.dumps({
            "block": name, "fit_end_dt": int(fit_end_dt), "n_rows": int(len(out)),
            "n_features": int(out.shape[1] - 1), "build_seconds": round(seconds, 1),
            "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 1),
        }) + "\n")
    return out
