"""Arm registry for the p4 selection protocol.

An arm is a module `arms/<name>.py` exposing `ARM = {...}` (contract: see validate()).
  discover()      -> {name: ARM dict}; import errors warn, they do not crash. Extra arm folders can be
                     listed in env P4_EXTRA_ARM_DIR (os.pathsep separated; scratch/debug arms only).
  validate(arm)   -> raises ValueError on unknown keys or a broken contract.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import warnings
from pathlib import Path

ARM_DIR = Path(__file__).resolve().parent
MODELS = ("lgbm", "xgb", "cat", "xgb2", "cat2", "custom")
KEYS = {"name", "description", "blocks", "block_cols", "drop_cols", "model", "params",
        "early_stop", "max_rounds", "fit", "blend_of"}
MODEL_KEYS = {"blocks", "block_cols", "drop_cols", "model", "params", "early_stop", "max_rounds", "fit"}


def validate(arm, known=None):
    """Check one ARM dict against the contract. `known` (optional): names of all arms, for blend_of."""
    if not isinstance(arm, dict):
        raise ValueError("ARM must be a dict")
    unknown = set(arm) - KEYS
    if unknown:
        raise ValueError(f"unknown ARM key(s) {sorted(unknown)}; allowed: {sorted(KEYS)}")
    name = arm.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("ARM['name'] (non-empty str) is required")
    if "description" in arm and not isinstance(arm["description"], str):
        raise ValueError("description must be a str")
    for k in ("blocks", "drop_cols", "blend_of"):
        if k in arm and not (isinstance(arm[k], (list, tuple)) and all(isinstance(x, str) for x in arm[k])):
            raise ValueError(f"{k} must be a list of str")
    if "block_cols" in arm:
        bc = arm["block_cols"]
        if not (isinstance(bc, dict) and all(isinstance(v, (list, tuple)) for v in bc.values())):
            raise ValueError("block_cols must be {block: [columns]}")
        extra = set(bc) - set(arm.get("blocks", []))
        if extra:
            raise ValueError(f"block_cols names blocks not in `blocks`: {sorted(extra)}")
    if "params" in arm and not isinstance(arm["params"], dict):
        raise ValueError("params must be a dict")
    for k in ("early_stop", "max_rounds"):
        if k in arm and not (isinstance(arm[k], int) and not isinstance(arm[k], bool) and arm[k] > 0):
            raise ValueError(f"{k} must be a positive int")
    model = arm.get("model", "lgbm")
    if "blend_of" in arm:
        bad = MODEL_KEYS & set(arm)
        if bad:
            raise ValueError(f"blend_of arm {name!r} must not have model keys {sorted(bad)}")
        if not arm["blend_of"]:
            raise ValueError("blend_of must list at least one arm")
        if name in arm["blend_of"]:
            raise ValueError("blend_of lists the arm itself")
        if known is not None:
            missing = [a for a in arm["blend_of"] if a not in known]
            if missing:
                raise ValueError(f"blend_of names unknown arm(s) {missing}")
        return arm
    if model not in MODELS:
        raise ValueError(f"model {model!r} not in {MODELS}")
    if model == "custom":
        if not callable(arm.get("fit")):
            raise ValueError("model 'custom' needs a callable ARM['fit']")
    elif "fit" in arm:
        raise ValueError("ARM['fit'] is only allowed with model 'custom'")
    return arm


def _module_files():
    for p in sorted(ARM_DIR.glob("*.py")):
        if not p.name.startswith("_"):
            yield p, f"ecommerce_fraud.arms.{p.stem}"
    for d in filter(None, os.environ.get("P4_EXTRA_ARM_DIR", "").split(os.pathsep)):
        for p in sorted(Path(d).glob("*.py")):
            if not p.name.startswith("_"):
                yield p, f"ecommerce_fraud.arms._extra_{p.stem}"


def discover() -> dict:
    """Import every non-underscore module of the arms folder; returns {name: ARM}."""
    found = {}
    for path, modname in _module_files():
        try:
            if modname in sys.modules:
                mod = sys.modules[modname]
            elif path.parent == ARM_DIR:
                mod = importlib.import_module(modname)
            else:
                spec = importlib.util.spec_from_file_location(modname, path)
                mod = importlib.util.module_from_spec(spec)
                sys.modules[modname] = mod
                spec.loader.exec_module(mod)
            arm = getattr(mod, "ARM", None)
            if arm is None:
                warnings.warn(f"arm module {path} has no ARM; ignored", stacklevel=2)
                continue
            validate(arm)
            if arm["name"] != path.stem:
                raise ValueError(f"ARM['name']={arm['name']!r} differs from module name {path.stem!r}")
        except Exception as exc:  # a broken arm must not hide the others
            warnings.warn(f"arm module {path} failed to import/validate: {exc!r}", stacklevel=2)
            continue
        if arm["name"] in found:
            raise RuntimeError(f"duplicate arm name {arm['name']!r}")
        found[arm["name"]] = arm
    return found
