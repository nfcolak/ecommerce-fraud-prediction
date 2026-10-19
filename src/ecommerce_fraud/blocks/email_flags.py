"""Block email_flags: row-level consistency flags between purchaser (P) and recipient (R) e-mail domain.

Pure row-level function of the row's own P_emaildomain / R_emaildomain: no history, no labels,
no fitted statistics (fit_end_dt unused), and no counts over the domains (an all-time count of a
low-cardinality key acts as a clock under the time split). Flags are 0/1 float32, never NaN.

Flags: p_eq_r (both present and equal), p_only (P present, R missing), r_only (R present, P missing),
both_missing, same_family_diff_tld (both present, same first label, different domain; gmail vs
gmail.com, yahoo.com vs yahoo.com.mx), p_free, r_free (domain family is a free web-mail provider).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NAME = "email_flags"
DESCRIPTION = "Row-level P/R e-mail domain consistency flags (P==R, one missing, same family other TLD, free-mail)"
USES_LABELS = False

FREE_FAMILIES = frozenset({
    "gmail", "yahoo", "ymail", "rocketmail", "hotmail", "outlook", "live", "msn", "aol", "aim",
    "icloud", "me", "mac", "mail", "gmx", "web", "protonmail"})
FEATS = ["p_eq_r", "p_only", "r_only", "both_missing", "same_family_diff_tld", "p_free", "r_free"]


def _family(s: pd.Series) -> pd.Series:
    return s.str.split(".").str[0]


def build(df: pd.DataFrame, fit_end_dt: int = 0) -> pd.DataFrame:
    p = df["P_emaildomain"].astype("object").str.lower()
    r = df["R_emaildomain"].astype("object").str.lower()
    pn, rn = p.notna().to_numpy(), r.notna().to_numpy()
    both = pn & rn
    eq = both & (p.fillna("").to_numpy() == r.fillna("").to_numpy())
    fam_eq = both & (_family(p).fillna("").to_numpy() == _family(r).fillna("").to_numpy())
    vals = {
        "p_eq_r": eq,
        "p_only": pn & ~rn,
        "r_only": ~pn & rn,
        "both_missing": ~pn & ~rn,
        "same_family_diff_tld": fam_eq & ~eq,
        "p_free": _family(p).isin(FREE_FAMILIES).to_numpy(),
        "r_free": _family(r).isin(FREE_FAMILIES).to_numpy(),
    }
    out = {"TransactionID": df["TransactionID"].to_numpy()}
    for k in FEATS:
        out[f"{NAME}__{k}"] = vals[k].astype(np.float32)
    return pd.DataFrame(out)


def _main() -> None:
    from ecommerce_fraud import blocks as blk
    from ecommerce_fraud import data

    df = data.load_joined().sort_values("TransactionDT", kind="stable").reset_index(drop=True).iloc[:80000]
    out = build(df, 0)
    blk.check_frame(NAME, out, df)  # one row per input row, same order, prefix
    feats = out.drop(columns="TransactionID")
    assert not feats.isna().any().any(), "NaN in flags"
    assert set(np.unique(feats.to_numpy())) <= {0.0, 1.0}
    assert (feats[[f"{NAME}__{k}" for k in ("p_eq_r", "p_only", "r_only", "both_missing")]].sum(axis=1) <= 1).all()
    # hand check on a few rows
    rng = np.random.default_rng(0)
    for i in rng.choice(len(df), 2000, replace=False):
        a, b = df["P_emaildomain"].iat[i], df["R_emaildomain"].iat[i]
        assert out[f"{NAME}__p_eq_r"].iat[i] == float(isinstance(a, str) and a == b)
        assert out[f"{NAME}__both_missing"].iat[i] == float(not isinstance(a, str) and not isinstance(b, str))
    print(f"rows {len(out)} == {len(df)}, order OK, no NaN")
    y = df["isFraud"].to_numpy()  # reporting only, never a feature
    for c in feats.columns:
        m = feats[c].to_numpy() == 1
        print(f"FLAG {c}: rate {m.mean():.4f}, fraud rate when 1 {y[m].mean() if m.any() else float('nan'):.4f}, "
              f"when 0 {y[~m].mean():.4f}")
    print(f"BLOCK {NAME} OK")


if __name__ == "__main__":
    _main()
