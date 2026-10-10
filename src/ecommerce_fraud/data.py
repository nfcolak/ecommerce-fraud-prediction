"""Load IEEE-CIS train_transaction + train_identity and build the time split."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DIR = Path(
    "/Users/necatifurkancolak/AI-Workplace/Projects/current/"
    "ecommerce-fraud-prediction/data/raw/ieee-cis"
)
WORKTREE = Path(__file__).resolve().parents[2]
CACHE_PATH = WORKTREE / "data" / "processed" / "ieee_cis_train_joined.parquet"

GAP_SECONDS = 7 * 24 * 3600  # 604800, label-maturity gap before test start
TEST_FRAC = 0.20
VALID_FRAC = 0.15


def load_joined(use_cache: bool = True, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Left-join identity onto transaction on TransactionID. Optional parquet cache."""
    if use_cache and CACHE_PATH.exists():
        return pd.read_parquet(CACHE_PATH)
    trans = pd.read_csv(raw_dir / "train_transaction.csv")
    ident = pd.read_csv(raw_dir / "train_identity.csv")
    df = trans.merge(ident, on="TransactionID", how="left")
    if use_cache:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(CACHE_PATH, index=False)
    return df


def split_stats(part: pd.DataFrame) -> dict:
    n_fraud = int(part["isFraud"].sum())
    return {
        "n_rows": int(len(part)),
        "n_fraud": n_fraud,
        "fraud_rate": float(n_fraud / len(part)),
        "dt_min": int(part["TransactionDT"].min()),
        "dt_max": int(part["TransactionDT"].max()),
    }


def time_split(df: pd.DataFrame):
    """Return (train, valid, test, stats). Test = last 20% by time; rows in the
    7 days before test start are dropped; valid = last 15% of the remainder."""
    df = df.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    n = len(df)
    n_test = int(round(n * TEST_FRAC))
    test = df.iloc[n - n_test:]
    test_start = test["TransactionDT"].iloc[0]
    earlier = df.iloc[: n - n_test]
    earlier = earlier[earlier["TransactionDT"] < test_start - GAP_SECONDS]
    n_valid = int(round(len(earlier) * VALID_FRAC))
    valid = earlier.iloc[len(earlier) - n_valid:]
    train = earlier.iloc[: len(earlier) - n_valid]
    stats = {name: split_stats(part) for name, part in
             (("train", train), ("valid", valid), ("test", test))}
    return train, valid, test, stats
