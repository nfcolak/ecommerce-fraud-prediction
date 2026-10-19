"""Arm: final's LightGBM with a per-seed parameter jitter (diversity for bagging)."""
from ecommerce_fraud.jitter import fit_jitter

ARM = {
    "name": "final_jitter",
    "description": "final's LightGBM with per-seed jitter of num_leaves, feature/bagging fraction, min_child_samples",
    "model": "custom",
    "fit": fit_jitter,
}
