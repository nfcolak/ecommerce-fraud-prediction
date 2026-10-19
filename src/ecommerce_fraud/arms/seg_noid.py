"""Arm: separate LightGBM for rows without an identity record, merged by Platt mapping."""
from ecommerce_fraud.segment import fit_seg_noid

ARM = {
    "name": "seg_noid",
    "description": "final's global LGBM + a no-identity-only LGBM, segment scores mapped to the global scale (Platt on ES rows)",
    "model": "custom",
    "fit": fit_seg_noid,
}
