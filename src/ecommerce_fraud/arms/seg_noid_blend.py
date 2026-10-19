"""Arm: as seg_noid, but no-identity rows get the average of calibrated global and segment probabilities."""
from ecommerce_fraud.segment import fit_seg_noid_blend

ARM = {
    "name": "seg_noid_blend",
    "description": "seg_noid with no-identity rows = mean of calibrated global and segment probabilities",
    "model": "custom",
    "fit": fit_seg_noid_blend,
}
