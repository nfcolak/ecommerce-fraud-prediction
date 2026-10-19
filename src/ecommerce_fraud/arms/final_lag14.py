"""Arm: final when labels arrive 14 days later (labelled rows end at S-22; robustness control for refit)."""
from ecommerce_fraud.lag import fit_final_lag

ARM = {"name": "final_lag14", "description": "final with labels only up to S-22 (fit <= S-36, ES S-35..S-22)",
       "model": "custom", "fit": fit_final_lag}
