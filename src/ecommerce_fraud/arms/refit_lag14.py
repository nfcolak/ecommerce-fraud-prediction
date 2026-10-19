"""Arm: refit_x100 when labels arrive 14 days later (stage 1 as final_lag14, refit on all rows <= S-22)."""
from ecommerce_fraud.lag import fit_refit_lag

ARM = {"name": "refit_lag14", "description": "refit_x100 with labels only up to S-22 (refit on all rows <= S-22)",
       "model": "custom", "fit": fit_refit_lag}
