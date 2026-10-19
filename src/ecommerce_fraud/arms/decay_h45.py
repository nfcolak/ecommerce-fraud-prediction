"""Arm: final's LightGBM with time-decay sample weights (half-life 45 days) on fit rows."""
from ecommerce_fraud.decay import fit_decay

ARM = {
    "name": "decay_h45",
    "description": "final LightGBM; fit rows weighted 0.5**((max_day-day)/45), mean 1, ES rows unweighted",
    "model": "custom",
    "fit": fit_decay(45),
}
