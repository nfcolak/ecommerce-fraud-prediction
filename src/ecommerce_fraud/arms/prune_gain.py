"""Arm: cheap pruning from one stage-1 model's gain (drop zero gain + bottom 20%), then final's LightGBM."""
from ecommerce_fraud.prune5 import fit_prune_gain

ARM = {"name": "prune_gain",
       "description": "drop zero-gain and bottom-20% gain features of one stage-1 model (fit/ES rows only), then final LightGBM",
       "model": "custom", "fit": fit_prune_gain}
