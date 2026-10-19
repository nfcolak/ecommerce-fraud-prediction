"""Arm: refit on fit+ES rows with the early-stopped round count (x1.00)."""
from ecommerce_fraud.refit import make_arm

ARM = make_arm("refit_x100", 1.00, "final, then refit on fit+ES rows for best_iteration rounds (x1.00)")
