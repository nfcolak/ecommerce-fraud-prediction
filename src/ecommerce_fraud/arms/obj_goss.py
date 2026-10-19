"""Arm: GOSS sampling instead of row bagging (bagging off; top_rate 0.2, other_rate 0.1 are LightGBM defaults)."""
ARM = {"name": "obj_goss", "description": "final with data_sample_strategy goss instead of bagging 0.8",
       "params": {"data_sample_strategy": "goss", "bagging_fraction": 1.0, "bagging_freq": 0}}
