# E-Commerce Fraud Prediction

Predict fraudulent e-commerce transactions. The motivating case is Pickup and Return Point (PARP) abuse: customer return fraud, false "not received" claims and pickup-point partner abuse. The first stage uses a public card-transaction fraud dataset to build an honest, time-split modelling baseline.

## Data

A copy of the IEEE-CIS Fraud Detection competition data (Vesta) from the Kaggle dataset lnasiri007/ieeecis-fraud-detection, used for educational purposes; the data is not redistributed here.

Only `train_transaction.csv` and `train_identity.csv` are used (left join on `TransactionID`). Place them under `data/raw/ieee-cis/` (git-ignored) or set `IEEE_CIS_RAW_DIR` to their folder.

Split by time: test = last 20% of rows; the 7 days before the test start are dropped (label-maturity gap); of the remaining earlier rows, valid = last 15%, train = the rest.

## Run

```
pip install -r requirements.txt
env -u PYTHONPATH PYTHONPATH=src python -m ecommerce_fraud.baseline_lgbm
```

Outputs go to `results/baseline_lgbm/` (`metrics.json`, `run_info.json`, `feature_importance.csv`).

## Results

`baseline_lgbm`: LightGBM on raw features (no feature engineering, no class weights), early stopping on valid PR-AUC, test scored once. `lgbm_fe1`: the same split and params plus 53 past-only entity-history features (counts, recency, amount stats and distinct-value counts per card/address/client/email key, built only from strictly earlier transactions). Numbers are from `results/baseline_lgbm/metrics.json` and `results/lgbm_fe1/metrics.json`.

| Metric (test) | baseline_lgbm | lgbm_fe1 |
|---|---|---|
| PR-AUC | 0.5232 | 0.5166 |
| ROC-AUC | 0.8993 | 0.9078 |
| Precision at top 1% | 0.8799 | 0.8756 |
| Recall at top 1% | 0.2559 | 0.2547 |
| Precision at top 5% | 0.3888 | 0.3879 |
| Recall at top 5% | 0.5650 | 0.5637 |
| Fraud rate (test) | 0.0344 | 0.0344 |
