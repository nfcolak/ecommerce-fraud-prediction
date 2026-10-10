# E-Commerce Fraud Prediction

Predict fraudulent e-commerce transactions. The motivating case is Pickup and Return Point (PARP) abuse: customer return fraud, false "not received" claims and pickup-point partner abuse. The first stage uses a public card-transaction fraud dataset to build an honest, time-split modelling baseline.

## Data

A copy of the IEEE-CIS Fraud Detection competition data (Vesta) from the Kaggle dataset lnasiri007/ieeecis-fraud-detection, used for educational purposes; the data is not redistributed here.

Only `train_transaction.csv` and `train_identity.csv` are used (left join on `TransactionID`). Place them under `data/raw/ieee-cis/` (git-ignored) or adjust `RAW_DIR` in `src/ecommerce_fraud/data.py`.

Split by time: test = last 20% of rows; the 7 days before the test start are dropped (label-maturity gap); of the remaining earlier rows, valid = last 15%, train = the rest.

## Run

```
pip install -r requirements.txt
env -u PYTHONPATH PYTHONPATH=src python -m ecommerce_fraud.baseline_lgbm
```

Outputs go to `results/baseline_lgbm/` (`metrics.json`, `run_info.json`, `feature_importance.csv`).

## Results

`baseline_lgbm`: LightGBM on raw features (no feature engineering, no class weights), early stopping on valid PR-AUC, test scored once. Numbers are from `results/baseline_lgbm/metrics.json`.

| Metric (test) | Value |
|---|---|
| PR-AUC | 0.5232 |
| ROC-AUC | 0.8993 |
| Precision at top 1% | 0.8799 |
| Recall at top 1% | 0.2559 |
| Precision at top 5% | 0.3888 |
| Recall at top 5% | 0.5650 |
| Fraud rate (test) | 0.0344 |
