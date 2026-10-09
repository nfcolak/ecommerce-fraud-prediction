# E-Commerce Fraud Prediction

Predict fraudulent e-commerce transactions and returns, with Pickup and Return Point (PARP) abuse as the motivating case (customer return fraud, false "not received" claims, pickup-point partner abuse). Portfolio project for data science roles in Switzerland.

## Status

Skeleton only. Dataset, targets and models are not decided yet.

## Layout

| Path | Use |
|---|---|
| `src/ecommerce_fraud/` | Python package; entry points run as `python -m ecommerce_fraud.<module>` with `PYTHONPATH=src` |
| `data/` | Raw and processed data, git-ignored |
| `results/` | Committed metrics, plots and run info |

## Non-negotiable rules

| Rule | Why |
|---|---|
| Split by time (train on older, test on newer); fit preprocessing on train only. | No leakage; honest fraud numbers. |
| Report PR-AUC and precision at top-k, never accuracy alone. | Fraud classes are highly imbalanced. |
| Every result number in README comes from a committed file under `results/`. | Evidence. |
| Data download and training need user approval first. | User rule. |
| No notes in this repo: only `AGENTS.md` and `README.md`. Plans and decisions go to ProjectOS `10-Projects/ecommerce-fraud-prediction/`, rules to Jev-Mem `Notes/ecommerce-fraud-prediction/`. | User rule. |
| Run Python with `env -u PYTHONPATH` (Hermes exports a PYTHONPATH that breaks venvs). | Environment. |
