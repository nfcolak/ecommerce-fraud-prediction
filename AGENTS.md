# E-Commerce Fraud Prediction

Predict fraudulent e-commerce transactions and returns, with Pickup and Return Point (PARP) abuse as the motivating case (customer return fraud, false "not received" claims, pickup-point partner abuse). Portfolio project for data science roles in Switzerland.

## Status

Data: IEEE-CIS (Vesta) copy from the Kaggle mirror `lnasiri007/ieeecis-fraud-detection` in `data/raw/ieee-cis/` (git-ignored; set `IEEE_CIS_RAW_DIR` in worktrees). Models on main: `baseline_lgbm`, `lgbm_fe1`, `lgbm_fe2` (best, test PR-AUC 0.5628). The plan for the remaining work lives in ProjectOS `10-Projects/ecommerce-fraud-prediction/Work/2026-10-10 - Plan - Remaining work v2.md`; the evaluation protocol is ADR-002 there.

## Layout

| Path | Use |
|---|---|
| `src/ecommerce_fraud/` | Python package; entry points run as `python -m ecommerce_fraud.<module>` with `PYTHONPATH=src` |
| `data/` | Raw and processed data, git-ignored |
| `models/` | Saved boosters and per-row scores, git-ignored |
| `results/` | Committed metrics, plots and run info |
| `.venv/` | Project venv (Python 3.12), git-ignored |

## Non-negotiable rules

| Rule | Why |
|---|---|
| Split by time with `data.time_split` (train / valid / 7-day gap / test); fit preprocessing on train only. | No leakage; honest fraud numbers. |
| Every choice (features, parameters, threshold, calibration) is made on valid or on forward folds before the test period. Test is scored once per final model; intermediate candidates report valid only. | The test period was already viewed in 3 rounds (baseline, fe1, fe2). |
| No absolute-time features (`TransactionDT`, day index, all-time counts on low-cardinality keys). History features use strictly earlier rows only and no labels. | Test days lie outside the training range. |
| Report PR-AUC and precision at top-k, never accuracy alone; give seed sd or bootstrap CIs with comparisons. | Imbalanced classes; small differences are noise. |
| Earlier `results/<model>/` files stay byte-identical; new work adds new model names. | Comparability. |
| Save boosters and per-row valid/test scores to `models/<model>/` for every trained model. | Re-scoring without retraining. |
| Every result number in README comes from a committed file under `results/`. | Evidence. |
| Data download and training need user approval first. | User rule. |
| No notes in this repo: only `AGENTS.md` and `README.md`. Plans and decisions go to ProjectOS `10-Projects/ecommerce-fraud-prediction/`, rules to Jev-Mem `Notes/ecommerce-fraud-prediction/`. | User rule. |
| Run Python with `env -u PYTHONPATH` (Hermes exports a PYTHONPATH that breaks venvs). | Environment. |
| One heavy LightGBM run at a time (peak ~10 GB on a 24 GB Mac); keep float64 and `n_jobs=12`; start long runs with `nohup` so an agent timeout does not kill them. | Memory and comparability. |
