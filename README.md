# E-Commerce Fraud Prediction

Fraud scoring for e-commerce transactions, motivated by Pickup and Return Point (PARP) abuse (customer return fraud, false "not received" claims, pickup-point partner abuse). No public labelled PARP data exists, so the models are built and evaluated on the public IEEE-CIS card-transaction fraud data with a strict time split, and the return/PARP side is written up as a design.

The final model is `final_bag5_refit`: five LightGBM seeds, each retrained on train + valid for the round count that early stopping found on valid, rank-averaged. It reaches test PR-AUC 0.6234; its top 1% of test orders has precision 0.920 and recall 0.267, its top 5% precision 0.454 and recall 0.660. Against `final_bag5` (the same recipe trained on train only, test PR-AUC 0.5662) the paired gain is +0.0572 (95% CI 0.0320 to 0.0891); against the single-seed `final` (0.5605) it is +0.0629 (0.0380 to 0.0937). Paired day-block bootstrap, 7-day blocks (6 over the 42 test days), 2000 resamples, p < 0.0005 (0 of 2000 resamples). This is the 6th view of the same test period: a comparative result, not an unbiased absolute estimate. Sources: `results/final_bag5_refit/test_view.json`, `results/final_bag5/test_view.json`.

Earlier single-model result (`final`, test PR-AUC 0.5605, day-block CI 0.5274-0.5903, +0.0373 over the raw-feature baseline, paired 95% CI 0.0274 to 0.0476): `results/final/metrics.json`. The decision-layer sections below (calibration, capacity, cost, SHAP, stability) were computed for this single-seed `final`, not for `final_bag5_refit`.

The project is closed (2026-10-19). `git_sha` fields inside the results JSON files refer to the local development history; the public repository holds one squash commit.

## Data

A copy of the IEEE-CIS Fraud Detection competition data (Vesta) from the Kaggle dataset lnasiri007/ieeecis-fraud-detection, used for educational purposes; the data is not redistributed here.

Only `train_transaction.csv` and `train_identity.csv` (the labelled rows) are used, left-joined on `TransactionID`. Place them under `data/raw/ieee-cis/` (git-ignored) or set `IEEE_CIS_RAW_DIR`. `TransactionAmt` is treated as currency units; the real currency is not documented.

## Evaluation protocol

Split by time with `data.time_split`: train, valid, a 7-day gap (label maturity), then test. Numbers from `results/final/metrics.json` (`split`):

| Part | Days (TransactionDT // 86400) | Rows | Fraud rate |
|---|---|---|---|
| train | 1-111 | 385,712 | 0.0342 |
| valid | 111-134 | 68,067 | 0.0412 |
| gap | 7 days, dropped | - | - |
| test | 141-182 (42 days) | 118,108 | 0.0344 |

- Every choice (features, parameters, threshold, calibration) is made on valid or on forward folds, never on test.
- History features use strictly earlier rows only and no labels. There are no absolute-time features (`TransactionDT`, day index, all-time counts on low-cardinality keys).
- Seeds: LightGBM seed 42 for the single-seed models; early selection runs use 3 seeds and report mean and sd, rounds p4/p5 use 5 seeds. Test CIs in the results table are day-block bootstraps (1,000 resamples over the 42 test days, seed 0). Paired comparisons of the bag models use a paired day-block bootstrap with 7-day blocks (6 over the 42 test days) and 2000 resamples; the selection windows of rounds p4/p5 are 16 days, so each has 3 blocks (7, 7 and 2 days).
- The test period (days 141-182) has been viewed 6 times: baseline_lgbm, lgbm_fe1, lgbm_fe2, final, final_bag5, final_bag5_refit. Each candidate was fixed before its view. Every other model and arm in this repository was judged on valid or on forward folds only.
- Protocol: ADR-002 (time split, valid-only choices) up to `final`; ADR-005 (six forward windows, five seeds, paired day-block bootstrap, Holm correction) for rounds p4 and p5.
- Reproducibility: baseline_lgbm, lgbm_fe1 and lgbm_fe2 retrain to identical best iteration and PR-AUC (243, 118, 2524; `exact: true` for all three in `results/repro/repro.json`).

## Results

Test PR-AUC and precision at top 1% for all scored models. Sources: `results/final_bag5_refit/test_view.json`, `results/final_bag5/test_view.json` (bag models), `results/final/metrics.json` (CIs, final), `results/baseline_lgbm/metrics.json`, `results/lgbm_fe1/metrics.json`, `results/lgbm_fe2/metrics.json`, `results/baselines/metrics.json`. CIs are 95% day-block bootstrap; "-" means none was computed. No per-model day-block CI exists in the committed files for the two bag models; their paired deltas are given below.

| Model | What it is | Test PR-AUC | 95% CI | Test precision at top 1% |
|---|---|---|---|---|
| final_bag5_refit | 5 seeds of final retrained on train + valid, rank average (final model) | 0.6234 | - | 0.9196 |
| final_bag5 | 5 seeds of final, rank average | 0.5662 | - | 0.8917 |
| final | LightGBM, 483 features, nan_keys fix, scale_pos_weight 28 | 0.5605 | 0.5274-0.5903 | 0.8824 |
| lgbm_fe2 | LightGBM + past-only history features (A_hist) | 0.5628 | 0.5293-0.5932 | 0.8883 |
| baseline_lgbm | LightGBM on raw features | 0.5232 | 0.4910-0.5544 | 0.8799 |
| lgbm_fe1 | baseline + 53 history features (incl. absolute day) | 0.5166 | 0.4838-0.5460 | 0.8756 |
| lr_raw_clip | logistic regression, clipped to train percentiles | 0.3938 | 0.3577-0.4300 | 0.7606 |
| lr_hist_clip | same with history features | 0.3849 | 0.3482-0.4196 | 0.7470 |
| lr_raw | logistic regression, unclipped | 0.1882 | - | 0.0956 |
| iforest | Isolation Forest | 0.1182 | - | 0.0000 |
| rule | velocity rule (uid count 1d, amount ratio) | 0.0776 | - | 0.1988 |

Reading:

- Final vs baseline_lgbm, paired over the same test days: PR-AUC +0.0373, 95% CI 0.0274 to 0.0476 (positive in all resamples). Precision at top 1% differs by +0.0025, CI -0.0126 to 0.0193: not distinguishable.
- Final vs lgbm_fe2: PR-AUC -0.0023, CI -0.0061 to 0.0016; precision at top 1% -0.0059, CI -0.0145 to 0.0043. Not distinguishable.
- The NaN-key fix and the class weight did not change ranking quality beyond seed noise (table below).
- final_bag5 vs final: PR-AUC +0.0056, CI 0.0035 to 0.0076 (`results/final_bag5/test_view.json`). final_bag5_refit vs final_bag5: +0.0572, CI 0.0320 to 0.0891; vs final: +0.0629, CI 0.0380 to 0.0937 (`results/final_bag5_refit/test_view.json`).
- Linear models are far below boosted trees. Going from fe1 to fe2 (dropping the absolute `day` feature from the history set) is where the history features start to help.

Validation-only experiments (3 seeds, mean +- sd, valid PR-AUC; no test scoring). Sources: `results/lgbm_fe2/metrics.json`, `results/lgbm_fe3_graph/metrics.json`, `results/lgbm_cw/metrics.json`.

| Candidate | Valid PR-AUC | n_features |
|---|---|---|
| A_hist (chosen in fe2) | 0.6929 +- 0.0054 | 483 |
| B_hist_uidagg | 0.6652 +- 0.0151 | 507 |
| C_uidagg | 0.6744 +- 0.0154 | 461 |
| A_hist_noall | 0.6776 +- 0.0301 | 480 |
| A_hist_noall_graph | 0.6868 +- 0.0021 | 488 |
| A_hist_nanfix | 0.6953 +- 0.0028 | 483 |
| A_hist_nanfix, spw5 | 0.6960 +- 0.0014 | 483 |
| A_hist_nanfix, spw28 (final) | 0.6961 +- 0.0014 | 483 |

All differences are within seed noise. The final configuration follows the pre-registered rule (highest mean valid PR-AUC): A_hist_nanfix with scale_pos_weight 28.

## Final model: final_bag5_refit

Recipe: for each seed 42-46, take the number of boosting rounds B that early stopping found on valid for `final` (`results/final_bag5/final_bag5_s<seed>.json`), retrain the same configuration on train + valid (453,779 rows) for B rounds with no early stopping, score test, and rank-average the five seeds. Source: `results/final_bag5_refit/test_view.json`.

What the gain is, and what it is not: its gain over `final_bag5` is the 14 most recent labelled days that early stopping had held out. A model trained on the same rows as `final` but without a hold-out (`refit_lag14`) ties `final` (+0.0016 mean PR-AUC over windows w2-w6, seed 42; `results/p5/refit_lag14/`, `results/p4/final/`). So this is a recency effect, not evidence that early stopping is wasteful in general. The ranking of the two recipes also holds when labels arrive 14 days later (`refit_lag14` vs `final_lag14`, +0.0431, 95% CI 0.0353 to 0.0486, 5 of 5 windows; `results/p5/stage_b.json`), but both lose about 0.04 PR-AUC in that case (`final_lag14` vs `final`: -0.0415 mean over w2-w6, seed 42; `results/p5/final_lag14/`). The absolute level does not survive the delay; the margin over the incumbent under the same delay does.

Per-seed results (test PR-AUC of each refit seed, `results/final_bag5_refit/test_view.json`, `per_seed`):

| Seed | Rounds B | Test PR-AUC |
|---|---|---|
| 42 | 2822 | 0.6233 |
| 43 | 2826 | 0.6255 |
| 44 | 2899 | 0.6234 |
| 45 | 669 | 0.5885 |
| 46 | 2995 | 0.6257 |
| rank average | - | 0.6234 |

Seed 45 early-stopped at a plateau after 669 rounds on valid (patience 100); it is kept as pre-registered, in both bags. Early stopping with patience 100 at learning rate 0.05 can stop at a plateau: a bagging recipe tolerates one such seed, a single-seed deployment would not.

Footnote on categorical codes: B per seed comes from `final_bag5`, whose codes were fitted on train; the refit fits its codes on train + valid. Codes use no labels, so this is not leakage; B was chosen with a slightly different vocabulary.

`final_bag5_refit` is used as an uncalibrated ranker: it trains on valid, so there is no held-out set left to fit a calibrator or set cost thresholds. `models/final_bag5_refit_s*/scores_valid.parquet` holds in-sample valid scores and must not be used for selection or calibration.

## Rounds after final

All numbers below are valid or forward-window numbers; none of these arms was scored on test. The only test views after `final` are the two bag models above.

### Rounds imp, graph, GNN, stack, drift pruning (2026-10-16): nothing beat final

- Feature blocks, screened on valid, seed 42 (`results/improve/combine.json`, `results/drift/screen.json` for the drift-pruned versions; delta vs final 0.6950): uid_vagg +0.0028, time_cyclic +0.0025, anomaly_score +0.0022 passed the 0.002 screen gate; amount_patterns -0.0388, graph_emb -0.0472, graph_hop2 -0.0203, identity_velocity -0.0180, graph_hop1 -0.0075 hurt.
- Confirmation (`results/improve/confirm.json`): the best candidate, LightGBM + uid_vagg, scored 0.6971 vs 0.6961 (+0.0010) on valid, below the rule's threshold of 0.0028 (2 sd, 3 seeds), and above final on 1 of 3 forward folds. XGBoost, CatBoost, the GNN member, the blend and the stack all scored below final on valid and on every fold (stack: 0.6634 valid). Selected: none.

### Round p4 (2026-10-17, ADR-005)

Sources: `results/p4/screen/*.json` (valid, seed 42), `results/p4/confirm.json` (6 windows x 5 seeds), `results/p4/bag_check.json`.

- Screened, valid PR-AUC vs final 0.6950: seg_noid_blend 0.7069, seg_noid 0.7044, email_flags 0.6962, es500 0.6964, prune_identity 0.6900, no_lowcard_all 0.6794, prune_hop2 0.6812, xgb2 0.6721, blend2 0.6693, prune_amount 0.6692, cat2 0.6189.
- Confirmed (pooled delta over 6 windows, 5 seeds): the no-identity segment blend `seg_noid_blend` +0.0024, 5 of 6 windows, Holm p = 0.0315, passed; es500 +0.0026 in 3 of 6 windows (needs 4), email_flags +0.0010 (Holm p 0.064), seg_noid -0.0040: not selected. The blend's test period stayed closed.
- Seed bagging: `final_bag5` (rank average of final seeds 42-46) vs single-seed final, +0.0068, 6 of 6 windows, CI 0.0062 to 0.0074 (`results/p4/bag_check.json`); bagged blend vs bag +0.0004, CI -0.0012 to 0.0019, not selected. Selected: `final_bag5`. Its test view (5th): 0.5662 vs 0.5605, paired +0.0056, CI 0.0035 to 0.0076 (`results/final_bag5/test_view.json`).

### Round p5 (2026-10-17 to 2026-10-19)

Sources: `results/p5/stage_a.json`, `results/p5/stage_b.json`, `results/p5/<arm>/<window>_s<seed>/result.json`. Stage A: nine arms on six windows (seeds 42 and 43; solo = arm seed 42 vs final seed 42, mix = rank average of arm and final seed 42 vs final seeds 42 and 43). An arm advances if its solo or mix delta is positive.

| Arm | Solo delta | Windows won (solo) | Mix delta | Windows won (mix) | Advanced |
|---|---|---|---|---|---|
| refit_x100 | +0.0567 | 6 | +0.0304 | 6 | yes |
| fastrp_past | +0.0023 | 3 | +0.0039 | 5 | yes |
| obj_goss | +0.0022 | 4 | +0.0006 | 3 | yes |
| prune_gain | +0.0019 | 3 | +0.0005 | 4 | yes |
| obj_spw1 | +0.0008 | 4 | -0.0001 | 4 | yes |
| final_jitter | +0.0003 | 3 | -0.0008 | 1 | yes |
| decay_h45 | -0.0001 | 4 | -0.0012 | 3 | no |
| uid2 | -0.0010 | 1 | -0.0008 | 3 | no |
| decay_h90 | -0.0017 | 3 | -0.0016 | 3 | no |

Stage B (paired day-block bootstrap, 7-day blocks, 2000 resamples, Holm; `results/p5/stage_b.json`). Only `refit_x100` was confirmed:

| Contrast | Pooled delta | 95% CI | Windows won |
|---|---|---|---|
| refit_x100 bag (5 seeds) vs final bag (5 seeds) | +0.0556 | 0.0489 to 0.0618 | 6 of 6 |
| final + refit_x100 (10 models) vs final bag of 10 seeds | +0.0305 | 0.0281 to 0.0328 | 6 of 6 |
| final bag of 10 seeds vs final bag of 5 seeds | +0.0016 | 0.0006 to 0.0021 | 6 of 6 |
| refit_lag14 vs final_lag14 (labels 14 days later, seed 42, w2-w6) | +0.0431 | 0.0353 to 0.0486 | 5 of 5 |

All four have p < 0.0005 (0 of 2000 resamples).

The five other advanced arms (fastrp_past, obj_goss, prune_gain, obj_spw1, final_jitter) were not confirmed in Stage B: their Stage A gains (+0.0003 to +0.0023, one seed) are at seed-noise level, so they are reported as not confirmed, not as negative. The 6th test view above was then taken for `final_bag5_refit`.

Per-row selection scores are kept locally, not published; the committed `result.json`, `stage_a.json`, `stage_b.json`, `confirm.json`, `bag_check.json` and `test_view.json` carry every reported number.

### Registered but not run

These arms and modules were written and registered for round p5 but never run, and were deleted at close: arms `final_lr02`, `nn_tabm`, `obj_extratrees`, `obj_focal`, `prune_null`, `recent90`, `refit_x115`, `tuned5`; modules `tune.py`, `tune5.py`, `nn_tabm.py`, `bag_curve.py`. Focal-loss objective: not run. `EF_SMOKE_MAX_ROUNDS` and `EF_SMOKE_N_JOBS` exist for subset smoke runs and were unset for every published run.

## Decision layer

Computed for the single-seed `final` model (`results/final/metrics.json`), not for `final_bag5_refit`; `final_bag5_refit` is used as an uncalibrated ranker because it trains on valid (no held-out calibration set). Its top-1% and top-5% precision and recall are in the headline. Source for this section: `results/final/metrics.json` (`calibration`, `capacity`, `cost`).

Calibration: raw scores are under-confident (mean 0.0149 on test vs observed 0.0344). Isotonic regression fitted on the first half of valid was chosen over Platt scaling (valid second half: ECE 0.0022 vs 0.0040, Brier 0.0180 vs 0.0182). Test: ECE 0.0031 and Brier 0.0211 (raw: 0.0230 and 0.0236; Platt: 0.0023 and 0.0211).

![Calibration](results/final/calibration.png)

Capacity: review the top k% of orders, either over the whole test period (global) or within each day (42 test days). "Fraud amount share" is the share of fraud `TransactionAmt` caught.

| k | Mode | Alerts per day | Precision | Recall | Fraud amount share |
|---|---|---|---|---|---|
| 0.5% | global | 14.1 | 0.939 | 0.137 | 0.075 |
| 0.5% | per day | 14.5 | 0.900 | 0.135 | 0.086 |
| 1% | global | 28.1 | 0.882 | 0.257 | 0.151 |
| 1% | per day | 28.6 | 0.855 | 0.252 | 0.167 |
| 2% | global | 56.3 | 0.734 | 0.427 | 0.323 |
| 2% | per day | 56.7 | 0.705 | 0.414 | 0.303 |
| 5% | global | 140.6 | 0.416 | 0.605 | 0.558 |
| 5% | per day | 141.0 | 0.413 | 0.601 | 0.555 |

![Capacity curve](results/final/capacity_curve.png)

Cost sensitivity: a held order costs a review fee (2, 5 or 15 currency units) and a missed fraud costs its amount. Thresholds are set on valid; the test loss is compared with holding nothing (loss 609,934). Ranking by p x amount beats ranking by p at every fee.

| Review cost | Ranking | Held share | Precision | Recall | Savings vs no model |
|---|---|---|---|---|---|
| 2 | p | 0.198 | 0.145 | 0.834 | 0.703 |
| 2 | p x amount | 0.110 | 0.206 | 0.658 | 0.748 |
| 5 | p | 0.106 | 0.240 | 0.740 | 0.598 |
| 5 | p x amount | 0.101 | 0.218 | 0.638 | 0.683 |
| 15 | p | 0.089 | 0.275 | 0.714 | 0.411 |
| 15 | p x amount | 0.051 | 0.314 | 0.467 | 0.533 |

![Cost curve](results/final/cost_curve.png)

## Explainability

Computed for the single-seed `final` model (`results/final/metrics.json`), not for `final_bag5_refit`; `final_bag5_refit` is used as an uncalibrated ranker because it trains on valid (no held-out calibration set). Sources: `results/final/metrics.json` (`shap.top20`), `results/final/top_alerts_reasons.csv`.

![SHAP summary](results/final/shap_summary.png)

Top features by mean |SHAP| (2,000 sampled rows): C1 (1.16), C13 (1.05), C14 (0.63), C11 (0.59), D1n (0.43), then C2, D15n, amt_cents, D10n, card1. The C columns are Vesta count features and the D columns are time deltas; their exact meaning is not documented.

`top_alerts_reasons.csv` lists, for each top-ranked alert, its score, calibrated probability and the top reason codes. A reason code is a feature name with its value and SHAP contribution; the first alert's reasons are C1, DeviceInfo and V45.

Governance note (not legal advice):

- The EU AI Act (Regulation (EU) 2024/1689) lists high-risk uses in Annex III. Scoring merchant payment fraud is not listed there; Recital 58 concerns fraud detection in financial services and the creditworthiness category. https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- GDPR Art. 22 (https://eur-lex.europa.eu/eli/reg/2016/679/oj) and the Swiss revised FADP Art. 21 (https://www.fedlex.admin.ch/eli/cc/2022/491/en) apply to decisions based solely on automated processing with a significant effect on the person. Held orders therefore go to human review with the reason codes; the model does not auto-decline.

## Stability over time

Computed for the single-seed `final` model (`results/final/metrics.json`, `results/backtest/metrics.json`), not for `final_bag5_refit`; `final_bag5_refit` is used as an uncalibrated ranker because it trains on valid (no held-out calibration set). The label-lag evidence for the final model is the round p5 lag-14 contrast above.

Forward folds: train on days 1 to the fold's train end (the last 14 train days are used for early stopping), score the next 21 days. The raw-feature baseline and the final configuration use the same folds.

| Fold | Scored days | Train rows | Baseline PR-AUC | Final PR-AUC | Delta |
|---|---|---|---|---|---|
| 1 | 63-83 | 209,660 | 0.5613 | 0.5954 | +0.0341 |
| 2 | 87-107 | 283,222 | 0.5164 | 0.5325 | +0.0161 |
| 3 | 111-131 | 362,483 | 0.6215 | 0.6362 | +0.0147 |

The final configuration is ahead in all three folds, by a smaller margin as training data grows (no fold-level CI, so this is descriptive).

Label delay (valid PR-AUC, final configuration retrained; N = days of most recent train labels affected). Variant B marks fraud in the last N days as legit; variant A drops those rows.

| N days | B: late fraud labelled legit | A: late rows dropped |
|---|---|---|
| 0 | 0.6950 | 0.6950 |
| 7 | 0.4160 | 0.6677 |
| 14 | 0.3021 | 0.6513 |
| 30 | 0.1891 | 0.5909 |

Marking the last N days' fraud as legit (variant B) collapses valid PR-AUC (0.695 -> 0.416 at 7 days) because valid follows train without a gap and the same fraudulent clients recur, so the model learns "recent = legit"; dropping unmatured rows instead (variant A) costs much less (0.668 at 7 days). Practical rule: train only on matured labels.

## Return/PARP side: design

No public labelled PARP data exists, so this is a design, not a result.

- Label: a return or "not received" claim confirmed as abuse after investigation. Label delay is weeks, so training uses matured cases only (see label delay above).
- Scoring moment: when the return or "not received" claim is filed.
- Past-only features: customer return rate so far, days from delivery to claim, claim count in the last 30 days, pickup-point sharing counts (distinct customers per pickup point, device and address in the last 7-30 days), recipient vs buyer mismatch.
- Transfer from this repo: the `features_history.py` entity keys map to customer, pickup point, address and device keys with the same strictly-earlier windows; the graph windows (7d / 30d distinct-entity counts) become distinct customers per pickup point.

Delivery-side proxies in IEEE-CIS (`results/final/parp_signals.json`; rank 1 = most important of 483 features):

| Signal | Gain rank | SHAP rank |
|---|---|---|
| addr1 | 15 | 11 |
| dist1 | 39 | 15 |
| R_emaildomain (recipient email) | 38 | 55 |
| M3 / M4 / M5 / M6 (match flags) | 106 / 80 / 84 / 82 | 85 / 78 / 37 / 52 |
| M1 | 431 | 391 |

Address-sharing counts were tested as graph features in `results/lgbm_fe3_graph/metrics.json` (`graph_features`: addr1_nuniq_uid_7d/30d, card1_addr1_nuniq_uid_7d/30d, fp_nuniq_uid_7d/30d, uid_nuniq_fp_30d, uid_fp_component_size). A_hist_noall_graph (0.6868) did not beat A_hist (0.6929) on validation, so they are not in the final set; the addr1 entries above are history features on the card1_addr1 and uid keys plus addr1 itself.

Real-data path, not run: ASOS GraphReturns (OSF c793h, CC-BY 4.0). Its label is return, not fraud.

## Findings and limitations

- Unclipped linear models and Isolation Forest put their top 1% on an out-of-range legit cluster on days 163-173 (top-1% fraud rate 0.0948 for lr_raw, 0.0 for iforest, vs 0.7604 for lr_raw_clip; `results/baselines/diagnostics.json`). Values are clipped to the train 1st-99th percentile.
- NaN keys were lumped into one pseudo-entity in the history features; fixed via `nan_keys="nan"`, no measurable change (valid 0.6953 vs 0.6929, within seed noise).
- Prevalence shifts: fraud rate 4.12% on valid vs 3.44% on test (`results/final/metrics.json`); test PR-AUC (0.5605) is well below valid (0.6950). The calibrator is fitted on valid, so mean calibrated test probability is close but slightly low (0.0336 vs 0.0344).
- Seed noise is about 0.005-0.015 PR-AUC (sd over 3 seeds, table above); differences below that are not evidence.
- Amounts are currency units; the cost table is relative, not money.
- IEEE-CIS has no device id, so device/graph features are weak; the PARP section is a design and is not validated.
- Valid metrics for the final model use full-valid early stopping and are optimistic. `final_bag5_refit` has no valid metric at all (valid is in its training set).
- The test period was viewed 6 times; 0.6234 is a comparative number, not an unbiased absolute estimate.
- The refit gain is a recency effect (see Final model), and under a 14-day later label arrival both recipes lose about 0.04 PR-AUC.
- Selection windows have 3 day-blocks each, so the bootstrap CIs rest on few blocks.
- Seed 45 of the refit bag stopped at 669 rounds (see Final model).

## How to run

```
pip install -r requirements.txt
export IEEE_CIS_RAW_DIR=/path/to/ieee-cis     # folder with train_transaction.csv and train_identity.csv
export MODELS_DIR=/path/to/models             # boosters and per-row scores (git-ignored)
env -u PYTHONPATH PYTHONPATH=src python -m ecommerce_fraud.baseline_lgbm   # results/baseline_lgbm
env -u PYTHONPATH PYTHONPATH=src python -m ecommerce_fraud.model_fe1       # results/lgbm_fe1
env -u PYTHONPATH PYTHONPATH=src python -m ecommerce_fraud.model_fe2       # results/lgbm_fe2
env -u PYTHONPATH PYTHONPATH=src python -m ecommerce_fraud.reproduce       # results/repro
env -u PYTHONPATH PYTHONPATH=src python -m ecommerce_fraud.baselines       # results/baselines
env -u PYTHONPATH PYTHONPATH=src python -m ecommerce_fraud.model_fe3       # results/lgbm_fe3_graph
env -u PYTHONPATH PYTHONPATH=src python -m ecommerce_fraud.model_cw        # results/lgbm_cw
env -u PYTHONPATH PYTHONPATH=src python -m ecommerce_fraud.final train     # results/final (4th test view)
env -u PYTHONPATH PYTHONPATH=src python -m ecommerce_fraud.final analyze   # calibration, capacity, cost, SHAP, bootstrap
env -u PYTHONPATH PYTHONPATH=src python -m ecommerce_fraud.backtest        # results/backtest
# rounds p4 / p5 (need the data and hours of compute; arms in src/ecommerce_fraud/arms/)
env -u PYTHONPATH PYTHONPATH=src python -m ecommerce_fraud.protocol --screen --arms <a,b>             # results/p4/screen
env -u PYTHONPATH PYTHONPATH=src python -m ecommerce_fraud.protocol --confirm --arms <a,b>            # 6 windows x 5 seeds
env -u PYTHONPATH PYTHONPATH=src python -m ecommerce_fraud.protocol --analyze --arms <a,b> --power    # results/p4/confirm.json
env -u PYTHONPATH PYTHONPATH=src python -m ecommerce_fraud.bag_check                                  # results/p4/bag_check.json
env -u PYTHONPATH PYTHONPATH=src python -m ecommerce_fraud.final_bag                                  # results/final_bag5 (5th test view)
env -u PYTHONPATH PYTHONPATH=src python -m ecommerce_fraud.protocol5 --run --arms <a,b> --seeds 42,43  # results/p5/<arm>
env -u PYTHONPATH PYTHONPATH=src python -m ecommerce_fraud.protocol5 --stage-a --arms <a,b>           # results/p5/stage_a.json
env -u PYTHONPATH PYTHONPATH=src python -m ecommerce_fraud.protocol5 --make-spec --from <stage_a.json> --out <spec.json>
env -u PYTHONPATH PYTHONPATH=src python -m ecommerce_fraud.protocol5 --contrasts <spec.json>          # results/p5/stage_b.json
env -u PYTHONPATH PYTHONPATH=src python -m ecommerce_fraud.final_bag_refit                           # results/final_bag5_refit (6th test view)
```

Training commands need the data and time (the final_bag_refit run took 2,505 s, `runtime_seconds` in its `test_view.json`; the p4/p5 rounds took many hours). Per-row selection scores written under `results/p4` and `results/p5` are git-ignored and not published.

`env -u PYTHONPATH` is only needed where the shell exports a conflicting PYTHONPATH. Training runs are heavy (up to about 10 GB RAM for LightGBM); run one at a time.

| Path | Use |
|---|---|
| `src/ecommerce_fraud/` | Python package; entry points run as `python -m ecommerce_fraud.<module>` with `PYTHONPATH=src` |
| `data/` | Raw and processed data, git-ignored |
| `models/` | Saved boosters and per-row scores, git-ignored |
| `results/` | Committed metrics, plots and run info per model |
| `AGENTS.md` | Binding rules for contributors and agents |
