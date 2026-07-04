# Online Payment Fraud Detection

A fraud detection project built on the [PaySim synthetic mobile-money dataset](https://www.kaggle.com/datasets/ealaxi/paysim1) (~6.36M transactions, ~0.13% fraud rate), designed and evaluated the way a real payments risk team would: time-based validation instead of random splits, business-framed metrics instead of accuracy, and a documented leakage investigation rather than a single headline number.

## 1. Business framing

Payment fraud is a two-sided cost problem, not a pure classification exercise. Every fraudulent transaction that slips through is a direct financial loss, chargeback liability, and a hit to customer trust — but a fraud filter that's too aggressive creates its own damage: declined legitimate purchases, held transfers, and customers who churn because their money got stuck. The goal of this project is to find a model and a decision threshold that **catches the large majority of fraud losses while flagging as small a slice of legitimate transaction volume as possible**, on a dataset (PaySim, synthetic, 6,362,620 transactions, 8,213 of them fraudulent — a 0.13% fraud rate) that is severely imbalanced and where fraud only ever occurs in two transaction types (`TRANSFER`, `CASH_OUT`).

## 2. The leakage discovery

An early version of this project reported precision, recall, and F1 all **above 0.99 for every model type and every imbalance-handling strategy tested** — six different model/technique combinations, all effectively perfect. That is not a result to celebrate on a fraud problem this imbalanced; it's a signal to stop and check for leakage.

### What was investigated

[04_advanced_modeling.ipynb](04_advanced_modeling.ipynb) treats the suspiciously perfect result as a bug report against itself and checks three things directly, in order:

1. **Split-before-resample:** confirmed the train/test split is created first, and SMOTE/undersampling are only ever fit on the training fold (`Rows in both train AND test: 0`). Not the source of the problem.
2. **Time-based split integrity:** confirmed via explicit assertion that training covers steps 1–354 and testing covers steps 355–743, with zero overlap. Not an accidental shuffle.
3. **Feature-level leakage — confirmed, and this was the real issue.** A diagnostic Random Forest's feature importances:

   | Feature | Importance |
   |---|---|
   | `errorBalanceOrig` | 0.4261 |
   | `oldbalanceOrg` | 0.1786 |
   | `newbalanceOrig` | 0.1024 |
   | `newbalanceDest` | 0.0909 |
   | `amount` | 0.0472 |
   | `errorBalanceDest` | 0.0449 |
   | `hour_of_day` | 0.0358 |
   | `oldbalanceDest` | 0.0306 |
   | `dest_txn_count_prior` | 0.0151 |
   | `type_TRANSFER` | 0.0128 |

   `errorBalanceOrig` alone drives 42.6% of the model's decisions. Combined with `oldbalanceOrg`, `newbalanceOrig`, `newbalanceDest`, and `errorBalanceDest`, post-transaction-derived fields account for the overwhelming majority of feature importance.

### Why this happened, specifically

`newbalanceOrig` and `newbalanceDest` are only known **after** a transaction completes — a real-time authorization system doesn't have these values when it needs to make a decision. `errorBalanceOrig` and `errorBalanceDest` are computed directly from them (`newbalanceOrig + amount - oldbalanceOrg` and `oldbalanceDest + amount - newbalanceDest`), so they inherit the same problem.

PaySim's fraud-simulation logic has a mechanical quirk: fraudulent transfers drain the sender's account to (near) exactly zero, regardless of the stated amount, far more consistently than legitimate transactions do. Checked directly on the data:

- **98.05%** of fraud cases have `newbalanceOrig == 0` (the account fully drained)
- **97.82%** of fraud cases have `oldbalanceOrg == amount` (drained by exactly the transaction amount)

A model trained on these fields isn't predicting fraud before it happens — it's reading a signature left behind by simulated fraud's *outcome*, which is available only after the fact. It's measuring "can I detect PaySim's fraud-generation logic," not "can I detect fraud before it happens."

### The fix, and the before/after gap it exposed

`newbalanceOrig`, `newbalanceDest`, `errorBalanceOrig`, and `errorBalanceDest` were excluded, and the full 6-way model/strategy comparison was rerun on the reduced ("pre-transaction") feature set, side by side with the original ("full-feature") one:

| Model + strategy | Full-feature precision | recall | f1 | pr_auc | Pre-transaction precision | recall | f1 | pr_auc |
|---|---|---|---|---|---|---|---|---|
| Random Forest + class_weight | 0.9998 | 0.9988 | 0.9993 | 1.0000 | 0.3512 | 0.8713 | 0.5006 | 0.8204 |
| XGBoost + class_weight | 0.9991 | 0.9995 | 0.9993 | 0.9998 | 0.8318 | 0.8283 | 0.8301 | 0.8898 |
| Random Forest + smote | 0.9991 | 0.9995 | 0.9993 | 1.0000 | 0.7327 | 0.6980 | 0.7149 | 0.7872 |
| XGBoost + smote | 0.9986 | 0.9984 | 0.9985 | 0.9998 | 0.9450 | 0.7384 | 0.8290 | 0.9073 |
| Random Forest + undersample | 0.9970 | 0.9998 | 0.9984 | 0.9998 | 0.5630 | 0.8372 | 0.6733 | 0.8305 |
| XGBoost + undersample | 0.9937 | 0.9986 | 0.9961 | 0.9992 | 0.6497 | 0.8967 | 0.7534 | 0.9114 |

Every row drops substantially once the leaky fields are removed — that drop **is the size of the leak**. The full-feature column is retained only as a labeled upper-bound/diagnostic benchmark ([05_interpretability.ipynb](05_interpretability.ipynb) uses SHAP to independently confirm the same leak). The pre-transaction column is this project's real, deployable result.

## 3. Model & threshold selection

### Chosen model: XGBoost + undersampling

Among the six pre-transaction candidates above, **XGBoost + undersampling has the highest PR-AUC (0.9114)** — the metric that best summarizes ranking quality across all thresholds — while also delivering strong recall (89.67% at the default 0.5 cutoff), the metric this business cares about most (missed fraud is a direct loss). XGBoost + SMOTE is close behind on PR-AUC (0.9073) but trades away meaningfully more recall (73.84%) for higher precision, which doesn't match this project's stated priorities.

### Threshold comparison

Every metric elsewhere in this project uses the default 0.5 cutoff, which is arbitrary — just the midpoint of the probability range, not a decision about this business's actual costs. [07_threshold_tuning.ipynb](07_threshold_tuning.ipynb) sweeps thresholds from 0.10 to 0.90 in steps of 0.05 and compares three candidates:

| Scenario | Threshold | Precision | Recall | F1 | False positives | False negatives |
|---|---|---|---|---|---|---|
| Default threshold | 0.50 | 0.6497 | 0.8967 | 0.7534 | 2,059 | 440 |
| F1-optimal threshold | 0.90 | 0.8687 | 0.8314 | 0.8496 | 535 | 718 |
| **Recall-prioritized threshold (recall ≥ 0.90) — recommended** | **0.45** | 0.6233 | 0.9030 | 0.7375 | 2,324 | 413 |

(Within the requested 0.10–0.90 grid, F1 is maximized at the top edge, 0.90; a finer check confirms the true nearby optimum sits around threshold ≈ 0.93, F1 ≈ 0.8528, before turning back over — the conclusion doesn't change either way.)

**Core reasoning:** F1 treats a false positive and a false negative as equally costly. That assumption doesn't hold in fraud detection — a missed fraud case is a direct, generally unrecoverable financial loss, while a false positive costs one bounded, plannable manual review. Optimizing for F1 therefore quietly bakes in a cost assumption this business doesn't actually hold, which is why the F1-optimal threshold (0.90) was rejected in favor of the recall-prioritized one.

**Recommendation: threshold 0.45 for production.** It catches more fraud than the default (413 missed vs. 440 at 0.5 — 27 fewer missed cases) at a bounded cost of ~265 additional manual reviews (2,324 vs. 2,059 false positives), roughly **9 extra reviews per additional fraud case caught**.

**Assumptions this rests on — flagged explicitly, not settled fact:**
- **(a) The relative dollar cost of a missed fraud case vs. a false alarm was not quantified.** The recommendation assumes missed fraud is *more* costly than a false alarm, but not by how much — a formal cost model could shift the optimal threshold in either direction.
- **(b) Review team capacity to absorb the additional ~265 flagged transactions (per test period) was not modeled.** If reviewer capacity is tightly constrained, the F1-optimal or default threshold could be the more operationally realistic choice despite catching less fraud.

## 4. Key results

### a. Naive full-feature model — INVALID (data leakage)

| Model + strategy | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| Random Forest + class_weight | 0.9998 | 0.9988 | 0.9993 | 1.0000 |
| XGBoost + class_weight | 0.9991 | 0.9995 | 0.9993 | 0.9998 |
| Random Forest + smote | 0.9991 | 0.9995 | 0.9993 | 1.0000 |
| XGBoost + smote | 0.9986 | 0.9984 | 0.9985 | 0.9998 |
| Random Forest + undersample | 0.9970 | 0.9998 | 0.9984 | 0.9998 |
| XGBoost + undersample | 0.9937 | 0.9986 | 0.9961 | 0.9992 |

Presented deliberately, not hidden: this is what a model trained on post-transaction fields (`newbalanceOrig`, `newbalanceDest`, `errorBalanceOrig`, `errorBalanceDest`) looks like. Every single combination clears 0.99 on every metric — the diagnostic finding described in Section 2, not a claim about deployable performance.

### b. Corrected pre-transaction model comparison (deployable / realistic)

| Model + strategy | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| Random Forest + class_weight | 0.3512 | 0.8713 | 0.5006 | 0.8204 |
| XGBoost + class_weight | 0.8318 | 0.8283 | 0.8301 | 0.8898 |
| Random Forest + smote | 0.7327 | 0.6980 | 0.7149 | 0.7872 |
| XGBoost + smote | 0.9450 | 0.7384 | 0.8290 | 0.9073 |
| Random Forest + undersample | 0.5630 | 0.8372 | 0.6733 | 0.8305 |
| **XGBoost + undersample** | **0.6497** | **0.8967** | **0.7534** | **0.9114** |

### c. Final threshold-tuned result (chosen production model + threshold)

**XGBoost + undersampling at threshold 0.45:** Precision 0.6233, Recall 0.9030, F1 0.7375 — 3,845 true positives, 2,324 false positives, 413 false negatives, on a test period of 552,504 transactions (4,258 of them fraud).

## 5. Methodology notes

- **Time-based split, not random.** Training uses steps 1–354 (2,217,905 rows, 80.1% of the scoped data, 0.1783% fraud rate), testing uses steps 355–743 (552,504 rows, 19.9%, 0.7707% fraud rate) — a real temporal holdout simulating deployment on transactions the model hasn't seen, confirmed to have zero overlap.
- **No accuracy used as a primary metric.** With an overall fraud rate of 0.13% (and even the test period's elevated 0.77%), a model predicting "not fraud" for everything would score above 99% accuracy while catching zero fraud. Precision, recall, F1, and PR-AUC are used throughout instead.
- **`isFlaggedFraud` excluded as a model feature.** This is PaySim's own naive rule (flag `TRANSFER` > 200,000); it's used only as a baseline to beat ([03_baseline_modeling.ipynb](03_baseline_modeling.ipynb): precision 1.0, recall 0.0031, F1 0.0061, PR-AUC 0.0107 — it misses 99.7% of fraud), never as a trainable input, since including it would let a model partially "cheat" off a rule instead of learning generalizable patterns.

## 6. Interpretability

[05_interpretability.ipynb](05_interpretability.ipynb) runs SHAP on the primary pre-transaction model (XGBoost + undersample). Mean absolute SHAP value by feature, computed on a 3,000-row test sample:

| Feature | Mean \|SHAP\| |
|---|---|
| `oldbalanceOrg` | 4.010 |
| `amount` | 3.287 |
| `oldbalanceDest` | 0.941 |
| `dest_time_since_last_txn` | 0.648 |
| `day` | 0.600 |
| `hour_of_day` | 0.523 |
| `type_TRANSFER` | 0.383 |
| `dest_txn_count_prior` | 0.265 |
| `orig_txn_count_prior`, `orig_time_since_last_txn`, `orig_dual_role_prior`, `dest_dual_role_prior` | ~0.000 |

The model relies mainly on the sender's pre-transaction balance and the transaction amount, with the destination account's balance, recency, and prior activity as secondary signals. Notably, **all four of the sender-side ("orig") behavioral features engineered in 02 — plus the destination dual-role flag — contribute essentially nothing** to this particular model; see Section 7. A short SHAP pass on the old full-feature model independently reproduces the leakage finding from Section 2 (`errorBalanceOrig` dominates), agreeing with the plain `feature_importances_` ranking via a second, independent method.

[05_interpretability.ipynb](05_interpretability.ipynb) also walks through individual true positive, false negative, and false positive cases with SHAP waterfall explanations, and notes why this kind of case-level explainability is a model-risk-management expectation in financial services, not just a diagnostic nicety.

## 7. Limitations / future work

- **No dollar-cost quantification of the precision/recall trade-off** (flagged in Section 3): the threshold recommendation assumes, but doesn't quantify, that a missed fraud case costs more than a false alarm.
- **Review team capacity was not modeled** (flagged in Section 3): whether ~265 additional manual reviews per test period is operationally absorbable is unknown.
- **`CASH_OUT` fraud is caught far less reliably than `TRANSFER` fraud.** [06_error_analysis.ipynb](06_error_analysis.ipynb) finds 99.8% recall on `TRANSFER` fraud vs. only 79.5% on `CASH_OUT` fraud at the default threshold — the model's strongest remaining signal ("destination account is empty and brand-new") doesn't transfer as well to `CASH_OUT`.
- **The account-behavior features from 02 are mostly unused by the winning model.** Per the SHAP ranking in Section 6, `orig_txn_count_prior`, `orig_time_since_last_txn`, `orig_dual_role_prior`, and `dest_dual_role_prior` all show ~0 contribution — real engineering effort that isn't currently paying off for this specific model/strategy combination.
- **Concrete next steps identified in 06_error_analysis.ipynb:** type-specific modeling or interaction terms (to close the `CASH_OUT` gap); a graduated "destination account novelty" feature instead of the current near-binary empty/new signal (to reduce false positives from legitimate new-payee transactions); network/mule-account features (whether a destination has ever received funds connected to prior fraud); and amount normalization against an account's own historical transaction size (to catch smaller-scale fraud that currently blends into the noise).

## 8. Project structure

| Notebook | What it covers |
|---|---|
| [01_eda.ipynb](01_eda.ipynb) | Exploratory analysis of the raw PaySim data: structural checks, transaction-type breakdown and fraud rate by type, numeric feature distributions, balance-consistency ("error balance") features, and zero-balance/drained-account patterns — the observations that motivated later feature engineering. |
| [02_feature_engineering.ipynb](02_feature_engineering.ipynb) | Scopes the data to `TRANSFER`/`CASH_OUT`, drops `isFlaggedFraud` as a feature (kept only as a baseline reference), engineers accounting-identity error-balance features, time-of-day features, and leakage-safe point-in-time account behavioral features (transaction frequency, dual-role flag, recency). Saves the modeling table to disk. |
| [03_baseline_modeling.ipynb](03_baseline_modeling.ipynb) | Establishes the time-based train/test split and the precision/recall/F1/PR-AUC evaluation approach. Scores the naive `isFlaggedFraud` rule, then trains a class-weighted Logistic Regression as the first real model. |
| [04_advanced_modeling.ipynb](04_advanced_modeling.ipynb) | Investigates a suspiciously perfect early result as a first-class diagnostic (split order, time-split integrity, feature-level leakage via feature importances), confirms the leak is post-transaction balance fields, then runs the Random Forest / XGBoost × class-weighting / SMOTE / undersampling comparison twice — full-feature (upper bound) and pre-transaction (deployable) — side by side, plus a first pass at business-framed threshold tuning. |
| [05_interpretability.ipynb](05_interpretability.ipynb) | SHAP global feature importance and individual case explanations (true positive, false negative, false positive) for the primary pre-transaction model, plus a short SHAP cross-check on the old full-feature model that independently confirms the leakage finding from 04. |
| [06_error_analysis.ipynb](06_error_analysis.ipynb) | Characterizes the pre-transaction model's false negatives and false positives (440 and 2,059 cases respectively), finds a large `TRANSFER`/`CASH_OUT` recall gap and reliance on "is the destination account empty and brand-new," and ties both back to concrete feature engineering recommendations. |
| [07_threshold_tuning.ipynb](07_threshold_tuning.ipynb) | A dedicated threshold-tuning deep dive on the primary model: a 0.05-increment sweep from 0.10 to 0.90 with raw TP/FP/FN counts, precision/recall-vs-threshold and full PR-curve plots, an F1-optimal threshold and a recall-prioritized (≥90% recall) threshold, and an explicit, business-justified recommendation of 0.45 as the production threshold. |

## 9. Tech stack & data source

**Libraries:** pandas, numpy, scikit-learn, xgboost, imbalanced-learn (SMOTE / undersampling), shap, matplotlib, seaborn, joblib.

**Data source:** [PaySim — Synthetic Financial Datasets For Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/ealaxi/paysim1).

### Reproducing this

The raw dataset (`online_payment_fraud.csv`) and the derived `data/processed/` and `models/` artifacts are not committed to this repo (see `.gitignore`) — they're regenerated by running the notebooks in order.

1. Download the PaySim dataset and place it at `online_payment_fraud.csv` in the project root.
2. `pip install -r requirements.txt`
3. Run the notebooks in order, 01 through 07 — each one depends on artifacts saved by the previous ones.
