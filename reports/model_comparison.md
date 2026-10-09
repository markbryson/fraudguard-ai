# FraudGuard AI Model Comparison

## Executive summary

FraudGuard AI evaluated three binary fraud-classification baselines using the training dataset and stratified five-fold cross-validation. Random Forest is the provisional leading candidate because it achieved the strongest fold-mean PR-AUC and the strongest reported training-only candidate F1. It is not yet conclusively superior: the candidate thresholds were selected from training-only out-of-fold predictions, and the final choice must be confirmed once the model and threshold are frozen and evaluated exactly once on the untouched holdout set.

## Model comparison

| Model | Mean five-fold PR-AUC | CV standard deviation | Pooled out-of-fold PR-AUC | Candidate threshold | Precision | Recall | F1-score | TP | FP | FN | Training duration |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.738234 | 0.026329 | Not reported | 0.90 | Not reported | Not reported | 0.399030 | Not reported | Not reported | Not reported | 5.14 s |
| Random Forest | **0.839862** | 0.035980 | 0.836907 | 0.40 | 0.907463 | 0.804233 | **0.852735** | 304 | 31 | 74 | 236.56 s |
| XGBoost | 0.835169 | 0.029087 | 0.832490 | 0.90 | 0.868715 | **0.822751** | 0.845109 | 311 | 47 | 67 | **20.53 s** |

The Logistic Regression results supplied for this report include its best evaluated F1 but not the corresponding precision, recall, or confusion counts. Those fields are therefore marked as not reported rather than inferred.

## Why PR-AUC is the primary comparison metric

The dataset is highly imbalanced: fraudulent transactions are a small minority of all transactions. In this setting, accuracy can be misleading because a classifier could obtain a very high accuracy by predicting almost every transaction as legitimate while missing a large share of fraud.

Average precision, commonly used as the area under the precision-recall curve (PR-AUC), focuses on the positive fraud class. It summarizes how well a model retrieves fraud while controlling the amount of legitimate activity incorrectly flagged as fraud. This makes PR-AUC more informative than accuracy for comparing models on this problem.

## Precision, recall, and operational trade-offs

At their reported training-only candidate thresholds, Random Forest has higher precision and F1 than XGBoost, while XGBoost has slightly higher recall:

- Random Forest: precision 0.907463, recall 0.804233, 31 false positives, and 74 false negatives.
- XGBoost: precision 0.868715, recall 0.822751, 47 false positives, and 67 false negatives.

Random Forest therefore produces fewer false alarms and a higher F1 at the evaluated candidate threshold, while XGBoost catches seven more fraud cases in the reported out-of-fold candidate results and produces 16 additional false positives.

The business choice depends on the cost of each error. A false negative is missed fraud and may create direct financial loss, chargebacks, investigation costs, and customer harm. A false positive is a legitimate transaction incorrectly flagged, which can create manual-review workload, transaction friction, customer frustration, or lost revenue. A fraud operations team that prioritizes catching more fraud may accept more false alarms; a team with limited review capacity may prefer higher precision. These costs should be made explicit before selecting a production threshold.

## Interpreting the PR-AUC figures

Two different PR-AUC summaries are shown:

- **Mean five-fold PR-AUC** is the arithmetic mean of the five validation-fold average-precision scores. The accompanying standard deviation describes variation across those folds.
- **Pooled out-of-fold PR-AUC** is calculated once by concatenating the prediction for each observation from the fold where that observation was held out, then applying `average_precision_score` to all out-of-fold predictions together.

They answer related but different questions and should not be expected to match exactly. The fold mean is useful for understanding fold-to-fold stability; the pooled out-of-fold value is useful as a single training-only summary over one prediction per training observation.

## Threshold-analysis caveat

The candidate thresholds were selected from training-only out-of-fold predictions. The reported candidate F1-scores are useful for model comparison and operational discussion, but they may be optimistic because the same training data contributed to threshold selection. They are not final validated production estimates.

The correct next step is to freeze the selected model and threshold, then perform one evaluation on the untouched holdout test set. That holdout must not be used to search across thresholds or revise the model after observing its results.

## Provisional recommendation

Random Forest is the provisional leading candidate based on the highest mean five-fold PR-AUC (0.839862), highest reported pooled out-of-fold PR-AUC (0.836907), and highest reported candidate F1 (0.852735). This is a provisional conclusion, not proof that Random Forest is conclusively superior.

XGBoost is a strong alternative. Its mean five-fold PR-AUC (0.835169) and pooled out-of-fold PR-AUC (0.832490) are close to Random Forest, it has slightly higher recall at the evaluated candidate threshold, and it trains much faster (20.53 seconds versus 236.56 seconds). The faster training time may be valuable for iteration, retraining, or operational constraints. The final choice should therefore consider the untouched holdout results, error costs, review capacity, reproducibility, and retraining requirements - not PR-AUC alone.

## Limitations and next validation steps

The current experiment has several limitations:

1. The data was prepared with an initial random stratified 80/20 split. Financial transactions can contain temporal drift, so a random split may not represent deployment conditions.
2. Exact duplicate rows were removed before the split. This is appropriate for the initial experiment, but the effect of the duplicate policy should be tested through sensitivity analysis, including comparisons with duplicate retention or alternative deduplication rules.
3. Threshold candidates were evaluated on training-only out-of-fold predictions and have not yet been validated on untouched data.
4. The current comparison does not yet quantify the financial or operational cost of false positives versus false negatives.

Future work should include a time-based holdout or rolling validation design using the `Time` feature, sensitivity analysis for duplicate handling, cost-based threshold analysis, and a single frozen-model evaluation on the reserved test set.

## Reproducibility references

Training scripts:

- [Logistic Regression training](../src/models/train_baseline.py)
- [Random Forest training](../src/models/train_random_forest.py)
- [XGBoost training](../src/models/train_xgboost.py)

Training-only threshold-analysis scripts:

- [Logistic Regression threshold analysis](../src/models/analyze_thresholds.py)
- [Random Forest threshold analysis](../src/models/analyze_random_forest_thresholds.py)
- [XGBoost threshold analysis](../src/models/analyze_xgboost_thresholds.py)

Evaluation and figure references:

- [Holdout evaluation script](../src/models/evaluate_baseline.py)
- [Logistic Regression confusion matrix](./figures/logistic_regression_confusion_matrix.png)
- [Logistic Regression threshold analysis](./figures/logistic_regression_threshold_analysis.png)
- [Random Forest threshold analysis](./figures/random_forest_threshold_analysis.png)
- [XGBoost threshold analysis](./figures/xgboost_threshold_analysis.png)
