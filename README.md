# FraudGuard AI

FraudGuard AI is an explainable machine-learning system for financial transaction fraud detection. It combines a reproducible model-development workflow with a professional Streamlit interface for historical analysis, single-transaction scoring, batch scoring, model-performance review, and model explanations.

This repository is an educational and research portfolio project. It is not a production banking system, does not connect to live payment infrastructure, and must not be used for unsupervised financial decisions.

## Project motivation

Fraud detection is a highly imbalanced classification problem: legitimate transactions greatly outnumber fraudulent ones. A useful system must therefore look beyond accuracy and make the precision, recall, false-alarm, and missed-fraud trade-offs visible. FraudGuard AI demonstrates that workflow while keeping the model, threshold, data split, holdout evaluation, and explainability assumptions explicit.

## Application features

The Streamlit application contains six sections:

1. **Dashboard** — Historical holdout summary and confusion-matrix overview.
2. **Single Transaction Prediction** — Score one transaction using the frozen Random Forest candidate.
3. **Batch Fraud Detection** — Validate, score, search, visualize, and download predictions for uploaded CSV files.
4. **Model Performance** — Review saved holdout metrics and training-only model comparisons.
5. **Explainable AI** — Explore prepared SHAP feature-importance, beeswarm, and waterfall figures.
6. **About** — Review the workflow, technology stack, dataset context, and responsible-use limitations.

The application uses a fixed fraud classification threshold of **0.40**. A score at or above the threshold is flagged for review; this is a model decision rule, not proof that fraud occurred.

## Technology stack

- Python 3.12.4
- pandas and NumPy for data handling
- scikit-learn for validation, metrics, and Random Forest
- XGBoost for model comparison
- SHAP for explainability
- Streamlit for the application
- Plotly for interactive charts
- Matplotlib and Seaborn for saved analysis figures
- pytest for automated regression tests
- joblib for model artifact serialization

The full local-development dependencies are listed in [requirements.txt](./requirements.txt). The smaller deployment environment is listed in [requirements-deploy.txt](./requirements-deploy.txt).

## Workflow and architecture

```text
Public credit-card fraud dataset
        |
        v
Schema checks -> exact-duplicate removal for the initial experiment
        |
        v
80/20 stratified random train/test split (random_state=42)
        |
        +--> Logistic Regression baseline
        +--> Random Forest
        +--> XGBoost
        |
        v
Training-only five-fold cross-validation and out-of-fold threshold analysis
        |
        v
Frozen Random Forest candidate + threshold 0.40
        |
        +--> One-time untouched holdout evaluation
        +--> Training-data SHAP explanations
        +--> Streamlit application
```

The application imports the frozen schema and threshold from [src/models/final_model_config.py](./src/models/final_model_config.py). Runtime inference preserves the exact feature order: `Time`, `V1` through `V28`, and `Amount`.

## Dataset and preprocessing

The project uses the public **Kaggle / ULB Credit Card Fraud Detection** dataset. The underlying dataset is credited to its public source; FraudGuard AI does not claim ownership of the data.

The original dataset contains:

- **284,807 transactions**
- **492 originally labeled fraudulent transactions**
- 30 input features plus the binary `Class` target
- Severe class imbalance, making accuracy alone a poor primary measure

For the initial experiment, exact duplicate rows were removed in memory before splitting. The original CSV is preserved locally and is not modified. The resulting data was divided with an 80/20 stratified random split using `random_state=42`:

- Training set: 226,980 transactions, including 378 fraud cases
- Holdout test set: 56,746 transactions, including 95 fraud cases

The random split and duplicate-removal policy are limitations. Identical rows are not necessarily erroneous transactions, and a random split is not a substitute for future time-based validation using the `Time` feature. Results may also change under different duplicate policies, temporal windows, or transaction populations.

## Model development

Three models were compared using training-only five-fold stratified cross-validation:

| Model | Mean CV PR-AUC |
|---|---:|
| Logistic Regression | 0.738234 |
| Random Forest | 0.839862 |
| XGBoost | 0.835169 |

PR-AUC was emphasized because fraud is rare and the precision-recall trade-off is more operationally informative than accuracy. Random Forest was selected as the **provisional leading candidate** because it achieved the highest mean cross-validation PR-AUC in this experiment. This does not establish permanent superiority under future data or temporal validation.

The threshold was selected using training-only out-of-fold predictions. A threshold of **0.40** maximized F1 only among the evaluated Random Forest threshold values; it was frozen before holdout evaluation and was not optimized on the holdout set.

## Final historical holdout results

The following values come from the saved one-time evaluation in [reports/final_model_metrics.json](./reports/final_model_metrics.json). They are historical test results, not live monitoring metrics and not a guarantee of future performance.

| Metric | Result |
|---|---:|
| Precision | 0.945946 |
| Recall | 0.736842 |
| F1-score | 0.828402 |
| PR-AUC | 0.809520 |
| ROC-AUC | 0.953164 |

Confusion matrix at threshold 0.40:

|  | Predicted legitimate | Predicted fraud |
|---|---:|---:|
| **Actual legitimate** | TN: 56,647 | FP: 4 |
| **Actual fraud** | FN: 25 | TP: 70 |

The model caught 70 of 95 fraud cases in this holdout evaluation and missed 25. It raised four false alarms among legitimate transactions. Production monitoring, calibration, temporal validation, drift analysis, and human investigation controls would be required before operational use.

## Explainability

SHAP figures are generated from a reproducible sample of training data and are displayed as prepared artifacts in the application. They describe model behavior; they do not establish why fraud actually occurred.

`V1`–`V28` are anonymized principal-component features. They cannot be interpreted directly as named financial behaviors and cannot be reconstructed from a card number, merchant, or customer name. SHAP contributions are expressed in the explainer's output units and should not automatically be interpreted as percentage-point changes in fraud probability.

## Synthetic demonstrations

The deployed application does not require the original training CSV. Its optional single-transaction example and downloadable batch template use small, deterministic, artificial feature values with the correct 30-column schema. These values are not copied from the Kaggle dataset, do not contain labels, and are provided only to test the interface. They are not genuine transactions and are not evidence of real-world fraud-detection performance.

Users may still enter feature values manually or upload their own compatible CSV files. Uploaded files are validated for schema, duplicate columns, numeric and finite values, nonnegative amounts, file size, column count, and row count before inference.

## Local setup

Create and activate a virtual environment, then install the full development dependencies:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
```

To run the application locally:

```powershell
streamlit run app/main.py
```

For a lean deployment environment, install [requirements-deploy.txt](./requirements-deploy.txt) instead:

```powershell
python -m pip install -r requirements-deploy.txt
```

Run the regression suite with:

```powershell
python -m pytest tests/test_regression.py -v
```

The tests use synthetic data and mocked estimators. They do not require the large dataset or production model artifact.

## Repository structure

```text
app/
  main.py                         # Streamlit application
data/
  raw/                            # Local source data; excluded from publication
  processed/                      # Local train/test CSVs; excluded from publication
models/
  random_forest.joblib            # Approved frozen deployment artifact
reports/
  final_model_metrics.json        # Publishable historical holdout metrics
  figures/                        # Approved SHAP figures and local analysis outputs
src/
  data/                           # Exploration, visualization, validation, preparation
  models/                         # Training, evaluation, threshold analysis, configuration
  explainability/                 # SHAP generation code
tests/
  test_regression.py              # Fast deterministic regression tests
requirements.txt                  # Full development environment
requirements-deploy.txt           # Lean Streamlit runtime environment
```

## Privacy and responsible use

- Educational and research demonstration only; not a production banking system.
- No real-time banking integrations or authorization controls are included.
- Model scores are not guaranteed calibrated probabilities.
- False positives and false negatives remain possible.
- Uploaded transaction files may contain sensitive information; use synthetic or properly governed data.
- The original Kaggle dataset, training CSVs, and holdout CSV are excluded from publication.
- Historical holdout metrics describe one frozen experiment and must not be presented as live monitoring results.
- The initial random stratified split is not temporal validation.
- Exact duplicate removal is an experimental policy, not proof that duplicate transactions were invalid.
- Future deployment would require privacy review, access control, audit logging, calibration, temporal validation, drift monitoring, and human oversight.

## Author

**Mark Bryson Mutuma**

FraudGuard AI is presented as a transparent, reproducible explainable-ML portfolio project.

## License

This project is licensed under the [MIT License](./LICENSE).
