"""Train and evaluate a reproducible XGBoost fraud baseline."""

from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost.joblib"

TARGET_COLUMN = "Class"
FEATURE_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]


def print_heading(title: str) -> None:
    """Print a readable section heading."""
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def validate_training_data(data: pd.DataFrame) -> None:
    """Validate the expected features and binary training target."""
    required_columns = [*FEATURE_COLUMNS, TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(
            "Training data is missing required column(s): "
            + ", ".join(missing_columns)
        )

    target_values = set(data[TARGET_COLUMN].dropna().unique())
    if data[TARGET_COLUMN].isna().any() or not target_values.issubset({0, 1}):
        raise ValueError(
            f"The '{TARGET_COLUMN}' target must be non-missing and binary with "
            f"values 0 and 1; found: {sorted(target_values)}."
        )

    missing_features = data[FEATURE_COLUMNS].isna().sum()
    columns_with_missing = missing_features[missing_features > 0]
    if not columns_with_missing.empty:
        details = ", ".join(
            f"{column} ({count})" for column, count in columns_with_missing.items()
        )
        raise ValueError(f"Training features contain missing values: {details}")


def calculate_scale_pos_weight(target: pd.Series) -> float:
    """Calculate the legitimate-to-fraud ratio for one training partition."""
    legitimate_count = int(target.eq(0).sum())
    fraud_count = int(target.eq(1).sum())
    if fraud_count == 0:
        raise ValueError("A training partition contains no fraudulent transactions.")
    return legitimate_count / fraud_count


def build_model(scale_pos_weight: float) -> XGBClassifier:
    """Build the regularized XGBoost classifier for one training partition."""
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        min_child_weight=2,
        reg_alpha=0.1,
        reg_lambda=1.0,
        max_bin=256,
        verbosity=0,
    )


def evaluate_with_cross_validation(
    features: pd.DataFrame,
    target: pd.Series,
    cross_validator: StratifiedKFold,
) -> list[float]:
    """Evaluate average precision with fold-specific class weighting."""
    scores = []
    for fold_number, (train_indices, validation_indices) in enumerate(
        cross_validator.split(features, target),
        start=1,
    ):
        fold_features = features.iloc[train_indices]
        fold_target = target.iloc[train_indices]
        validation_features = features.iloc[validation_indices]
        validation_target = target.iloc[validation_indices]
        fold_scale_pos_weight = calculate_scale_pos_weight(fold_target)

        model = build_model(fold_scale_pos_weight)
        model.fit(fold_features, fold_target)
        validation_probabilities = model.predict_proba(validation_features)[:, 1]
        score = average_precision_score(validation_target, validation_probabilities)
        scores.append(score)
        print(
            f"Fold {fold_number} PR-AUC: {score:.6f} "
            f"(scale_pos_weight={fold_scale_pos_weight:.6f})"
        )

    return scores


def main() -> None:
    """Evaluate and fit the XGBoost baseline on training data only."""
    print_heading("FraudGuard AI - XGBoost Baseline")
    print(f"Training dataset: {TRAIN_PATH}")
    print("Holdout test data is intentionally not loaded.")
    print("Memory-conscious note: five-fold XGBoost cross-validation may take several minutes.")

    if not TRAIN_PATH.is_file():
        print(
            "\nError: Training dataset file not found."
            f"\nExpected file: {TRAIN_PATH}"
            "\nRun the preparation step first and try again."
        )
        return

    try:
        training_data = pd.read_csv(TRAIN_PATH)
        validate_training_data(training_data)
    except (OSError, pd.errors.ParserError, ValueError) as error:
        print(f"\nError loading or validating training data: {error}")
        return

    features = training_data[FEATURE_COLUMNS]
    target = training_data[TARGET_COLUMN]
    cross_validator = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    print_heading("Five-Fold Stratified Cross-Validation")
    start_time = perf_counter()
    try:
        scores = evaluate_with_cross_validation(features, target, cross_validator)
    except (ValueError, RuntimeError) as error:
        print(f"\nError during cross-validation: {error}")
        return

    score_series = pd.Series(scores)
    print(f"Mean PR-AUC: {score_series.mean():.6f}")
    print(f"Standard deviation: {score_series.std(ddof=0):.6f}")

    print_heading("Final Model Fit")
    full_scale_pos_weight = calculate_scale_pos_weight(target)
    print(f"Full-training scale_pos_weight: {full_scale_pos_weight:.6f}")
    model = build_model(full_scale_pos_weight)
    model.fit(features, target)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    elapsed_seconds = perf_counter() - start_time
    print(f"Training duration: {elapsed_seconds:.2f} seconds")
    print(f"Model artifact: {MODEL_PATH}")


if __name__ == "__main__":
    main()
