"""Train and evaluate a reproducible Logistic Regression fraud baseline."""

from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "logistic_regression.joblib"

TARGET_COLUMN = "Class"
FEATURE_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]


def print_heading(title: str) -> None:
    """Print a readable section heading."""
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


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


def build_pipeline() -> Pipeline:
    """Build the baseline pipeline so scaling is learned within each fold."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2_000,
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate_with_cross_validation(features: pd.DataFrame, target: pd.Series) -> list[float]:
    """Evaluate average precision using shuffled stratified five-fold CV."""
    cross_validator = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )
    pipeline = build_pipeline()
    average_precision_scorer = make_scorer(
        average_precision_score,
        response_method="predict_proba",
    )
    scores = cross_val_score(
        pipeline,
        features,
        target,
        cv=cross_validator,
        scoring=average_precision_scorer,
    )
    return scores.tolist()


def main() -> None:
    """Evaluate and fit the Logistic Regression baseline on training data only."""
    print_heading("FraudGuard AI - Logistic Regression Baseline")
    print(f"Training dataset: {TRAIN_PATH}")

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

    print_heading("Five-Fold Stratified Cross-Validation")
    start_time = perf_counter()
    try:
        scores = evaluate_with_cross_validation(features, target)
    except (ValueError, RuntimeError) as error:
        print(f"\nError during cross-validation: {error}")
        return

    for fold_number, score in enumerate(scores, start=1):
        print(f"Fold {fold_number} average precision (PR-AUC): {score:.6f}")
    print(f"Mean PR-AUC: {sum(scores) / len(scores):.6f}")
    print(
        "Standard deviation: "
        f"{pd.Series(scores).std(ddof=0):.6f}"
    )

    print_heading("Final Model Fit")
    model = build_pipeline()
    model.fit(features, target)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    elapsed_seconds = perf_counter() - start_time
    print(f"Training duration: {elapsed_seconds:.2f} seconds")
    print(f"Model artifact: {MODEL_PATH}")


if __name__ == "__main__":
    main()
