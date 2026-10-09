"""Analyze Logistic Regression precision-recall tradeoffs using training data only."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
FIGURE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "logistic_regression_threshold_analysis.png"
)

TARGET_COLUMN = "Class"
FEATURE_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def print_heading(title: str) -> None:
    """Print a readable section heading."""
    print(f"\n{'=' * 92}\n{title}\n{'=' * 92}")


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
    """Build the same scaled, class-weighted baseline pipeline."""
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


def evaluate_thresholds(
    target: pd.Series, fraud_probabilities: pd.Series
) -> pd.DataFrame:
    """Calculate threshold metrics from out-of-fold fraud probabilities."""
    results = []
    for threshold in THRESHOLDS:
        predictions = (fraud_probabilities >= threshold).astype(int)
        true_positives = int(((target == 1) & (predictions == 1)).sum())
        false_positives = int(((target == 0) & (predictions == 1)).sum())
        false_negatives = int(((target == 1) & (predictions == 0)).sum())
        results.append(
            {
                "threshold": threshold,
                "precision": precision_score(target, predictions, zero_division=0),
                "recall": recall_score(target, predictions, zero_division=0),
                "f1": f1_score(target, predictions, zero_division=0),
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
            }
        )
    return pd.DataFrame(results)


def save_threshold_chart(results: pd.DataFrame) -> None:
    """Save a precision-recall versus threshold chart."""
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(11, 7))
    sns.lineplot(
        data=results,
        x="threshold",
        y="precision",
        marker="o",
        linewidth=2.5,
        label="Precision",
        color="#2f6f9f",
        ax=axis,
    )
    sns.lineplot(
        data=results,
        x="threshold",
        y="recall",
        marker="o",
        linewidth=2.5,
        label="Recall",
        color="#c44e52",
        ax=axis,
    )
    axis.set_title("Logistic Regression Precision-Recall Tradeoff by Threshold")
    axis.set_xlabel("Fraud probability threshold")
    axis.set_ylabel("Score")
    axis.set_xticks(THRESHOLDS)
    axis.set_ylim(0, 1.05)
    axis.legend(title="Metric")
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Generate training-only out-of-fold threshold analysis."""
    print_heading("FraudGuard AI - Training-Only Threshold Analysis")
    print(f"Training dataset: {TRAIN_PATH}")
    print("Holdout test data is intentionally not loaded.")

    if not TRAIN_PATH.is_file():
        print(
            "\nError: Training dataset file not found."
            f"\nExpected file: {TRAIN_PATH}"
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

    print_heading("Generating Out-of-Fold Fraud Probabilities")
    print("Scaling is fitted separately inside each cross-validation training fold.")
    try:
        fraud_probabilities = cross_val_predict(
            build_pipeline(),
            features,
            target,
            cv=cross_validator,
            method="predict_proba",
        )[:, 1]
    except (ValueError, RuntimeError) as error:
        print(f"\nError during cross-validation: {error}")
        return

    results = evaluate_thresholds(target, pd.Series(fraud_probabilities, index=target.index))

    print_heading("Threshold Metrics")
    print(
        results.to_string(
            index=False,
            formatters={
                "threshold": "{:.2f}".format,
                "precision": "{:.6f}".format,
                "recall": "{:.6f}".format,
                "f1": "{:.6f}".format,
            },
        )
    )

    best_row = results.loc[results["f1"].idxmax()]
    print_heading("Training-Only Candidate")
    print(
        f"Highest evaluated F1: {best_row['f1']:.6f} "
        f"at threshold {best_row['threshold']:.2f}"
    )
    print("This is a training-only candidate, not a final validated optimum.")

    save_threshold_chart(results)
    print(f"Threshold analysis figure: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
