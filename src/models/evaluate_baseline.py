"""Evaluate the saved Logistic Regression baseline on the reserved test set."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "logistic_regression.joblib"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
FIGURE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "logistic_regression_confusion_matrix.png"
)

TARGET_COLUMN = "Class"
FEATURE_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
DEFAULT_THRESHOLD = 0.50


def print_heading(title: str) -> None:
    """Print a readable section heading."""
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def validate_test_data(data: pd.DataFrame) -> None:
    """Validate the expected features and binary test target."""
    required_columns = [*FEATURE_COLUMNS, TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(
            "Test data is missing required column(s): " + ", ".join(missing_columns)
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
        raise ValueError(f"Test features contain missing values: {details}")


def save_confusion_matrix(matrix: pd.DataFrame) -> None:
    """Save a professional high-resolution confusion-matrix visualization."""
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        linewidths=1,
        linecolor="white",
        annot_kws={"size": 16},
        ax=axis,
    )
    axis.set_title("Logistic Regression Confusion Matrix (Threshold = 0.50)")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_xticklabels(["Legitimate", "Fraudulent"])
    axis.set_yticklabels(["Legitimate", "Fraudulent"], rotation=0)
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Evaluate the existing baseline without retraining or threshold tuning."""
    print_heading("FraudGuard AI - Logistic Regression Holdout Evaluation")
    print(f"Model: {MODEL_PATH}")
    print(f"Test dataset: {TEST_PATH}")
    print(f"Decision threshold: {DEFAULT_THRESHOLD:.2f}")

    if not MODEL_PATH.is_file():
        print(f"\nError: Saved model not found at {MODEL_PATH}")
        return
    if not TEST_PATH.is_file():
        print(f"\nError: Test dataset not found at {TEST_PATH}")
        return

    try:
        model = joblib.load(MODEL_PATH)
        test_data = pd.read_csv(TEST_PATH)
        validate_test_data(test_data)
    except (OSError, pd.errors.ParserError, ValueError) as error:
        print(f"\nError loading or validating evaluation inputs: {error}")
        return

    features = test_data[FEATURE_COLUMNS]
    target = test_data[TARGET_COLUMN]
    fraud_probabilities = model.predict_proba(features)[:, 1]
    predictions = (fraud_probabilities >= DEFAULT_THRESHOLD).astype(int)

    matrix_values = confusion_matrix(target, predictions, labels=[0, 1])
    true_negatives, false_positives, false_negatives, true_positives = matrix_values.ravel()
    matrix = pd.DataFrame(
        matrix_values,
        index=["Legitimate", "Fraudulent"],
        columns=["Legitimate", "Fraudulent"],
    )

    print_heading("Evaluation Metrics")
    print(f"Precision: {precision_score(target, predictions, zero_division=0):.6f}")
    print(f"Recall: {recall_score(target, predictions, zero_division=0):.6f}")
    print(f"F1-score: {f1_score(target, predictions, zero_division=0):.6f}")
    print(f"Accuracy: {accuracy_score(target, predictions):.6f}")
    print(f"Average precision (PR-AUC): {average_precision_score(target, fraud_probabilities):.6f}")
    print(f"ROC-AUC: {roc_auc_score(target, fraud_probabilities):.6f}")

    print_heading("Confusion Matrix Counts")
    print(f"True positives: {true_positives:,}")
    print(f"False positives: {false_positives:,}")
    print(f"True negatives: {true_negatives:,}")
    print(f"False negatives: {false_negatives:,}")

    print_heading("Confusion Matrix")
    print(matrix.to_string())

    print_heading("Classification Report")
    print(
        classification_report(
            target,
            predictions,
            target_names=["Legitimate", "Fraudulent"],
            zero_division=0,
        )
    )

    save_confusion_matrix(matrix)
    print(f"Confusion matrix figure: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
