"""Evaluate the frozen Random Forest candidate once on the untouched holdout set."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from final_model_config import (
        CLASSIFICATION_THRESHOLD,
        FEATURE_COLUMNS,
        MODEL_ARTIFACT_PATH,
        SELECTED_MODEL_NAME,
        TARGET_COLUMN,
    )
except ImportError:
    from src.models.final_model_config import (
        CLASSIFICATION_THRESHOLD,
        FEATURE_COLUMNS,
        MODEL_ARTIFACT_PATH,
        SELECTED_MODEL_NAME,
        TARGET_COLUMN,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "final_model_confusion_matrix.png"
METRICS_PATH = PROJECT_ROOT / "reports" / "final_model_metrics.json"


def print_heading(title: str) -> None:
    """Print a readable section heading."""
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def validate_holdout_data(data: pd.DataFrame) -> None:
    """Validate schema, numeric features, finite values, and binary target."""
    required_columns = [*FEATURE_COLUMNS, TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in data.columns]
    unexpected_columns = [column for column in data.columns if column not in required_columns]
    if missing_columns or unexpected_columns:
        details = []
        if missing_columns:
            details.append("missing: " + ", ".join(missing_columns))
        if unexpected_columns:
            details.append("unexpected: " + ", ".join(unexpected_columns))
        raise ValueError("Invalid holdout schema (" + "; ".join(details) + ").")

    feature_data = data[FEATURE_COLUMNS]
    non_numeric_features = [
        column
        for column in FEATURE_COLUMNS
        if not pd.api.types.is_numeric_dtype(feature_data[column])
    ]
    if non_numeric_features:
        raise ValueError(
            "Feature columns must be numeric: " + ", ".join(non_numeric_features)
        )

    if feature_data.isna().any().any():
        missing_features = feature_data.columns[feature_data.isna().any()].tolist()
        raise ValueError(
            "Feature columns contain missing values: " + ", ".join(missing_features)
        )

    if not np.isfinite(feature_data.to_numpy(dtype=float)).all():
        raise ValueError("Feature columns contain non-finite numeric values.")

    target = data[TARGET_COLUMN]
    if not pd.api.types.is_numeric_dtype(target):
        raise ValueError(f"The '{TARGET_COLUMN}' target must be numeric and binary.")
    target_values = set(target.dropna().unique())
    if target.isna().any() or not target_values.issubset({0, 1}):
        raise ValueError(
            f"The '{TARGET_COLUMN}' target must be non-missing and binary with "
            f"values 0 and 1; found: {sorted(target_values)}."
        )


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
    axis.set_title(
        f"Final {SELECTED_MODEL_NAME} Confusion Matrix "
        f"(Threshold = {CLASSIFICATION_THRESHOLD:.2f})"
    )
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_xticklabels(["Legitimate", "Fraudulent"])
    axis.set_yticklabels(["Legitimate", "Fraudulent"], rotation=0)
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_metrics(metrics: dict[str, object], matrix: pd.DataFrame) -> None:
    """Save JSON-safe metrics and frozen configuration details."""
    payload = {
        "evaluation_scope": "One-time untouched holdout evaluation",
        "model_name": SELECTED_MODEL_NAME,
        "model_artifact": str(MODEL_ARTIFACT_PATH.relative_to(PROJECT_ROOT)),
        "threshold": float(CLASSIFICATION_THRESHOLD),
        "feature_columns": list(FEATURE_COLUMNS),
        "target_column": TARGET_COLUMN,
        "metrics": {key: float(value) for key, value in metrics.items()},
        "confusion_matrix": {
            "true_negatives": int(matrix.iloc[0, 0]),
            "false_positives": int(matrix.iloc[0, 1]),
            "false_negatives": int(matrix.iloc[1, 0]),
            "true_positives": int(matrix.iloc[1, 1]),
        },
        "threshold_selection_note": (
            "Threshold was frozen before holdout evaluation; no threshold tuning "
            "was performed on the holdout set."
        ),
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Evaluate the frozen model without retraining or threshold optimization."""
    print_heading("FraudGuard AI - Final Model Holdout Evaluation")
    print("This is a one-time evaluation of the untouched holdout dataset.")
    print("The model and threshold are frozen; no retraining or threshold tuning will occur.")
    print(f"Model: {SELECTED_MODEL_NAME}")
    print(f"Model artifact: {MODEL_ARTIFACT_PATH}")
    print(f"Holdout dataset: {TEST_PATH}")
    print(f"Fixed threshold: {CLASSIFICATION_THRESHOLD:.2f}")

    if not TEST_PATH.is_file():
        print(f"\nError: Untouched holdout dataset not found at {TEST_PATH}")
        return
    if not MODEL_ARTIFACT_PATH.is_file():
        print(f"\nError: Frozen model artifact not found at {MODEL_ARTIFACT_PATH}")
        return

    try:
        holdout_data = pd.read_csv(TEST_PATH)
        validate_holdout_data(holdout_data)
        model = joblib.load(MODEL_ARTIFACT_PATH)
    except (OSError, pd.errors.ParserError, ValueError) as error:
        print(f"\nError loading or validating evaluation inputs: {error}")
        return

    features = holdout_data[FEATURE_COLUMNS]
    target = holdout_data[TARGET_COLUMN]

    try:
        fraud_probabilities = model.predict_proba(features)[:, 1]
    except (AttributeError, ValueError, RuntimeError) as error:
        print(f"\nError generating holdout predictions: {error}")
        return

    predictions = (fraud_probabilities >= CLASSIFICATION_THRESHOLD).astype(int)
    matrix_values = confusion_matrix(target, predictions, labels=[0, 1])
    true_negatives, false_positives, false_negatives, true_positives = matrix_values.ravel()
    matrix = pd.DataFrame(
        matrix_values,
        index=["Legitimate", "Fraudulent"],
        columns=["Legitimate", "Fraudulent"],
    )

    metrics = {
        "precision": precision_score(target, predictions, zero_division=0),
        "recall": recall_score(target, predictions, zero_division=0),
        "f1_score": f1_score(target, predictions, zero_division=0),
        "accuracy": accuracy_score(target, predictions),
        "pr_auc": average_precision_score(target, fraud_probabilities),
        "roc_auc": roc_auc_score(target, fraud_probabilities),
    }

    print_heading("Holdout Metrics")
    print(f"Precision: {metrics['precision']:.6f}")
    print(f"Recall: {metrics['recall']:.6f}")
    print(f"F1-score: {metrics['f1_score']:.6f}")
    print(f"Accuracy: {metrics['accuracy']:.6f}")
    print(f"PR-AUC: {metrics['pr_auc']:.6f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.6f}")

    print_heading("Confusion Matrix Counts")
    print(f"True positives: {true_positives:,}")
    print(f"False positives: {false_positives:,}")
    print(f"True negatives: {true_negatives:,}")
    print(f"False negatives: {false_negatives:,}")
    print(matrix.to_string())

    try:
        save_confusion_matrix(matrix)
        save_metrics(metrics, matrix)
    except (OSError, TypeError, ValueError) as error:
        print(f"\nError saving evaluation outputs: {error}")
        return

    print_heading("Saved Evaluation Outputs")
    print(f"Confusion matrix figure: {FIGURE_PATH}")
    print(f"Metrics and configuration: {METRICS_PATH}")
    print("Holdout evaluation complete; no further model or threshold changes were made.")


if __name__ == "__main__":
    main()
