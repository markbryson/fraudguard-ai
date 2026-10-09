"""Analyze XGBoost precision-recall tradeoffs using training data only."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

try:
    from train_xgboost import (
        build_model,
        calculate_scale_pos_weight,
        validate_training_data,
    )
except ImportError:
    from src.models.train_xgboost import (
        build_model,
        calculate_scale_pos_weight,
        validate_training_data,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
FIGURE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "xgboost_threshold_analysis.png"
)

TARGET_COLUMN = "Class"
FEATURE_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def print_heading(title: str) -> None:
    """Print a readable section heading."""
    print(f"\n{'=' * 106}\n{title}\n{'=' * 106}")


def generate_out_of_fold_probabilities(
    features: pd.DataFrame,
    target: pd.Series,
    cross_validator: StratifiedKFold,
) -> tuple[pd.Series, list[float]]:
    """Fit one fold-specific model at a time and return OOF probabilities."""
    probabilities = pd.Series(index=target.index, dtype=float)
    fold_scale_weights = []

    for fold_number, (train_indices, validation_indices) in enumerate(
        cross_validator.split(features, target),
        start=1,
    ):
        fold_features = features.iloc[train_indices]
        fold_target = target.iloc[train_indices]
        validation_features = features.iloc[validation_indices]
        fold_scale_pos_weight = calculate_scale_pos_weight(fold_target)
        fold_scale_weights.append(fold_scale_pos_weight)

        model = build_model(fold_scale_pos_weight)
        model.fit(fold_features, fold_target)
        probabilities.iloc[validation_indices] = model.predict_proba(validation_features)[:, 1]
        print(
            f"Completed fold {fold_number}/5 "
            f"(scale_pos_weight={fold_scale_pos_weight:.6f})"
        )

    return probabilities, fold_scale_weights


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
    """Save a high-resolution precision-recall threshold chart."""
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
    sns.lineplot(
        data=results,
        x="threshold",
        y="f1",
        marker="o",
        linestyle="--",
        linewidth=2,
        label="F1-score",
        color="#55a868",
        ax=axis,
    )
    axis.set_title("XGBoost Precision-Recall Tradeoff by Threshold")
    axis.set_xlabel("Fraud probability threshold")
    axis.set_ylabel("Score")
    axis.set_xticks(THRESHOLDS)
    axis.set_ylim(0, 1.05)
    axis.legend(title="Metric")
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Generate training-only out-of-fold XGBoost threshold analysis."""
    print_heading("FraudGuard AI - XGBoost Threshold Analysis")
    print(f"Training dataset: {TRAIN_PATH}")
    print("Holdout test data is intentionally not loaded.")
    print("Memory-conscious note: five-fold XGBoost analysis may take several minutes.")

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
    print("The exact XGBoost configuration is reused for every fold.")
    try:
        fraud_probabilities, fold_scale_weights = generate_out_of_fold_probabilities(
            features,
            target,
            cross_validator,
        )
    except (ValueError, RuntimeError) as error:
        print(f"\nError during cross-validation: {error}")
        return

    average_precision = average_precision_score(target, fraud_probabilities)
    results = evaluate_thresholds(target, fraud_probabilities)

    print_heading("Out-of-Fold Evaluation")
    print(f"Average precision (PR-AUC): {average_precision:.6f}")
    print(
        "Fold scale_pos_weight values: "
        + ", ".join(f"{weight:.6f}" for weight in fold_scale_weights)
    )

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
