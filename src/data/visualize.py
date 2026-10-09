"""Generate visual summaries of the credit card fraud dataset."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
FIGURES_PATH = PROJECT_ROOT / "reports" / "figures"


def save_figure(filename: str) -> None:
    """Save the current figure as a high-resolution PNG."""
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_PATH / filename, dpi=300, bbox_inches="tight")
    plt.close()


def add_bar_labels(axis: plt.Axes, values: list[int], percentages: list[float]) -> None:
    """Add count and percentage labels above each bar."""
    offset = max(values) * 0.01 if values and max(values) else 1
    for index, (value, percentage) in enumerate(zip(values, percentages)):
        axis.text(
            index,
            value + offset,
            f"{value:,}\n({percentage:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )


def main() -> None:
    """Load the dataset and save the requested visual summaries."""
    sns.set_theme(style="whitegrid", context="talk")

    if not DATASET_PATH.is_file():
        print(
            "Error: Dataset file not found."
            f"\nExpected file: {DATASET_PATH}"
            "\nPlease place creditcard.csv in data/raw/ and try again."
        )
        return

    data = pd.read_csv(DATASET_PATH)
    required_columns = {"Class", "Amount"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        print(f"Error: Dataset is missing required column(s): {missing}")
        return

    # Class distribution with direct count and percentage labels.
    class_counts = data["Class"].value_counts().reindex([0, 1], fill_value=0)
    total_transactions = len(data)
    class_summary = pd.DataFrame(
        {
            "class_label": ["Legitimate", "Fraudulent"],
            "count": class_counts.to_numpy(),
            "percentage": (
                class_counts.to_numpy() / total_transactions * 100
                if total_transactions
                else np.zeros(2)
            ),
        }
    )

    figure, axis = plt.subplots(figsize=(10, 7))
    sns.barplot(
        data=class_summary,
        x="class_label",
        y="count",
        hue="class_label",
        palette=["#2f6f9f", "#c44e52"],
        legend=False,
        ax=axis,
    )
    axis.set_title("Transaction Class Distribution")
    axis.set_xlabel("Transaction class")
    axis.set_ylabel("Number of transactions")
    add_bar_labels(
        axis,
        class_summary["count"].tolist(),
        class_summary["percentage"].tolist(),
    )
    figure.tight_layout()
    save_figure("class_distribution.png")

    # log1p keeps all transactions visible while reducing the effect of outliers.
    log_amount = np.log1p(data["Amount"].clip(lower=0))
    figure, axis = plt.subplots(figsize=(11, 7))
    sns.histplot(log_amount, bins=60, color="#4c72b0", edgecolor="none", ax=axis)
    axis.set_title("Transaction Amount Distribution (log1p scale)")
    axis.set_xlabel("log1p(Transaction amount)")
    axis.set_ylabel("Number of transactions")
    figure.tight_layout()
    save_figure("transaction_amount_distribution.png")

    # A log1p-scaled boxplot makes the two classes comparable despite extreme values.
    comparison = pd.DataFrame(
        {
            "Class": data["Class"].map({0: "Legitimate", 1: "Fraudulent"}),
            "log_amount": log_amount,
        }
    )
    figure, axis = plt.subplots(figsize=(10, 7))
    sns.boxplot(
        data=comparison,
        x="Class",
        y="log_amount",
        hue="Class",
        order=["Legitimate", "Fraudulent"],
        hue_order=["Legitimate", "Fraudulent"],
        palette=["#2f6f9f", "#c44e52"],
        legend=False,
        showfliers=False,
        ax=axis,
    )
    axis.set_title("Transaction Amount Comparison by Class (log1p scale)")
    axis.set_xlabel("Transaction class")
    axis.set_ylabel("log1p(Transaction amount)")
    figure.tight_layout()
    save_figure("transaction_amount_by_class.png")

    # Include numerical features and Class; mask the redundant upper triangle.
    numerical_data = data.select_dtypes(include="number")
    correlation = numerical_data.corr()
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    figure, axis = plt.subplots(figsize=(20, 16))
    sns.heatmap(
        correlation,
        mask=mask,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.25,
        cbar_kws={"label": "Correlation coefficient"},
        ax=axis,
    )
    axis.set_title("Correlation Heatmap of Numerical Features")
    axis.set_xlabel("Numerical features")
    axis.set_ylabel("Numerical features")
    figure.tight_layout()
    save_figure("correlation_heatmap.png")

    print(f"Saved figures to: {FIGURES_PATH}")


if __name__ == "__main__":
    main()
