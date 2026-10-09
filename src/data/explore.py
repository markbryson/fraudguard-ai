"""Basic exploratory inspection of the credit card fraud dataset."""

from pathlib import Path

import pandas as pd


DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "creditcard.csv"


def print_heading(title: str) -> None:
    """Print a readable section heading."""
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    """Load the dataset and print basic exploratory information."""
    print_heading("FraudGuard AI - Dataset Exploration")
    print(f"Dataset: {DATASET_PATH}")

    if not DATASET_PATH.is_file():
        print(
            "\nError: Dataset file not found."
            f"\nExpected file: {DATASET_PATH}"
            "\nPlease place creditcard.csv in data/raw/ and try again."
        )
        return

    data = pd.read_csv(DATASET_PATH)

    print_heading("Dataset Dimensions")
    print(f"Rows: {data.shape[0]:,}")
    print(f"Columns: {data.shape[1]:,}")

    print_heading("First Five Transactions")
    print(data.head(5).to_string())

    print_heading("Column Names")
    for column in data.columns:
        print(f"- {column}")

    print_heading("Data Types")
    print(data.dtypes.to_string())

    print_heading("Missing Values")
    missing_values = data.isna().sum()
    print(missing_values.to_string())
    print(f"Total missing values: {missing_values.sum():,}")

    print_heading("Transaction Class Counts")
    legitimate_count = int((data["Class"] == 0).sum())
    fraudulent_count = int((data["Class"] == 1).sum())
    print(f"Legitimate transactions (Class = 0): {legitimate_count:,}")
    print(f"Fraudulent transactions (Class = 1): {fraudulent_count:,}")

    print_heading("Fraud Percentage")
    fraud_percentage = (fraudulent_count / len(data) * 100) if len(data) else 0.0
    print(f"Fraud percentage: {fraud_percentage:.4f}%")

    print_heading("Amount Descriptive Statistics")
    print(data["Amount"].describe().to_string())


if __name__ == "__main__":
    main()
