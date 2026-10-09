"""Check for exact duplicate transactions in the credit card dataset."""

from pathlib import Path

import pandas as pd


DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "creditcard.csv"


def print_heading(title: str) -> None:
    """Print a readable section heading."""
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    """Load the dataset and report exact duplicate-row statistics."""
    print_heading("FraudGuard AI - Duplicate Transaction Check")
    print(f"Dataset: {DATASET_PATH}")

    if not DATASET_PATH.is_file():
        print(
            "\nError: Dataset file not found."
            f"\nExpected file: {DATASET_PATH}"
            "\nPlease place creditcard.csv in data/raw/ and try again."
        )
        return

    data = pd.read_csv(DATASET_PATH)
    if "Class" not in data.columns:
        print("\nError: Dataset does not contain the required 'Class' column.")
        return

    total_transactions = len(data)
    duplicate_mask = data.duplicated(keep="first")
    duplicate_count = int(duplicate_mask.sum())
    duplicate_percentage = (
        duplicate_count / total_transactions * 100 if total_transactions else 0.0
    )

    legitimate_duplicates = int(
        (duplicate_mask & data["Class"].eq(0)).sum()
    )
    fraudulent_duplicates = int(
        (duplicate_mask & data["Class"].eq(1)).sum()
    )
    unique_rows_remaining = len(data.drop_duplicates())

    print_heading("Duplicate Summary")
    print(f"Total transactions: {total_transactions:,}")
    print(f"Exact duplicate rows: {duplicate_count:,}")
    print(f"Duplicate percentage: {duplicate_percentage:.4f}%")

    print_heading("Duplicate Rows by Transaction Class")
    print(f"Legitimate duplicates (Class = 0): {legitimate_duplicates:,}")
    print(f"Fraudulent duplicates (Class = 1): {fraudulent_duplicates:,}")

    print_heading("Unique Rows After Removing Exact Duplicates")
    print(f"Unique rows remaining: {unique_rows_remaining:,}")


if __name__ == "__main__":
    main()
