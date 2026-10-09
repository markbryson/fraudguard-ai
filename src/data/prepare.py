"""Prepare a deduplicated train/test dataset for an initial experiment."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"

TARGET_COLUMN = "Class"
EXPECTED_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount", TARGET_COLUMN]


def print_heading(title: str) -> None:
    """Print a readable section heading."""
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def validate_dataset(data: pd.DataFrame) -> None:
    """Validate the expected schema and binary target values."""
    missing_columns = [column for column in EXPECTED_COLUMNS if column not in data.columns]
    unexpected_columns = [column for column in data.columns if column not in EXPECTED_COLUMNS]
    if missing_columns or unexpected_columns:
        details = []
        if missing_columns:
            details.append(f"missing columns: {', '.join(missing_columns)}")
        if unexpected_columns:
            details.append(f"unexpected columns: {', '.join(unexpected_columns)}")
        raise ValueError("Invalid dataset columns (" + "; ".join(details) + ").")

    target_values = set(data[TARGET_COLUMN].dropna().unique())
    if not target_values.issubset({0, 1}):
        raise ValueError(
            f"The '{TARGET_COLUMN}' target must be binary with values 0 and 1; "
            f"found: {sorted(target_values)}."
        )
    if data[TARGET_COLUMN].isna().any():
        raise ValueError(f"The '{TARGET_COLUMN}' target contains missing values.")


def print_split_summary(name: str, split: pd.DataFrame) -> None:
    """Print row counts and class balance for one dataset split."""
    total = len(split)
    legitimate = int(split[TARGET_COLUMN].eq(0).sum())
    fraudulent = int(split[TARGET_COLUMN].eq(1).sum())
    fraud_percentage = fraudulent / total * 100 if total else 0.0

    print_heading(f"{name} Split Summary")
    print(f"Rows: {total:,}")
    print(f"Legitimate transactions: {legitimate:,}")
    print(f"Fraudulent transactions: {fraudulent:,}")
    print(f"Fraud percentage: {fraud_percentage:.4f}%")


def main() -> None:
    """Deduplicate in memory, split the data, and save processed CSV files."""
    print_heading("FraudGuard AI - Dataset Preparation")
    print(f"Source dataset: {DATASET_PATH}")

    if not DATASET_PATH.is_file():
        print(
            "\nError: Dataset file not found."
            f"\nExpected file: {DATASET_PATH}"
            "\nPlease place creditcard.csv in data/raw/ and try again."
        )
        return

    try:
        data = pd.read_csv(DATASET_PATH)
        validate_dataset(data)
    except (OSError, pd.errors.ParserError, ValueError) as error:
        print(f"\nError preparing dataset: {error}")
        return

    original_rows = len(data)
    deduplicated_data = data.drop_duplicates().copy()
    removed_duplicates = original_rows - len(deduplicated_data)

    print_heading("Deduplication Summary")
    print(f"Original rows: {original_rows:,}")
    print(f"Removed exact duplicates: {removed_duplicates:,}")
    print(f"Remaining rows: {len(deduplicated_data):,}")

    # Initial stratified random-split experiment; evaluate a time-based holdout later
    # because this dataset contains a Time feature.
    features = deduplicated_data.drop(columns=TARGET_COLUMN)
    target = deduplicated_data[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        stratify=target,
        random_state=42,
    )

    train_data = x_train.copy()
    train_data[TARGET_COLUMN] = y_train
    test_data = x_test.copy()
    test_data[TARGET_COLUMN] = y_test

    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    train_path = PROCESSED_PATH / "train.csv"
    test_path = PROCESSED_PATH / "test.csv"
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

    print_split_summary("Training", train_data)
    print_split_summary("Testing", test_data)

    overlapping_rows = len(train_data.merge(test_data, how="inner"))
    print_heading("Full-Row Overlap Check")
    print(f"Exact full rows overlapping between splits: {overlapping_rows:,}")
    print(f"Overlap detected: {'Yes' if overlapping_rows else 'No'}")

    print_heading("Saved Processed Datasets")
    print(f"Training data: {train_path}")
    print(f"Testing data: {test_path}")


if __name__ == "__main__":
    main()
