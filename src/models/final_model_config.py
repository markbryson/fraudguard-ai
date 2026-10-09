"""Frozen configuration for the FraudGuard AI production candidate.

Random Forest was selected provisionally using training-only cross-validation,
and the fraud classification threshold is frozen at 0.40 for the next
untouched-holdout evaluation. The 0.40 threshold maximized F1 only among the
evaluated Random Forest thresholds; it is not necessarily optimal across all
possible thresholds.

Importing this module defines configuration constants only. It does not load
datasets or model artifacts.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SELECTED_MODEL_NAME = "Random Forest"
SELECTED_MODEL_TYPE = "random_forest"
MODEL_ARTIFACT_PATH = PROJECT_ROOT / "models" / "random_forest.joblib"

TARGET_COLUMN = "Class"
NEGATIVE_CLASS = 0
POSITIVE_CLASS = 1
CLASSIFICATION_THRESHOLD = 0.40

FEATURE_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
EXPECTED_INPUT_COLUMNS = FEATURE_COLUMNS.copy()

MODEL_SELECTION_NOTE = (
    "Random Forest and threshold 0.40 were selected using training-only "
    "cross-validation and training-only out-of-fold threshold analysis."
)
THRESHOLD_SELECTION_NOTE = (
    "Threshold 0.40 maximized F1 among the evaluated Random Forest thresholds "
    "only; it is not necessarily optimal among all possible thresholds."
)
