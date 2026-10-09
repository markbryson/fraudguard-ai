"""Generate SHAP explanations for the frozen Random Forest model.

The V1-V28 columns are anonymized principal components and cannot be
interpreted directly as named financial behaviors.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

try:
    from final_model_config import (
        FEATURE_COLUMNS,
        MODEL_ARTIFACT_PATH,
        SELECTED_MODEL_NAME,
        TARGET_COLUMN,
    )
except ImportError:
    from src.models.final_model_config import (
        FEATURE_COLUMNS,
        MODEL_ARTIFACT_PATH,
        SELECTED_MODEL_NAME,
        TARGET_COLUMN,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
FIGURES_PATH = PROJECT_ROOT / "reports" / "figures"

GLOBAL_SAMPLE_SIZE = 500
RANDOM_STATE = 42
POSITIVE_CLASS_INDEX = 1

GLOBAL_IMPORTANCE_PATH = FIGURES_PATH / "random_forest_shap_feature_importance.png"
BEESWARM_PATH = FIGURES_PATH / "random_forest_shap_beeswarm.png"
WATERFALL_PATH = FIGURES_PATH / "random_forest_shap_waterfall.png"


def print_heading(title: str) -> None:
    """Print a readable section heading."""
    print(f"\n{'=' * 82}\n{title}\n{'=' * 82}")


def validate_training_data(data: pd.DataFrame) -> None:
    """Validate the training schema and numeric, finite feature values."""
    required_columns = [*FEATURE_COLUMNS, TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(
            "Training data is missing required column(s): "
            + ", ".join(missing_columns)
        )

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

    target_values = set(data[TARGET_COLUMN].dropna().unique())
    if data[TARGET_COLUMN].isna().any() or not target_values.issubset({0, 1}):
        raise ValueError(
            f"The '{TARGET_COLUMN}' target must be non-missing and binary with "
            f"values 0 and 1; found: {sorted(target_values)}."
        )


def select_positive_class_values(shap_values: object) -> np.ndarray:
    """Normalize SHAP output and select Class 1 values across SHAP versions."""
    if isinstance(shap_values, list):
        if len(shap_values) <= POSITIVE_CLASS_INDEX:
            raise ValueError("SHAP output does not contain a Class 1 explanation.")
        values = np.asarray(shap_values[POSITIVE_CLASS_INDEX])
    else:
        values = np.asarray(shap_values)
        if values.ndim == 3:
            if values.shape[-1] > POSITIVE_CLASS_INDEX:
                values = values[:, :, POSITIVE_CLASS_INDEX]
            elif values.shape[0] > POSITIVE_CLASS_INDEX:
                values = values[POSITIVE_CLASS_INDEX, :, :]
            else:
                raise ValueError("SHAP output does not contain a Class 1 explanation.")

    if values.ndim != 2 or values.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError(
            "Unexpected SHAP value shape: "
            f"{values.shape}; expected (samples, {len(FEATURE_COLUMNS)})."
        )
    return values


def select_positive_class_base_value(expected_value: object) -> float:
    """Normalize the TreeExplainer expected value for Class 1."""
    values = np.asarray(expected_value).reshape(-1)
    if values.size == 0:
        raise ValueError("SHAP expected value is empty.")
    if values.size > POSITIVE_CLASS_INDEX:
        return float(values[POSITIVE_CLASS_INDEX])
    return float(values[0])


def save_global_importance_plot(shap_values: np.ndarray, features: pd.DataFrame) -> None:
    """Save a global mean-absolute-SHAP feature-importance bar chart."""
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(
        shap_values,
        features,
        feature_names=FEATURE_COLUMNS,
        plot_type="bar",
        max_display=20,
        show=False,
    )
    figure = plt.gcf()
    figure.suptitle("Random Forest SHAP Feature Importance (Class 1)", y=1.02)
    figure.tight_layout()
    figure.savefig(GLOBAL_IMPORTANCE_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_beeswarm_plot(shap_values: np.ndarray, features: pd.DataFrame) -> None:
    """Save a SHAP beeswarm summary plot for Class 1."""
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(
        shap_values,
        features,
        feature_names=FEATURE_COLUMNS,
        max_display=20,
        show=False,
    )
    figure = plt.gcf()
    figure.suptitle("Random Forest SHAP Beeswarm Summary (Class 1)", y=1.02)
    figure.tight_layout()
    figure.savefig(BEESWARM_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_waterfall_plot(
    shap_values: np.ndarray,
    features: pd.DataFrame,
    expected_value: object,
) -> None:
    """Save a local SHAP waterfall explanation for one training transaction."""
    base_value = select_positive_class_base_value(expected_value)
    explanation = shap.Explanation(
        values=shap_values[0],
        base_values=base_value,
        data=features.iloc[0].to_numpy(),
        feature_names=FEATURE_COLUMNS,
    )
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    shap.plots.waterfall(explanation, max_display=15, show=False)
    figure = plt.gcf()
    figure.suptitle("Random Forest SHAP Local Explanation (Class 1)", y=1.02)
    figure.tight_layout()
    figure.savefig(WATERFALL_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Generate memory-conscious SHAP explanations from training data only."""
    print_heading("FraudGuard AI - SHAP Explainability")
    print(f"Model: {SELECTED_MODEL_NAME}")
    print(f"Model artifact: {MODEL_ARTIFACT_PATH}")
    print(f"Training dataset: {TRAIN_PATH}")
    print("Holdout data is intentionally not loaded or inspected.")
    print(
        "V1-V28 are anonymized features; their SHAP effects cannot be mapped "
        "directly to named financial behaviors."
    )

    if not TRAIN_PATH.is_file():
        print(f"\nError: Training dataset not found at {TRAIN_PATH}")
        return
    if not MODEL_ARTIFACT_PATH.is_file():
        print(f"\nError: Model artifact not found at {MODEL_ARTIFACT_PATH}")
        return

    try:
        training_data = pd.read_csv(TRAIN_PATH)
        validate_training_data(training_data)
        model = joblib.load(MODEL_ARTIFACT_PATH)
    except (OSError, pd.errors.ParserError, ValueError) as error:
        print(f"\nError loading or validating explainability inputs: {error}")
        return

    features = training_data[FEATURE_COLUMNS]
    fraud_indices = training_data.index[training_data[TARGET_COLUMN].eq(1)]
    if fraud_indices.empty:
        print("\nError: No fraudulent training transaction is available for the local explanation.")
        return

    sample_size = min(GLOBAL_SAMPLE_SIZE, len(features))
    explanation_features = features.sample(
        n=sample_size,
        random_state=RANDOM_STATE,
    )
    local_index = fraud_indices[0]
    local_features = features.loc[[local_index]]
    print(f"Generating SHAP values for {sample_size:,} training transactions.")
    print(f"Local explanation transaction index: {local_index}")

    try:
        explainer = shap.TreeExplainer(model)
        sample_output = explainer.shap_values(explanation_features)
        local_output = explainer.shap_values(local_features)
        sample_shap_values = select_positive_class_values(sample_output)
        local_shap_values = select_positive_class_values(local_output)
        expected_value = explainer.expected_value
        save_global_importance_plot(sample_shap_values, explanation_features)
        save_beeswarm_plot(sample_shap_values, explanation_features)
        save_waterfall_plot(local_shap_values, local_features, expected_value)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"\nError generating SHAP explanations: {error}")
        return

    print_heading("Saved SHAP Figures")
    print(f"Global feature importance: {GLOBAL_IMPORTANCE_PATH}")
    print(f"Beeswarm summary: {BEESWARM_PATH}")
    print(f"Local waterfall explanation: {WATERFALL_PATH}")


if __name__ == "__main__":
    main()
