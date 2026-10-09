"""FraudGuard AI Streamlit application foundation.

This module contains presentation structure only. Model loading, data loading,
prediction, evaluation, and SHAP logic will be added in later iterations.
"""

import csv
import hashlib
import json
import sys
from collections import Counter
from numbers import Integral, Real
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted


def ensure_project_root_on_import_path() -> Path:
    """Make the repository root importable when launched from any directory."""
    project_root = Path(__file__).resolve().parents[1]
    project_root_string = str(project_root)
    if project_root_string not in sys.path:
        sys.path.insert(0, project_root_string)
    return project_root


PROJECT_ROOT = ensure_project_root_on_import_path()

from src.models.final_model_config import (  # noqa: E402
    CLASSIFICATION_THRESHOLD,
    FEATURE_COLUMNS,
    MODEL_ARTIFACT_PATH,
    SELECTED_MODEL_NAME,
)


PROJECT_NAME = "FraudGuard AI"
PROJECT_SUBTITLE = "Explainable machine learning for financial transaction fraud detection"
SELECTED_MODEL = SELECTED_MODEL_NAME
FROZEN_THRESHOLD = CLASSIFICATION_THRESHOLD
METRICS_PATH = PROJECT_ROOT / "reports" / "final_model_metrics.json"
MAX_BATCH_ROWS = 100_000
MAX_BATCH_FILE_BYTES = 10 * 1024 * 1024
MAX_BATCH_COLUMNS = 100


def inject_custom_css() -> None:
    """Apply the shared visual design system for the application."""
    st.markdown(
        """
        <style>
        :root {
            --fg-navy: #071426;
            --fg-navy-soft: #0d1d33;
            --fg-blue: #2f80ed;
            --fg-blue-bright: #62a5ff;
            --fg-cyan: #43d6c5;
            --fg-text: #edf4ff;
            --fg-muted: #93a4bd;
            --fg-border: rgba(135, 169, 211, 0.20);
            --fg-panel: rgba(13, 29, 51, 0.82);
        }

        .stApp {
            background:
                radial-gradient(circle at 85% 0%, rgba(47, 128, 237, 0.14), transparent 32rem),
                linear-gradient(135deg, var(--fg-navy) 0%, #091a2f 52%, #06101f 100%);
            color: var(--fg-text);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #08172a 0%, #061120 100%);
            border-right: 1px solid var(--fg-border);
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label {
            color: var(--fg-text);
        }

        .fg-brand {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin: 0.25rem 0 2rem 0;
        }

        .fg-brand-mark {
            display: grid;
            place-items: center;
            width: 2.6rem;
            height: 2.6rem;
            border-radius: 0.8rem;
            background: linear-gradient(135deg, var(--fg-blue), var(--fg-cyan));
            color: #041020;
            font-size: 1.35rem;
            font-weight: 800;
            box-shadow: 0 0 24px rgba(67, 214, 197, 0.18);
        }

        .fg-brand-name {
            color: var(--fg-text);
            font-size: 1.15rem;
            font-weight: 750;
            letter-spacing: -0.02em;
        }

        .fg-brand-caption {
            color: var(--fg-muted);
            font-size: 0.72rem;
            margin-top: 0.15rem;
        }

        .fg-header {
            padding: 1.35rem 1.5rem 1.5rem 1.5rem;
            border: 1px solid var(--fg-border);
            border-radius: 1.25rem;
            background: linear-gradient(120deg, rgba(15, 41, 72, 0.92), rgba(8, 22, 41, 0.80));
            box-shadow: 0 1.5rem 3rem rgba(0, 0, 0, 0.16);
            margin-bottom: 1.5rem;
        }

        .fg-eyebrow {
            color: var(--fg-cyan);
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 0.55rem;
        }

        .fg-title {
            color: var(--fg-text);
            font-size: clamp(2rem, 4vw, 3.35rem);
            font-weight: 800;
            letter-spacing: -0.05em;
            line-height: 1.04;
            margin: 0;
        }

        .fg-subtitle {
            color: var(--fg-muted);
            font-size: 1rem;
            line-height: 1.6;
            margin: 0.8rem 0 0 0;
            max-width: 48rem;
        }

        .fg-section-title {
            color: var(--fg-text);
            font-size: 1.55rem;
            font-weight: 750;
            letter-spacing: -0.03em;
            margin: 0 0 0.35rem 0;
        }

        .fg-section-description {
            color: var(--fg-muted);
            line-height: 1.6;
            margin: 0 0 1.25rem 0;
        }

        .fg-card {
            min-height: 8.25rem;
            padding: 1.15rem;
            border: 1px solid var(--fg-border);
            border-radius: 1rem;
            background: var(--fg-panel);
            box-shadow: 0 0.8rem 2rem rgba(0, 0, 0, 0.10);
        }

        .fg-card-label {
            color: var(--fg-muted);
            font-size: 0.78rem;
            font-weight: 650;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .fg-card-value {
            color: var(--fg-text);
            font-size: 1.65rem;
            font-weight: 750;
            margin-top: 0.55rem;
        }

        .fg-card-note {
            color: var(--fg-muted);
            font-size: 0.78rem;
            margin-top: 0.25rem;
        }

        .fg-empty {
            padding: 2rem 1.25rem;
            border: 1px dashed rgba(98, 165, 255, 0.35);
            border-radius: 1rem;
            background: rgba(11, 30, 54, 0.52);
            text-align: center;
        }

        .fg-empty-icon {
            color: var(--fg-blue-bright);
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }

        .fg-empty-title {
            color: var(--fg-text);
            font-size: 1.05rem;
            font-weight: 700;
        }

        .fg-empty-copy {
            color: var(--fg-muted);
            line-height: 1.55;
            margin: 0.45rem auto 0;
            max-width: 38rem;
        }

        .fg-status {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.38rem 0.72rem;
            border: 1px solid rgba(67, 214, 197, 0.28);
            border-radius: 999px;
            background: rgba(67, 214, 197, 0.08);
            color: var(--fg-cyan);
            font-size: 0.78rem;
            font-weight: 650;
        }

        .fg-footer {
            color: var(--fg-muted);
            border-top: 1px solid var(--fg-border);
            font-size: 0.76rem;
            line-height: 1.55;
            margin-top: 3rem;
            padding: 1.25rem 0 0.5rem 0;
        }

        .fg-footer strong {
            color: var(--fg-text);
        }

        div[data-testid="stMetric"] {
            background: var(--fg-panel);
            border: 1px solid var(--fg-border);
            border-radius: 1rem;
            padding: 0.9rem 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    """Render navigation and frozen model metadata, returning the active section."""
    with st.sidebar:
        st.markdown(
            """
            <div class="fg-brand">
                <div class="fg-brand-mark">F</div>
                <div>
                    <div class="fg-brand-name">FraudGuard AI</div>
                    <div class="fg-brand-caption">Risk intelligence workspace</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Workspace")
        section = st.radio(
            "Application section",
            [
                "Dashboard",
                "Single Transaction Prediction",
                "Batch Fraud Detection",
                "Model Performance",
                "Explainable AI",
                "About",
            ],
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("#### Frozen model")
        st.markdown(
            f"""
            <div class="fg-card" style="min-height: auto; padding: 0.9rem;">
                <div class="fg-card-label">Selected candidate</div>
                <div style="color: var(--fg-text); font-weight: 700; margin-top: 0.4rem;">
                    {SELECTED_MODEL}
                </div>
                <div class="fg-card-note">Production threshold: <strong style="color: var(--fg-cyan);">{FROZEN_THRESHOLD:.2f}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption("Model and threshold are frozen for the current evaluation cycle.")

    return section


def render_header() -> None:
    """Render the application header."""
    st.markdown(
        f"""
        <div class="fg-header">
            <div class="fg-eyebrow">Financial risk intelligence</div>
            <h1 class="fg-title">{PROJECT_NAME}</h1>
            <p class="fg-subtitle">{PROJECT_SUBTITLE}. Explore model signals, review transaction risk, and understand the reasoning behind each decision.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_intro(title: str, description: str) -> None:
    """Render a consistent section heading and description."""
    st.markdown(f'<div class="fg-section-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="fg-section-description">{description}</div>',
        unsafe_allow_html=True,
    )


def render_empty_state(icon: str, title: str, message: str) -> None:
    """Render a polished placeholder for functionality planned for later."""
    st.markdown(
        f"""
        <div class="fg-empty">
            <div class="fg-empty-icon">{icon}</div>
            <div class="fg-empty-title">{title}</div>
            <div class="fg-empty-copy">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_evaluation_metrics(metrics_path: str) -> dict | None:
    """Load the saved holdout metrics without loading data or model artifacts."""
    try:
        with open(metrics_path, encoding="utf-8") as metrics_file:
            metrics = json.load(metrics_file)
        return metrics if isinstance(metrics, dict) else None
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def _validate_unit_interval(value: object, label: str) -> float:
    """Return a finite metric value in the inclusive [0, 1] interval."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a numeric value between 0 and 1.")
    try:
        numeric_value = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite value between 0 and 1.") from error
    if not np.isfinite(numeric_value) or not 0 <= numeric_value <= 1:
        raise ValueError(f"{label} must be a finite value between 0 and 1.")
    return numeric_value


def _validate_nonnegative_integer(value: object, label: str) -> int:
    """Return a nonnegative integer count."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be a nonnegative integer.")
    integer_value = int(value)
    if integer_value < 0:
        raise ValueError(f"{label} must be a nonnegative integer.")
    return integer_value


def validate_saved_metrics(metrics: dict) -> None:
    """Validate the saved historical metrics against the frozen configuration."""
    if not isinstance(metrics, dict):
        raise ValueError("The saved evaluation metrics must be a JSON object.")

    if "model_name" in metrics and metrics["model_name"] != SELECTED_MODEL:
        raise ValueError(
            "The saved evaluation metrics identify a different model than the "
            f"frozen {SELECTED_MODEL} configuration."
        )
    if "threshold" in metrics:
        threshold = _validate_unit_interval(metrics["threshold"], "Saved threshold")
        if threshold != FROZEN_THRESHOLD:
            raise ValueError(
                f"The saved classification threshold ({threshold:.2f}) does not "
                f"match the frozen threshold ({FROZEN_THRESHOLD:.2f})."
            )
    if "feature_columns" in metrics:
        if metrics["feature_columns"] != list(FEATURE_COLUMNS):
            raise ValueError(
                "The saved feature schema or ordering does not match the frozen "
                "30-feature configuration."
            )
    if "target_column" in metrics and metrics["target_column"] != "Class":
        raise ValueError("The saved target column does not match the frozen Class target.")

    metric_values = metrics.get("metrics")
    confusion = metrics.get("confusion_matrix")
    if not isinstance(metric_values, dict):
        raise ValueError("The saved metrics object is missing or malformed.")
    if not isinstance(confusion, dict):
        raise ValueError("The saved confusion_matrix object is missing or malformed.")

    required_metric_keys = ["precision", "recall", "f1_score", "pr_auc"]
    required_confusion_keys = [
        "true_positives",
        "false_positives",
        "true_negatives",
        "false_negatives",
    ]
    if not all(key in metric_values for key in required_metric_keys):
        raise ValueError("The metrics object is missing a required performance key.")
    if not all(key in confusion for key in required_confusion_keys):
        raise ValueError("The confusion_matrix object is missing a required count.")

    sample_count_keys = (
        "n_samples",
        "sample_count",
        "total_samples",
        "total_transactions",
        "transactions_evaluated",
    )
    for metric_name, metric_value in metric_values.items():
        if metric_name not in sample_count_keys:
            _validate_unit_interval(metric_value, f"Metric '{metric_name}'")

    counts = {
        key: _validate_nonnegative_integer(confusion[key], f"Count '{key}'")
        for key in required_confusion_keys
    }
    total_transactions = sum(counts.values())
    for container_name, container in (("root", metrics), ("metrics", metric_values)):
        for key in sample_count_keys:
            if key in container:
                reported_count = _validate_nonnegative_integer(
                    container[key], f"{container_name}.{key}"
                )
                if reported_count != total_transactions:
                    raise ValueError(
                        f"The saved sample count '{container_name}.{key}' "
                        "does not equal the total confusion-matrix count."
                    )

    true_positives = counts["true_positives"]
    false_positives = counts["false_positives"]
    true_negatives = counts["true_negatives"]
    false_negatives = counts["false_negatives"]


def extract_dashboard_values(metrics: dict) -> dict:
    """Read and validate the existing final-model metrics JSON structure."""
    validate_saved_metrics(metrics)
    metric_values = metrics["metrics"]
    counts = metrics["confusion_matrix"]
    true_positives = int(counts["true_positives"])
    false_positives = int(counts["false_positives"])
    true_negatives = int(counts["true_negatives"])
    false_negatives = int(counts["false_negatives"])
    total_transactions = (
        true_positives + false_positives + true_negatives + false_negatives
    )
    return {
        "transactions_evaluated": total_transactions,
        "fraud_alerts": true_positives + false_positives,
        "recall": float(metric_values["recall"]),
        "precision": float(metric_values["precision"]),
        "f1_score": float(metric_values["f1_score"]),
        "pr_auc": float(metric_values["pr_auc"]),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "evaluation_scope": str(metrics.get("evaluation_scope", "Historical holdout evaluation")),
        "model_name": str(metrics.get("model_name", SELECTED_MODEL)),
        "threshold": float(metrics.get("threshold", FROZEN_THRESHOLD)),
    }


def build_confusion_matrix_figure(values: dict) -> go.Figure:
    """Build an interactive Plotly confusion matrix from saved counts."""
    counts = [
        [values["true_negatives"], values["false_positives"]],
        [values["false_negatives"], values["true_positives"]],
    ]
    labels = ["Legitimate", "Fraudulent"]
    figure = go.Figure(
        data=go.Heatmap(
            z=counts,
            x=labels,
            y=labels,
            text=[[f"{count:,}" for count in row] for row in counts],
            texttemplate="%{text}",
            textfont={"size": 18},
            colorscale=[
                [0.0, "#0d1d33"],
                [0.35, "#1b4f86"],
                [1.0, "#43d6c5"],
            ],
            hovertemplate=(
                "Actual: %{y}<br>Predicted: %{x}<br>Count: %{z:,}<extra></extra>"
            ),
            colorbar={"title": "Count"},
        )
    )
    figure.update_layout(
        title="Historical holdout confusion matrix",
        xaxis_title="Predicted label",
        yaxis_title="Actual label",
        height=430,
        margin={"l": 20, "r": 20, "t": 70, "b": 30},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#edf4ff"},
    )
    return figure


def render_dashboard() -> None:
    """Render the dashboard using the saved historical holdout evaluation."""
    render_section_intro(
        "Risk monitoring dashboard",
        "Historical holdout evaluation results for the frozen Random Forest candidate. This is not live transaction monitoring.",
    )
    metrics = load_evaluation_metrics(str(METRICS_PATH))
    if metrics is None:
        st.warning(
            "Historical evaluation metrics are unavailable or malformed. "
            "The dashboard cannot display verified performance values yet."
        )
        render_empty_state(
            "!",
            "Evaluation results unavailable",
            "Check reports/final_model_metrics.json and ensure it contains the saved final-model metrics structure.",
        )
        return

    try:
        values = extract_dashboard_values(metrics)
    except (KeyError, TypeError, ValueError):
        st.warning(
            "The historical evaluation metrics file does not contain the expected keys. "
            "No metrics have been inferred or substituted."
        )
        render_empty_state(
            "!",
            "Evaluation results could not be read",
            "The dashboard expects the existing metrics and confusion_matrix objects from the final holdout evaluation.",
        )
        return

    st.markdown(
        f'<div class="fg-status">● Historical holdout · {values["evaluation_scope"]}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    columns = st.columns(4)
    cards = [
        ("Transactions evaluated", f'{values["transactions_evaluated"]:,}', "Untouched holdout set"),
        ("Fraud alerts", f'{values["fraud_alerts"]:,}', "Predicted positive transactions"),
        ("Fraud detection recall", f'{values["recall"] * 100:.2f}%', "Fraud cases detected"),
        ("Model status", "Evaluated", f'{values["model_name"]} · threshold {values["threshold"]:.2f}'),
    ]
    for column, (label, value, note) in zip(columns, cards):
        with column:
            st.markdown(
                f"""
                <div class="fg-card">
                    <div class="fg-card-label">{label}</div>
                    <div class="fg-card-value">{value}</div>
                    <div class="fg-card-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1, 1.35])
    with left:
        st.markdown('<div class="fg-section-title">Model performance summary</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="fg-section-description">Metrics calculated once on the frozen holdout evaluation.</div>',
            unsafe_allow_html=True,
        )
        performance_columns = st.columns(2)
        for column, label, value in [
            (performance_columns[0], "Precision", f'{values["precision"] * 100:.2f}%'),
            (performance_columns[1], "Recall", f'{values["recall"] * 100:.2f}%'),
            (performance_columns[0], "F1-score", f'{values["f1_score"] * 100:.2f}%'),
            (performance_columns[1], "PR-AUC", f'{values["pr_auc"]:.4f}'),
        ]:
            with column:
                st.metric(label, value)
    with right:
        st.plotly_chart(
            build_confusion_matrix_figure(values),
            use_container_width=True,
            config={"displayModeBar": False, "responsive": True},
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="fg-section-title">How to read these results</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        The frozen Random Forest correctly identified <strong>{values["true_positives"]:,}</strong> fraudulent transactions and missed <strong>{values["false_negatives"]:,}</strong>. It raised <strong>{values["false_positives"]:,}</strong> false alarms among legitimate transactions. This reflects a high-precision operating point, while the remaining missed fraud shows why model outputs should support investigation workflows rather than replace human judgment.

        These figures describe one historical holdout evaluation at threshold <strong>{values["threshold"]:.2f}</strong>; they are not live monitoring metrics and may not represent future transaction behavior or drift.
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_frozen_model(model_path: str):
    """Load and cache the frozen model artifact without retraining it."""
    model = joblib.load(model_path)
    validate_frozen_model(model)
    return model


def validate_frozen_model(model: object) -> None:
    """Validate the frozen Random Forest artifact before model inference."""
    if not isinstance(model, RandomForestClassifier):
        raise ValueError(
            "The loaded model artifact is incompatible: expected a fitted "
            "sklearn RandomForestClassifier."
        )

    try:
        check_is_fitted(model, attributes=["classes_", "n_features_in_"])
    except NotFittedError as error:
        raise ValueError(
            "The loaded Random Forest artifact is not fitted and cannot be used "
            "for predictions."
        ) from error

    if not callable(getattr(model, "predict_proba", None)):
        raise ValueError(
            "The loaded Random Forest artifact does not support predict_proba()."
        )

    try:
        feature_count = int(model.n_features_in_)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(
            "The loaded Random Forest artifact has an invalid feature-count "
            "metadata value."
        ) from error
    if feature_count != len(FEATURE_COLUMNS):
        raise ValueError(
            "The loaded Random Forest artifact expects "
            f"{feature_count} features, but FraudGuard AI requires "
            f"{len(FEATURE_COLUMNS)} features."
        )

    if hasattr(model, "feature_names_in_"):
        try:
            feature_names = list(model.feature_names_in_)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "The loaded Random Forest artifact has invalid feature-name "
                "metadata."
            ) from error
        if feature_names != list(FEATURE_COLUMNS):
            raise ValueError(
                "The loaded Random Forest artifact feature names or ordering do "
                "not match the frozen FraudGuard AI feature schema."
            )

    class_values = np.asarray(getattr(model, "classes_", [])).reshape(-1).tolist()
    if len(class_values) != 2 or set(class_values) != {0, 1}:
        raise ValueError(
            "The loaded Random Forest artifact must contain exactly the binary "
            "classes 0 and 1."
        )


def predict_fraud_probabilities(model: object, features: pd.DataFrame) -> np.ndarray:
    """Return Class 1 probabilities using the model's class labels."""
    validate_frozen_model(model)
    class_values = np.asarray(model.classes_).reshape(-1)
    fraud_indices = np.flatnonzero(class_values == 1)
    if len(fraud_indices) != 1:
        raise ValueError(
            "The loaded Random Forest artifact does not expose a unique Class 1 "
            "probability column."
        )

    probabilities = np.asarray(model.predict_proba(features))
    if probabilities.ndim != 2 or probabilities.shape != (
        len(features),
        len(class_values),
    ):
        raise ValueError(
            "The loaded Random Forest artifact returned an invalid predict_proba "
            "shape for the frozen feature schema."
        )

    fraud_probabilities = probabilities[:, fraud_indices[0]]
    if not np.isfinite(fraud_probabilities).all() or (
        (fraud_probabilities < 0).any() or (fraud_probabilities > 1).any()
    ):
        raise ValueError("The model returned invalid fraud probabilities.")
    return fraud_probabilities


@st.cache_data(show_spinner=False)
def build_synthetic_demo_transactions() -> pd.DataFrame:
    """Build deterministic artificial feature rows for interface demonstrations."""
    demo_values = np.zeros((3, len(FEATURE_COLUMNS)), dtype=float)
    demo_values[:, 0] = [0.0, 3_600.0, 86_400.0]
    demo_values[:, 1:29] = np.array(
        [
            [0.001 * (feature_index + 1) for feature_index in range(28)],
            [-0.002 * (feature_index + 1) for feature_index in range(28)],
            [0.003 * (feature_index + 1) for feature_index in range(28)],
        ],
        dtype=float,
    )
    demo_values[:, -1] = [25.0, 120.0, 750.0]
    return pd.DataFrame(demo_values, columns=FEATURE_COLUMNS)


@st.cache_data(show_spinner=False)
def load_example_transaction() -> dict:
    """Return one deterministic artificial example for interface demonstrations."""
    return build_synthetic_demo_transactions().iloc[0].to_dict()


def safe_default_value(example: dict | None, feature: str) -> float:
    """Return a finite numeric widget default."""
    if example is None:
        return 0.0
    try:
        value = float(example.get(feature, 0.0))
        return value if np.isfinite(value) else 0.0
    except (TypeError, ValueError):
        return 0.0


def render_feature_inputs(example: dict | None, widget_suffix: str) -> dict[str, float]:
    """Render grouped numerical inputs in the frozen feature order."""
    values: dict[str, float] = {}
    with st.expander("Transaction context", expanded=True):
        time_value = safe_default_value(example, "Time")
        amount_value = max(0.0, safe_default_value(example, "Amount"))
        time_column, amount_column = st.columns(2)
        with time_column:
            values["Time"] = st.number_input(
                "Time",
                value=time_value,
                step=0.01,
                format="%.6f",
                key=f"single_time_{widget_suffix}",
                help="Elapsed time value from the source dataset.",
            )
        with amount_column:
            values["Amount"] = st.number_input(
                "Amount",
                min_value=0.0,
                value=amount_value,
                step=0.01,
                format="%.6f",
                key=f"single_amount_{widget_suffix}",
                help="Transaction amount. Negative values are not accepted.",
            )

    feature_groups = [
        ("Anonymized PCA features · V1–V10", FEATURE_COLUMNS[1:11]),
        ("Anonymized PCA features · V11–V20", FEATURE_COLUMNS[11:21]),
        ("Anonymized PCA features · V21–V28", FEATURE_COLUMNS[21:29]),
    ]
    for group_label, group_features in feature_groups:
        with st.expander(group_label, expanded=False):
            feature_columns = st.columns(2)
            for index, feature in enumerate(group_features):
                with feature_columns[index % 2]:
                    values[feature] = st.number_input(
                        feature,
                        value=safe_default_value(example, feature),
                        step=0.01,
                        format="%.6f",
                        key=f"single_{feature}_{widget_suffix}",
                    )
    return {feature: values[feature] for feature in FEATURE_COLUMNS}


def validate_transaction_values(values: dict[str, float]) -> None:
    """Validate the submitted transaction values before model inference."""
    numeric_values = np.asarray([values[feature] for feature in FEATURE_COLUMNS], dtype=float)
    if not np.isfinite(numeric_values).all():
        raise ValueError("All transaction inputs must be finite numeric values.")
    if values["Amount"] < 0:
        raise ValueError("Amount must be nonnegative.")


def single_prediction_matches_mode(result: object, input_mode: str) -> bool:
    """Return whether a stored result belongs to the currently selected mode."""
    return isinstance(result, dict) and result.get("input_mode") == input_mode


def render_prediction_result(result: dict) -> None:
    """Render the latest submitted prediction until another prediction replaces it."""
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="fg-section-title">Prediction result</div>', unsafe_allow_html=True)
    st.caption(
        "Last submitted transaction. Edits made inside the form are applied only "
        "after selecting Score transaction."
    )
    probability = result["fraud_probability"]
    is_fraudulent = result["is_fraudulent"]
    result_columns = st.columns(3)
    with result_columns[0]:
        st.metric("Predicted classification", "Fraudulent" if is_fraudulent else "Legitimate")
    with result_columns[1]:
        st.metric("Model fraud score", f"{probability * 100:.2f}%")
    with result_columns[2]:
        st.metric("Frozen threshold", f"{FROZEN_THRESHOLD:.2f}")

    if is_fraudulent:
        st.error(
            "This transaction is at or above the frozen threshold and is flagged "
            "for fraud review by the current model rule."
        )
    else:
        st.success(
            "This transaction is below the frozen threshold and is classified as "
            "legitimate by the current model rule."
        )
    st.caption(
        "The fraud score is the model's output for this input, not a guaranteed or "
        "perfectly calibrated real-world probability. Review it with appropriate "
        "controls and human judgment."
    )


def render_single_transaction() -> None:
    """Render the single-transaction prediction interface."""
    render_section_intro(
        "Single transaction prediction",
        "Score one transaction with the frozen Random Forest candidate at the fixed 0.40 decision threshold.",
    )
    st.info(
        "V1–V28 are anonymized PCA features from the source dataset. They are not "
        "ordinary transaction details and cannot be inferred from a card number, "
        "merchant, or customer name."
    )

    st.caption(
        "The optional example below is an artificial demonstration created for "
        "interface testing. It is not a genuine transaction and is not evidence "
        "of real-world fraud-detection performance."
    )
    use_example = st.checkbox(
        "Populate a reproducible synthetic demonstration example",
        value=True,
        key="single_use_training_example",
        help="Uses artificial feature values only; no original dataset transaction is loaded.",
    )
    input_mode = "synthetic_example" if use_example else "manual"
    previous_mode = st.session_state.get("single_input_mode")
    if previous_mode is not None and previous_mode != input_mode:
        st.session_state.pop("single_prediction_result", None)
    st.session_state["single_input_mode"] = input_mode
    st.caption(
        "Transaction fields are inside a form. Streamlit applies their edits when "
        "you submit the form, so any visible result remains tied to the last submitted input."
    )
    example = load_example_transaction() if use_example else None

    widget_suffix = "example" if use_example else "manual"
    with st.form(f"single_transaction_form_{widget_suffix}"):
        input_values = render_feature_inputs(example, widget_suffix)
        submitted = st.form_submit_button(
            "Score transaction",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        st.session_state.pop("single_prediction_result", None)
        try:
            validate_transaction_values(input_values)
            ordered_features = pd.DataFrame(
                [[input_values[feature] for feature in FEATURE_COLUMNS]],
                columns=FEATURE_COLUMNS,
            )
            model = load_frozen_model(str(MODEL_ARTIFACT_PATH))
            fraud_probability = float(
                predict_fraud_probabilities(model, ordered_features)[0]
            )
            st.session_state["single_prediction_result"] = {
                "fraud_probability": fraud_probability,
                "is_fraudulent": fraud_probability >= CLASSIFICATION_THRESHOLD,
                "input_mode": input_mode,
            }
        except (OSError, ValueError, TypeError, AttributeError, IndexError) as error:
            st.error(f"Prediction could not be completed: {error}")

    result = st.session_state.get("single_prediction_result")
    if single_prediction_matches_mode(result, input_mode):
        render_prediction_result(result)

    st.caption(
        "Educational demonstration only. FraudGuard AI is not a production banking "
        "system and should not be used for unsupervised financial decisions."
    )


def render_batch_detection_placeholder() -> None:
    """Render the batch fraud detection placeholder."""
    render_section_intro(
        "Batch fraud detection",
        "Upload a compatible transaction file to score multiple records and prepare an investigation-ready output.",
    )
    render_empty_state(
        "⇧",
        "Batch scoring is not connected yet",
        "A secure upload flow, schema checks, batch predictions, and downloadable results will be implemented here later.",
    )


@st.cache_data(show_spinner=False)
def load_batch_template() -> pd.DataFrame:
    """Return a deterministic artificial CSV template with feature columns only."""
    return build_synthetic_demo_transactions()


def read_uploaded_batch(uploaded_file) -> pd.DataFrame:
    """Read a bounded CSV upload after lightweight size and header checks."""
    file_size = getattr(uploaded_file, "size", None)
    if file_size is None:
        raise ValueError(
            "The uploaded file size could not be determined safely. "
            "Please upload the CSV again."
        )
    if file_size == 0:
        raise ValueError("The uploaded CSV file is empty.")
    if file_size > MAX_BATCH_FILE_BYTES:
        maximum_megabytes = MAX_BATCH_FILE_BYTES // (1024 * 1024)
        raise ValueError(
            f"This CSV file is too large ({file_size / (1024 * 1024):.1f} MB). "
            f"Please upload a file no larger than {maximum_megabytes} MB."
        )

    uploaded_file.seek(0)
    header_bytes = uploaded_file.readline()
    if not header_bytes:
        raise ValueError("The uploaded CSV file is empty.")
    try:
        header = next(csv.reader([header_bytes.decode("utf-8-sig")]))
    except (UnicodeDecodeError, StopIteration, csv.Error) as error:
        raise ValueError("The uploaded file does not have a readable CSV header.") from error

    if len(header) > MAX_BATCH_COLUMNS:
        raise ValueError(
            f"This CSV contains {len(header):,} columns. "
            f"Please upload a file with no more than {MAX_BATCH_COLUMNS:,} columns."
        )

    duplicate_columns = [
        name for name, count in Counter(header).items() if count > 1
    ]
    if duplicate_columns:
        raise ValueError(
            "Duplicate column name(s) are not allowed: "
            + ", ".join(duplicate_columns)
        )

    uploaded_file.seek(0)
    try:
        data = pd.read_csv(uploaded_file, nrows=MAX_BATCH_ROWS + 1)
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
        raise ValueError(f"The uploaded file could not be parsed as CSV: {error}") from error

    if data.empty:
        raise ValueError("The uploaded CSV file contains no transaction rows.")
    if len(data) > MAX_BATCH_ROWS:
        raise ValueError(
            f"The upload contains more than {MAX_BATCH_ROWS:,} rows. "
            "Please upload a smaller batch."
        )
    if len(data.columns) > MAX_BATCH_COLUMNS:
        raise ValueError(
            f"This CSV contains {len(data.columns):,} columns. "
            f"Please upload a file with no more than {MAX_BATCH_COLUMNS:,} columns."
        )
    return data


def calculate_uploaded_file_sha256(uploaded_file) -> str:
    """Calculate a bounded, content-based identity without copying the full file."""
    digest = hashlib.sha256()
    uploaded_file.seek(0)
    while chunk := uploaded_file.read(1024 * 1024):
        digest.update(chunk)
    uploaded_file.seek(0)
    return digest.hexdigest()


def get_batch_upload_identity(uploaded_file) -> str:
    """Return a cached or freshly calculated content identity for an upload."""
    upload_token = getattr(uploaded_file, "file_id", None)
    cached_identity = st.session_state.get("batch_upload_hash_cache")
    if (
        upload_token is not None
        and isinstance(cached_identity, dict)
        and cached_identity.get("file_id") == upload_token
        and cached_identity.get("size") == getattr(uploaded_file, "size", None)
    ):
        return cached_identity["sha256"]

    sha256 = calculate_uploaded_file_sha256(uploaded_file)
    if upload_token is not None:
        st.session_state["batch_upload_hash_cache"] = {
            "file_id": upload_token,
            "size": getattr(uploaded_file, "size", None),
            "sha256": sha256,
        }
    return sha256


def validate_batch_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate and numerically coerce only the model input columns."""
    missing_columns = [column for column in FEATURE_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(
            "Missing required feature column(s): " + ", ".join(missing_columns)
        )

    validated = data.copy()
    nonnumeric_columns = []
    for feature in FEATURE_COLUMNS:
        original = validated[feature]
        numeric = pd.to_numeric(original, errors="coerce")
        invalid_values = numeric.isna() & original.notna()
        if invalid_values.any():
            nonnumeric_columns.append(feature)
        validated[feature] = numeric
    if nonnumeric_columns:
        raise ValueError(
            "Required feature columns contain nonnumeric values: "
            + ", ".join(nonnumeric_columns)
        )

    feature_data = validated[FEATURE_COLUMNS]
    missing_columns = feature_data.columns[feature_data.isna().any()].tolist()
    if missing_columns:
        raise ValueError(
            "Required feature columns contain missing values: "
            + ", ".join(missing_columns)
        )
    if not np.isfinite(feature_data.to_numpy(dtype=float)).all():
        raise ValueError("Required feature columns contain infinite or non-finite values.")
    if (feature_data["Amount"] < 0).any():
        raise ValueError("Amount must be nonnegative for every uploaded transaction.")
    return validated


def score_batch_data(data: pd.DataFrame) -> pd.DataFrame:
    """Score validated batch rows while preserving all uploaded columns."""
    validated = validate_batch_data(data)
    model = load_frozen_model(str(MODEL_ARTIFACT_PATH))
    model_inputs = validated.loc[:, list(FEATURE_COLUMNS)]
    fraud_scores = predict_fraud_probabilities(model, model_inputs)

    results = validated.copy()
    results["fraud_score"] = fraud_scores
    results["fraud_score_percent"] = fraud_scores * 100
    results["predicted_class"] = (fraud_scores >= FROZEN_THRESHOLD).astype(int)
    results["predicted_label"] = results["predicted_class"].map(
        {0: "Legitimate", 1: "Fraudulent"}
    )
    return results


def build_score_distribution_figure(results: pd.DataFrame) -> go.Figure:
    """Build an interactive Plotly distribution of model fraud scores."""
    figure = go.Figure(
        data=go.Histogram(
            x=results["fraud_score"],
            xbins={"start": 0.0, "end": 1.0, "size": 0.05},
            marker={"color": "#43d6c5", "line": {"color": "#071426", "width": 1}},
            hovertemplate="Fraud score: %{x:.3f}<br>Transactions: %{y}<extra></extra>",
        )
    )
    figure.add_vline(
        x=FROZEN_THRESHOLD,
        line_width=2,
        line_dash="dash",
        line_color="#62a5ff",
        annotation_text=f"Frozen threshold {FROZEN_THRESHOLD:.2f}",
        annotation_position="top right",
    )
    figure.update_layout(
        title="Predicted fraud-score distribution",
        xaxis_title="Model fraud score",
        xaxis={
            "range": [0.0, 1.0],
            "dtick": 0.1,
            "tickformat": ".1f",
        },
        yaxis_title="Transactions",
        height=420,
        margin={"l": 20, "r": 20, "t": 70, "b": 30},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#edf4ff"},
        bargap=0.04,
    )
    return figure


def render_batch_results(results: pd.DataFrame) -> None:
    """Render batch summary cards, chart, searchable table, and download."""
    total = len(results)
    fraudulent = int(results["predicted_class"].eq(1).sum())
    legitimate = total - fraudulent
    alert_rate = fraudulent / total * 100 if total else 0.0

    columns = st.columns(4)
    cards = [
        ("Transactions analyzed", f"{total:,}"),
        ("Flagged as fraudulent", f"{fraudulent:,}"),
        ("Classified as legitimate", f"{legitimate:,}"),
        ("Fraud alert rate", f"{alert_rate:.2f}%"),
    ]
    for column, (label, value) in zip(columns, cards):
        with column:
            st.metric(label, value)

    st.markdown("<br>", unsafe_allow_html=True)
    st.plotly_chart(
        build_score_distribution_figure(results),
        use_container_width=True,
        config={"displayModeBar": False, "responsive": True},
    )

    st.markdown('<div class="fg-section-title">Prediction results</div>', unsafe_allow_html=True)
    st.caption(
        "The table preserves uploaded columns and appends model outputs. Use the search "
        "box to filter rows; the table also supports column sorting."
    )
    search_query = st.text_input(
        "Search prediction results",
        placeholder="Search any uploaded or prediction column...",
        key="batch_results_search",
    )
    display_results = results
    if search_query.strip():
        query = search_query.strip().lower()
        matches = results.astype(str).apply(
            lambda column: column.str.lower().str.contains(query, na=False)
        )
        display_results = results[matches.any(axis=1)]
    st.dataframe(display_results, use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(display_results):,} of {len(results):,} analyzed transactions.")

    st.download_button(
        "Download complete prediction results",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="fraudguard_batch_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.caption(
        "A fraud alert is a model prediction for review, not confirmation that a transaction is fraudulent. "
        "Scores are not guaranteed calibrated real-world probabilities."
    )


def render_batch_detection() -> None:
    """Render batch upload, validation, scoring, and result download."""
    render_section_intro(
        "Batch fraud detection",
        "Upload compatible transactions to score a batch with the frozen Random Forest candidate at threshold 0.40.",
    )
    st.info(
        "Required model inputs are Time, V1–V28, and Amount. Additional columns such as "
        "transaction IDs are preserved in the results but are not sent to the model."
    )

    template = load_batch_template()
    st.download_button(
        "Download synthetic CSV template",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="fraudguard_synthetic_batch_template.csv",
        mime="text/csv",
        help="Contains all required feature columns and three artificial demonstration rows.",
    )
    st.caption(
        "The template contains artificial demonstration values only. It is useful "
        "for testing the interface, not for measuring real-world fraud performance."
    )

    uploaded_file = st.file_uploader(
        "Upload transaction CSV",
        type=["csv"],
        key="batch_transaction_uploader",
        help=(
            f"Maximum upload size: {MAX_BATCH_FILE_BYTES // (1024 * 1024)} MB, "
            f"{MAX_BATCH_ROWS:,} rows, and {MAX_BATCH_COLUMNS:,} columns."
        ),
    )
    if uploaded_file is None:
        render_empty_state(
            "⇧",
            "Upload a transaction file to begin",
            "The file must include all 30 required model features. Any additional identifiers will be carried through to the downloadable results.",
        )
        return

    try:
        upload_identity = get_batch_upload_identity(uploaded_file)
    except (AttributeError, OSError, TypeError, ValueError) as error:
        st.error(f"The uploaded file identity could not be calculated safely: {error}")
        return

    if st.session_state.get("batch_upload_identity") != upload_identity:
        st.session_state["batch_upload_identity"] = upload_identity
        st.session_state.pop("batch_prediction_results", None)

    st.caption(f"Selected file: {uploaded_file.name} · {uploaded_file.size:,} bytes")
    if st.button("Analyze uploaded transactions", type="primary", use_container_width=True):
        st.session_state.pop("batch_prediction_results", None)
        try:
            uploaded_data = read_uploaded_batch(uploaded_file)
            results = score_batch_data(uploaded_data)
            st.session_state["batch_prediction_results"] = results
        except (
            OSError,
            UnicodeDecodeError,
            ValueError,
            TypeError,
            AttributeError,
            IndexError,
            RuntimeError,
        ) as error:
            st.error(f"Batch analysis could not be completed: {error}")

    if "batch_prediction_results" in st.session_state:
        render_batch_results(st.session_state["batch_prediction_results"])

    st.caption(
        "Educational demonstration only. FraudGuard AI is not a production banking system; "
        "alerts require appropriate human review and operational controls."
    )


def render_model_performance_placeholder() -> None:
    """Render the model performance placeholder."""
    render_section_intro(
        "Model performance",
        "Review frozen holdout metrics and the operating characteristics of the selected Random Forest candidate.",
    )
    metric_columns = st.columns(3)
    for column, label in zip(
        metric_columns,
        ["PR-AUC", "Recall", "Precision"],
    ):
        with column:
            st.metric(label, "—", help="Holdout metrics will be connected in a later step.")
    st.markdown("<br>", unsafe_allow_html=True)
    render_empty_state(
        "◒",
        "Performance visualizations will appear here",
        "Confusion matrices, precision-recall views, threshold context, and validation notes will be connected without changing the frozen model configuration.",
    )


def render_model_performance() -> None:
    """Render historical holdout metrics and training-only model comparison."""
    render_section_intro(
        "Model performance",
        "Review the frozen Random Forest on its one-time untouched holdout evaluation and compare training-only validation results.",
    )
    metrics = load_evaluation_metrics(str(METRICS_PATH))
    if metrics is None:
        st.warning(
            "Historical evaluation metrics are unavailable or malformed. "
            "No performance values have been inferred or substituted."
        )
        render_empty_state(
            "!",
            "Performance results unavailable",
            "Check reports/final_model_metrics.json and ensure it contains the saved final-model metrics structure.",
        )
        return

    try:
        values = extract_dashboard_values(metrics)
        metric_values = metrics["metrics"]
        accuracy = float(metric_values["accuracy"])
        roc_auc = float(metric_values["roc_auc"])
    except (KeyError, TypeError, ValueError):
        st.warning(
            "The historical evaluation metrics file does not contain the expected performance keys."
        )
        render_empty_state(
            "!",
            "Performance results could not be read",
            "The page expects the existing metrics and confusion_matrix objects from the final holdout evaluation.",
        )
        return

    st.markdown(
        f'<div class="fg-status">● {values["evaluation_scope"]} · {values["model_name"]} · threshold {values["threshold"]:.2f}</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "These are historical one-time holdout results, not live monitoring metrics. "
        "The model and threshold were frozen before this evaluation."
    )

    performance_cards = [
        ("Precision", f'{values["precision"] * 100:.2f}%', "Of alerts, how many were fraud"),
        ("Recall", f'{values["recall"] * 100:.2f}%', "Of fraud cases, how many were found"),
        ("F1-score", f'{values["f1_score"] * 100:.2f}%', "Balance of precision and recall"),
        ("Accuracy", f"{accuracy * 100:.2f}%", "All classifications correct"),
        ("PR-AUC", f'{values["pr_auc"]:.4f}', "Ranking quality for rare fraud"),
        ("ROC-AUC", f"{roc_auc:.4f}", "Overall ranking separation"),
    ]
    for row_start in (0, 3):
        columns = st.columns(3)
        for column, (label, value, help_text) in zip(columns, performance_cards[row_start : row_start + 3]):
            with column:
                st.metric(label, value, help=help_text)
    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1.05, 1.35])
    with left:
        st.markdown('<div class="fg-section-title">Confusion matrix</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="fg-section-description">How the frozen threshold classified the holdout transactions.</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            build_confusion_matrix_figure(values),
            use_container_width=True,
            config={"displayModeBar": False, "responsive": True},
        )
    with right:
        st.markdown('<div class="fg-section-title">What the metrics mean</div>', unsafe_allow_html=True)
        with st.expander("Beginner-friendly metric guide", expanded=True):
            st.markdown(
                """
                - **Precision**: When the model raises a fraud alert, precision is the share of those alerts that are actually fraud.
                - **Recall**: Recall is the share of all known fraud cases that the model successfully catches.
                - **F1-score**: F1 combines precision and recall into one balance; it is high only when both are reasonably strong.
                - **PR-AUC**: This measures how well the model ranks rare fraud cases while keeping false alerts under control. It is especially useful for imbalanced fraud data.
                - **ROC-AUC**: This measures how well the model separates legitimate and fraudulent transactions across possible score cutoffs.
                """
            )
        st.warning(
            "Accuracy alone can mislead here because fraud is rare. A model that labels nearly every transaction legitimate could appear highly accurate while missing important fraud. PR-AUC, recall, precision, and the confusion matrix provide a more useful operational view."
        )

    with st.expander("Model comparison: training-only cross-validation", expanded=False):
        st.caption(
            "These are five-fold cross-validation results from the training dataset only. "
            "They are distinct from the final holdout metrics shown above."
        )
        comparison = pd.DataFrame(
            {
                "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
                "Mean CV PR-AUC": [0.738234, 0.839862, 0.835169],
            }
        )
        comparison_columns = st.columns([1, 1.5])
        with comparison_columns[0]:
            st.dataframe(
                comparison.style.format({"Mean CV PR-AUC": "{:.6f}"}),
                use_container_width=True,
                hide_index=True,
            )
        with comparison_columns[1]:
            comparison_figure = go.Figure(
                go.Bar(
                    x=comparison["Model"],
                    y=comparison["Mean CV PR-AUC"],
                    marker_color=["#6b7f9e", "#43d6c5", "#62a5ff"],
                    text=[f"{score:.6f}" for score in comparison["Mean CV PR-AUC"]],
                    textposition="outside",
                    hovertemplate="%{x}<br>Mean CV PR-AUC: %{y:.6f}<extra></extra>",
                )
            )
            comparison_figure.update_layout(
                title="Training-only mean CV PR-AUC",
                yaxis_title="Mean CV PR-AUC",
                yaxis_range=[0.65, 0.9],
                height=360,
                margin={"l": 20, "r": 20, "t": 65, "b": 30},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#edf4ff"},
            )
            st.plotly_chart(
                comparison_figure,
                use_container_width=True,
                config={"displayModeBar": False, "responsive": True},
            )

    st.caption(
        "Educational evaluation view only. FraudGuard AI is not a production banking "
        "system, and historical performance may change with future transaction patterns."
    )


def render_explainable_ai_placeholder() -> None:
    """Render the explainable AI placeholder."""
    render_section_intro(
        "Explainable AI",
        "Understand which model signals influence a fraud-risk assessment at portfolio and transaction level.",
    )
    render_empty_state(
        "✦",
        "SHAP explanations will appear here",
        "Global importance, beeswarm summaries, and local waterfall explanations will be connected to the prepared explainability outputs.",
    )


def render_shap_figure(path: Path, title: str, description: str) -> None:
    """Render one prepared SHAP figure or a helpful missing-file state."""
    st.markdown(f'<div class="fg-section-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="fg-section-description">{description}</div>',
        unsafe_allow_html=True,
    )
    if not path.is_file():
        st.warning(f"This SHAP figure is unavailable: {path}")
        return
    try:
        st.image(str(path), use_container_width=True)
    except Exception as error:
        st.warning(f"This SHAP figure could not be displayed: {error}")


def render_explainable_ai() -> None:
    """Render the prepared SHAP explainability dashboard."""
    render_section_intro(
        "Explainable AI",
        "Understand which model signals influence the frozen Random Forest fraud-class output.",
    )
    st.markdown(
        f'<div class="fg-status">● {SELECTED_MODEL} · frozen threshold {FROZEN_THRESHOLD:.2f}</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "These figures were generated once by the prepared SHAP explainer from a reproducible "
        "training-data sample of up to 500 transactions. They are explanations of model behavior, "
        "not a rerun of evaluation or proof of why fraud actually occurred."
    )

    figures_path = Path(__file__).resolve().parents[1] / "reports" / "figures"
    tabs = st.tabs(["Global importance", "Beeswarm summary", "Local waterfall"])
    with tabs[0]:
        render_shap_figure(
            figures_path / "random_forest_shap_feature_importance.png",
            "Global SHAP feature importance",
            "This bar chart ranks features by their average absolute SHAP contribution across the sampled training transactions. Larger bars indicate greater influence on the model output, not greater real-world importance.",
        )
    with tabs[1]:
        render_shap_figure(
            figures_path / "random_forest_shap_beeswarm.png",
            "SHAP beeswarm summary",
            "Each point represents one sampled training transaction. Position shows contribution direction and size; color shows whether the feature value was relatively low or high within the sample.",
        )
    with tabs[2]:
        render_shap_figure(
            figures_path / "random_forest_shap_waterfall.png",
            "Local SHAP waterfall explanation",
            "This waterfall explains one example training transaction by showing how individual feature contributions moved the model output from its baseline toward the final Class 1 output.",
        )

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        with st.expander("How to interpret SHAP contributions", expanded=True):
            st.markdown(
                """
                - A **positive contribution** pushes the model's fraud-class output higher for the transaction being explained.
                - A **negative contribution** pushes the model's fraud-class output lower.
                - The direction is relative to the explainer's baseline and the model output; it does not mean a feature caused fraud.
                - SHAP contributions are expressed in the explainer's output units. They must not automatically be read as percentage-point changes in fraud probability.
                """
            )
    with right:
        with st.expander("Important feature context", expanded=True):
            st.markdown(
                """
                **V1–V28 are anonymized PCA features.** They are transformed statistical components, not ordinary transaction details. Their SHAP values cannot be translated directly into named financial behaviors and cannot be inferred from a card number, merchant, or customer name.

                The local waterfall is for the reproducible example transaction used when generating the saved figure. It is **not** the transaction currently entered in Single Transaction Prediction, and it does not update when that form is used.
                """
            )

    st.info(
        "Prediction answers what the model decided for an input. SHAP explains how the model arrived at that output. Neither one establishes why a transaction was genuinely fraudulent; investigation and domain evidence are still required."
    )
    st.caption(
        "Educational explainability view only. FraudGuard AI is not a production banking system, "
        "and SHAP explanations should not be treated as causal or regulatory findings."
    )


def render_about_placeholder() -> None:
    """Render project context and model governance notes."""
    render_section_intro(
        "About FraudGuard AI",
        "A portfolio project demonstrating an explainable fraud-detection workflow from data preparation through model review.",
    )
    left, right = st.columns(2)
    with left:
        st.markdown(
            """
            <div class="fg-card">
                <div class="fg-card-label">Technology foundation</div>
                <div style="color: var(--fg-text); line-height: 1.8; margin-top: 0.65rem;">
                    Python · pandas · NumPy · scikit-learn<br>
                    Random Forest · SHAP · Streamlit · Plotly
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="fg-card">
                <div class="fg-card-label">Governance posture</div>
                <div style="color: var(--fg-text); line-height: 1.8; margin-top: 0.65rem;">
                    Frozen model candidate<br>
                    Fixed threshold · Explainability planned
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        "This interface is an educational demonstration and is not a substitute "
        "for regulated financial-risk controls, human review, or production monitoring."
    )


def render_about() -> None:
    """Render the project overview, workflow, governance, and authorship page."""
    render_section_intro(
        "About FraudGuard AI",
        "An explainable machine learning system for understanding and reviewing financial transaction fraud risk.",
    )

    st.markdown(
        """
        FraudGuard AI explores how machine learning can help identify suspicious
        credit card transactions while keeping model behavior visible to reviewers.
        The project combines careful evaluation, a frozen decision policy, and
        explainability so that a score can be investigated rather than treated as
        an unquestionable decision.
        """
    )

    dataset_columns = st.columns(3)
    dataset_facts = [
        ("284,807", "Original transactions", "Public credit card fraud dataset"),
        ("492", "Originally labeled fraud", "A highly imbalanced positive class"),
        ("30", "Model input features", "Time, V1-V28, and Amount"),
    ]
    for column, (value, label, note) in zip(dataset_columns, dataset_facts):
        with column:
            st.markdown(
                f"""
                <div class="fg-card">
                    <div class="fg-card-label">{label}</div>
                    <div class="fg-card-value">{value}</div>
                    <div class="fg-card-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="fg-section-title">Why this problem matters</div>', unsafe_allow_html=True)
    st.markdown(
        """
        Financial fraud creates direct losses for customers and institutions and
        can also produce costly chargebacks, investigation work, and loss of trust.
        Suspicious-transaction detection is difficult because fraudulent behavior
        is rare, patterns can change over time, and an overly aggressive system can
        disrupt legitimate customers. FraudGuard AI is designed as an educational
        framework for examining that balance between catching fraud and limiting
        unnecessary alerts.
        """
    )

    with st.expander("Project workflow", expanded=True):
        st.markdown(
            """
            1. **Dataset and preparation** - Start with the public credit card transaction dataset, validate its schema, remove exact duplicate rows for the initial experiment, and create a stratified train/test split.
            2. **Baseline modeling** - Train Logistic Regression, Random Forest, and XGBoost models using the 30 input features and the binary `Class` target.
            3. **Training-only comparison** - Compare models with stratified cross-validation and use training-only out-of-fold predictions for threshold analysis.
            4. **Frozen candidate** - Select Random Forest as the provisional production candidate and freeze the classification threshold at 0.40 before the holdout evaluation.
            5. **Holdout evaluation** - Run one-time evaluation on the untouched holdout set and preserve the resulting metrics as the historical reference.
            6. **Explainability and application** - Generate SHAP explanations from training data and present the workflow through this Streamlit interface.
            """
        )

    st.markdown('<div class="fg-section-title">Data policy and final model</div>', unsafe_allow_html=True)
    st.markdown(
        """
        Exact duplicate rows were removed during preprocessing for the initial
        experiment. This does not mean that every identical record was an erroneous
        transaction: repeated legitimate transactions can be genuinely identical
        across the available columns. The duplicate policy is therefore documented
        as an experimental choice and should be tested through future sensitivity
        analysis.
        """
    )
    model_columns = st.columns(2)
    with model_columns[0]:
        st.markdown(
            f"""
            <div class="fg-card">
                <div class="fg-card-label">Frozen production candidate</div>
                <div class="fg-card-value">{SELECTED_MODEL}</div>
                <div class="fg-card-note">Classification threshold: {FROZEN_THRESHOLD:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with model_columns[1]:
        st.markdown(
            """
            <div class="fg-card">
                <div class="fg-card-label">Holdout policy</div>
                <div class="fg-card-value">One-time evaluation</div>
                <div class="fg-card-note">Metrics are historical, not live monitoring.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="fg-section-title">Final holdout snapshot</div>', unsafe_allow_html=True)
    metrics = load_evaluation_metrics(str(METRICS_PATH))
    metric_values = None
    if metrics is not None:
        try:
            metric_values = extract_dashboard_values(metrics)
            accuracy = float(metrics["metrics"]["accuracy"])
        except (KeyError, TypeError, ValueError):
            metric_values = None
    if metric_values is None:
        st.warning(
            "Saved final metrics are unavailable or malformed. The About page cannot display verified holdout values."
        )
    else:
        st.caption(
            "Historical one-time holdout results from the saved evaluation JSON; these are not live transaction metrics."
        )
        final_metrics = [
            ("Precision", f'{metric_values["precision"] * 100:.2f}%'),
            ("Recall", f'{metric_values["recall"] * 100:.2f}%'),
            ("F1-score", f'{metric_values["f1_score"] * 100:.2f}%'),
            ("PR-AUC", f'{metric_values["pr_auc"]:.4f}'),
        ]
        metric_columns = st.columns(4)
        for column, (label, value) in zip(metric_columns, final_metrics):
            with column:
                st.metric(label, value)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="fg-section-title">Application sections</div>', unsafe_allow_html=True)
    application_sections = [
        ("Dashboard", "Historical holdout summary, confusion matrix, and model comparison."),
        ("Single Transaction Prediction", "Score one transaction with the frozen model and threshold."),
        ("Batch Fraud Detection", "Validate, score, search, and download results for uploaded CSV files."),
        ("Model Performance", "Review holdout metrics and training-only cross-validation context."),
        ("Explainable AI", "Explore prepared SHAP feature-importance, beeswarm, and waterfall views."),
        ("About", "Understand the project purpose, workflow, technology, and responsible-use boundaries."),
    ]
    for row_start in range(0, len(application_sections), 3):
        columns = st.columns(3)
        for column, (section, purpose) in zip(columns, application_sections[row_start : row_start + 3]):
            with column:
                st.markdown(
                    f"""
                    <div class="fg-card">
                        <div class="fg-card-label">{section}</div>
                        <div class="fg-card-note" style="margin-top: 0.7rem; line-height: 1.6;">{purpose}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="fg-section-title">Technology stack</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="fg-card">
                <div style="color: var(--fg-text); line-height: 2.0;">
                    <strong>Python</strong> for the project workflow<br>
                    <strong>pandas</strong> for tabular data handling<br>
                    <strong>scikit-learn</strong> for evaluation and Random Forest<br>
                    <strong>XGBoost</strong> for model comparison<br>
                    <strong>SHAP</strong> for model explainability<br>
                    <strong>Streamlit</strong> for the application interface<br>
                    <strong>Plotly</strong> for interactive visualizations
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown('<div class="fg-section-title">Dataset credit</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="fg-card">
                <div style="color: var(--fg-text); line-height: 1.65;">
                    This project uses the public credit card fraud dataset associated
                    with the Kaggle community and the Université de la Libre de
                    Bruxelles (ULB) machine learning research context. The dataset
                    is credited as a public source; FraudGuard AI does not claim
                    ownership of the underlying data.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Limitations and responsible use", expanded=True):
        st.warning(
            "FraudGuard AI is an educational and research demonstration only, not a production banking system."
        )
        st.markdown(
            """
            - There are no real-time banking integrations or live authorization controls.
            - Model scores are not guaranteed calibrated real-world probabilities.
            - V1-V28 are anonymized PCA features and cannot be interpreted as named financial behaviors.
            - False negatives and false positives remain possible; alerts require appropriate human review.
            - The initial random stratified split is not a substitute for temporal validation or drift testing.
            - Results are specific to this dataset, preprocessing policy, frozen model, threshold, and evaluation setup.
            """
        )

    st.markdown("<br>", unsafe_allow_html=True)
    author_columns = st.columns([1.8, 1])
    with author_columns[0]:
        st.markdown(
            """
            <div class="fg-card">
                <div class="fg-card-label">Project author</div>
                <div class="fg-card-value">Mark Bryson Mutuma</div>
                <div class="fg-card-note">FraudGuard AI · Explainable machine learning portfolio project</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with author_columns[1]:
        st.markdown(
            """
            <div class="fg-card">
                <div class="fg-card-label">Project posture</div>
                <div class="fg-card-note" style="margin-top: 0.7rem; line-height: 1.6;">
                    Transparent experimentation<br>
                    Reproducible evaluation<br>
                    Responsible-use boundaries
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_footer() -> None:
    """Render the shared application footer."""
    st.markdown(
        f"""
        <div class="fg-footer">
            <strong>{PROJECT_NAME}</strong> · Educational machine learning demonstration.<br>
            Not financial advice and not intended for unsupervised production decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Render the Streamlit application foundation."""
    st.set_page_config(
        page_title=PROJECT_NAME,
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()
    selected_section = render_sidebar()
    render_header()

    section_renderers = {
        "Dashboard": render_dashboard,
        "Single Transaction Prediction": render_single_transaction,
        "Batch Fraud Detection": render_batch_detection,
        "Model Performance": render_model_performance,
        "Explainable AI": render_explainable_ai,
        "About": render_about,
    }
    section_renderers[selected_section]()
    render_footer()


if __name__ == "__main__":
    main()
