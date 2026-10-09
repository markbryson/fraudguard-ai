"""Fast regression tests for FraudGuard AI application contracts."""

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import app.main as app_main
from src.models.final_model_config import (
    CLASSIFICATION_THRESHOLD,
    FEATURE_COLUMNS,
    SELECTED_MODEL_NAME,
)


class InMemoryUpload(io.BytesIO):
    """Small Streamlit UploadedFile-like object for parser tests."""

    def __init__(self, content: bytes):
        super().__init__(content)
        self.size = len(content)
        self.name = "synthetic.csv"


def make_feature_frame(rows: int = 2) -> pd.DataFrame:
    """Create deterministic synthetic model inputs in the frozen feature order."""
    return pd.DataFrame(
        np.zeros((rows, len(FEATURE_COLUMNS))),
        columns=FEATURE_COLUMNS,
    )


def make_upload(data: pd.DataFrame) -> InMemoryUpload:
    """Serialize a small synthetic frame as an in-memory CSV upload."""
    return InMemoryUpload(data.to_csv(index=False).encode("utf-8"))


def make_valid_metrics_payload() -> dict:
    """Create a complete valid historical metrics payload for validation tests."""
    return {
        "evaluation_scope": "One-time untouched holdout evaluation",
        "model_name": "Random Forest",
        "threshold": 0.40,
        "feature_columns": list(FEATURE_COLUMNS),
        "target_column": "Class",
        "n_samples": 56746,
        "metrics": {
            "precision": 0.9459459459,
            "recall": 0.7368421053,
            "f1_score": 0.8284023669,
            "accuracy": 0.9994889508,
            "pr_auc": 0.8095198448,
            "roc_auc": 0.9531642030,
        },
        "confusion_matrix": {
            "true_positives": 70,
            "false_positives": 4,
            "true_negatives": 56647,
            "false_negatives": 25,
        },
    }


def test_frozen_model_configuration() -> None:
    assert SELECTED_MODEL_NAME == "Random Forest"
    assert CLASSIFICATION_THRESHOLD == 0.40
    assert FEATURE_COLUMNS == ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]


def test_app_bootstrap_imports_src_from_repository_root() -> None:
    expected_root = Path(app_main.__file__).resolve().parents[1]

    assert app_main.PROJECT_ROOT == expected_root
    assert str(expected_root) in sys.path
    assert app_main.MODEL_ARTIFACT_PATH == expected_root / "models" / "random_forest.joblib"
    assert len(FEATURE_COLUMNS) == 30


def test_synthetic_demo_transactions_have_valid_frozen_schema_and_values() -> None:
    first = app_main.build_synthetic_demo_transactions()
    second = app_main.build_synthetic_demo_transactions()

    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == FEATURE_COLUMNS
    assert first.shape == (3, 30)
    assert np.isfinite(first.to_numpy(dtype=float)).all()
    assert (first["Time"] >= 0).all()
    assert (first["Amount"] >= 0).all()
    assert "Class" not in first.columns


def test_synthetic_example_and_template_work_without_train_csv() -> None:
    example = app_main.load_example_transaction()
    template = app_main.load_batch_template()

    assert list(example) == FEATURE_COLUMNS
    assert list(template.columns) == FEATURE_COLUMNS
    assert len(template) == 3
    assert example == template.iloc[0].to_dict()


def test_valid_batch_csv_is_read_and_validated() -> None:
    data = make_feature_frame()
    data.insert(0, "transaction_id", ["txn-1", "txn-2"])

    uploaded = app_main.read_uploaded_batch(make_upload(data))
    validated = app_main.validate_batch_data(uploaded)

    assert list(validated.columns) == ["transaction_id", *FEATURE_COLUMNS]
    assert list(validated.loc[:, FEATURE_COLUMNS].columns) == FEATURE_COLUMNS


def test_batch_validation_rejects_missing_required_feature() -> None:
    data = make_feature_frame().drop(columns=["V1"])

    with pytest.raises(ValueError, match="Missing required feature"):
        app_main.validate_batch_data(data)


def test_batch_csv_rejects_duplicate_column_names() -> None:
    data = make_feature_frame()
    header = [*FEATURE_COLUMNS[:-1], "Amount", "Amount"]
    content = ",".join(header) + "\n" + ",".join(["0"] * len(header)) + "\n"

    with pytest.raises(ValueError, match="Duplicate column"):
        app_main.read_uploaded_batch(InMemoryUpload(content.encode("utf-8")))


def test_batch_validation_rejects_nonnumeric_values() -> None:
    data = make_feature_frame()
    data["V1"] = data["V1"].astype(object)
    data.loc[0, "V1"] = "not-a-number"

    with pytest.raises(ValueError, match="nonnumeric"):
        app_main.validate_batch_data(data)


def test_batch_validation_rejects_missing_values() -> None:
    data = make_feature_frame()
    data.loc[0, "V1"] = np.nan

    with pytest.raises(ValueError, match="missing values"):
        app_main.validate_batch_data(data)


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_batch_validation_rejects_nan_and_infinite_values(invalid_value: float) -> None:
    data = make_feature_frame()
    data.loc[0, "V1"] = invalid_value

    with pytest.raises(ValueError):
        app_main.validate_batch_data(data)


def test_batch_validation_rejects_negative_amount() -> None:
    data = make_feature_frame()
    data.loc[0, "Amount"] = -0.01

    with pytest.raises(ValueError, match="Amount must be nonnegative"):
        app_main.validate_batch_data(data)


@pytest.mark.parametrize(
    "content, message",
    [
        (b"", "empty"),
        (b'Time,V1\n"unterminated\n', "could not be parsed"),
    ],
)
def test_batch_csv_rejects_empty_and_malformed_files(
    content: bytes, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        app_main.read_uploaded_batch(InMemoryUpload(content))


def test_batch_csv_rejects_more_than_100_columns() -> None:
    columns = [f"column_{i}" for i in range(app_main.MAX_BATCH_COLUMNS + 1)]
    content = (",".join(columns) + "\n" + ",".join(["0"] * len(columns)) + "\n").encode(
        "utf-8"
    )

    with pytest.raises(ValueError, match="no more than 100 columns"):
        app_main.read_uploaded_batch(InMemoryUpload(content))


def test_batch_csv_rejects_files_over_10_mb() -> None:
    uploaded = InMemoryUpload(b"Time\n")
    uploaded.size = app_main.MAX_BATCH_FILE_BYTES + 1

    with pytest.raises(ValueError, match="no larger than 10 MB"):
        app_main.read_uploaded_batch(uploaded)


def test_batch_csv_rejects_more_than_100000_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    uploaded = InMemoryUpload(b"Time\n0\n")
    oversized_frame = pd.DataFrame({"Time": range(app_main.MAX_BATCH_ROWS + 1)})

    def fake_read_csv(*args, **kwargs) -> pd.DataFrame:
        return oversized_frame

    monkeypatch.setattr(app_main.pd, "read_csv", fake_read_csv)

    with pytest.raises(ValueError, match="more than 100,000 rows"):
        app_main.read_uploaded_batch(uploaded)


class StubModel(app_main.RandomForestClassifier):
    """Mock classifier that records inference column order."""

    def __init__(
        self,
        score: float,
        classes: tuple[int, int] = (0, 1),
        feature_names: list[str] | None = None,
        feature_count: int | None = None,
        include_feature_names: bool = True,
    ):
        super().__init__(n_estimators=1, random_state=42)
        self.score = score
        self.seen_columns: list[str] | None = None
        self.classes_ = np.asarray(classes)
        self.n_features_in_ = feature_count or len(FEATURE_COLUMNS)
        if include_feature_names:
            self.feature_names_in_ = np.asarray(
                feature_names or FEATURE_COLUMNS,
                dtype=object,
            )

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        self.seen_columns = list(features.columns)
        if list(self.classes_) == [1, 0]:
            probabilities = [self.score, 1.0 - self.score]
        else:
            probabilities = [1.0 - self.score, self.score]
        return np.tile(probabilities, (len(features), 1))


def test_compatible_model_artifact_passes_validation() -> None:
    app_main.validate_frozen_model(StubModel(0.25))


def test_model_without_optional_feature_names_metadata_passes_validation() -> None:
    app_main.validate_frozen_model(StubModel(0.25, include_feature_names=False))


def test_model_validation_rejects_wrong_estimator_type() -> None:
    with pytest.raises(ValueError, match="expected a fitted"):
        app_main.validate_frozen_model(object())


def test_model_validation_rejects_unfitted_random_forest() -> None:
    with pytest.raises(ValueError, match="not fitted"):
        app_main.validate_frozen_model(app_main.RandomForestClassifier())


def test_model_validation_rejects_missing_predict_proba() -> None:
    model = StubModel(0.25)
    model.predict_proba = None

    with pytest.raises(ValueError, match="does not support predict_proba"):
        app_main.validate_frozen_model(model)


def test_model_validation_rejects_wrong_feature_count() -> None:
    with pytest.raises(ValueError, match="expects 29 features"):
        app_main.validate_frozen_model(StubModel(0.25, feature_count=29))


def test_model_validation_rejects_wrong_feature_names_or_order() -> None:
    wrong_order = [*FEATURE_COLUMNS[1:], FEATURE_COLUMNS[0]]

    with pytest.raises(ValueError, match="feature names or ordering"):
        app_main.validate_frozen_model(StubModel(0.25, feature_names=wrong_order))


def test_model_validation_rejects_nonbinary_classes() -> None:
    with pytest.raises(ValueError, match="binary classes 0 and 1"):
        app_main.validate_frozen_model(StubModel(0.25, classes=(0, 2)))


@pytest.mark.parametrize(
    "score, expected_class",
    [
        (0.399999, 0),
        (0.40, 1),
        (0.400001, 1),
    ],
)
def test_prediction_uses_frozen_threshold(
    monkeypatch: pytest.MonkeyPatch,
    score: float,
    expected_class: int,
) -> None:
    model = StubModel(score)
    monkeypatch.setattr(app_main, "load_frozen_model", lambda _: model)

    results = app_main.score_batch_data(make_feature_frame())

    assert results["predicted_class"].tolist() == [expected_class, expected_class]
    assert model.seen_columns == FEATURE_COLUMNS


def test_prediction_preserves_exact_feature_order(monkeypatch: pytest.MonkeyPatch) -> None:
    model = StubModel(0.25)
    monkeypatch.setattr(app_main, "load_frozen_model", lambda _: model)
    data = make_feature_frame()
    data.insert(0, "transaction_id", ["txn-1", "txn-2"])
    data = data[["transaction_id", *reversed(FEATURE_COLUMNS)]]

    app_main.score_batch_data(data)

    assert model.seen_columns == FEATURE_COLUMNS


def test_prediction_selects_probability_by_fraud_class_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = StubModel(0.75, classes=(1, 0))
    monkeypatch.setattr(app_main, "load_frozen_model", lambda _: model)

    results = app_main.score_batch_data(make_feature_frame(rows=1))

    assert results.loc[0, "fraud_score"] == pytest.approx(0.75)
    assert results.loc[0, "predicted_class"] == 1


def test_single_prediction_result_is_scoped_to_selected_input_mode() -> None:
    result = {"input_mode": "synthetic_example", "fraud_probability": 0.25}

    assert app_main.single_prediction_matches_mode(result, "synthetic_example")
    assert not app_main.single_prediction_matches_mode(result, "manual")
    assert not app_main.single_prediction_matches_mode(None, "synthetic_example")


def test_different_same_size_files_have_different_content_identities() -> None:
    first_file = InMemoryUpload(b"alpha")
    second_file = InMemoryUpload(b"bravo")
    first_file.name = second_file.name = "transactions.csv"

    assert first_file.size == second_file.size
    assert app_main.calculate_uploaded_file_sha256(first_file) != (
        app_main.calculate_uploaded_file_sha256(second_file)
    )


def test_saved_metrics_are_extracted_from_existing_structure(tmp_path) -> None:
    metrics = {
        "evaluation_scope": "One-time untouched holdout evaluation",
        "model_name": "Random Forest",
        "threshold": 0.4,
        "metrics": {
            "precision": 0.9459459459,
            "recall": 0.7368421053,
            "f1_score": 0.8284023669,
            "pr_auc": 0.8095198448,
        },
        "confusion_matrix": {
            "true_positives": 70,
            "false_positives": 4,
            "true_negatives": 56647,
            "false_negatives": 25,
        },
    }
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(metrics), encoding="utf-8")

    values = app_main.extract_dashboard_values(
        app_main.load_evaluation_metrics(str(path))
    )

    assert values["transactions_evaluated"] == 56746
    assert values["fraud_alerts"] == 74
    assert values["precision"] == pytest.approx(metrics["metrics"]["precision"])
    assert values["true_positives"] == 70


def test_complete_saved_metrics_payload_is_valid() -> None:
    app_main.validate_saved_metrics(make_valid_metrics_payload())


@pytest.mark.parametrize(
    "mutate, expected_message",
    [
        (
            lambda payload: payload["metrics"].update(precision=1.01),
            "Metric 'precision' must be a finite value",
        ),
        (
            lambda payload: payload["metrics"].update(recall=float("nan")),
            "Metric 'recall' must be a finite value",
        ),
        (
            lambda payload: payload["confusion_matrix"].update(true_positives=-1),
            "Count 'true_positives' must be a nonnegative integer",
        ),
        (
            lambda payload: payload["confusion_matrix"].update(false_positives=4.0),
            "Count 'false_positives' must be a nonnegative integer",
        ),
        (
            lambda payload: payload["metrics"].pop("recall"),
            "missing a required performance key",
        ),
        (
            lambda payload: payload.update(n_samples=56745),
            "does not equal the total confusion-matrix count",
        ),
        (
            lambda payload: payload.update(model_name="XGBoost"),
            "different model",
        ),
        (
            lambda payload: payload.update(threshold=0.50),
            "does not match the frozen threshold",
        ),
        (
            lambda payload: payload.update(feature_columns=["Amount", *FEATURE_COLUMNS[:-1]]),
            "feature schema or ordering",
        ),
    ],
)
def test_saved_metrics_reject_invalid_values_and_metadata(
    mutate, expected_message: str
) -> None:
    payload = make_valid_metrics_payload()
    mutate(payload)

    with pytest.raises(ValueError, match=expected_message):
        app_main.validate_saved_metrics(payload)


def test_missing_or_malformed_metrics_are_handled_gracefully(tmp_path) -> None:
    missing_path = tmp_path / "missing.json"
    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text("{not valid json", encoding="utf-8")

    assert app_main.load_evaluation_metrics(str(missing_path)) is None
    assert app_main.load_evaluation_metrics(str(malformed_path)) is None


def test_score_histogram_is_bounded_to_probability_range() -> None:
    results = pd.DataFrame({"fraud_score": [0.01, 0.40, 0.99]})

    figure = app_main.build_score_distribution_figure(results)

    assert list(figure.layout.xaxis.range) == [0.0, 1.0]
    assert figure.data[0].xbins.start == 0.0
    assert figure.data[0].xbins.end == 1.0


def test_confusion_matrix_maps_counts_correctly() -> None:
    values = {
        "true_negatives": 56647,
        "false_positives": 4,
        "false_negatives": 25,
        "true_positives": 70,
    }

    figure = app_main.build_confusion_matrix_figure(values)

    assert [list(row) for row in figure.data[0].z] == [[56647, 4], [25, 70]]
