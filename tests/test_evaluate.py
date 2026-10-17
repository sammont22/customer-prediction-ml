from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from customer_prediction.evaluate import (
    calculate_classification_metrics,
    evaluate_model,
    load_model_artifact,
    predict_subscriptions,
)


class FixedProbabilityModel:
    """Return fixed probabilities for predictable tests."""

    def predict_proba(
        self,
        data: pd.DataFrame,
    ) -> np.ndarray:
        probabilities = np.array(
            [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.6, 0.4],
                [0.3, 0.7],
            ]
        )

        return probabilities[: len(data)]


@pytest.fixture
def evaluation_data() -> pd.DataFrame:
    """Create a small labeled evaluation dataset."""

    return pd.DataFrame(
        {
            "age": [25, 40, 55, 32],
            "job": [
                "student",
                "management",
                "retired",
                "technician",
            ],
            "marital": [
                "single",
                "married",
                "married",
                "single",
            ],
            "education": [
                "secondary",
                "tertiary",
                "primary",
                "secondary",
            ],
            "default": ["no", "no", "no", "yes"],
            "balance": [100, 1500, 3000, -50],
            "housing": ["no", "yes", "no", "yes"],
            "loan": ["no", "no", "yes", "no"],
            "contact": [
                "cellular",
                "telephone",
                "unknown",
                "cellular",
            ],
            "day": [5, 12, 20, 28],
            "month": ["jan", "may", "aug", "nov"],
            "duration": [100, 300, 250, 150],
            "campaign": [1, 2, 1, 3],
            "pdays": [-1, 100, 50, -1],
            "previous": [0, 2, 1, 0],
            "poutcome": [
                "unknown",
                "success",
                "failure",
                "unknown",
            ],
            "y": ["no", "yes", "yes", "no"],
        },
        index=[10, 11, 12, 13],
    )


@pytest.fixture
def model_artifact() -> dict[str, object]:
    """Create an artifact with deterministic predictions."""

    return {
        "pipeline": FixedProbabilityModel(),
        "decision_threshold": 0.5,
    }


def test_load_model_artifact(
    tmp_path: Path,
) -> None:
    """A valid serialized artifact can be loaded."""

    destination = tmp_path / "model.joblib"
    artifact = {
        "pipeline": None,
        "decision_threshold": 0.5,
    }

    joblib.dump(artifact, destination)
    loaded_artifact = load_model_artifact(destination)

    assert loaded_artifact == artifact


def test_load_model_artifact_requires_dictionary(
    tmp_path: Path,
) -> None:
    """Artifacts that are not dictionaries are rejected."""

    destination = tmp_path / "model.joblib"
    joblib.dump(["invalid"], destination)

    with pytest.raises(
        TypeError,
        match="must be a dictionary",
    ):
        load_model_artifact(destination)


def test_load_model_artifact_requires_keys(
    tmp_path: Path,
) -> None:
    """Artifacts missing required keys are rejected."""

    destination = tmp_path / "model.joblib"
    joblib.dump(
        {"pipeline": None},
        destination,
    )

    with pytest.raises(
        ValueError,
        match="missing keys",
    ):
        load_model_artifact(destination)


def test_load_model_artifact_validates_threshold(
    tmp_path: Path,
) -> None:
    """Decision thresholds outside zero and one are rejected."""

    destination = tmp_path / "model.joblib"
    joblib.dump(
        {
            "pipeline": None,
            "decision_threshold": 1.5,
        },
        destination,
    )

    with pytest.raises(
        ValueError,
        match="between zero and one",
    ):
        load_model_artifact(destination)


def test_predict_subscriptions(
    evaluation_data: pd.DataFrame,
    model_artifact: dict[str, object],
) -> None:
    """Predictions use the saved decision threshold."""

    results = predict_subscriptions(
        evaluation_data,
        model_artifact,
    )

    assert results.index.equals(evaluation_data.index)
    assert results["prediction"].tolist() == [
        0,
        1,
        0,
        1,
    ]
    assert results["prediction_label"].tolist() == [
        "no",
        "yes",
        "no",
        "yes",
    ]
    assert results[
        "subscription_probability"
    ].tolist() == pytest.approx(
        [0.1, 0.8, 0.4, 0.7]
    )


def test_calculate_classification_metrics() -> None:
    """Classification metrics and error counts are correct."""

    target = pd.Series([0, 1, 1, 0])
    predictions = pd.Series([0, 1, 0, 1])
    probabilities = pd.Series([0.1, 0.8, 0.4, 0.7])

    metrics = calculate_classification_metrics(
        target,
        predictions,
        probabilities,
    )

    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1_score"] == pytest.approx(0.5)
    assert metrics["roc_auc"] == pytest.approx(0.75)
    assert metrics["average_precision"] == pytest.approx(
        5 / 6
    )
    assert metrics["true_negatives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["true_positives"] == 1


def test_evaluate_model(
    evaluation_data: pd.DataFrame,
    model_artifact: dict[str, object],
) -> None:
    """Evaluation returns metrics and prediction details."""

    metrics, results = evaluate_model(
        evaluation_data,
        model_artifact,
    )

    assert metrics["accuracy"] == pytest.approx(0.5)
    assert len(results) == len(evaluation_data)
    assert results.index.equals(evaluation_data.index)


def test_evaluate_model_requires_target(
    evaluation_data: pd.DataFrame,
    model_artifact: dict[str, object],
) -> None:
    """Evaluation requires the target column."""

    data_without_target = evaluation_data.drop(
        columns="y"
    )

    with pytest.raises(
        ValueError,
        match="target column",
    ):
        evaluate_model(
            data_without_target,
            model_artifact,
        )