import joblib
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from customer_prediction.preprocessing import (
    MODEL_FEATURES,
    prepare_features,
)
from customer_prediction.train import (
    DECISION_THRESHOLD,
    build_model_pipeline,
    create_model_artifact,
    save_model_artifact,
    train_model,
)


@pytest.fixture
def training_data() -> pd.DataFrame:
    """Create a small labeled dataset for training tests."""

    return pd.DataFrame(
        {
            "age": [
                25,
                34,
                42,
                51,
                29,
                47,
                38,
                56,
                31,
                45,
                36,
                60,
            ],
            "job": [
                "student",
                "technician",
                "management",
            ]
            * 4,
            "marital": [
                "single",
                "married",
            ]
            * 6,
            "education": [
                "secondary",
                "tertiary",
                "primary",
            ]
            * 4,
            "default": ["no"] * 11 + ["yes"],
            "balance": [
                100,
                850,
                1200,
                -50,
                430,
                2100,
                700,
                3300,
                90,
                1500,
                620,
                2800,
            ],
            "housing": [
                "no",
                "yes",
            ]
            * 6,
            "loan": [
                "no",
                "no",
                "yes",
            ]
            * 4,
            "contact": [
                "cellular",
                "telephone",
                "unknown",
            ]
            * 4,
            "day": list(range(1, 13)),
            "month": [
                "jan",
                "feb",
                "mar",
                "apr",
                "may",
                "jun",
                "jul",
                "aug",
                "sep",
                "oct",
                "nov",
                "dec",
            ],
            "duration": [
                100,
                210,
                85,
                450,
                175,
                320,
                140,
                510,
                95,
                275,
                190,
                600,
            ],
            "campaign": [
                1,
                2,
                1,
                3,
                2,
                1,
                4,
                2,
                1,
                3,
                2,
                1,
            ],
            "pdays": [
                -1,
                120,
            ]
            * 6,
            "previous": [
                0,
                2,
            ]
            * 6,
            "poutcome": [
                "unknown",
                "success",
                "failure",
            ]
            * 4,
            "y": [
                "no",
                "yes",
            ]
            * 6,
        }
    )


def test_build_model_pipeline() -> None:
    """The final pipeline contains preprocessing and classifier."""

    model = build_model_pipeline()

    assert isinstance(model, Pipeline)
    assert list(model.named_steps) == [
        "preprocessor",
        "classifier",
    ]


def test_train_model_returns_fitted_pipeline(
    training_data: pd.DataFrame,
) -> None:
    """The training function returns a usable fitted model."""

    model = train_model(training_data)
    features = prepare_features(training_data)

    probabilities = model.predict_proba(features)[:, 1]

    assert probabilities.shape == (len(training_data),)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_train_model_requires_target(
    training_data: pd.DataFrame,
) -> None:
    """Training without the target column is rejected."""

    data_without_target = training_data.drop(columns="y")

    with pytest.raises(
        ValueError,
        match="target column",
    ):
        train_model(data_without_target)


def test_create_model_artifact_contains_metadata(
    training_data: pd.DataFrame,
) -> None:
    """The model artifact contains the required metadata."""

    model = train_model(training_data)
    artifact = create_model_artifact(model)

    assert artifact["pipeline"] is model
    assert artifact["decision_threshold"] == DECISION_THRESHOLD
    assert artifact["feature_columns"] == list(MODEL_FEATURES)
    assert artifact["excluded_features"] == ["duration"]
    assert "sklearn_version" in artifact


def test_save_model_artifact(
    training_data: pd.DataFrame,
    tmp_path,
) -> None:
    """The complete artifact can be saved and loaded."""

    model = train_model(training_data)
    destination = tmp_path / "model.joblib"

    returned_path = save_model_artifact(
        model,
        destination,
    )
    loaded_artifact = joblib.load(destination)

    assert returned_path == destination
    assert destination.exists()
    assert loaded_artifact["decision_threshold"] == DECISION_THRESHOLD
    assert isinstance(loaded_artifact["pipeline"], Pipeline)