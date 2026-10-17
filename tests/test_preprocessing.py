"""Tests for preprocessing utilities."""

import numpy as np
import pandas as pd
import pytest

from customer_prediction.preprocessing import (
    MODEL_FEATURES,
    build_preprocessor,
    encode_target,
    prepare_features,
)


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create a small representative raw dataset."""

    return pd.DataFrame({
        "age": [30, 50],
        "job": ["student", "management"],
        "marital": ["single", "married"],
        "education": ["tertiary", "secondary"],
        "default": ["no", "no"],
        "balance": [1000, 2500],
        "housing": ["no", "yes"],
        "loan": ["no", "yes"],
        "contact": ["cellular", "telephone"],
        "day": [10, 20],
        "month": ["mar", "may"],
        "duration": [200, 400],
        "campaign": [1, 3],
        "pdays": [-1, 120],
        "previous": [0, 2],
        "poutcome": ["unknown", "success"],
        "y": ["yes", "no"],
    })


def test_prepare_features_removes_excluded_columns(
    sample_data: pd.DataFrame,
) -> None:
    features = prepare_features(sample_data)

    assert "y" not in features.columns
    assert "duration" not in features.columns
    assert tuple(features.columns) == MODEL_FEATURES


def test_prepare_features_handles_pdays(
    sample_data: pd.DataFrame,
) -> None:
    features = prepare_features(sample_data)

    assert features["pdays"].tolist() == [0, 120]
    assert features[
        "previously_contacted"
    ].tolist() == [0, 1]


def test_prepare_features_rejects_missing_columns(
    sample_data: pd.DataFrame,
) -> None:
    invalid_data = sample_data.drop(
        columns=["balance"]
    )

    with pytest.raises(
        ValueError,
        match="balance",
    ):
        prepare_features(invalid_data)


def test_encode_target() -> None:
    target = pd.Series([
        "no",
        "yes",
        "no",
    ])

    encoded = encode_target(target)

    assert encoded.tolist() == [0, 1, 0]
    assert str(encoded.dtype) == "int8"


def test_encode_target_rejects_unknown_labels() -> None:
    target = pd.Series([
        "no",
        "maybe",
    ])

    with pytest.raises(
        ValueError,
        match="yes.*no",
    ):
        encode_target(target)


def test_preprocessor_produces_finite_values(
    sample_data: pd.DataFrame,
) -> None:
    features = prepare_features(sample_data)
    preprocessor = build_preprocessor()

    transformed = preprocessor.fit_transform(
        features
    )

    assert transformed.shape[0] == len(sample_data)
    assert np.isfinite(transformed).all()