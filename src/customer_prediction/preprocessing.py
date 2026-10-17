"""Feature preparation and preprocessing utilities."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET_COLUMN = "y"

TARGET_MAPPING = {
    "no": 0,
    "yes": 1,
}

NUMERIC_FEATURES = (
    "age",
    "balance",
    "day",
    "campaign",
    "pdays",
    "previous",
)

CATEGORICAL_FEATURES = (
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
)

BINARY_FEATURES = (
    "previously_contacted",
)

RAW_REQUIRED_FEATURES = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)

MODEL_FEATURES = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
    + BINARY_FEATURES
)


def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create model-ready input features from the raw dataset."""

    missing_columns = sorted(
        set(RAW_REQUIRED_FEATURES)
        - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required feature columns: "
            + ", ".join(missing_columns)
        )

    features = data.drop(
        columns=[TARGET_COLUMN, "duration"],
        errors="ignore",
    ).copy()

    features["previously_contacted"] = (
        features["pdays"] != -1
    ).astype("int8")

    features["pdays"] = features["pdays"].replace(
        -1,
        0,
    )

    return features.loc[:, list(MODEL_FEATURES)]


def encode_target(target: pd.Series) -> pd.Series:
    """Encode the target labels as zero and one."""

    encoded_target = target.map(TARGET_MAPPING)

    if encoded_target.isna().any():
        raise ValueError(
            "Target values must contain only 'yes' or 'no'."
        )

    return encoded_target.astype("int8")


def build_preprocessor() -> ColumnTransformer:
    """Build the preprocessing pipeline used by the final model."""

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                list(NUMERIC_FEATURES),
            ),
            (
                "categorical",
                categorical_transformer,
                list(CATEGORICAL_FEATURES),
            ),
            (
                "binary",
                "passthrough",
                list(BINARY_FEATURES),
            ),
        ]
    )