"""Customer subscription prediction package."""

from customer_prediction.evaluate import (
    calculate_classification_metrics,
    evaluate_model,
    load_model_artifact,
    predict_subscriptions,
)
from customer_prediction.preprocessing import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    TARGET_MAPPING,
    build_preprocessor,
    encode_target,
    prepare_features,
)

__version__ = "1.0.0"

__all__ = [
    "CATEGORICAL_FEATURES",
    "MODEL_FEATURES",
    "NUMERIC_FEATURES",
    "TARGET_COLUMN",
    "TARGET_MAPPING",
    "build_preprocessor",
    "calculate_classification_metrics",
    "encode_target",
    "evaluate_model",
    "load_model_artifact",
    "predict_subscriptions",
    "prepare_features",
]