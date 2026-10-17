"""Prediction and evaluation utilities."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from customer_prediction.preprocessing import (
    TARGET_COLUMN,
    encode_target,
    prepare_features,
)

REQUIRED_ARTIFACT_KEYS = {
    "pipeline",
    "decision_threshold",
}


def load_model_artifact(
    model_path: str | Path,
) -> dict[str, Any]:
    """Load and validate a saved model artifact."""

    artifact = joblib.load(model_path)

    if not isinstance(artifact, dict):
        raise TypeError(
            "The model artifact must be a dictionary."
        )

    missing_keys = sorted(
        REQUIRED_ARTIFACT_KEYS
        - set(artifact)
    )

    if missing_keys:
        raise ValueError(
            "Model artifact is missing keys: "
            + ", ".join(missing_keys)
        )

    threshold = float(
        artifact["decision_threshold"]
    )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "Decision threshold must be "
            "between zero and one."
        )

    return artifact


def predict_subscriptions(
    data: pd.DataFrame,
    artifact: Mapping[str, Any],
) -> pd.DataFrame:
    """Generate probabilities and subscription predictions."""

    features = prepare_features(data)

    model = artifact["pipeline"]

    threshold = float(
        artifact["decision_threshold"]
    )

    probabilities = model.predict_proba(
        features
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype("int8")

    results = pd.DataFrame(
        {
            "subscription_probability": probabilities,
            "prediction": predictions,
        },
        index=data.index,
    )

    results["prediction_label"] = (
        results["prediction"].map({
            0: "no",
            1: "yes",
        })
    )

    return results


def calculate_classification_metrics(
    target: pd.Series,
    predictions: pd.Series,
    probabilities: pd.Series,
) -> pd.Series:
    """Calculate classification metrics and error counts."""

    true_negative, false_positive, false_negative, true_positive = (
        confusion_matrix(
            target,
            predictions,
            labels=[0, 1],
        ).ravel()
    )

    return pd.Series({
        "accuracy": accuracy_score(
            target,
            predictions,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            target,
            predictions,
        ),
        "precision": precision_score(
            target,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            target,
            predictions,
            zero_division=0,
        ),
        "f1_score": f1_score(
            target,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            target,
            probabilities,
        ),
        "average_precision": average_precision_score(
            target,
            probabilities,
        ),
        "predicted_positives": int(
            predictions.sum()
        ),
        "true_negatives": int(true_negative),
        "false_positives": int(false_positive),
        "false_negatives": int(false_negative),
        "true_positives": int(true_positive),
    })


def evaluate_model(
    data: pd.DataFrame,
    artifact: Mapping[str, Any],
) -> tuple[pd.Series, pd.DataFrame]:
    """Evaluate an artifact using a labeled dataset."""

    if TARGET_COLUMN not in data.columns:
        raise ValueError(
            f"Evaluation data must contain "
            f"the target column '{TARGET_COLUMN}'."
        )

    target = encode_target(
        data[TARGET_COLUMN]
    )

    prediction_results = predict_subscriptions(
        data,
        artifact,
    )

    metrics = calculate_classification_metrics(
        target=target,
        predictions=prediction_results["prediction"],
        probabilities=prediction_results[
            "subscription_probability"
        ],
    )

    return metrics, prediction_results