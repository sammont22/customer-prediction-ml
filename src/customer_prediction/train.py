"""Training utilities and command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from customer_prediction.preprocessing import (
    MODEL_FEATURES,
    TARGET_COLUMN,
    TARGET_MAPPING,
    build_preprocessor,
    encode_target,
    prepare_features,
)

DECISION_THRESHOLD = 0.645058

MODEL_PARAMETERS = {
    "learning_rate": 0.05,
    "max_iter": 300,
    "max_leaf_nodes": 63,
    "min_samples_leaf": 10,
    "l2_regularization": 10.0,
    "class_weight": "balanced",
    "random_state": 42,
}


def build_model_pipeline() -> Pipeline:
    """Build the final preprocessing and classification pipeline."""

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    **MODEL_PARAMETERS,
                ),
            ),
        ]
    )


def train_model(data: pd.DataFrame) -> Pipeline:
    """Train the final model using a labeled raw dataset."""

    if TARGET_COLUMN not in data.columns:
        raise ValueError(
            f"Training data must contain "
            f"the target column '{TARGET_COLUMN}'."
        )

    features = prepare_features(data)
    target = encode_target(data[TARGET_COLUMN])

    model = build_model_pipeline()
    model.fit(features, target)

    return model


def create_model_artifact(
    model: Pipeline,
) -> dict[str, object]:
    """Package the trained pipeline with prediction metadata."""

    return {
        "pipeline": model,
        "decision_threshold": DECISION_THRESHOLD,
        "feature_columns": list(MODEL_FEATURES),
        "target_mapping": dict(TARGET_MAPPING),
        "excluded_features": ["duration"],
        "model_parameters": dict(MODEL_PARAMETERS),
        "sklearn_version": sklearn.__version__,
    }


def save_model_artifact(
    model: Pipeline,
    output_path: str | Path,
) -> Path:
    """Save the trained model and its metadata."""

    destination = Path(output_path)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact = create_model_artifact(model)

    joblib.dump(
        artifact,
        destination,
    )

    return destination


def train_and_save(
    data_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Load the raw dataset, train the model, and save it."""

    data = pd.read_csv(
        data_path,
        sep=";",
    )

    model = train_model(data)

    return save_model_artifact(
        model,
        output_path,
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Train the customer subscription "
            "prediction model."
        )
    )

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to bank-full.csv.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "models/final_model.joblib"
        ),
        help="Destination for the trained model artifact.",
    )

    return parser.parse_args()


def main() -> None:
    """Run model training from the command line."""

    arguments = parse_arguments()

    saved_path = train_and_save(
        data_path=arguments.data,
        output_path=arguments.output,
    )

    print(
        f"Model artifact saved to {saved_path}"
    )


if __name__ == "__main__":
    main()