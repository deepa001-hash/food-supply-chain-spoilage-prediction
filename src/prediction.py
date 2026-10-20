"""Load saved model artifacts and prepare inputs for predictions."""

import os
import math
from typing import Any

import joblib
import pandas as pd
from PIL import Image

from src.image_training import FEATURE_VERSION, extract_image_features
from utils.config import CONDITION_MODEL_DIR, IMAGE_MODEL_DIR


def load_condition_model(model_directory: str = CONDITION_MODEL_DIR) -> dict[str, Any]:
    """Load the condition preprocessing/model pipeline saved by training."""
    model_path = os.path.join(model_directory, "condition_model.joblib")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"No trained condition model found at {model_path}")
    artifact = joblib.load(model_path)
    required_keys = {"pipeline", "feature_columns", "input_schema", "target_column"}
    if not isinstance(artifact, dict) or not required_keys.issubset(artifact):
        raise ValueError(f"The condition model artifact at {model_path} is incomplete.")
    return artifact


def predict_condition(
    inputs: dict[str, Any],
    model_artifact: dict[str, Any] | None = None,
) -> str:
    """Predict the dataset's original class label after validating every input."""
    artifact = model_artifact or load_condition_model()
    feature_columns = artifact["feature_columns"]
    schema_by_name = {field["name"]: field for field in artifact["input_schema"]}
    missing = [name for name in feature_columns if name not in inputs]
    if missing:
        raise ValueError(f"Missing required input(s): {', '.join(missing)}.")

    validated: dict[str, Any] = {}
    for name in feature_columns:
        field = schema_by_name.get(name)
        if field is None:
            raise ValueError(f"The saved model has no input validation schema for '{name}'.")
        value = inputs[name]
        if field["kind"] == "categorical":
            allowed_values = field["options"]
            if value not in allowed_values:
                raise ValueError(
                    f"'{name}' must be one of the dataset values: "
                    f"{', '.join(str(option) for option in allowed_values)}."
                )
            validated[name] = value
            continue

        if isinstance(value, bool):
            raise ValueError(f"'{name}' must be a numeric value.")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"'{name}' must be a numeric value.") from error
        if not math.isfinite(numeric_value):
            raise ValueError(f"'{name}' must be a finite numeric value.")
        if not field["minimum"] <= numeric_value <= field["maximum"]:
            raise ValueError(
                f"'{name}' must be between {field['minimum']:g} and "
                f"{field['maximum']:g}, matching the training-data range."
            )
        if field["integer"] and not numeric_value.is_integer():
            raise ValueError(f"'{name}' must be a whole number.")
        validated[name] = int(numeric_value) if field["integer"] else numeric_value

    input_frame = pd.DataFrame([{name: validated[name] for name in feature_columns}])
    pipeline = artifact["pipeline"]
    return str(pipeline.predict(input_frame)[0])


def load_image_model(model_directory: str = IMAGE_MODEL_DIR) -> dict[str, Any]:
    """Load the saved image feature pipeline and classifier."""
    model_path = os.path.join(model_directory, "image_model.joblib")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"No trained image model found at {model_path}")
    artifact = joblib.load(model_path)
    required_keys = {"classifier", "classes", "feature_version"}
    if not isinstance(artifact, dict) or not required_keys.issubset(artifact):
        raise ValueError(f"The image model artifact at {model_path} is incomplete.")
    if artifact["feature_version"] != FEATURE_VERSION:
        raise ValueError(
            f"The image model at {model_path} uses unsupported feature version "
            f"{artifact['feature_version']!r}; expected {FEATURE_VERSION!r}."
        )
    classifier = artifact["classifier"]
    if not callable(getattr(classifier, "predict", None)) or not callable(
        getattr(classifier, "predict_proba", None)
    ):
        raise ValueError(
            f"The image model at {model_path} must support prediction and class probabilities."
        )
    classifier_classes = [str(label) for label in getattr(classifier, "classes_", [])]
    saved_classes = [str(label) for label in artifact["classes"]]
    if not classifier_classes or classifier_classes != saved_classes:
        raise ValueError(
            f"The image model class metadata at {model_path} does not match its classifier."
        )
    return artifact


def predict_image(
    image: Image.Image,
    model_artifact: dict[str, Any] | None = None,
) -> tuple[str, dict[str, float]]:
    """Classify an uploaded image using the saved baseline and return probabilities."""
    artifact = model_artifact or load_image_model()
    features = extract_image_features(image)
    classifier = artifact["classifier"]
    prediction = str(classifier.predict(features)[0])
    probabilities = classifier.predict_proba(features)[0]
    return prediction, {
        str(class_name): float(probability)
        for class_name, probability in zip(classifier.classes_, probabilities)
    }
