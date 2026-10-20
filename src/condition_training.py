"""Train and evaluate a leakage-safe baseline for shipment conditions."""

import json
import os
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils.config import CONDITION_CSV, CONDITION_MODEL_DIR, RANDOM_SEED

TARGET_COLUMN = "Class"
MODEL_FILENAME = "condition_model.joblib"
EVALUATION_FILENAME = "evaluation_results.json"
TEST_SIZE = 0.2


def _make_input_schema(features: pd.DataFrame) -> list[dict[str, Any]]:
    """Describe training inputs so the app can build safe, data-backed widgets."""
    schema: list[dict[str, Any]] = []
    for name in features.columns:
        column = features[name]
        if pd.api.types.is_numeric_dtype(column):
            schema.append(
                {
                    "name": str(name),
                    "kind": "numeric",
                    "minimum": float(column.min()),
                    "maximum": float(column.max()),
                    "default": float(column.median()),
                    "integer": pd.api.types.is_integer_dtype(column),
                }
            )
        else:
            options = sorted(column.dropna().astype(str).unique().tolist())
            schema.append(
                {
                    "name": str(name),
                    "kind": "categorical",
                    "options": options,
                    "default": options[0] if options else "",
                }
            )
    return schema


def build_condition_pipeline(
    features: pd.DataFrame,
) -> Pipeline:
    """Build preprocessing and logistic-regression steps fitted only on training data."""
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = [
        column for column in features.columns if column not in numeric_columns
    ]

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocessing", preprocessing),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def train_condition_model(
    csv_path: str = CONDITION_CSV,
    model_directory: str = CONDITION_MODEL_DIR,
) -> dict[str, Any]:
    """Train, evaluate, and save the tabular pipeline and genuine test metrics."""
    data = pd.read_csv(csv_path)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(
            f"Expected target column '{TARGET_COLUMN}' in {csv_path}; "
            f"found columns: {data.columns.tolist()}"
        )

    rows_before_deduplication = len(data)
    data = data.drop_duplicates().reset_index(drop=True)
    rows_after_deduplication = len(data)
    if data[TARGET_COLUMN].isna().any():
        raise ValueError(
            f"The target column '{TARGET_COLUMN}' contains missing values. "
            "Resolve these labels before training."
        )

    feature_columns = [column for column in data.columns if column != TARGET_COLUMN]
    if not feature_columns:
        raise ValueError("The dataset has no feature columns after removing the target.")

    features = data[feature_columns]
    labels = data[TARGET_COLUMN]
    class_counts = labels.value_counts()
    if len(class_counts) < 2 or class_counts.min() < 2:
        raise ValueError("A stratified split requires at least two examples per class.")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    pipeline = build_condition_pipeline(x_train)
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    classes = pipeline.named_steps["classifier"].classes_.tolist()

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_macro": float(
            precision_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "precision_weighted": float(
            precision_score(y_test, predictions, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_test, predictions, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_test, predictions, average="weighted", zero_division=0)
        ),
    }
    matrix = confusion_matrix(y_test, predictions, labels=classes)

    os.makedirs(model_directory, exist_ok=True)
    artifact = {
        "pipeline": pipeline,
        "target_column": TARGET_COLUMN,
        "feature_columns": feature_columns,
        "input_schema": _make_input_schema(x_train),
        "classes": [str(value) for value in classes],
    }
    model_path = os.path.join(model_directory, MODEL_FILENAME)
    joblib.dump(artifact, model_path)

    evaluation = {
        "model_name": "Class-balanced logistic regression",
        "target_column": TARGET_COLUMN,
        "feature_columns": feature_columns,
        "classes": [str(value) for value in classes],
        "risk_assessment": {
            "spoilage_risk_score_available": False,
            "risk_categories_available": False,
            "probability_calibration": "not performed",
            "reason": (
                "The source does not define the meaning of its distinct target labels "
                "('BAD', 'Bad', and 'Good'), so there is no validated mapping from "
                "model classes to a spoilage event. Model probabilities are also "
                "uncalibrated and are not displayed as risk."
            ),
            "feature_contributions_or_recommendations_available": False,
            "feature_explanation_reason": (
                "The fitted classifier and observational dataset do not establish "
                "causal factor contributions or validated handling recommendations."
            ),
            "measurement_units": "not documented in the source CSV",
        },
        "split": {
            "method": "stratified random holdout",
            "test_fraction": TEST_SIZE,
            "random_seed": RANDOM_SEED,
            "rows_before_exact_deduplication": rows_before_deduplication,
            "exact_duplicate_rows_removed_before_split": (
                rows_before_deduplication - rows_after_deduplication
            ),
            "rows_after_exact_deduplication": rows_after_deduplication,
            "train_rows": len(x_train),
            "test_rows": len(x_test),
        },
        "metrics": metrics,
        "confusion_matrix": {
            "labels": [str(value) for value in classes],
            "values": matrix.tolist(),
        },
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=classes,
            output_dict=True,
            zero_division=0,
        ),
    }
    evaluation_path = os.path.join(model_directory, EVALUATION_FILENAME)
    with open(evaluation_path, "w", encoding="utf-8") as evaluation_file:
        json.dump(evaluation, evaluation_file, indent=2, allow_nan=False)

    return {
        "model_path": model_path,
        "evaluation_path": evaluation_path,
        "evaluation": evaluation,
    }


if __name__ == "__main__":
    result = train_condition_model()
    print(json.dumps(result["evaluation"], indent=2))
    print(f"Saved model: {result['model_path']}")
    print(f"Saved evaluation: {result['evaluation_path']}")
