"""Train and evaluate a lightweight food-image freshness baseline."""

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from utils.config import IMAGE_DATA_DIR, IMAGE_MODEL_DIR, RANDOM_SEED

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
FEATURE_VERSION = "rgb-12x12-plus-16-bin-rgb-histogram-v1"
FEATURE_COUNT = (12 * 12 * 3) + (16 * 3)
MODEL_FILENAME = "image_model.joblib"
EVALUATION_FILENAME = "evaluation_results.json"
TEST_FRACTION_TARGET = 0.2


def extract_image_features(image: Image.Image) -> np.ndarray:
    """Create a compact RGB thumbnail and color histogram for one image."""
    thumbnail = image.convert("RGB").resize((12, 12), Image.Resampling.BILINEAR)
    pixels = np.asarray(thumbnail, dtype=np.uint8)
    thumbnail_features = pixels.astype(np.float32).reshape(-1) / 255.0
    histograms = [
        np.histogram(pixels[:, :, channel], bins=16, range=(0, 256))[0].astype(np.float32)
        / pixels.shape[0]
        / pixels.shape[1]
        for channel in range(3)
    ]
    return np.concatenate([thumbnail_features, *histograms]).reshape(1, FEATURE_COUNT)


def _collect_image_records(image_root: str) -> list[dict[str, str]]:
    """Read labels and group names from the real category/food/label folder layout."""
    root = Path(image_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Image dataset folder does not exist: {root}")

    records: list[dict[str, str]] = []
    unsupported_layouts: list[str] = []
    for image_path in sorted(root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        parts = image_path.relative_to(root).parts
        if len(parts) != 4:
            unsupported_layouts.append(str(image_path))
            continue
        category, food_class, label, _ = parts
        records.append(
            {
                "path": str(image_path),
                "label": label,
                "group": f"{category}/{food_class}",
            }
        )

    if unsupported_layouts:
        preview = ", ".join(unsupported_layouts[:3])
        raise ValueError(
            f"Found {len(unsupported_layouts)} image files outside the expected "
            f"category/food/label/file layout (for example: {preview})."
        )
    if not records:
        raise ValueError(f"No JPG, JPEG, or PNG images were found under {root}.")
    return records


def _make_grouped_split(
    labels: np.ndarray,
    groups: np.ndarray,
    test_fraction_target: float = TEST_FRACTION_TARGET,
) -> tuple[np.ndarray, np.ndarray]:
    """Select a stratified group fold with every label represented in both sets."""
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    placeholder = np.zeros((len(labels), 1), dtype=np.uint8)
    all_classes = set(labels.tolist())
    candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
    for train_indices, test_indices in splitter.split(placeholder, labels, groups):
        if set(labels[train_indices].tolist()) != all_classes:
            continue
        if set(labels[test_indices].tolist()) != all_classes:
            continue
        test_fraction = len(test_indices) / len(labels)
        candidates.append(
            (abs(test_fraction - test_fraction_target), train_indices, test_indices)
        )

    if not candidates:
        raise ValueError(
            "Could not create a five-fold stratified split by food type with every "
            "label present in both train and test. Each image label needs support "
            "across multiple food-type folders."
        )
    _, train_indices, test_indices = min(candidates, key=lambda candidate: candidate[0])
    return train_indices, test_indices


def train_image_model(
    image_root: str = IMAGE_DATA_DIR,
    model_directory: str = IMAGE_MODEL_DIR,
) -> dict[str, Any]:
    """Train the image baseline on real images with food-type-grouped evaluation."""
    records = _collect_image_records(image_root)
    labels = np.asarray([record["label"] for record in records], dtype=str)
    groups = np.asarray([record["group"] for record in records], dtype=str)
    train_indices, test_indices = _make_grouped_split(labels, groups)

    features = np.empty((len(records), FEATURE_COUNT), dtype=np.float32)
    for index, record in enumerate(records):
        try:
            with Image.open(record["path"]) as image:
                features[index] = extract_image_features(image)
        except (OSError, ValueError) as error:
            raise ValueError(
                f"Could not read image {record['path']}: {error}"
            ) from error

    classifier = SGDClassifier(
        loss="log_loss",
        class_weight="balanced",
        max_iter=300,
        tol=1e-3,
        random_state=RANDOM_SEED,
        average=True,
    )
    classifier.fit(features[train_indices], labels[train_indices])
    predictions = classifier.predict(features[test_indices])
    classes = classifier.classes_.tolist()

    metrics = {
        "accuracy": float(accuracy_score(labels[test_indices], predictions)),
        "precision_macro": float(
            precision_score(
                labels[test_indices], predictions, average="macro", zero_division=0
            )
        ),
        "recall_macro": float(
            recall_score(
                labels[test_indices], predictions, average="macro", zero_division=0
            )
        ),
        "f1_macro": float(
            f1_score(labels[test_indices], predictions, average="macro", zero_division=0)
        ),
        "precision_weighted": float(
            precision_score(
                labels[test_indices], predictions, average="weighted", zero_division=0
            )
        ),
        "recall_weighted": float(
            recall_score(
                labels[test_indices], predictions, average="weighted", zero_division=0
            )
        ),
        "f1_weighted": float(
            f1_score(
                labels[test_indices], predictions, average="weighted", zero_division=0
            )
        ),
    }
    matrix = confusion_matrix(labels[test_indices], predictions, labels=classes)
    train_groups = sorted(set(groups[train_indices].tolist()))
    test_groups = sorted(set(groups[test_indices].tolist()))

    os.makedirs(model_directory, exist_ok=True)
    model_path = os.path.join(model_directory, MODEL_FILENAME)
    joblib.dump(
        {
            "classifier": classifier,
            "classes": [str(value) for value in classes],
            "feature_version": FEATURE_VERSION,
        },
        model_path,
    )

    evaluation = {
        "model_name": "Balanced SGD image-feature classifier",
        "target_column": "label folder name",
        "feature_description": (
            "12x12 RGB thumbnail values plus normalized per-channel RGB histograms"
        ),
        "classes": [str(value) for value in classes],
        "split": {
            "method": "five-fold stratified group holdout by category/food class",
            "random_seed": RANDOM_SEED,
            "target_test_fraction": TEST_FRACTION_TARGET,
            "actual_test_fraction": len(test_indices) / len(records),
            "train_images": len(train_indices),
            "test_images": len(test_indices),
            "train_food_groups": train_groups,
            "test_food_groups": test_groups,
        },
        "metrics": metrics,
        "confusion_matrix": {
            "labels": [str(value) for value in classes],
            "values": matrix.tolist(),
        },
        "classification_report": classification_report(
            labels[test_indices],
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
    result = train_image_model()
    print(json.dumps(result["evaluation"], indent=2))
    print(f"Saved model: {result['model_path']}")
    print(f"Saved evaluation: {result['evaluation_path']}")
