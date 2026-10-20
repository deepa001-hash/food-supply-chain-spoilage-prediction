"""Shared data loading and page rendering for the Streamlit application."""

import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.prediction import (
    load_condition_model,
    load_image_model,
    predict_condition,
    predict_image,
)
from utils.config import CONDITION_CSV, CONDITION_MODEL_DIR, IMAGE_DATA_DIR, IMAGE_MODEL_DIR

IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
MODEL_EXTENSIONS = {".h5", ".joblib", ".keras", ".model", ".onnx", ".pkl", ".pickle", ".pt", ".pth", ".sav"}
EVALUATION_FILENAME = "evaluation_results.json"
DISPLAY_METRICS = {
    "accuracy",
    "precision",
    "precision_macro",
    "precision_weighted",
    "recall",
    "recall_macro",
    "recall_weighted",
    "f1",
    "f1_score",
    "f1_macro",
    "f1_weighted",
    "roc_auc",
    "auc",
}


def apply_page_style() -> None:
    """Apply a light-green accent theme while keeping the page background white."""
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #ffffff;
            color: #1f2933;
        }
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
            color: #174c2b !important;
        }
        .stApp p, .stApp li, .stApp label,
        .stApp [data-testid="stCaptionContainer"],
        .stApp [data-testid="stWidgetLabel"] {
            color: #26352c;
        }
        [data-testid="stSidebar"] {
            background-color: #f1f8f2;
            color: #1f2933;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: #174c2b !important;
        }
        [data-testid="stSidebar"] h1 {
            font-size: 1.35rem;
        }
        [data-testid="stSidebarNav"] a {
            background: #ffffff;
            border: 1px solid #dcebdd;
            border-radius: 0.6rem;
            margin: 0.2rem 0;
            color: #20352a !important;
        }
        [data-testid="stSidebarNav"] a p,
        [data-testid="stSidebarNav"] a span {
            color: #20352a !important;
        }
        [data-testid="stSidebarNav"] a[aria-current="page"] {
            background: #e1f1e3;
            border-color: #88bd91;
            color: #174c2b !important;
        }
        [data-testid="stSidebarNav"] a[aria-current="page"] p,
        [data-testid="stSidebarNav"] a[aria-current="page"] span {
            color: #174c2b !important;
        }
        div[data-testid="stMetric"] {
            background: #f3f8f3;
            border: 1px solid #d7e7d9;
            border-radius: 0.75rem;
            padding: 0.8rem;
        }
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricLabel"],
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] p {
            color: #34483a !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"],
        div[data-testid="stMetric"] [data-testid="stMetricValue"] div {
            color: #173d25 !important;
        }
        .stButton > button, .stDownloadButton > button {
            background-color: #e4f3e6;
            border: 1px solid #9bcaa1;
            color: #245b32;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_condition_data(csv_path: str) -> tuple[pd.DataFrame | None, str | None]:
    """Load the condition CSV, returning a readable error when it is unavailable."""
    try:
        return pd.read_csv(csv_path), None
    except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        return None, str(error)


@st.cache_data(show_spinner=False)
def scan_image_folders(image_root: str) -> tuple[pd.DataFrame, list[str]]:
    """Count supported image files in each folder that directly contains images."""
    rows: list[dict[str, Any]] = []
    errors: list[str] = []

    if not os.path.isdir(image_root):
        return pd.DataFrame(columns=["Folder", "Food category", "Food class", "Label", "Images"]), errors

    def record_walk_error(error: OSError) -> None:
        errors.append(str(error))

    for current_root, _, filenames in os.walk(image_root, onerror=record_walk_error):
        image_count = sum(
            Path(filename).suffix.lower() in IMAGE_EXTENSIONS for filename in filenames
        )
        if not image_count:
            continue

        relative_parts = Path(current_root).relative_to(image_root).parts
        rows.append(
            {
                "Folder": " / ".join(relative_parts) or ".",
                "Food category": relative_parts[0] if relative_parts else "(root)",
                "Food class": relative_parts[1] if len(relative_parts) > 1 else "(not specified)",
                "Label": relative_parts[2] if len(relative_parts) > 2 else "(not specified)",
                "Images": image_count,
            }
        )

    folder_counts = pd.DataFrame(
        rows,
        columns=["Folder", "Food category", "Food class", "Label", "Images"],
    ).sort_values("Folder", ignore_index=True)
    return folder_counts, errors


def find_model_artifacts(model_directory: str) -> list[str]:
    """List likely model files without loading or executing them."""
    if not os.path.isdir(model_directory):
        return []

    matches: list[str] = []
    for current_root, _, filenames in os.walk(model_directory):
        matches.extend(
            os.path.join(current_root, filename)
            for filename in filenames
            if Path(filename).suffix.lower() in MODEL_EXTENSIONS
        )
    return sorted(matches)


def load_evaluation_document(
    model_directory: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Load a model's saved evaluation report, if it exists."""
    results_path = os.path.join(model_directory, EVALUATION_FILENAME)
    if not os.path.isfile(results_path):
        return None, None
    try:
        modified_time = os.path.getmtime(results_path)
    except OSError as error:
        return None, f"Could not inspect {results_path}: {error}"
    return _read_evaluation_document(results_path, modified_time)


@st.cache_data(show_spinner=False)
def _read_evaluation_document(
    results_path: str,
    modified_time: float,
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        if os.path.getmtime(results_path) != modified_time:
            return None, f"The evaluation report changed while it was being read: {results_path}"
        with open(results_path, encoding="utf-8") as results_file:
            results = json.load(results_file)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return None, f"Could not read {results_path}: {error}"

    if not isinstance(results, dict):
        return None, f"{results_path} must contain a JSON object of evaluation results."
    return results, None


def load_evaluation_results(model_directory: str) -> tuple[dict[str, float] | None, str | None]:
    """Read explicitly named evaluation output and keep recognized numeric metrics only."""
    results, error = load_evaluation_document(model_directory)
    if error or results is None:
        return None, error

    source = results.get("metrics", results)
    if not isinstance(source, dict):
        results_path = os.path.join(model_directory, EVALUATION_FILENAME)
        return None, f"The 'metrics' value in {results_path} must be a JSON object."

    metrics = {
        str(name): float(value)
        for name, value in source.items()
        if str(name).lower() in DISPLAY_METRICS
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    }
    if not metrics:
        return None, f"No recognized numeric evaluation metrics were found in {results_path}."
    return metrics, None


def _show_data_error(error: str | None) -> None:
    if error:
        st.error(f"The condition dataset could not be read: {error}")
    else:
        st.warning(
            "Condition dataset not found. Add Dataset.csv to data/condition_data/ "
            "to enable dataset exploration."
        )


def _get_data() -> pd.DataFrame | None:
    data, error = load_condition_data(CONDITION_CSV)
    if error:
        _show_data_error(error)
    elif data is None:
        _show_data_error(None)
    return data


def _get_image_counts() -> tuple[pd.DataFrame, list[str]]:
    return scan_image_folders(IMAGE_DATA_DIR)


def _model_status(model_directory: str) -> tuple[str, list[str]]:
    artifacts = find_model_artifacts(model_directory)
    if artifacts:
        return "Trained model artifact found.", artifacts
    return "Not trained yet — no model artifact found.", artifacts


def _show_image_scan_warnings(errors: list[str]) -> None:
    for error in errors:
        st.warning(f"Some image folders could not be scanned: {error}")


def render_home() -> None:
    st.title("Food Supply Chain Spoilage Prediction")
    st.write(
        "Explore the condition dataset and food-image collection. Predictions and "
        "performance scores are shown only after a model has been trained and evaluated."
    )

    data, data_error = load_condition_data(CONDITION_CSV)
    if data_error:
        _show_data_error(data_error)
    elif data is None:
        _show_data_error(None)

    image_counts, scan_errors = _get_image_counts()
    _show_image_scan_warnings(scan_errors)
    condition_rows = len(data) if data is not None else 0
    image_total = int(image_counts["Images"].sum()) if not image_counts.empty else 0
    condition_columns = len(data.columns) if data is not None else 0

    first, second, third = st.columns(3)
    first.metric("Condition records", f"{condition_rows:,}")
    second.metric("Condition fields", condition_columns)
    image_data_available = os.path.isdir(IMAGE_DATA_DIR)
    third.metric(
        "Food images",
        f"{image_total:,}" if image_data_available else "Not bundled",
    )
    if not image_data_available:
        st.caption(
            "The full image-training dataset is not included in this deployment. "
            "Image uploads can still be classified by the saved model."
        )

    st.subheader("Model status")
    image_status, _ = _model_status(IMAGE_MODEL_DIR)
    condition_status, _ = _model_status(CONDITION_MODEL_DIR)
    left, right = st.columns(2)
    with left:
        st.markdown("**Food image classification**")
        st.info(image_status)
    with right:
        st.markdown("**Shipment condition prediction**")
        st.info(condition_status)

    st.caption(
        "Prediction outputs use the dataset's original labels. They are model estimates, "
        "not a substitute for food-safety inspection or shipment-quality controls."
    )


def render_food_image_analysis() -> None:
    st.title("Food Image Analysis")
    st.write("Upload a food image to preview it and classify its folder label.")
    uploaded_image = st.file_uploader(
        "Choose a food image",
        type=["jpg", "jpeg", "png"],
        help="Supported file types: JPG, JPEG, and PNG.",
    )

    image_for_prediction: Image.Image | None = None
    if uploaded_image is not None:
        image_bytes = uploaded_image.getvalue()
        try:
            with Image.open(BytesIO(image_bytes)) as checked_image:
                checked_image.verify()
            with Image.open(BytesIO(image_bytes)) as preview:
                st.image(preview, caption=uploaded_image.name, width="stretch")
                st.caption(
                    f"Image format: {preview.format or 'Unknown'} · "
                    f"Dimensions: {preview.width} × {preview.height} pixels"
                )
                image_for_prediction = preview.copy()
        except (UnidentifiedImageError, OSError, ValueError) as error:
            st.error(f"This file could not be opened as an image: {error}")

    _, artifacts = _model_status(IMAGE_MODEL_DIR)
    if artifacts:
        image_report, report_error = load_evaluation_document(IMAGE_MODEL_DIR)
        if report_error:
            st.error(report_error)
        elif image_report is not None:
            image_metrics = image_report.get("metrics", {})
            image_accuracy = image_metrics.get("accuracy") if isinstance(image_metrics, dict) else None
            class_report = image_report.get("classification_report", {})
            class_labels = image_report.get("classes", [])
            supports = (
                [
                    class_report[label].get("support", 0)
                    for label in class_labels
                    if isinstance(class_report, dict)
                    and isinstance(class_report.get(label), dict)
                ]
                if isinstance(class_labels, list)
                else []
            )
            total_support = sum(supports)
            majority_accuracy = max(supports, default=0) / total_support if total_support else 0
            if (
                isinstance(image_accuracy, (int, float))
                and total_support
                and image_accuracy < majority_accuracy
            ):
                st.warning(
                    f"This baseline's held-out accuracy ({image_accuracy:.1%}) is "
                    f"below the test set's majority-label baseline ({majority_accuracy:.1%}). "
                    "Treat image predictions as experimental, not as food-safety guidance."
                )
        if image_for_prediction is None:
            st.info("A trained image model is available. Upload an image to classify it.")
        else:
            try:
                model_artifact = _load_image_model(
                    IMAGE_MODEL_DIR,
                    _get_model_modified_time(IMAGE_MODEL_DIR, "image_model.joblib"),
                )
                prediction, probabilities = predict_image(
                    image_for_prediction,
                    model_artifact,
                )
            except (FileNotFoundError, OSError, ValueError, KeyError) as error:
                st.error(f"The saved image model could not be used: {error}")
            else:
                st.success(f"Predicted folder label: **{prediction}**")
                st.caption(
                    "The prediction uses the label text from the image folders; "
                    "the label meanings are not reinterpreted. The model's class "
                    "probabilities are uncalibrated and are not validated confidence "
                    "or food-safety risk."
                )
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Folder label": label,
                                "Uncalibrated model probability": f"{probability:.1%}",
                            }
                            for label, probability in sorted(
                                probabilities.items(),
                                key=lambda item: item[1],
                                reverse=True,
                            )
                        ]
                    ),
                    width="stretch",
                    hide_index=True,
                )
    else:
        st.info(
            "Classification is not ready because no trained image model exists yet. "
            "Upload and preview are available; no prediction will be generated."
        )


@st.cache_resource(show_spinner=False)
def _load_condition_model(model_directory: str, modified_time: float) -> dict[str, Any]:
    model_path = os.path.join(model_directory, "condition_model.joblib")
    if os.path.getmtime(model_path) != modified_time:
        raise OSError(f"The condition model changed while it was being loaded: {model_path}")
    return load_condition_model(model_directory)


@st.cache_resource(show_spinner=False)
def _load_image_model(model_directory: str, modified_time: float) -> dict[str, Any]:
    model_path = os.path.join(model_directory, "image_model.joblib")
    if os.path.getmtime(model_path) != modified_time:
        raise OSError(f"The image model changed while it was being loaded: {model_path}")
    return load_image_model(model_directory)


def _get_model_modified_time(model_directory: str, model_filename: str) -> float:
    return os.path.getmtime(os.path.join(model_directory, model_filename))


def render_shipment_risk_analysis() -> None:
    st.title("Shipment Risk Analysis")
    st.write("The fields below come directly from the condition dataset.")
    data = _get_data()
    if data is not None:
        target_column = "Class"
        input_columns = [column for column in data.columns if column != target_column]
        if input_columns:
            summary = pd.DataFrame(
                {
                    "Field": input_columns,
                    "Data type": [str(data[column].dtype) for column in input_columns],
                    "Missing values": [int(data[column].isna().sum()) for column in input_columns],
                    "Example values": [
                        ", ".join(data[column].dropna().astype(str).unique()[:3])
                        for column in input_columns
                    ],
                }
            )
            st.dataframe(summary, width="stretch", hide_index=True)
        else:
            st.warning("No input fields are available in the current CSV.")

        if target_column in data.columns:
            st.caption(f"'{target_column}' is the dataset label, not an input field.")
        else:
            st.caption("No 'Class' label column was found; all CSV columns are listed as inputs.")

    _, artifacts = _model_status(CONDITION_MODEL_DIR)
    if artifacts:
        try:
            model_artifact = _load_condition_model(
                CONDITION_MODEL_DIR,
                _get_model_modified_time(
                    CONDITION_MODEL_DIR,
                    "condition_model.joblib",
                ),
            )
        except (FileNotFoundError, OSError, ValueError, KeyError) as error:
            st.error(f"The saved condition model could not be loaded: {error}")
            return

        if data is not None:
            dataset_features = [column for column in data.columns if column != "Class"]
            if dataset_features != model_artifact["feature_columns"]:
                st.error(
                    "The current CSV inputs do not match the trained model's feature "
                    "columns. Retrain the condition model before making predictions."
                )
                return

        class_labels = ", ".join(model_artifact.get("classes", []))
        st.subheader("Predict the dataset class")
        st.caption(
            "Inputs and permitted ranges come from the trained model's dataset. "
            "The source CSV does not document measurement units; values are shown "
            "exactly in its recorded scale."
        )
        st.info(
            "**Validated spoilage risk percentage: not available.** "
            f"The model predicts the dataset labels ({class_labels}), but the source "
            "does not define their spoilage meaning. The model probabilities have not "
            "been calibrated, so they cannot be presented as spoilage risk. "
            "Validated Low/Medium/High thresholds are also unavailable."
        )
        with st.form("shipment_condition_prediction"):
            inputs: dict[str, Any] = {}
            for field in model_artifact["input_schema"]:
                name = field["name"]
                if field["kind"] == "categorical":
                    options = field["options"]
                    if not options:
                        st.error(f"No trained options are available for {name}.")
                        continue
                    inputs[name] = st.selectbox(
                        name,
                        options,
                        index=options.index(field["default"])
                        if field["default"] in options
                        else 0,
                        help=f"Values observed in the training data: {', '.join(options)}.",
                    )
                elif field["integer"]:
                    range_help = (
                        f"Observed training-data range: {field['minimum']:g} to "
                        f"{field['maximum']:g}. Unit not documented by the source CSV."
                    )
                    inputs[name] = st.number_input(
                        name,
                        min_value=int(field["minimum"]),
                        max_value=int(field["maximum"]),
                        value=int(field["default"]),
                        step=1,
                        help=range_help,
                    )
                else:
                    range_help = (
                        f"Observed training-data range: {field['minimum']:g} to "
                        f"{field['maximum']:g}. Unit not documented by the source CSV."
                    )
                    numeric_range = field["maximum"] - field["minimum"]
                    inputs[name] = st.number_input(
                        name,
                        min_value=float(field["minimum"]),
                        max_value=float(field["maximum"]),
                        value=float(field["default"]),
                        step=max(numeric_range / 100.0, 0.01),
                        format="%.4f",
                        help=range_help,
                    )
            submitted = st.form_submit_button("Predict dataset class")

        if submitted:
            try:
                prediction = predict_condition(inputs, model_artifact)
            except (ValueError, KeyError, TypeError) as error:
                st.error(f"Could not generate a condition prediction: {error}")
            else:
                st.success(f"Predicted dataset class: **{prediction}**")
                st.caption(
                    "Low/Medium/High risk bands, per-factor contributions, and "
                    "handling recommendations are not provided because this model "
                    "and dataset do not establish causal contributions or validated "
                    "interventions."
                )
    else:
        st.info(
            "Predictions are not ready because no trained condition model exists yet. "
            "This page only inspects the available inputs and will not estimate spoilage risk."
        )


def render_dataset_explorer() -> None:
    st.title("Dataset Explorer")
    data = _get_data()

    if data is not None:
        st.subheader("Condition dataset")
        first, second, third = st.columns(3)
        first.metric("Rows", f"{len(data):,}")
        second.metric("Columns", len(data.columns))
        third.metric("Duplicate rows", f"{int(data.duplicated().sum()):,}")

        st.markdown("**Columns**")
        st.write(", ".join(str(column) for column in data.columns))
        st.markdown("**Sample rows**")
        st.dataframe(data.head(10), width="stretch", hide_index=True)

        st.markdown("**Missing values by column**")
        missing_values = data.isna().sum().rename("Missing values").rename_axis("Column")
        st.dataframe(
            missing_values.reset_index(),
            width="stretch",
            hide_index=True,
        )

        if "Class" in data.columns:
            st.markdown("**Class distribution**")
            class_distribution = (
                data["Class"].fillna("(missing)").astype(str).value_counts().rename("Records")
            )
            st.bar_chart(class_distribution)
            st.dataframe(
                class_distribution.rename_axis("Class").reset_index(),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No 'Class' column is available to show a class distribution.")

    st.subheader("Food image folders")
    image_counts, scan_errors = _get_image_counts()
    _show_image_scan_warnings(scan_errors)
    if image_counts.empty:
        if not os.path.isdir(IMAGE_DATA_DIR):
            st.info(
                "The image-training folders are not included in this deployment. "
                "The saved image model can still classify uploaded JPG, JPEG, and PNG files."
            )
        else:
            st.info("No JPG, JPEG, or PNG images were found in the image folders.")
    else:
        category_counts = (
            image_counts.groupby("Food category", as_index=False)["Images"]
            .sum()
            .sort_values("Food category")
        )
        st.markdown("**Image totals by top-level category**")
        st.dataframe(category_counts, width="stretch", hide_index=True)
        st.markdown("**Counts by folder containing images**")
        st.dataframe(image_counts, width="stretch", hide_index=True)


def render_model_performance() -> None:
    st.title("Model Performance")
    st.write("Metrics are read only from saved test-set evaluation output; this page does not train models.")

    models = [
        ("Food image model", IMAGE_MODEL_DIR),
        ("Shipment condition model", CONDITION_MODEL_DIR),
    ]
    for model_name, model_directory in models:
        st.subheader(model_name)
        report, error = load_evaluation_document(model_directory)
        if error:
            st.error(error)
        elif report is None:
            st.info("Not evaluated yet.")
        else:
            metrics = report.get("metrics", {})
            if not isinstance(metrics, dict):
                st.error("The saved evaluation report has an invalid metrics section.")
                continue
            st.caption(
                f"{report.get('model_name', 'Model')} · "
                f"Target: {report.get('target_column', 'not specified')}"
            )
            st.dataframe(
                pd.DataFrame(
                    [{"Metric": name, "Value": f"{value:.4f}"} for name, value in metrics.items()]
                ),
                width="stretch",
                hide_index=True,
            )
            split = report.get("split", {})
            if isinstance(split, dict):
                split_details = [str(split.get("method", "Evaluation split"))]
                if "train_rows" in split and "test_rows" in split:
                    split_details.append(
                        f"train: {split['train_rows']:,}, test: {split['test_rows']:,}"
                    )
                elif "train_images" in split and "test_images" in split:
                    split_details.append(
                        f"train images: {split['train_images']:,}, "
                        f"test images: {split['test_images']:,}"
                    )
                if "actual_test_fraction" in split:
                    split_details.append(
                        f"test fraction: {split['actual_test_fraction']:.1%}"
                    )
                if "exact_duplicate_rows_removed_before_split" in split:
                    split_details.append(
                        "exact duplicates removed before split: "
                        f"{split['exact_duplicate_rows_removed_before_split']:,}"
                    )
                if "test_food_groups" in split:
                    split_details.append(
                        "held-out food groups: "
                        + ", ".join(str(group) for group in split["test_food_groups"])
                    )
                st.caption(" · ".join(split_details))
            confusion = report.get("confusion_matrix")
            if isinstance(confusion, dict):
                labels = confusion.get("labels")
                values = confusion.get("values")
                if (
                    isinstance(labels, list)
                    and isinstance(values, list)
                    and len(values) == len(labels)
                    and all(
                        isinstance(row, list) and len(row) == len(labels)
                        for row in values
                    )
                ):
                    st.markdown("**Confusion matrix** (rows: actual, columns: predicted)")
                    matrix = pd.DataFrame(
                        values,
                        index=[f"Actual: {label}" for label in labels],
                        columns=[f"Predicted: {label}" for label in labels],
                    )
                    st.dataframe(matrix, width="stretch")
                else:
                    st.warning("The saved confusion matrix has an invalid shape.")
            st.caption(f"Source: {os.path.join(model_directory, EVALUATION_FILENAME)}")
