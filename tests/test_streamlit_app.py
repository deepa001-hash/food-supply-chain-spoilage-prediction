"""Tests for the Streamlit app's data and artifact helpers."""

import json
import os
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from streamlit.testing.v1 import AppTest

from src.condition_training import train_condition_model
from src.image_training import (
    FEATURE_COUNT,
    _make_grouped_split,
    extract_image_features,
)
from src.prediction import (
    load_condition_model,
    load_image_model,
    predict_condition,
)
from src.streamlit_app import (
    find_model_artifacts,
    load_condition_data,
    load_evaluation_results,
    scan_image_folders,
)
from utils.config import CONDITION_CSV, CONDITION_MODEL_DIR


class TestStreamlitAppHelpers(unittest.TestCase):
    def test_theme_uses_light_background_and_dark_text(self):
        project_root = Path(__file__).resolve().parents[1]
        theme_path = project_root / ".streamlit" / "config.toml"
        theme = theme_path.read_text(encoding="utf-8")

        self.assertIn('base = "light"', theme)
        self.assertIn('backgroundColor = "#ffffff"', theme)
        self.assertIn('textColor = "#1f2933"', theme)
        self.assertIn('textColor = "#20352a"', theme)

    def test_cloud_ignore_rules_keep_runtime_artifacts_and_exclude_training_images(self):
        project_root = Path(__file__).resolve().parents[1]
        ignore_rules = (project_root / ".gitignore").read_text(encoding="utf-8")

        self.assertIn("data/images/", ignore_rules)
        self.assertIn("!data/condition_data/Dataset.csv", ignore_rules)
        self.assertIn("!models/image_model/image_model.joblib", ignore_rules)
        self.assertIn("!models/condition_model/condition_model.joblib", ignore_rules)
        self.assertIn(".env", ignore_rules)
        self.assertIn(".streamlit/secrets.toml", ignore_rules)

    def test_cloud_pages_explain_when_image_training_data_is_not_bundled(self):
        project_root = Path(__file__).resolve().parents[1]
        missing_image_root = os.path.join(tempfile.gettempdir(), "not-bundled-images")

        with patch("src.streamlit_app.IMAGE_DATA_DIR", missing_image_root):
            home = AppTest.from_file(
                str(project_root / "pages" / "0_Home.py"),
                default_timeout=60,
            ).run()
            explorer = AppTest.from_file(
                str(project_root / "pages" / "4_Dataset_Explorer.py"),
                default_timeout=60,
            ).run()

        self.assertFalse(home.exception, [error.message for error in home.exception])
        self.assertIn("Not bundled", [metric.value for metric in home.metric])
        self.assertFalse(
            explorer.exception,
            [error.message for error in explorer.exception],
        )
        self.assertTrue(
            any("not included in this deployment" in item.value.lower() for item in explorer.info)
        )

    def test_loads_project_condition_dataset(self):
        data, error = load_condition_data(CONDITION_CSV)

        self.assertIsNone(error)
        self.assertIsNotNone(data)
        self.assertGreater(len(data), 0)
        self.assertIn("Class", data.columns)

    def test_missing_dataset_returns_error(self):
        data, error = load_condition_data(os.path.join(tempfile.gettempdir(), "missing.csv"))

        self.assertIsNone(data)
        self.assertTrue(error)

    def test_counts_images_in_actual_leaf_folders(self):
        with tempfile.TemporaryDirectory() as image_root:
            fresh_folder = os.path.join(image_root, "Fruit", "Apple", "Fresh")
            rotten_folder = os.path.join(image_root, "Fruit", "Apple", "Rotten")
            os.makedirs(fresh_folder)
            os.makedirs(rotten_folder)
            for filename in ("one.jpg", "two.PNG", "three.webp", "ignore.txt"):
                with open(os.path.join(fresh_folder, filename), "wb") as image_file:
                    image_file.write(b"test")
            with open(os.path.join(rotten_folder, "three.jpeg"), "wb") as image_file:
                image_file.write(b"test")

            counts, errors = scan_image_folders(image_root)

        self.assertEqual(errors, [])
        self.assertEqual(counts["Images"].sum(), 4)
        self.assertEqual(
            counts.set_index("Folder").loc["Fruit / Apple / Fresh", "Images"],
            3,
        )

    def test_only_recognized_model_files_are_listed(self):
        with tempfile.TemporaryDirectory() as model_directory:
            with open(os.path.join(model_directory, "model.joblib"), "wb") as model_file:
                model_file.write(b"placeholder")
            with open(os.path.join(model_directory, "notes.txt"), "w", encoding="utf-8") as notes:
                notes.write("not a model")

            artifacts = find_model_artifacts(model_directory)

        self.assertEqual(len(artifacts), 1)
        self.assertTrue(artifacts[0].endswith("model.joblib"))

    def test_reads_recognized_evaluation_metrics(self):
        with tempfile.TemporaryDirectory() as model_directory:
            results_path = os.path.join(model_directory, "evaluation_results.json")
            with open(results_path, "w", encoding="utf-8") as results_file:
                json.dump({"metrics": {"accuracy": 0.91, "ignored": 3}}, results_file)

            metrics, error = load_evaluation_results(model_directory)

        self.assertIsNone(error)
        self.assertEqual(metrics, {"accuracy": 0.91})

    def test_missing_evaluation_results_are_not_evaluated(self):
        with tempfile.TemporaryDirectory() as model_directory:
            metrics, error = load_evaluation_results(model_directory)

        self.assertIsNone(error)
        self.assertIsNone(metrics)

    def test_condition_training_saves_loadable_model_and_real_metrics(self):
        with tempfile.TemporaryDirectory() as temp_directory:
            csv_path = os.path.join(temp_directory, "conditions.csv")
            model_directory = os.path.join(temp_directory, "model")
            rows = []
            labels = ["BAD", "Bad", "Good"]
            for index in range(90):
                rows.append(
                    {
                        "Fruit": "Apple" if index % 2 else "Banana",
                        "Temp": 20 + index % 9,
                        "Class": labels[index % len(labels)],
                    }
                )
            pd.DataFrame(rows).to_csv(csv_path, index=False)

            result = train_condition_model(csv_path, model_directory)
            artifact = load_condition_model(model_directory)
            prediction = predict_condition(
                {"Fruit": "Apple", "Temp": 22},
                artifact,
            )

            self.assertTrue(os.path.isfile(result["model_path"]))
            self.assertTrue(os.path.isfile(result["evaluation_path"]))
            self.assertEqual(result["evaluation"]["split"]["test_rows"], 4)
            self.assertEqual(
                result["evaluation"]["split"]["exact_duplicate_rows_removed_before_split"],
                72,
            )
            self.assertIn(prediction, {"BAD", "Bad", "Good"})
            self.assertIn("f1_macro", result["evaluation"]["metrics"])
            self.assertFalse(
                result["evaluation"]["risk_assessment"]["spoilage_risk_score_available"]
            )

    def test_saved_condition_metrics_match_recomputed_stratified_holdout(self):
        artifact = load_condition_model()
        report_path = Path(CONDITION_MODEL_DIR) / "evaluation_results.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        dataset = pd.read_csv(CONDITION_CSV).drop_duplicates().reset_index(drop=True)
        features = dataset[artifact["feature_columns"]]
        labels = dataset[artifact["target_column"]]
        x_train, x_test, _, y_test = train_test_split(
            features,
            labels,
            test_size=report["split"]["test_fraction"],
            random_state=report["split"]["random_seed"],
            stratify=labels,
        )

        pipeline = artifact["pipeline"]
        classifier = pipeline.named_steps["classifier"]
        predictions = pipeline.predict(x_test)
        classes = classifier.classes_.tolist()
        metrics = report["metrics"]

        self.assertTrue(hasattr(classifier, "n_iter_"))
        self.assertGreater(int(classifier.n_iter_.max()), 0)
        self.assertEqual(len(x_train), report["split"]["train_rows"])
        self.assertEqual(len(x_test), report["split"]["test_rows"])
        self.assertFalse(
            {tuple(row) for row in x_train.itertuples(index=False, name=None)}
            & {tuple(row) for row in x_test.itertuples(index=False, name=None)}
        )
        self.assertEqual(
            accuracy_score(y_test, predictions),
            metrics["accuracy"],
        )
        self.assertEqual(
            precision_score(y_test, predictions, average="macro", zero_division=0),
            metrics["precision_macro"],
        )
        self.assertEqual(
            recall_score(y_test, predictions, average="macro", zero_division=0),
            metrics["recall_macro"],
        )
        self.assertEqual(
            f1_score(y_test, predictions, average="macro", zero_division=0),
            metrics["f1_macro"],
        )
        self.assertEqual(
            confusion_matrix(y_test, predictions, labels=classes).tolist(),
            report["confusion_matrix"]["values"],
        )
        self.assertFalse(report["risk_assessment"]["spoilage_risk_score_available"])

    def test_condition_prediction_rejects_missing_input(self):
        artifact = load_condition_model()
        with self.assertRaisesRegex(ValueError, "Missing required input"):
            predict_condition(
                {
                    "Fruit": "Banana",
                    "Temp": 24,
                    "Humid (%)": 90,
                    "Light (Fux)": 12.0,
                },
                artifact,
            )

    def test_condition_prediction_rejects_unseen_category_and_out_of_range_value(self):
        artifact = load_condition_model()
        valid_inputs = {
            "Fruit": "Banana",
            "Temp": 24,
            "Humid (%)": 90,
            "Light (Fux)": 12.0,
            "CO2 (pmm)": 330,
        }
        with self.assertRaisesRegex(ValueError, "dataset values"):
            predict_condition({**valid_inputs, "Fruit": "Unknown food"}, artifact)
        with self.assertRaisesRegex(ValueError, "training-data range"):
            predict_condition({**valid_inputs, "Temp": 500}, artifact)
        with self.assertRaisesRegex(ValueError, "finite numeric"):
            predict_condition({**valid_inputs, "Light (Fux)": float("nan")}, artifact)
        with self.assertRaisesRegex(ValueError, "whole number"):
            predict_condition({**valid_inputs, "Temp": 22.5}, artifact)

    def test_shipment_page_runs_prediction_without_showing_risk_percentage(self):
        page_path = Path(__file__).resolve().parents[1] / "pages" / "2_Shipment_Risk_Analysis.py"
        app = AppTest.from_file(
            str(page_path),
            default_timeout=30,
        ).run()
        self.assertFalse(app.exception, [error.message for error in app.exception])
        self.assertEqual(
            {widget.label for widget in app.number_input},
            {"Temp", "Humid (%)", "Light (Fux)", "CO2 (pmm)"},
        )
        self.assertEqual([widget.label for widget in app.selectbox], ["Fruit"])
        self.assertTrue(
            any("Validated spoilage risk percentage: not available" in item.value for item in app.info)
        )

        app.button[0].click().run()
        self.assertFalse(app.exception, [error.message for error in app.exception])
        self.assertTrue(
            any(item.value.startswith("Predicted dataset class:") for item in app.success)
        )
        self.assertFalse(
            any("Probability" in str(frame.value) for frame in app.dataframe)
        )

    def test_all_five_pages_render_without_exceptions(self):
        project_root = Path(__file__).resolve().parents[1]
        page_paths = sorted((project_root / "pages").glob("*.py"))

        self.assertEqual(len(page_paths), 5)
        for page_path in page_paths:
            with self.subTest(page=page_path.name):
                app = AppTest.from_file(str(page_path), default_timeout=60).run()
                self.assertFalse(
                    app.exception,
                    [error.message for error in app.exception],
                )

    def test_main_entry_point_loads_with_readable_theme(self):
        project_root = Path(__file__).resolve().parents[1]
        app = AppTest.from_file(
            str(project_root / "app.py"),
            default_timeout=60,
        ).run()

        self.assertFalse(app.exception, [error.message for error in app.exception])
        self.assertTrue(
            any("Food Supply Chain Spoilage Prediction" in item.value for item in app.title)
        )
        self.assertTrue(
            any("#174c2b" in item.value and "#1f2933" in item.value for item in app.markdown)
        )

    def test_image_upload_runs_saved_classifier_and_discloses_uncalibrated_scores(self):
        project_root = Path(__file__).resolve().parents[1]
        page_path = project_root / "pages" / "1_Food_Image_Analysis.py"
        image_model = load_image_model()
        self.assertEqual(
            list(image_model["classifier"].classes_),
            image_model["classes"],
        )
        image_bytes = BytesIO()
        Image.new("RGB", (24, 16), color=(120, 50, 200)).save(image_bytes, format="PNG")

        app = AppTest.from_file(str(page_path), default_timeout=60).run()
        self.assertFalse(app.exception, [error.message for error in app.exception])
        app.file_uploader[0].set_value(("sample.png", image_bytes.getvalue(), "image/png"))
        app.run()

        self.assertFalse(app.exception, [error.message for error in app.exception])
        self.assertTrue(
            any(item.value.startswith("Predicted folder label:") for item in app.success)
        )
        self.assertTrue(
            any("uncalibrated" in item.value.lower() for item in app.caption)
        )
        score_tables = [
            frame.value
            for frame in app.dataframe
            if "Uncalibrated model probability" in frame.value.columns
        ]
        self.assertEqual(len(score_tables), 1)
        self.assertEqual(len(score_tables[0]), len(image_model["classes"]))

    def test_image_features_and_grouped_split_are_valid(self):
        image = Image.new("RGB", (20, 12), color=(120, 50, 200))
        features = extract_image_features(image)
        self.assertEqual(features.shape, (1, FEATURE_COUNT))
        self.assertTrue(np.isfinite(features).all())

        labels = np.asarray(["Fresh", "Rotten"] * 5)
        groups = np.asarray([f"food-{group}" for group in range(5) for _ in range(2)])
        train_indices, test_indices = _make_grouped_split(labels, groups)
        self.assertEqual(set(labels[train_indices]), {"Fresh", "Rotten"})
        self.assertEqual(set(labels[test_indices]), {"Fresh", "Rotten"})
        self.assertFalse(set(groups[train_indices]) & set(groups[test_indices]))


if __name__ == "__main__":
    unittest.main()
