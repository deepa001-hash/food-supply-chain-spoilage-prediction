"""
config.py
---------
Central configuration for the project: file paths, model hyperparameters,
image dimensions, random seeds, etc.

TODO: Populate as the project develops.
"""

import os

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Data paths ────────────────────────────────────────────────────────────────
IMAGE_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "images")
CONDITION_CSV  = os.path.join(PROJECT_ROOT, "data", "condition_data", "Dataset.csv")

# ── Model save paths ─────────────────────────────────────────────────────────
IMAGE_MODEL_DIR     = os.path.join(PROJECT_ROOT, "models", "image_model")
CONDITION_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "condition_model")

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
