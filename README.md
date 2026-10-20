# 🍎 Food Supply Chain – Spoilage Prediction

> **College Project (FT-22)** – Predict food spoilage using image classification and supply-chain condition analysis.

---

## 📁 Project Structure

| Folder / File | Purpose |
|---|---|
| `data/images/` | Food images organised by category, food class, and freshness label. |
| `data/condition_data/` | Tabular shipment-condition dataset (`Dataset.csv`). |
| `models/image_model/` | Saved artifacts for the trained image classification model. |
| `models/condition_model/` | Saved artifacts for the trained condition-based prediction model. |
| `notebooks/` | Jupyter notebooks for interactive exploration and prototyping. |
| `src/` | Core Python modules – preprocessing, training, prediction, and evaluation. |
| `pages/` | Streamlit pages for image analysis, shipment risk, dataset exploration, and model performance. |
| `utils/` | Shared helpers and central configuration (`config.py`). |
| `tests/` | Unit tests for the prediction pipeline. |
| `app.py` | Main Streamlit entry point. |
| `requirements.txt` | Python dependencies. |

---

## 🚀 How to Run

### Run locally

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
streamlit run app.py
```

Run these commands from the project root. The app entry point and data/model paths resolve relative to the project files, not the shell's current folder.

## ☁️ Streamlit Community Cloud deployment

The app is prepared for Community Cloud with `app.py` as the entry point, a root-level `requirements.txt`, a light theme in `.streamlit/config.toml`, and paths resolved from the project root. Runtime dependencies are pinned to the versions in the verified model and application environment.

The full local image-training corpus contains **85,012 images totaling about 1.33 GB**, so it is intentionally excluded from Git. The saved image classifier and its evaluation report are included by the ignore rules and are sufficient for upload-based image predictions. When that corpus is absent in the cloud, the app labels its image count as **Not bundled** rather than reporting zero. The condition CSV (about 386 KB), both model artifacts, and both evaluation reports are small runtime files and are explicitly allowed by `.gitignore`.

1. Push this project to a GitHub repository. Confirm `data/condition_data/Dataset.csv`, both `models/*_model.joblib` files, and both `models/*/evaluation_results.json` files are present in the repository. Do not add `data/images/` or its `_folder_structure_backup.txt` inventory file.
2. Open [share.streamlit.io](https://share.streamlit.io), select **Create app**, and choose the repository, branch, and `app.py` as the main file.
3. In **Advanced settings**, select **Python 3.14**, matching the training environment used for these model artifacts, then deploy. No secrets or system packages are required.
4. Confirm the app opens on its Home page and test both prediction pages after the build completes.

This workspace is not a Git repository, so it cannot be pushed or deployed from this folder until it is initialized or uploaded to GitHub. Community Cloud deployment also requires a GitHub account and repository access; those steps have not been performed here.

### Train the baseline models

Run from the project root:

```bash
python -m src.condition_training
python -m src.image_training
```

The condition baseline uses `Class` as its target and every other CSV column as an input. It removes exact duplicate rows before a stratified 80/20 split so identical rows cannot appear in both sets. The labels `BAD`, `Bad`, and `Good` are preserved exactly as stored; the app does not infer what they mean. Preprocessing (numeric imputation/scaling and categorical imputation/one-hot encoding) is fitted only on the training split.

**Shipment risk-score limitation:** Shipment Risk Analysis currently reports only the model's predicted dataset class. It deliberately does not show a 0–100% spoilage score or Low/Medium/High band: the source does not define the distinct labels `BAD`, `Bad`, and `Good`, and the class-weighted logistic model's probabilities have not been calibrated against an independent validation set. Therefore no spoilage event can yet be mapped defensibly to a probability or threshold. The dataset also does not document measurement units (values are shown as recorded), causal factor contributions, or validated handling recommendations. Do not interpret the prediction as food-safety advice.

The image baseline uses the label-folder names as-is and a compact RGB thumbnail/histogram feature representation. Its stratified group split holds out complete category/food-type folders, so images of a food type do not occur in both train and test. The 85,012 images require time and memory to read and process.

Both commands save a model artifact and `evaluation_results.json` under their respective folders in `models/`. The app uses these artifacts for condition/image predictions and shows the saved test metrics and confusion matrices. A model retrain replaces only its own generated model and evaluation files; the datasets are not modified.

---

## 🧠 Two-Model Approach

| Model | Input | Output |
|---|---|---|
| **Image Model** | Food photograph | Exact label copied from its image folder |
| **Condition Model** | `Fruit`, `Temp`, `Humid (%)`, `Light (Fux)`, and `CO2 (pmm)` | Exact `Class` value predicted by the baseline |

The two models are developed **independently** and later surfaced together in the Streamlit dashboard.

---

## 📝 Notes

- The image classifier is a lightweight scikit-learn baseline, not a deep-learning model. Its visual features and held-out-by-food-type split should be considered when interpreting its metrics.
- Condition predictions reproduce the dataset's label strings; confirm class meaning with the dataset source before using results operationally.
- Image-model class probabilities are uncalibrated; the UI labels them clearly and warns that image accuracy is below the test majority-label baseline.
- The `.gitignore` excludes the large image corpus while allowing the small condition CSV and saved model/evaluation files needed at runtime. Never commit secrets.
