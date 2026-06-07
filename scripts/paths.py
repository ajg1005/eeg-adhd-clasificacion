from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "Figuras"

CSV_PATH = DATA_DIR / "adhdata.csv"
TRAINING_CURVES_DIR = FIGURES_DIR / "training_curves_tf"

MODELS_DIR = BASE_DIR / "models"
ML_MODELS_DIR = MODELS_DIR / "ml"
DL_MODELS_DIR = MODELS_DIR / "dl"

ML_BEST_CONFIG_PATH = RESULTS_DIR / "best_model_config.json"
DL_BEST_CONFIG_PATH = RESULTS_DIR / "dl_best_model_config.json"

ML_MODEL_PATH = ML_MODELS_DIR / "final_model.joblib"
ML_FEATURE_COLUMNS_PATH = ML_MODELS_DIR / "feature_columns.json"
ML_METADATA_PATH = ML_MODELS_DIR / "model_metadata.json"

DL_MODEL_PATH = DL_MODELS_DIR / "final_model.keras"
DL_METADATA_PATH = DL_MODELS_DIR / "model_metadata.json"
DL_METRICS_PATH = DL_MODELS_DIR / "model_metrics.json"
