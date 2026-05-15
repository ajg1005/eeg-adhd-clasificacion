from pathlib import Path
import sys

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from backend.modeling.common import (  # noqa: E402
    map_prediction_label,
    prepare_features_from_dataframe,
    validate_eeg_dataframe,
)
from backend.modeling.predictors import get_predictor  # noqa: E402


def load_model_artifacts():
    return get_predictor("ml_best").load_artifacts()


def predict_eeg_dataframe(df):
    return get_predictor("ml_best").predict(df)


def predict_eeg_file(file_path):
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {file_path}")

    if file_path.suffix.lower() != ".csv":
        raise ValueError("De momento solo se admiten archivos CSV.")

    df = pd.read_csv(file_path)

    return predict_eeg_dataframe(df)
