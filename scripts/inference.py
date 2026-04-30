from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from preprocessing import preprocess_dataset
from epochs import create_epochs
from features import extract_epoch_features
from spectral_features import extract_spectral_features
from signal_preprocessing import apply_basic_filtering


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "final_model.joblib"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.json"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
METRICS_PATH = MODELS_DIR / "model_metrics.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No existe el modelo: {MODEL_PATH}")

    if not FEATURE_COLUMNS_PATH.exists():
        raise FileNotFoundError(f"No existe feature_columns.json: {FEATURE_COLUMNS_PATH}")

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"No existe model_metadata.json: {METADATA_PATH}")

    model = joblib.load(MODEL_PATH)
    feature_columns = load_json(FEATURE_COLUMNS_PATH)
    metadata = load_json(METADATA_PATH)

    metrics = None
    if METRICS_PATH.exists():
        metrics = load_json(METRICS_PATH)

    return model, feature_columns, metadata, metrics


def map_prediction_label(prediction):
    mapping = {
        "0": "Control",
        "1": "ADHD",
        0: "Control",
        1: "ADHD",
    }

    return mapping.get(prediction, str(prediction))


def validate_eeg_dataframe(df, expected_channels):
    if df is None or df.empty:
        raise ValueError("El archivo está vacío.")

    missing_channels = [ch for ch in expected_channels if ch not in df.columns]

    if missing_channels:
        raise ValueError(f"Faltan canales EEG esperados: {missing_channels}")

    non_numeric = []

    for col in expected_channels:
        if not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric.append(col)

    if non_numeric:
        raise ValueError(f"Estas columnas EEG no son numéricas: {non_numeric}")

    return True


def prepare_features_from_dataframe(df, metadata, feature_columns):
    df = df.copy()

    channels = metadata["channels"]
    sfreq = metadata["sfreq"]
    epoch_size = metadata["epoch_size"]
    step_size = metadata["step_size"]
    nperseg = metadata.get("nperseg", epoch_size)
    feature_mode = metadata["feature_mode"]

    validate_eeg_dataframe(df, expected_channels=channels)

    # En inferencia, Class e ID son opcionales.
    # Class no se usa para predecir; solo permite reutilizar create_epochs().
    # ID permite agrupar la señal por sujeto. Si no existe, se crea uno temporal.
    if "Class" not in df.columns:
        df["Class"] = 0

    if "ID" not in df.columns:
        df["ID"] = "uploaded_file"

    df_clean, eeg_cols = preprocess_dataset(df)

    if metadata.get("apply_filtering", False):
        df_clean = apply_basic_filtering(
            df_clean,
            eeg_cols,
            subject_col="ID",
        )

    X_epochs, y_epochs, groups_epochs = create_epochs(
        df=df_clean,
        eeg_columns=eeg_cols,
        label_column="Class",
        group_column="ID",
        epoch_size=epoch_size,
        step_size=step_size,
    )

    if len(X_epochs) == 0:
        raise ValueError(
            "No se han podido generar epochs. El archivo puede ser demasiado corto."
        )

    X_time = extract_epoch_features(
        X_epochs,
        eeg_cols,
    )

    X_spectral = extract_spectral_features(
        X_epochs=X_epochs,
        channel_names=eeg_cols,
        sfreq=sfreq,
        nperseg=nperseg,
    )

    if feature_mode in ["time", "temporal"]:
        X_features = X_time

    elif feature_mode == "spectral":
        X_features = X_spectral

    elif feature_mode == "combined":
        X_features = pd.concat(
            [
                X_time.reset_index(drop=True),
                X_spectral.reset_index(drop=True),
            ],
            axis=1,
        )

    else:
        raise ValueError(f"feature_mode no válido: {feature_mode}")

    missing_features = [col for col in feature_columns if col not in X_features.columns]

    if missing_features:
        raise ValueError(
            f"Faltan features esperadas por el modelo: {missing_features[:20]}"
        )

    X_features = X_features[feature_columns]

    return X_features, X_epochs, y_epochs, groups_epochs


def predict_eeg_dataframe(df):
    model, feature_columns, metadata, metrics = load_model_artifacts()

    X_features, X_epochs, y_epochs, groups_epochs = prepare_features_from_dataframe(
        df=df,
        metadata=metadata,
        feature_columns=feature_columns,
    )

    epoch_predictions = model.predict(X_features)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_features)
        class_labels = list(model.classes_)

        mean_probabilities = probabilities.mean(axis=0)
        best_idx = int(np.argmax(mean_probabilities))

        final_prediction = class_labels[best_idx]
        confidence = float(mean_probabilities[best_idx])

    else:
        values, counts = np.unique(epoch_predictions, return_counts=True)
        best_idx = int(np.argmax(counts))

        final_prediction = values[best_idx]
        confidence = float(counts[best_idx] / len(epoch_predictions))

    unique_preds, pred_counts = np.unique(epoch_predictions, return_counts=True)

    epoch_count_by_class = {
        map_prediction_label(label): int(count)
        for label, count in zip(unique_preds, pred_counts)
    }

    epoch_percentage_by_class = {
        map_prediction_label(label): float(count / len(epoch_predictions))
        for label, count in zip(unique_preds, pred_counts)
    }

    result = {
        "prediction": str(final_prediction),
        "prediction_label": map_prediction_label(final_prediction),
        "confidence": confidence,
        "n_epochs": int(len(epoch_predictions)),
        "epoch_count_by_class": epoch_count_by_class,
        "epoch_percentage_by_class": epoch_percentage_by_class,
        "metrics": metrics,
        "metadata": metadata,
    }

    return result


def predict_eeg_file(file_path):
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {file_path}")

    if file_path.suffix.lower() != ".csv":
        raise ValueError("De momento solo se admiten archivos CSV.")

    df = pd.read_csv(file_path)

    return predict_eeg_dataframe(df)