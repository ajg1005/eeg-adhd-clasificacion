from pathlib import Path
import json

import numpy as np
import pandas as pd
from tensorflow import keras

from epochs import create_epochs
from inference import map_prediction_label, validate_eeg_dataframe
from preprocessing import preprocess_dataset
from signal_preprocessing import apply_basic_filtering, zscore_per_subject


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models" / "dl"

MODEL_PATH = MODELS_DIR / "final_model.keras"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
METRICS_PATH = MODELS_DIR / "model_metrics.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dl_model_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No existe el modelo DL: {MODEL_PATH}")

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"No existe model_metadata.json: {METADATA_PATH}")

    model = keras.models.load_model(MODEL_PATH)
    metadata = load_json(METADATA_PATH)

    metrics = None
    if METRICS_PATH.exists():
        metrics = load_json(METRICS_PATH)

    return model, metadata, metrics


def prepare_dl_epochs_from_dataframe(df, metadata):
    df = df.copy()

    channels = metadata["channels"]
    epoch_size = metadata["epoch_size"]
    step_size = metadata["step_size"]

    validate_eeg_dataframe(df, expected_channels=channels)

    # En inferencia, Class e ID son opcionales.
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
            sfreq=metadata["sfreq"],
        )

    if metadata.get("apply_zscore", False):
        df_clean = zscore_per_subject(
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

    X_epochs = np.asarray(X_epochs).astype(np.float32)

    return X_epochs, y_epochs, groups_epochs


def predict_eeg_dataframe_dl(df):
    model, metadata, metrics = load_dl_model_artifacts()

    X_epochs, y_epochs, groups_epochs = prepare_dl_epochs_from_dataframe(
        df=df,
        metadata=metadata,
    )

    threshold = float(metadata.get("threshold", 0.5))
    epoch_scores = model.predict(X_epochs, batch_size=32, verbose=0).ravel()
    epoch_predictions = (epoch_scores >= threshold).astype(int)

    mean_score = float(np.mean(epoch_scores))
    final_prediction = int(mean_score >= threshold)

    if final_prediction == 1:
        confidence = mean_score
    else:
        confidence = 1.0 - mean_score

    unique_preds, pred_counts = np.unique(epoch_predictions, return_counts=True)

    epoch_count_by_class = {
        map_prediction_label(int(label)): int(count)
        for label, count in zip(unique_preds, pred_counts)
    }

    epoch_percentage_by_class = {
        map_prediction_label(int(label)): float(count / len(epoch_predictions))
        for label, count in zip(unique_preds, pred_counts)
    }

    prediction_label = map_prediction_label(final_prediction)
    final_class_epoch_percentage = epoch_percentage_by_class.get(
        prediction_label,
        0.0,
    )

    result = {
        "model_id": metadata.get("model_id", "dl_best"),
        "model_name": metadata.get("model_name"),
        "model_family": metadata.get("model_family", "deep_learning"),
        "prediction": str(final_prediction),
        "prediction_label": prediction_label,
        "confidence": float(confidence),
        "decision_score": float(confidence),
        "final_class_epoch_percentage": final_class_epoch_percentage,
        "threshold": threshold,
        "n_epochs": int(len(epoch_predictions)),
        "epoch_count_by_class": epoch_count_by_class,
        "epoch_percentage_by_class": epoch_percentage_by_class,
        "metrics": metrics,
        "metadata": metadata,
    }

    return result


def predict_eeg_file_dl(file_path):
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {file_path}")

    if file_path.suffix.lower() != ".csv":
        raise ValueError("De momento solo se admiten archivos CSV.")

    df = pd.read_csv(file_path)

    return predict_eeg_dataframe_dl(df)
