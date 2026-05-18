from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.base import clone

from data_load import load_dataset
from epochs import create_epochs
from features import extract_epoch_features
from pipeline import get_models
from preprocessing import preprocess_dataset
from spectral_features import extract_spectral_features


# Rutas


BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "adhdata.csv"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models" / "ml"

CONFIG_PATH = RESULTS_DIR / "best_model_config.json"

MODEL_PATH = MODELS_DIR / "final_model.joblib"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.json"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)


# Funciones


def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"No existe {CONFIG_PATH}. Ejecuta primero train_ml.py."
        )

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_features(x_epochs, eeg_cols, config):
    feature_mode = config["feature_mode"]

    if feature_mode == "time":
        return extract_epoch_features(x_epochs, eeg_cols)

    if feature_mode == "spectral":
        return extract_spectral_features(
            x_epochs=x_epochs,
            channel_names=eeg_cols,
            sfreq=config["sfreq"],
            nperseg=config["nperseg"],
        )

    if feature_mode == "combined":
        x_time = extract_epoch_features(x_epochs, eeg_cols)

        x_spectral = extract_spectral_features(
            x_epochs=x_epochs,
            channel_names=eeg_cols,
            sfreq=config["sfreq"],
            nperseg=config["nperseg"],
        )

        return pd.concat(
            [
                x_time.reset_index(drop=True),
                x_spectral.reset_index(drop=True),
            ],
            axis=1,
        )

    raise ValueError(f"feature_mode no valido: {feature_mode}")


def main():
    print("Cargando configuracion del mejor modelo...")
    config = load_config()

    best_model_name = config["best_model"]

    print(f"Modelo elegido: {best_model_name}")
    print(f"Features: {config['feature_mode']}")
    print(f"Epoch size: {config['epoch_size']}")
    print(f"Step size: {config['step_size']}")

    print("\nCargando dataset...")
    df = load_dataset(CSV_PATH)

    print("Preprocesando dataset...")
    df_clean, eeg_cols = preprocess_dataset(df)

    print("Creando epochs...")
    x_epochs, y_epochs, groups_epochs = create_epochs(
        df=df_clean,
        eeg_columns=eeg_cols,
        label_column="Class",
        group_column="ID",
        epoch_size=config["epoch_size"],
        step_size=config["step_size"],
    )

    print("Shape X_epochs:", x_epochs.shape)
    print("Shape y_epochs:", y_epochs.shape)
    print("Shape groups_epochs:", groups_epochs.shape)

    print("Extrayendo features...")
    x_features = build_features(x_epochs, eeg_cols, config)

    print("Shape X_features:", x_features.shape)

    print("Cargando modelos disponibles...")
    models = get_models()

    if best_model_name not in models:
        raise ValueError(
            f"El modelo '{best_model_name}' no existe en get_models(). "
            f"Modelos disponibles: {list(models.keys())}"
        )

    print("Entrenando modelo final...")
    final_model = clone(models[best_model_name])
    final_model.fit(x_features, y_epochs)

    print("Guardando modelo...")
    joblib.dump(final_model, MODEL_PATH)

    print("Guardando columnas de features...")
    with open(FEATURE_COLUMNS_PATH, "w", encoding="utf-8") as f:
        json.dump(list(x_features.columns), f, indent=4)

    print("Guardando metadata...")
    metadata = {
        "model_name": best_model_name,
        "feature_mode": config["feature_mode"],

        "sfreq": config["sfreq"],
        "epoch_size": config["epoch_size"],
        "step_size": config["step_size"],
        "nperseg": config["nperseg"],

        "apply_zscore": config.get("apply_zscore", False),
        "apply_filtering": False,

        "channels": list(eeg_cols),
        "label_column": "Class",
        "group_column": "ID",

        "class_mapping": {
            "0": "Control",
            "1": "ADHD"
        },

        "n_epochs_training": int(len(x_features)),
        "n_features": int(x_features.shape[1]),
        "n_subjects_training": int(len(set(groups_epochs))),

        "cv_metrics": config.get("cv_metrics", {}),
        "dataset_summary": config.get("dataset_summary", {}),

        "training_strategy": (
            "Final model trained using the model selected by "
            "cross-subject cross-validation in train_ml.py."
        )
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("\nExportacion ML completada:")
    print(f"- {MODEL_PATH}")
    print(f"- {FEATURE_COLUMNS_PATH}")
    print(f"- {METADATA_PATH}")


if __name__ == "__main__":
    main()
