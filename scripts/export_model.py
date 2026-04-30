from pathlib import Path
import sys
import json
import joblib
import pandas as pd
from sklearn.base import clone


# Rutas


BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"

if SRC_DIR.exists():
    sys.path.append(str(SRC_DIR))

CSV_PATH = BASE_DIR / "data" / "adhdata.csv"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

CONFIG_PATH = RESULTS_DIR / "best_model_config.json"

MODEL_PATH = MODELS_DIR / "final_model.joblib"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.json"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)


# Imports del proyecto


from data_load import load_dataset
from epochs import create_epochs
from features import extract_epoch_features
from pipeline import get_models
from preprocessing import preprocess_dataset
from signal_preprocessing import apply_basic_filtering
from spectral_features import extract_spectral_features



# Funciones


def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"No existe {CONFIG_PATH}. Ejecuta primero train_ml.py."
        )

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_features(X_epochs, eeg_cols, config):
    feature_mode = config["feature_mode"]

    if feature_mode == "time":
        return extract_epoch_features(X_epochs, eeg_cols)

    if feature_mode == "spectral":
        return extract_spectral_features(
            X_epochs=X_epochs,
            channel_names=eeg_cols,
            sfreq=config["sfreq"],
            nperseg=config["nperseg"],
        )

    if feature_mode == "combined":
        X_time = extract_epoch_features(X_epochs, eeg_cols)

        X_spectral = extract_spectral_features(
            X_epochs=X_epochs,
            channel_names=eeg_cols,
            sfreq=config["sfreq"],
            nperseg=config["nperseg"],
        )

        return pd.concat(
            [
                X_time.reset_index(drop=True),
                X_spectral.reset_index(drop=True),
            ],
            axis=1,
        )

    raise ValueError(f"feature_mode no válido: {feature_mode}")


def main():
    print("Cargando configuración del mejor modelo...")
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

    # IMPORTANTE:
    # Esto debe coincidir con lo usado en train_ml.py.
    if config.get("apply_filtering", False):
        print("Aplicando filtrado básico...")
        df_model = apply_basic_filtering(
            df_clean,
            eeg_cols,
            subject_col="ID",
        )
    else:
        df_model = df_clean

    print("Creando epochs...")
    X_epochs, y_epochs, groups_epochs = create_epochs(
        df=df_model,
        eeg_columns=eeg_cols,
        label_column="Class",
        group_column="ID",
        epoch_size=config["epoch_size"],
        step_size=config["step_size"],
    )

    print("Shape X_epochs:", X_epochs.shape)
    print("Shape y_epochs:", y_epochs.shape)
    print("Shape groups_epochs:", groups_epochs.shape)

    print("Extrayendo features...")
    X_features = build_features(X_epochs, eeg_cols, config)

    print("Shape X_features:", X_features.shape)

    print("Cargando modelos disponibles...")
    models = get_models()

    if best_model_name not in models:
        raise ValueError(
            f"El modelo '{best_model_name}' no existe en get_models(). "
            f"Modelos disponibles: {list(models.keys())}"
        )

    print("Entrenando modelo final...")
    final_model = clone(models[best_model_name])
    final_model.fit(X_features, y_epochs)

    print("Guardando modelo...")
    joblib.dump(final_model, MODEL_PATH)

    print("Guardando columnas de features...")
    with open(FEATURE_COLUMNS_PATH, "w", encoding="utf-8") as f:
        json.dump(list(X_features.columns), f, indent=4)

    print("Guardando metadata...")
    metadata = {
        "model_name": best_model_name,
        "feature_mode": config["feature_mode"],

        "sfreq": config["sfreq"],
        "epoch_size": config["epoch_size"],
        "step_size": config["step_size"],
        "nperseg": config["nperseg"],

        "apply_zscore": config.get("apply_zscore", False),
        "lowcut": config.get("lowcut"),
        "highcut": config.get("highcut"),

        "channels": list(eeg_cols),
        "label_column": "Class",
        "group_column": "ID",

        "class_mapping": {
            "0": "Control",
            "1": "ADHD"
        },

        "n_epochs_training": int(len(X_features)),
        "n_features": int(X_features.shape[1]),
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

    print("\nExportación completada:")
    print(f"- {MODEL_PATH}")
    print(f"- {FEATURE_COLUMNS_PATH}")
    print(f"- {METADATA_PATH}")


if __name__ == "__main__":
    main()