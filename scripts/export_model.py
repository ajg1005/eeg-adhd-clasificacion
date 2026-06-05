"""Reentrena y exporta el modelo ML seleccionado por train_ml.py."""

import json
import joblib
from sklearn.base import clone

from data_load import load_dataset
from epochs import create_epochs
from feature_pipeline import build_features_from_config
from pipeline import get_models
from paths import (
    CSV_PATH,
    ML_BEST_CONFIG_PATH as CONFIG_PATH,
    ML_FEATURE_COLUMNS_PATH as FEATURE_COLUMNS_PATH,
    ML_METADATA_PATH as METADATA_PATH,
    ML_MODEL_PATH as MODEL_PATH,
    ML_MODELS_DIR as MODELS_DIR,
)
from preprocessing import preprocess_dataset


MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"No existe {CONFIG_PATH}. Ejecuta primero train_ml.py."
        )

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print("Cargando configuración del mejor modelo...")
    config = load_config()

    best_model_name = config["best_model"]

    print(f"Modelo elegido: {best_model_name}")
    print(f"Caracteristicas: {config['feature_mode']}")
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

    print("Extrayendo caracteristicas...")
    x_features = build_features_from_config(x_epochs, eeg_cols, config)

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

    print("Guardando columnas de caracteristicas...")
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
            "Modelo final entrenado usando el modelo seleccionado mediante "
            "validacion cruzada cross-subject StratifiedGroupKFold de 5 folds en train_ml.py."
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
