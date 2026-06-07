"""Reentrena y exporta el modelo DL seleccionado por train_dl.py."""

import json

import keras
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from scripts.constants import RANDOM_STATE
from scripts.data_load import load_dataset
from scripts.epochs import create_epochs
from scripts.preprocessing import preprocess_dataset
from scripts.paths import (
    CSV_PATH,
    DL_BEST_CONFIG_PATH as CONFIG_PATH,
    DL_METADATA_PATH as METADATA_PATH,
    DL_METRICS_PATH as METRICS_PATH,
    DL_MODEL_PATH as MODEL_PATH,
    DL_MODELS_DIR as MODELS_DIR,
)
from scripts.signal_preprocessing import apply_basic_filtering, zscore_per_subject
from scripts.split import make_group_shuffle_split
from scripts.tf_models import build_model


MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"No existe {CONFIG_PATH}. Ejecuta primero python -m scripts.train_dl."
        )

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed):
    keras.utils.set_random_seed(seed)


def build_callbacks(patience):
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=1,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def build_compiled_model(config, input_shape):
    model = build_model(
        model_name=config["best_model"],
        input_shape=input_shape,
        dropout=config["dropout"],
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=config["learning_rate"],
            clipnorm=1.0,
        ),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=0.05),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
        ],
    )

    return model


def prepare_dataset(config):
    df = load_dataset(CSV_PATH)
    df_clean, eeg_cols = preprocess_dataset(df)

    df_model = df_clean

    if config.get("apply_filtering", False):
        df_model = apply_basic_filtering(
            df_model,
            eeg_cols,
            subject_col="ID",
            sfreq=config["sfreq"],
        )

    if config.get("apply_zscore", False):
        df_model = zscore_per_subject(
            df_model,
            eeg_cols,
            subject_col="ID",
        )

    x_epochs, y_epochs, groups_epochs = create_epochs(
        df=df_model,
        eeg_columns=eeg_cols,
        label_column="Class",
        group_column="ID",
        epoch_size=config["epoch_size"],
        step_size=config["step_size"],
    )

    return x_epochs, y_epochs, groups_epochs, eeg_cols


def main():
    print("Version de TensorFlow:", tf.__version__)
    print("Cargando configuración del mejor modelo DL...")

    config = load_config()
    set_seed(42)

    print(f"Modelo DL elegido: {config['best_model']}")

    x_epochs, y_epochs, groups_epochs, eeg_cols = prepare_dataset(config)

    print("Shape X_epochs:", x_epochs.shape)
    print("Shape y_epochs:", y_epochs.shape)
    print("Sujetos:", len(set(groups_epochs)))

    X_train, x_val, y_train, y_val, groups_train, groups_val = make_group_shuffle_split(
        x_epochs,
        y_epochs,
        groups_epochs,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    print("Sujetos train:", len(set(groups_train)))
    print("Sujetos val:", len(set(groups_val)))
    print("Solapamiento train/val:", len(set(groups_train) & set(groups_val)))

    X_train = np.asarray(X_train).astype(np.float32)
    x_val = np.asarray(x_val).astype(np.float32)
    y_train = np.asarray(y_train).astype(np.float32)
    y_val = np.asarray(y_val).astype(np.float32)

    model = build_compiled_model(config, X_train.shape[1:])

    history = model.fit(
        X_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=config["n_epochs"],
        batch_size=config["batch_size"],
        callbacks=build_callbacks(config["patience"]),
        verbose=1,
    )

    threshold = float(config.get("threshold_cv_mean", 0.5))
    y_val_score = model.predict(x_val, batch_size=config["batch_size"], verbose=0).ravel()
    y_val_pred = (y_val_score >= threshold).astype(int)

    export_metrics = {
        "model_id": config["model_id"],
        "model_name": config["best_model"],
        "model_family": config["model_family"],
        "selection_metric": config["selection_metric"],
        "threshold": threshold,
        "accuracy_epoch_val": float(accuracy_score(y_val, y_val_pred)),
        "balanced_accuracy_epoch_val": float(balanced_accuracy_score(y_val, y_val_pred)),
        "f1_epoch_val": float(f1_score(y_val, y_val_pred, average="weighted", zero_division=0)),
        "best_epoch": int(np.argmin(history.history["val_loss"])) + 1,
        "val_loss_min": float(np.min(history.history["val_loss"])),
        "cv_metrics": config["cv_metrics"],
        "dataset_summary": config["dataset_summary"],
        "n_epochs_train": int(len(X_train)),
        "n_epochs_val": int(len(x_val)),
    }

    metadata = {
        "model_id": config["model_id"],
        "model_name": config["best_model"],
        "model_family": config["model_family"],
        "sfreq": config["sfreq"],
        "epoch_size": config["epoch_size"],
        "step_size": config["step_size"],
        "apply_filtering": config["apply_filtering"],
        "apply_zscore": config["apply_zscore"],
        "threshold": threshold,
        "channels": list(eeg_cols),
        "label_column": "Class",
        "group_column": "ID",
        "class_mapping": {
            "0": "Control",
            "1": "ADHD",
        },
        "input_shape": list(X_train.shape[1:]),
        "training_strategy": (
            "Modelo DL final entrenado usando el modelo seleccionado mediante "
            "validacion cruzada cross-subject en train_dl.py."
        ),
    }

    print("Guardando modelo DL...")
    model.save(MODEL_PATH)

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(export_metrics, f, indent=4)

    print("\nExportacion DL completada:")
    print(f"- {MODEL_PATH}")
    print(f"- {METADATA_PATH}")
    print(f"- {METRICS_PATH}")


if __name__ == "__main__":
    main()
