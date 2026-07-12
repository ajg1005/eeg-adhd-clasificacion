"""Serializa y registra el modelo final de un entrenamiento interactivo.

La evaluacion cross-subject (training_runners) entrena un modelo por fold solo
para medir y los descarta, asi que al terminar no queda ningun modelo reutilizable
en memoria. Este modulo cierra ese hueco: reentrena un modelo final sobre el
dataset (mismo criterio que scripts/export_model.py) y lo vuelca a disco junto con
los metadatos minimos para poder cargarlo despues en inferencia. Lo consume
run_training en backend/training/service.py tras persistir el experimento.

El binario pesado (.joblib / .keras) vive en disco bajo TRAINED_MODELS_DIR; en la BD
solo se guarda la ruta y los metadatos (ver model_registry.repository.save_trained_model).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

from backend.config import BASE_DIR, TRAINED_MODELS_DIR
from backend.model_registry import catalog
from backend.modeling.dl_factory import create_dl_model
from backend.modeling.model_factory import create_ml_model
from backend.training.data import PreparedEpochs, features_for_mode
from backend.training.runners import (
    _dl_callbacks,
    _force_single_thread,
    _release_keras_model,
)
from scripts.constants import RANDOM_STATE
from scripts.evaluation import find_best_threshold
from scripts.split import make_group_shuffle_split


def persist_final_model(
    experiment_id: int,
    model_type: str,
    model_name: str,
    eeg_params: dict[str, Any],
    model_params: dict[str, Any],
    training_params: dict[str, Any],
    prepared: PreparedEpochs,
) -> dict[str, Any]:
    """Reentrena el modelo final sobre todo el dataset y persiste sus artefactos.

    Delega en la variante ML o DL segun model_type. Devuelve el dict que
    model_registry.repository.save_trained_model insertara en la tabla trained_models (rutas
    relativas, metadatos y tamaño del artefacto). No lanza si el fit es costoso:
    el llamador (run_training) ya envuelve la llamada en try/except.
    """
    if model_type == "ml":
        return _persist_final_ml_model(
            experiment_id,
            model_name,
            eeg_params,
            model_params,
            prepared,
        )
    if model_type == "dl":
        return _persist_final_dl_model(
            experiment_id,
            model_name,
            eeg_params,
            model_params,
            training_params,
            prepared,
        )
    raise ValueError("model_type debe ser 'ml' o 'dl'.")


def _persist_final_ml_model(
    experiment_id: int,
    model_name: str,
    eeg_params: dict[str, Any],
    model_params: dict[str, Any],
    prepared: PreparedEpochs,
) -> dict[str, Any]:
    """Ajusta el pipeline ML sobre todos los epochs y guarda joblib + sidecars.

    Escribe tres ficheros en la carpeta del experimento: el modelo (model.joblib),
    las columnas de features (feature_columns.json) y los metadatos (metadata.json).
    feature_columns es imprescindible para reconstruir las features en el mismo
    orden al inferir; por eso se persiste junto al binario.
    """
    output_dir = _artifact_dir(experiment_id)
    features = features_for_mode(prepared.x_epochs, prepared.eeg_columns, eeg_params)
    model = create_ml_model(model_name, model_params)
    # Un solo hilo: la API puede atender varias peticiones a la vez y no queremos
    # que un fit acapare todos los nucleos. Mismo criterio que en training_runners.
    _force_single_thread(model)

    fit_kwargs = {}
    if model_name == "xgboost":
        fit_kwargs["model__sample_weight"] = compute_sample_weight(
            "balanced",
            prepared.y_epochs,
        )
    model.fit(features, prepared.y_epochs, **fit_kwargs)

    artifact_path = output_dir / "model.joblib"
    feature_columns_path = output_dir / "feature_columns.json"
    metadata_path = output_dir / "metadata.json"

    joblib.dump(model, artifact_path)
    _write_json(feature_columns_path, list(features.columns))

    metadata = _base_metadata(
        model_type="ml",
        model_name=model_name,
        eeg_params=eeg_params,
        prepared=prepared,
        n_features=int(features.shape[1]),
    )
    _write_json(metadata_path, metadata)

    return _record(
        model_type="ml",
        model_name=model_name,
        artifact_path=artifact_path,
        feature_columns_path=feature_columns_path,
        n_features=int(features.shape[1]),
        n_epochs_training=int(len(features)),
        n_subjects_training=int(len(set(prepared.groups_epochs))),
        threshold=None,
        metadata=metadata,
    )


def _persist_final_dl_model(
    experiment_id: int,
    model_name: str,
    eeg_params: dict[str, Any],
    model_params: dict[str, Any],
    training_params: dict[str, Any],
    prepared: PreparedEpochs,
) -> dict[str, Any]:
    """Entrena la red final y guarda model.keras + metadata.json (con umbral).

    Reserva un split interno cross-subject de validacion (sin solape de pacientes)
    para calcular el umbral de decision con find_best_threshold; ese umbral se
    guarda en los metadatos porque sin el la red no sabe donde cortar al predecir.
    Libera la sesion de Keras al terminar para no retener el grafo en memoria.
    """
    output_dir = _artifact_dir(experiment_id)
    x_train, x_val, y_train, y_val, groups_train, _ = make_group_shuffle_split(
        prepared.x_epochs,
        prepared.y_epochs,
        prepared.groups_epochs,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    x_train = np.asarray(x_train).astype(np.float32)
    x_val = np.asarray(x_val).astype(np.float32)
    y_train = np.asarray(y_train).astype(np.float32)
    y_val = np.asarray(y_val).astype(np.float32)

    model = None
    try:
        model = create_dl_model(
            model_name=model_name,
            input_shape=x_train.shape[1:],
            model_params=model_params,
            training_params=training_params,
        )
        classes = np.unique(y_train.astype(int))
        class_weights = compute_class_weight(
            "balanced",
            classes=classes,
            y=y_train.astype(int),
        )
        class_weight_dict = {
            int(cls): float(weight) for cls, weight in zip(classes, class_weights)
        }
        batch_size = int(training_params.get("batch_size", 32))
        model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=int(training_params.get("epochs", 40)),
            batch_size=batch_size,
            callbacks=_dl_callbacks(training_params),
            class_weight=class_weight_dict,
            verbose=0,
        )
        y_val_score = model.predict(x_val, batch_size=batch_size, verbose=0).reshape(-1)
        threshold = float(find_best_threshold(y_val.astype(int), y_val_score))

        artifact_path = output_dir / "model.keras"
        metadata_path = output_dir / "metadata.json"
        model.save(artifact_path)

        metadata = _base_metadata(
            model_type="dl",
            model_name=model_name,
            eeg_params=eeg_params,
            prepared=prepared,
            n_features=None,
        )
        metadata.update(
            {
                "threshold": threshold,
                "input_shape": list(x_train.shape[1:]),
                "n_epochs_training": int(len(x_train)),
                "n_subjects_training": int(len(set(groups_train))),
            }
        )
        _write_json(metadata_path, metadata)

        return _record(
            model_type="dl",
            model_name=model_name,
            artifact_path=artifact_path,
            feature_columns_path=None,
            n_features=None,
            n_epochs_training=int(len(x_train)),
            n_subjects_training=int(len(set(groups_train))),
            threshold=threshold,
            metadata=metadata,
        )
    finally:
        _release_keras_model(model)


def _base_metadata(
    model_type: str,
    model_name: str,
    eeg_params: dict[str, Any],
    prepared: PreparedEpochs,
    n_features: int | None,
) -> dict[str, Any]:
    """Construye el metadata.json con la misma forma que scripts/export_model.py.

    Mantener el mismo esquema (canales, sfreq, ventana, class_mapping...) permite
    que los predictores existentes carguen estos modelos sin cambios en la futura
    fase de inferencia.
    """
    metadata = {
        "model_name": model_name,
        "model_family": catalog.model_family(model_name),
        "sfreq": int(eeg_params.get("sfreq", 128)),
        "epoch_size": int(eeg_params.get("epoch_size", 1920)),
        "step_size": int(eeg_params.get("step_size", 960)),
        "apply_filtering": bool(eeg_params.get("use_filtering", False)),
        "apply_zscore": bool(eeg_params.get("use_filtering", False)),
        "channels": list(prepared.eeg_columns),
        "label_column": "Class",
        "group_column": "ID",
        "class_mapping": {
            "0": "Control",
            "1": "ADHD",
        },
        "n_epochs_training": int(len(prepared.y_epochs)),
        "n_subjects_training": int(len(set(prepared.groups_epochs))),
        "training_strategy": "Modelo final entrenado tras evaluacion cross-subject desde la API.",
    }
    if model_type == "ml":
        metadata.update(
            {
                "feature_mode": eeg_params.get("feature_mode", "combined"),
                "nperseg": int(eeg_params.get("nperseg", 960)),
                "n_features": n_features,
            }
        )
    return metadata


def _record(
    model_type: str,
    model_name: str,
    artifact_path: Path,
    feature_columns_path: Path | None,
    n_features: int | None,
    n_epochs_training: int,
    n_subjects_training: int,
    threshold: float | None,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_type": model_type,
        "model_name": model_name,
        "model_family": catalog.model_family(model_name),
        "artifact_path": _relative_path(artifact_path),
        "feature_columns_path": (
            _relative_path(feature_columns_path) if feature_columns_path is not None else None
        ),
        "n_features": n_features,
        "n_epochs_training": n_epochs_training,
        "n_subjects_training": n_subjects_training,
        "file_size_bytes": artifact_path.stat().st_size if artifact_path.exists() else None,
        "threshold": threshold,
        "model_metadata": metadata,
        "is_selected": False,
    }


def _artifact_dir(experiment_id: int) -> Path:
    output_dir = TRAINED_MODELS_DIR / str(experiment_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(BASE_DIR)).replace("\\", "/")
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4)