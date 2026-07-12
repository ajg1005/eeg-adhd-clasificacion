from __future__ import annotations

import logging
import time
from typing import Any

from sklearn.metrics import classification_report, confusion_matrix

from backend.modeling.dl_factory import DL_MODEL_OPTIONS
from backend.modeling.model_factory import ML_MODEL_OPTIONS
from backend.datasets.service import read_csv, validate_training_dataframe
from backend.db.repository import save_experiment
from backend.model_registry import catalog
from backend.model_registry.repository import save_trained_model
from backend.training.persistence import persist_final_model
from backend.training.data import prepare_epochs
from backend.training.runners import (
    metrics_dict,
    patient_results,
    run_dl_cross_subject_cv,
    run_ml_cross_subject_cv,
)


logger = logging.getLogger(__name__)


DEFAULT_EEG_PARAMS = {
    "ml": {
        "epoch_size": 1920,
        "step_size": 960,
        "sfreq": 128,
        "nperseg": 960,
        "feature_mode": "combined",
        "use_filtering": False,
    },
    "dl": {
        "epoch_size": 512,
        "step_size": 256,
        "sfreq": 128,
        "use_filtering": True,
    },
}
DEFAULT_MODEL_TYPE = "ml"
DEFAULT_MODELS = {"ml": "xgboost", "dl": "cnn_1d"}
EEG_PARAMS_BY_TYPE = {
    "ml": ["epoch_size", "step_size", "sfreq", "nperseg", "feature_mode", "use_filtering"],
    "dl": ["epoch_size", "step_size", "sfreq", "use_filtering"],
}
DEFAULT_TRAINING_PARAMS = {
    "epochs": 40,
    "batch_size": 32,
    "learning_rate": 0.0003,
    "early_stopping_patience": 4,
}
TRAINING_PARAMS_BY_TYPE = {
    "ml": [],
    "dl": ["epochs", "batch_size", "learning_rate", "early_stopping_patience"],
}


def _models_for_ui(models: dict[str, Any]) -> dict[str, Any]:
    """Sobrescribe el display_name de cada modelo con el del catalogo unico."""
    return {
        name: {**spec, "display_name": catalog.display_name(name)}
        for name, spec in models.items()
    }


def get_training_options() -> dict[str, Any]:
    """Devuelve los modelos disponibles y los rangos de parametros validos.

    Lo consume el frontend para construir los desplegables y validar los
    valores antes de mandar el entrenamiento. Centralizar aqui los rangos
    evita duplicar listas en JS.
    """
    return {
        "default_model_type": DEFAULT_MODEL_TYPE,
        "default_models": DEFAULT_MODELS,
        "default_eeg_params": DEFAULT_EEG_PARAMS,
        "eeg_params_by_type": EEG_PARAMS_BY_TYPE,
        "default_training_params": DEFAULT_TRAINING_PARAMS,
        "training_params_by_type": TRAINING_PARAMS_BY_TYPE,
        "model_types": {
            "ml": {"display_name": "Machine Learning", "models": _models_for_ui(ML_MODEL_OPTIONS)},
            "dl": {"display_name": "Deep Learning", "models": _models_for_ui(DL_MODEL_OPTIONS)},
        },
        "eeg_params": {
            "epoch_size": [512, 640, 1280, 1920],
            "step_size": [256, 320, 640, 960],
            "sfreq": [128],
            "nperseg": [256, 320, 640, 960],
            "feature_mode": ["temporal", "spectral", "combined"],
            "use_filtering": [False, True],
        },
        "training_params": {
            "epochs": [10, 25, 40, 50],
            "batch_size": [16, 32, 64],
            "learning_rate": [0.001, 0.0005, 0.0003, 0.0001],
            "early_stopping_patience": [3, 4, 5, 8],
        },
    }


def run_training(
    file_bytes: bytes,
    model_type: str,
    model_name: str,
    filename: str = "training.csv",
    eeg_params: dict[str, Any] | None = None,
    model_params: dict[str, Any] | None = None,
    training_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Lanza un entrenamiento cross-subject completo y persiste el experimento.

    Lee el CSV, lo valida, aplica StratifiedGroupKFold para no mezclar pacientes
    entre folds, entrena el modelo elegido (ML o DL) y agrega las metricas. Al
    terminar guarda el experimento en BD; si la BD falla se devuelven igualmente
    las metricas para que el usuario no pierda el resultado.
    """
    started_at = time.perf_counter()
    eeg_params = _merge_default_eeg_params(model_type, eeg_params or {})
    model_params = model_params or {}
    training_params = training_params or {}

    df = read_csv(file_bytes)
    validate_training_dataframe(df)
    prepared = prepare_epochs(df, eeg_params)
    evaluation = _evaluate_model(model_type, model_name, eeg_params, model_params, training_params, prepared)

    y_true = evaluation["y_true"]
    y_pred = evaluation["y_pred"]
    metrics = metrics_dict(y_true, y_pred)

    result = {
        **metrics,
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=["Control", "ADHD"],
            labels=[0, 1],
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "patient_results": patient_results(evaluation["groups"], y_true, y_pred),
        "fold_results": evaluation["fold_results"],
        "feature_importance": evaluation.get("feature_importance"),
        "configuration": {
            "model_type": model_type,
            "model_name": model_name,
            "eeg_params": eeg_params,
            "model_params": model_params,
            "training_params": training_params,
            "evaluation_mode": evaluation["evaluation_mode"],
        },
        "training_time_seconds": round(time.perf_counter() - started_at, 3),
        "trained_model_id": None,
        "model_saved": False,
    }
    # Si falla la BD, se devuelven las metricas igualmente.
    try:
        result["experiment_id"] = save_experiment(
            file_bytes=file_bytes,
            filename=filename,
            dataframe=df,
            result=result,
        )
    except Exception:
        logger.exception("No se pudo persistir el experimento; se devuelven las metricas igualmente.")
        result["experiment_id"] = None
        result["persisted"] = False
    else:
        result["persisted"] = True
        try:
            model_record = persist_final_model(
                experiment_id=result["experiment_id"],
                model_type=model_type,
                model_name=model_name,
                eeg_params=eeg_params,
                model_params=model_params,
                training_params=training_params,
                prepared=prepared,
            )
            result["trained_model_id"] = save_trained_model(
                experiment_id=result["experiment_id"],
                record=model_record,
            )
            result["model_saved"] = True
        except Exception:
            logger.exception("No se pudo persistir el modelo final; se devuelven las metricas igualmente.")
            result["trained_model_id"] = None
            result["model_saved"] = False
    return result


def _merge_default_eeg_params(model_type: str, eeg_params: dict[str, Any]) -> dict[str, Any]:
    if model_type not in DEFAULT_EEG_PARAMS:
        raise ValueError("model_type debe ser 'ml' o 'dl'.")
    return {**DEFAULT_EEG_PARAMS[model_type], **eeg_params}


def _evaluate_model(
    model_type: str,
    model_name: str,
    eeg_params: dict[str, Any],
    model_params: dict[str, Any],
    training_params: dict[str, Any],
    prepared,
) -> dict[str, Any]:
    if model_type == "ml":
        return run_ml_cross_subject_cv(model_name, model_params, eeg_params, prepared)
    return run_dl_cross_subject_cv(model_name, model_params, training_params, prepared)
