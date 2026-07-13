from functools import cached_property, lru_cache
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from backend.core.config import BASE_DIR, MODELS_DIR
from backend.model_registry import catalog
from backend.model_registry.repository import get_trained_model
from backend.modeling.common import (
    map_prediction_label,
    prepare_dl_epochs_from_dataframe,
    prepare_features_from_dataframe,
    validate_eeg_dataframe,
)


# Modelos que puede seleccionar el frontend.
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "ml_best": {
        "model_id": "ml_best",
        "display_name": "Mejor modelo ML",
        "model_family": catalog.MACHINE_LEARNING,
        "description": "Modelo clasico basado en caracteristicas temporales y espectrales.",
        "feature_mode": None,
        "enabled": True,
        "figures": [
            {
                "title": "Comparacion por F1",
                "url": "/figures/cv_model_comparison_f1.png",
            },
            {
                "title": "Comparacion por balanced accuracy",
                "url": "/figures/cv_model_comparison_balanced_accuracy.png",
            },
            {
                "title": "Matriz de confusion ML",
                "url": "/figures/random_forest_cv_confusion_matrix.png",
            },
            {
                "title": "Curva ROC ML",
                "url": "/figures/random_forest_cv_roc_curve.png",
            },
        ],
    },
    "dl_best": {
        "model_id": "dl_best",
        "display_name": "Mejor modelo Deep Learning",
        "model_family": catalog.DEEP_LEARNING,
        "description": "Modelo neuronal entrenado directamente sobre epochs EEG.",
        "feature_mode": "raw_epochs",
        "enabled": True,
        "figures": [
            {
                "title": "Matriz de confusion DL",
                "url": "/figures/cnn_1d_tf_cv_confusion_matrix.png",
            },
            {
                "title": "Curva ROC DL",
                "url": "/figures/cnn_1d_tf_cv_roc_curve.png",
            },
        ],
    },
}


TRAINED_MODEL_PREFIX = "trained_model_"


# Leer metadatos y metricas guardadas al exportar modelos.
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Buscar la configuracion del modelo seleccionado.
def get_model_config(model_id: str) -> dict[str, Any]:
    if model_id.startswith(TRAINED_MODEL_PREFIX):
        model_config = _trained_model_config(model_id)
    else:
        try:
            model_config = MODEL_REGISTRY[model_id]
        except KeyError as exc:
            raise ValueError("Modelo no encontrado.") from exc

    if not model_config.get("enabled", False):
        raise ValueError("Modelo no disponible.")

    return model_config


def _trained_model_config(model_id: str) -> dict[str, Any]:
    trained_model_id = _parse_trained_model_id(model_id)
    trained_model = get_trained_model(trained_model_id)
    if trained_model is None:
        raise ValueError("Modelo no encontrado.")

    artifact_path = _resolve_path(trained_model.artifact_path)
    feature_columns_path = _resolve_optional_path(trained_model.feature_columns_path)
    metadata = dict(trained_model.model_metadata or {})
    if trained_model.threshold is not None and "threshold" not in metadata:
        metadata["threshold"] = float(trained_model.threshold)
    family = catalog.model_family(trained_model.model_name, default=trained_model.model_family)

    return {
        "model_id": model_id,
        "display_name": f"{catalog.display_name(trained_model.model_name)} - experimento #{trained_model.experiment_id}",
        "model_name": trained_model.model_name,
        "model_family": family,
        "description": "Modelo entrenado desde la aplicacion",
        "feature_mode": metadata.get("feature_mode") or _default_feature_mode(family),
        "enabled": artifact_path.exists(),
        "artifact_path": artifact_path,
        "feature_columns_path": feature_columns_path,
        "metadata_path": artifact_path.parent / "metadata.json",
        "metrics_path": artifact_path.parent / "model_metrics.json",
        "metadata": metadata,
        "metrics": None,
    }


def _parse_trained_model_id(model_id: str) -> int:
    suffix = model_id.removeprefix(TRAINED_MODEL_PREFIX)
    try:
        return int(suffix)
    except ValueError as exc:
        raise ValueError("Modelo no encontrado.") from exc


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else BASE_DIR / path


def _resolve_optional_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    return _resolve_path(path_value)


def _default_feature_mode(model_family: str) -> str | None:
    if model_family == "deep_learning":
        return "raw_epochs"
    return None

def list_enabled_models() -> list[dict[str, Any]]:
    return [
        model_config
        for model_config in MODEL_REGISTRY.values()
        if model_config.get("enabled", False)
    ]


def validate_dataframe(df: pd.DataFrame, expected_channels: list[str]) -> dict[str, Any]:
    validate_eeg_dataframe(df, expected_channels)

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "available_channels": [
            channel for channel in expected_channels if channel in df.columns
        ],
        "expected_channels": expected_channels,
        "has_id": "ID" in df.columns,
        "has_class": "Class" in df.columns,
    }


class MLPredictor:
    """Encapsula la carga e inferencia del mejor modelo ML exportado.

    Lee `final_model.joblib`, sus feature_columns y los metadatos guardados
    cuando se entreno. Sirve el `info()` para la pestana Modelo, el
    `validate()` para comprobar el CSV antes de predecir y el `predict()`
    que devuelve la clase final agregando los votos por epoch.
    """

    def __init__(self, model_config: dict[str, Any]):
        self.model_config = model_config
        self.model_dir = MODELS_DIR / "ml"
        self.model_path = _config_path(
            model_config.get("artifact_path"),
            self.model_dir / "final_model.joblib",
        )
        self.feature_columns_path = _config_path(
            model_config.get("feature_columns_path"),
            self.model_dir / "feature_columns.json",
        )
        self.metadata_path = _config_path(
            model_config.get("metadata_path"),
            self.model_dir / "model_metadata.json",
        )
        self.metrics_path = _config_path(
            model_config.get("metrics_path"),
            self.model_dir / "model_metrics.json",
        )
        self.inline_metadata = model_config.get("metadata")
        self.inline_metrics = model_config.get("metrics")

    @cached_property
    def artifacts(self):
        """Carga una vez el modelo ML y sus metadatos."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"No existe el modelo: {self.model_path}")

        if not self.feature_columns_path.exists():
            raise FileNotFoundError(f"No existe feature_columns.json: {self.feature_columns_path}")

        model = joblib.load(self.model_path)
        feature_columns = load_json(self.feature_columns_path)
        metadata = _load_metadata(self.metadata_path, self.inline_metadata)
        metrics = _load_metrics(self.metrics_path, self.inline_metrics)

        return model, feature_columns, metadata, metrics

    def load_artifacts(self):
        """Atajo de lectura del modelo cargado en memoria.

        Existe para que los tests puedan monkeypatchear este metodo y forzar
        artefactos de prueba sin tocar `cached_property`.
        """
        return self.artifacts

    def info(self) -> dict[str, Any]:
        """Devuelve los metadatos y metricas del modelo activo.

        Pintar el panel de info en el frontend sin tener que abrir el modelo,
        solo leer los JSON guardados al exportar.
        """
        _, feature_columns, metadata, metrics = self.load_artifacts()

        return {
            "model_id": self.model_config["model_id"],
            "display_name": self.model_config["display_name"],
            "model_name": metadata.get("model_name"),
            "model_family": self.model_config["model_family"],
            "feature_mode": self.model_config.get("feature_mode") or metadata.get("feature_mode"),
            "sfreq": metadata.get("sfreq"),
            "epoch_size": metadata.get("epoch_size"),
            "step_size": metadata.get("step_size"),
            "channels": metadata.get("channels", []),
            "n_features": len(feature_columns),
            "metrics": metrics,
            "metadata": metadata,
        }

    def validate(self, df: pd.DataFrame) -> dict[str, Any]:
        """Comprueba que el CSV trae los canales que espera este modelo.

        Devuelve cuantos hay, cuales faltan y si las columnas Class/ID estan
        presentes. El frontend usa esto para avisar al usuario antes de
        ejecutar la prediccion.
        """
        _, _, metadata, _ = self.load_artifacts()
        return validate_dataframe(df, metadata.get("channels", []))

    def predict(self, df: pd.DataFrame) -> dict[str, Any]:
        """Predice la clase del paciente a partir de su CSV.

        Extrae features de cada epoch, las pasa por el modelo y agrega los
        votos: la clase mayoritaria gana. La confianza se calcula promediando
        las probabilidades de los epochs que votaron a la clase ganadora; si
        el modelo no expone predict_proba se cae al porcentaje de epochs.
        """
        model, feature_columns, metadata, metrics = self.load_artifacts()
        x_features, _, _, _ = prepare_features_from_dataframe(
            df=df,
            metadata=metadata,
            feature_columns=feature_columns,
        )
        epoch_predictions = model.predict(x_features)
        epoch_probabilities = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(x_features)
            epoch_probabilities = probabilities

        # Prediccion final agregando todas las epochs del archivo.
        unique_preds, pred_counts = np.unique(epoch_predictions, return_counts=True)
        best_idx = int(np.argmax(pred_counts))
        final_prediction = unique_preds[best_idx]
        epoch_count_by_class = {
            map_prediction_label(label): int(count)
            for label, count in zip(unique_preds, pred_counts)
        }
        epoch_percentage_by_class = {
            map_prediction_label(label): float(count / len(epoch_predictions))
            for label, count in zip(unique_preds, pred_counts)
        }
        prediction_label = map_prediction_label(final_prediction)

        final_label_confidences = []
        if epoch_probabilities is not None:
            class_labels = list(model.classes_)
            for index, prediction in enumerate(epoch_predictions):
                if map_prediction_label(prediction) != prediction_label:
                    continue
                class_index = class_labels.index(prediction)
                final_label_confidences.append(
                    float(epoch_probabilities[index][class_index])
                )
        confidence = (
            float(np.mean(final_label_confidences))
            if final_label_confidences
            else epoch_percentage_by_class.get(prediction_label, 0.0)
        )

        return {
            "model_id": self.model_config["model_id"],
            "model_name": metadata.get("model_name"),
            "model_family": self.model_config["model_family"],
            "prediction": str(final_prediction),
            "prediction_label": prediction_label,
            "confidence": confidence,
            "decision_score": confidence,
            "final_class_epoch_percentage": epoch_percentage_by_class.get(
                prediction_label,
                0.0,
            ),
            "n_epochs": int(len(epoch_predictions)),
            "epoch_count_by_class": epoch_count_by_class,
            "epoch_percentage_by_class": epoch_percentage_by_class,
            "metrics": metrics,
            "metadata": metadata,
        }


class DLPredictor:
    """Encapsula la carga e inferencia del mejor modelo DL exportado.

    A diferencia del ML, los modelos Keras pesan mas y reconstruirlos en cada
    request es caro, por eso se cachean en memoria. Predict trabaja sobre
    epochs crudas (no sobre features) y usa el threshold optimo aprendido en
    validacion en vez del 0.5 por defecto.
    """

    def __init__(self, model_config: dict[str, Any]):
        self.model_config = model_config
        self.model_dir = MODELS_DIR / "dl"
        self.model_path = _config_path(
            model_config.get("artifact_path"),
            self.model_dir / "final_model.keras",
        )
        self.metadata_path = _config_path(
            model_config.get("metadata_path"),
            self.model_dir / "model_metadata.json",
        )
        self.metrics_path = _config_path(
            model_config.get("metrics_path"),
            self.model_dir / "model_metrics.json",
        )
        self.inline_metadata = model_config.get("metadata")
        self.inline_metrics = model_config.get("metrics")

    @cached_property
    def artifacts(self):
        """Carga una vez el modelo DL y sus metadatos."""
        import keras

        if not self.model_path.exists():
            raise FileNotFoundError(f"No existe el modelo DL: {self.model_path}")

        model = keras.models.load_model(self.model_path)
        metadata = _load_metadata(self.metadata_path, self.inline_metadata)
        metrics = _load_metrics(self.metrics_path, self.inline_metrics)

        return model, metadata, metrics

    def load_artifacts(self):
        """Atajo de lectura del modelo DL cargado (analogo al ML)."""
        return self.artifacts

    def info(self) -> dict[str, Any]:
        """Devuelve los metadatos y metricas del modelo DL activo.

        Misma intencion que en MLPredictor: alimentar la pestana Modelo. La
        diferencia es que aqui no hay feature_columns porque DL trabaja con
        la senal cruda.
        """
        _, metadata, metrics = self.load_artifacts()

        return {
            "model_id": self.model_config["model_id"],
            "display_name": self.model_config["display_name"],
            "model_name": metadata.get("model_name"),
            "model_family": self.model_config["model_family"],
            "feature_mode": self.model_config.get("feature_mode") or metadata.get("feature_mode"),
            "sfreq": metadata.get("sfreq"),
            "epoch_size": metadata.get("epoch_size"),
            "step_size": metadata.get("step_size"),
            "channels": metadata.get("channels", []),
            "n_features": None,
            "metrics": metrics,
            "metadata": metadata,
        }

    def validate(self, df: pd.DataFrame) -> dict[str, Any]:
        """Comprueba que el CSV trae los canales que espera el modelo DL.

        Mismo contrato que en MLPredictor para que el frontend pueda llamar
        sin distinguir.
        """
        _, metadata, _ = self.load_artifacts()
        return validate_dataframe(df, metadata.get("channels", []))

    def predict(self, df: pd.DataFrame) -> dict[str, Any]:
        """Predice la clase del paciente con el modelo DL.

        Epocha la senal, normaliza por sujeto, pasa cada epoch por la red y
        aplica el threshold optimo guardado. La clase final sale por voto
        mayoritario y la confianza promedia los scores de los epochs que
        votaron a la clase ganadora.
        """
        model, metadata, metrics = self.load_artifacts()
        x_epochs, _, _ = prepare_dl_epochs_from_dataframe(df=df, metadata=metadata)

        threshold = float(metadata.get("threshold", 0.5))
        epoch_scores = model.predict(x_epochs, batch_size=32, verbose=0).ravel()
        epoch_predictions = (epoch_scores >= threshold).astype(int)

        # Prediccion final por voto mayoritario de las ventanas temporales.

        unique_preds, pred_counts = np.unique(epoch_predictions, return_counts=True)
        best_idx = int(np.argmax(pred_counts))
        final_prediction = int(unique_preds[best_idx])
        epoch_count_by_class = {
            map_prediction_label(int(label)): int(count)
            for label, count in zip(unique_preds, pred_counts)
        }
        epoch_percentage_by_class = {
            map_prediction_label(int(label)): float(count / len(epoch_predictions))
            for label, count in zip(unique_preds, pred_counts)
        }
        prediction_label = map_prediction_label(final_prediction)

        final_label_confidences = [
            float(score) if int(prediction) == 1 else 1.0 - float(score)
            for score, prediction in zip(epoch_scores, epoch_predictions)
            if map_prediction_label(int(prediction)) == prediction_label
        ]
        confidence = (
            float(np.mean(final_label_confidences))
            if final_label_confidences
            else epoch_percentage_by_class.get(prediction_label, 0.0)
        )

        return {
            "model_id": self.model_config["model_id"],
            "model_name": metadata.get("model_name"),
            "model_family": self.model_config["model_family"],
            "prediction": str(final_prediction),
            "prediction_label": prediction_label,
            "confidence": float(confidence),
            "decision_score": float(confidence),
            "final_class_epoch_percentage": epoch_percentage_by_class.get(
                prediction_label,
                0.0,
            ),
            "threshold": threshold,
            "n_epochs": int(len(epoch_predictions)),
            "epoch_count_by_class": epoch_count_by_class,
            "epoch_percentage_by_class": epoch_percentage_by_class,
            "metrics": metrics,
            "metadata": metadata,
        }


def _config_path(path_value: str | Path | None, default: Path) -> Path:
    if path_value is None:
        return default
    return Path(path_value)


def _load_metadata(metadata_path: Path, inline_metadata: dict[str, Any] | None) -> dict[str, Any]:
    if metadata_path.exists():
        return load_json(metadata_path)
    if inline_metadata:
        return dict(inline_metadata)
    raise FileNotFoundError(f"No existe metadata.json: {metadata_path}")


def _load_metrics(metrics_path: Path, inline_metrics: dict[str, Any] | None) -> dict[str, Any] | None:
    if metrics_path.exists():
        return load_json(metrics_path)
    return inline_metrics


@lru_cache(maxsize=None)
def get_predictor(model_id: str):
    model_config = get_model_config(model_id)
    model_family = model_config.get("model_family")

    if model_family == "machine_learning":
        return MLPredictor(model_config)

    if model_family == "deep_learning":
        return DLPredictor(model_config)

    raise ValueError("Modelo no encontrado.")
