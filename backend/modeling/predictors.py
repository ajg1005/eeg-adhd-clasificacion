from functools import cached_property, lru_cache
import json
from typing import Any

import joblib
import numpy as np
import pandas as pd

from backend.config import MODELS_DIR
from backend.modeling.common import (
    map_prediction_label,
    prepare_dl_epochs_from_dataframe,
    prepare_features_from_dataframe,
    validate_eeg_dataframe,
)


# Modelos que puede seleccionar el frontend
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "ml_best": {
        "model_id": "ml_best",
        "display_name": "Mejor modelo ML",
        "model_family": "machine_learning",
        "description": "Modelo clasico basado en features temporales y espectrales.",
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
        "model_family": "deep_learning",
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


# Leer metadatos y metricas guardadas al exportar modelos
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Buscar la configuracion del modelo seleccionado
def get_model_config(model_id: str) -> dict[str, Any]:
    try:
        model_config = MODEL_REGISTRY[model_id]
    except KeyError as exc:
        raise ValueError("Modelo no encontrado.") from exc

    if not model_config.get("enabled", False):
        raise ValueError("Modelo no disponible.")

    return model_config


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
    def __init__(self, model_config: dict[str, Any]):
        self.model_config = model_config
        self.model_dir = MODELS_DIR / "ml"
        self.model_path = self.model_dir / "final_model.joblib"
        self.feature_columns_path = self.model_dir / "feature_columns.json"
        self.metadata_path = self.model_dir / "model_metadata.json"
        self.metrics_path = self.model_dir / "model_metrics.json"

    @cached_property
    def artifacts(self):
        """Carga una vez el modelo ML y sus metadatos.

        cached_property evita leer los artefactos desde disco en cada llamada a
        /model/info, /validate o /predict.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"No existe el modelo: {self.model_path}")

        if not self.feature_columns_path.exists():
            raise FileNotFoundError(f"No existe feature_columns.json: {self.feature_columns_path}")

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"No existe model_metadata.json: {self.metadata_path}")

        model = joblib.load(self.model_path)
        feature_columns = load_json(self.feature_columns_path)
        metadata = load_json(self.metadata_path)
        metrics = load_json(self.metrics_path) if self.metrics_path.exists() else None

        return model, feature_columns, metadata, metrics

    def load_artifacts(self):
        return self.artifacts

    def info(self) -> dict[str, Any]:
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
        _, _, metadata, _ = self.load_artifacts()
        return validate_dataframe(df, metadata.get("channels", []))

    def predict(self, df: pd.DataFrame) -> dict[str, Any]:
        # Prediccion final agregando todas las epochs del archivo
        model, feature_columns, metadata, metrics = self.load_artifacts()
        x_features, _, _, _ = prepare_features_from_dataframe(
            df=df,
            metadata=metadata,
            feature_columns=feature_columns,
        )
        epoch_predictions = model.predict(x_features)

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(x_features)
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
        prediction_label = map_prediction_label(final_prediction)

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
    def __init__(self, model_config: dict[str, Any]):
        self.model_config = model_config
        self.model_dir = MODELS_DIR / "dl"
        self.model_path = self.model_dir / "final_model.keras"
        self.metadata_path = self.model_dir / "model_metadata.json"
        self.metrics_path = self.model_dir / "model_metrics.json"

    @cached_property
    def artifacts(self):
        """Carga una vez el modelo DL y sus metadatos.

        Los modelos de Keras son mas costosos de reconstruir desde disco, por
        lo que se mantienen en memoria durante la vida del predictor.
        """
        import keras

        if not self.model_path.exists():
            raise FileNotFoundError(f"No existe el modelo DL: {self.model_path}")

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"No existe model_metadata.json: {self.metadata_path}")

        model = keras.models.load_model(self.model_path)
        metadata = load_json(self.metadata_path)
        metrics = load_json(self.metrics_path) if self.metrics_path.exists() else None

        return model, metadata, metrics

    def load_artifacts(self):
        return self.artifacts

    def info(self) -> dict[str, Any]:
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
        _, metadata, _ = self.load_artifacts()
        return validate_dataframe(df, metadata.get("channels", []))

    def predict(self, df: pd.DataFrame) -> dict[str, Any]:
        # Prediccion DL agregando la probabilidad media de las epochs
        model, metadata, metrics = self.load_artifacts()
        x_epochs, _, _ = prepare_dl_epochs_from_dataframe(df=df, metadata=metadata)

        threshold = float(metadata.get("threshold", 0.5))
        epoch_scores = model.predict(x_epochs, batch_size=32, verbose=0).ravel()
        epoch_predictions = (epoch_scores >= threshold).astype(int)
        mean_score = float(np.mean(epoch_scores))
        final_prediction = int(mean_score >= threshold)
        confidence = mean_score if final_prediction == 1 else 1.0 - mean_score

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


@lru_cache(maxsize=None)
def get_predictor(model_id: str):
    model_config = get_model_config(model_id)

    if model_id == "ml_best":
        return MLPredictor(model_config)

    if model_id == "dl_best":
        return DLPredictor(model_config)

    raise ValueError("Modelo no encontrado.")
