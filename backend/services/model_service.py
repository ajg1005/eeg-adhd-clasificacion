from typing import Any

import pandas as pd

from backend.config import configure_import_paths
from backend.model_registry import get_model_config, list_enabled_models

configure_import_paths()

from inference import (  # noqa: E402
    load_model_artifacts,
    predict_eeg_dataframe,
    validate_eeg_dataframe,
)


def list_models() -> list[dict]:
    return list_enabled_models()


def _load_dl_artifacts():
    from inference_dl import load_dl_model_artifacts

    return load_dl_model_artifacts()


def _predict_dl_dataframe(df: pd.DataFrame) -> dict:
    from inference_dl import predict_eeg_dataframe_dl

    return predict_eeg_dataframe_dl(df)


def get_model_metadata(model_id: str) -> tuple[dict[str, Any], list[str], Any]:
    if model_id == "ml_best":
        _, feature_columns, metadata, metrics = load_model_artifacts()
        return metadata, feature_columns, metrics

    if model_id == "dl_best":
        _, metadata, metrics = _load_dl_artifacts()
        return metadata, [], metrics

    get_model_config(model_id)
    raise ValueError("Modelo no encontrado.")


def get_model_info(model_id: str) -> dict:
    model_config = get_model_config(model_id)
    metadata, feature_columns, metrics = get_model_metadata(model_id)

    return {
        "model_id": model_id,
        "display_name": model_config["display_name"],
        "model_name": metadata.get("model_name"),
        "model_family": model_config["model_family"],
        "feature_mode": model_config.get("feature_mode") or metadata.get("feature_mode"),
        "sfreq": metadata.get("sfreq"),
        "epoch_size": metadata.get("epoch_size"),
        "step_size": metadata.get("step_size"),
        "channels": metadata.get("channels", []),
        "n_features": len(feature_columns) if feature_columns else None,
        "metrics": metrics,
        "metadata": metadata,
    }


def validate_dataframe_for_model(df: pd.DataFrame, model_id: str) -> dict:
    metadata, _, _ = get_model_metadata(model_id)
    expected_channels = metadata.get("channels", [])

    validate_eeg_dataframe(df, expected_channels)

    available_channels = [
        channel for channel in expected_channels
        if channel in df.columns
    ]

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "available_channels": available_channels,
        "expected_channels": expected_channels,
        "has_id": "ID" in df.columns,
        "has_class": "Class" in df.columns,
    }


def predict_dataframe(df: pd.DataFrame, model_id: str) -> dict:
    model_config = get_model_config(model_id)

    if model_id == "ml_best":
        result = predict_eeg_dataframe(df)
        result["model_id"] = model_id
        result["model_family"] = model_config["model_family"]
        result["model_name"] = result["metadata"].get("model_name")
        return result

    if model_id == "dl_best":
        result = _predict_dl_dataframe(df)
        result["model_id"] = model_id
        result["model_family"] = model_config["model_family"]
        result["model_name"] = result.get("model_name") or result["metadata"].get("model_name")
        return result

    raise ValueError("Modelo no encontrado.")


def get_model_figures(model_id: str) -> list[dict]:
    return get_model_config(model_id).get("figures", [])
