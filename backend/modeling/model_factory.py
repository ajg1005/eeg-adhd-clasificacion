from typing import Any

from scripts.ml_model_registry import ML_MODEL_OPTIONS, merged_ml_params


# Wrapper UI: valida que el modelo esta expuesto en la UI, mezcla los defaults
# del registro compartido con los params del usuario y delega en scripts.pipeline.
def create_ml_model(model_name: str, params: dict[str, Any] | None = None):
    from scripts.pipeline import create_ml_model as _build_ml_model

    if model_name not in ML_MODEL_OPTIONS:
        raise ValueError(f"Modelo ML no disponible en la UI: {model_name}")

    merged = merged_ml_params(model_name, params)
    return _build_ml_model(model_name, merged)
