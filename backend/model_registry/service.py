from pathlib import Path

from backend.config import BASE_DIR
from backend.model_registry import catalog, repository
from backend.inference.predictors import (
    get_model_config,
    get_predictor,
    list_enabled_models,
)


# Listar modelos disponibles para el selector
def list_models() -> list[dict]:
    """Devuelve los modelos base y los entrenados disponibles para inferencia."""
    return [
        *list_enabled_models(),
        *[_trained_model_item(model) for model in repository.list_trained_models()],
    ]


def _trained_model_item(model) -> dict:
    artifact_path = _resolve_path(model.artifact_path)
    display_name = f"{catalog.display_name(model.model_name)} - experimento #{model.experiment_id}"

    return {
        "model_id": f"trained_model_{model.id}",
        "display_name": display_name,
        "model_family": catalog.model_family(model.model_name, default=model.model_family),
        "description": "Modelo entrenado desde la aplicacion",
        "enabled": artifact_path.exists(),
    }


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else BASE_DIR / path


# Devolver informacion y metricas del modelo seleccionado
def get_model_info(model_id: str) -> dict:
    """Devuelve metadatos, metricas y configuracion del modelo activo.

    Lo usa la pestana Modelo para mostrar al usuario que es lo que tiene
    cargado: tipo, hiperparametros y metricas de validacion.
    """
    return get_predictor(model_id).info()


def get_model_figures(model_id: str) -> list[dict]:
    """Devuelve las figuras de evaluacion que el frontend tiene que renderizar."""
    return get_model_config(model_id).get("figures", [])
