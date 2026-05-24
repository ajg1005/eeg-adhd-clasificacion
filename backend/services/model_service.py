import pandas as pd

from backend.modeling.model_catalog import get_model_catalog
from backend.modeling.predictors import (
    get_model_config,
    get_predictor,
    list_enabled_models,
)


# Listar modelos disponibles para el selector
def list_models() -> list[dict]:
    return list_enabled_models()


# Devolver modelos candidatos y parametros comunes para entrenamiento
def get_training_model_catalog() -> dict:
    return get_model_catalog()


# Devolver informacion y metricas del modelo seleccionado
def get_model_info(model_id: str) -> dict:
    return get_predictor(model_id).info()


def validate_dataframe_for_model(df: pd.DataFrame, model_id: str) -> dict:
    return get_predictor(model_id).validate(df)


# Ejecutar inferencia con el predictor elegido
def predict_dataframe(df: pd.DataFrame, model_id: str) -> dict:
    return get_predictor(model_id).predict(df)


def get_model_figures(model_id: str) -> list[dict]:
    return get_model_config(model_id).get("figures", [])
