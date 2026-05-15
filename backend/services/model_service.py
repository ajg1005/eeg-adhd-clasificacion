import pandas as pd

from backend.modeling.predictors import (
    get_model_config,
    get_predictor,
    list_enabled_models,
)


def list_models() -> list[dict]:
    return list_enabled_models()


def get_model_info(model_id: str) -> dict:
    return get_predictor(model_id).info()


def get_model_metadata(model_id: str) -> tuple[dict, list, dict | None]:
    info = get_model_info(model_id)
    n_features = info.get("n_features")
    feature_columns = [None] * n_features if n_features else []

    return info["metadata"], feature_columns, info.get("metrics")


def validate_dataframe_for_model(df: pd.DataFrame, model_id: str) -> dict:
    return get_predictor(model_id).validate(df)


def predict_dataframe(df: pd.DataFrame, model_id: str) -> dict:
    return get_predictor(model_id).predict(df)


def get_model_figures(model_id: str) -> list[dict]:
    return get_model_config(model_id).get("figures", [])
