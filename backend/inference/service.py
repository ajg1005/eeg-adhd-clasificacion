import pandas as pd

from backend.inference.predictors import get_predictor


def validate_dataframe_for_model(
    df: pd.DataFrame,
    model_id: str,
    owner_id: int,
) -> dict:
    """Comprueba que el CSV del paciente es compatible con el modelo."""
    return get_predictor(model_id, owner_id).validate(df)


def predict_dataframe(df: pd.DataFrame, model_id: str, owner_id: int) -> dict:
    """Ejecuta la prediccion sobre un CSV previamente validado."""
    return get_predictor(model_id, owner_id).predict(df)
