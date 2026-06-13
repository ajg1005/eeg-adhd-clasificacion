import pandas as pd

from backend.modeling.predictors import (
    get_model_config,
    get_predictor,
    list_enabled_models,
)


# Listar modelos disponibles para el selector
def list_models() -> list[dict]:
    """Devuelve los modelos exportados que el frontend puede ofrecer al usuario.

    Filtra el registro de predictores y solo devuelve los habilitados (los que
    tienen artefactos en disco). Lo usa el desplegable de la pestaña de
    inferencia.
    """
    return list_enabled_models()


# Devolver informacion y metricas del modelo seleccionado
def get_model_info(model_id: str) -> dict:
    """Devuelve metadatos, metricas y configuracion del modelo activo.

    Lo usa la pestaña Modelo para mostrar al usuario que es lo que tiene
    cargado (tipo, hiperparametros, accuracy/F1 de validacion).
    """
    return get_predictor(model_id).info()


def validate_dataframe_for_model(df: pd.DataFrame, model_id: str) -> dict:
    """Comprueba que el CSV del paciente es compatible con el modelo.

    Verifica que estan los 19 canales 10-20, las dimensiones encajan con la
    ventana que espera el modelo y no hay columnas raras. Se ejecuta antes
    de predecir para dar feedback inmediato al usuario.
    """
    return get_predictor(model_id).validate(df)


# Ejecutar inferencia con el predictor elegido
def predict_dataframe(df: pd.DataFrame, model_id: str) -> dict:
    """Lanza la prediccion sobre un CSV ya validado.

    Internamente epocha la señal, ejecuta el modelo en cada epoch y agrega
    los votos para devolver una clase final (ADHD/Control) con su nivel de
    confianza y el reparto de epochs por clase.
    """
    return get_predictor(model_id).predict(df)


def get_model_figures(model_id: str) -> list[dict]:
    """Devuelve las figuras de evaluacion que el frontend tiene que renderizar.

    Las rutas apuntan a `/figures/...` servidas como estaticos por FastAPI.
    Suelen ser matriz de confusion, ROC y, en DL, curvas de entrenamiento.
    """
    return get_model_config(model_id).get("figures", [])
