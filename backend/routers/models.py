from fastapi import APIRouter, HTTPException

from backend.routers.responses import MODEL_INFO_RESPONSES, NOT_FOUND_RESPONSES
from backend.schemas import FiguresResponse, ModelCatalogResponse, ModelInfoResponse, ModelsResponse
from backend.services.model_service import (
    get_model_figures,
    get_model_info,
    get_training_model_catalog,
    list_models,
)


router = APIRouter()


# Listar modelos disponibles
@router.get("/models", response_model=ModelsResponse)
def list_available_models():
    return {"models": list_models()}


# Listar modelos candidatos y parametros habituales de entrenamiento
@router.get("/model/catalog", response_model=ModelCatalogResponse)
def model_catalog():
    return get_training_model_catalog()


# Devolver informacion del modelo cargado
@router.get("/model/info", response_model=ModelInfoResponse, responses=MODEL_INFO_RESPONSES)
def model_info(model_id: str = "ml_best"):
    try:
        return get_model_info(model_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/model/figures", response_model=FiguresResponse, responses=NOT_FOUND_RESPONSES)
def model_figures(model_id: str = "ml_best"):
    try:
        return {"figures": get_model_figures(model_id)}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
