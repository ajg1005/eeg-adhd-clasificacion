from fastapi import APIRouter, HTTPException

from backend.model_registry import service
from backend.model_registry.schemas import (
    BestAvailableModelResponse,
    FiguresResponse,
    ModelInfoResponse,
    ModelsResponse,
)
from backend.api.responses import MODEL_INFO_RESPONSES, NOT_FOUND_RESPONSES


router = APIRouter()


# Listar modelos disponibles
@router.get("/models", response_model=ModelsResponse)
def list_available_models():
    """Devuelve los modelos de inferencia disponibles en el selector."""
    return {"models": service.list_models()}


@router.get("/models/best", response_model=BestAvailableModelResponse | None)
def best_available_model():
    """Devuelve el mejor modelo entrenado que puede utilizarse en inferencia."""
    return service.get_best_available_model()


# Devolver informacion del modelo cargado
@router.get("/model/info", response_model=ModelInfoResponse, responses=MODEL_INFO_RESPONSES)
def model_info(model_id: str = "ml_best"):
    """Devuelve metadatos y metricas de validacion de un modelo exportado."""
    try:
        return service.get_model_info(model_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/model/figures", response_model=FiguresResponse, responses=NOT_FOUND_RESPONSES)
def model_figures(model_id: str = "ml_best"):
    """Devuelve las figuras de diagnostico asociadas a un modelo."""
    try:
        return {"figures": service.get_model_figures(model_id)}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
