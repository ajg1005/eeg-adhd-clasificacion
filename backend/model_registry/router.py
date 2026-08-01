from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from backend.api.responses import MODEL_INFO_RESPONSES, NOT_FOUND_RESPONSES
from backend.auth.dependencies import get_current_user
from backend.db.models import User
from backend.model_registry import service
from backend.model_registry.schemas import (
    BestAvailableModelResponse,
    FiguresResponse,
    ModelInfoResponse,
    ModelsResponse,
)


router = APIRouter()


@router.get("/models", response_model=ModelsResponse)
def list_available_models(
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Devuelve los modelos disponibles para el usuario autenticado."""
    return {"models": service.list_models(int(current_user.id))}


@router.get("/models/best", response_model=BestAvailableModelResponse | None)
def best_available_model(
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Devuelve el mejor modelo entrenado disponible del usuario."""
    return service.get_best_available_model(int(current_user.id))


@router.get(
    "/model/info", response_model=ModelInfoResponse, responses=MODEL_INFO_RESPONSES
)
def model_info(
    current_user: Annotated[User, Depends(get_current_user)],
    model_id: str = "ml_best",
):
    """Devuelve metadatos y metricas de un modelo accesible por el usuario."""
    try:
        return service.get_model_info(model_id, int(current_user.id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get(
    "/model/figures", response_model=FiguresResponse, responses=NOT_FOUND_RESPONSES
)
def model_figures(
    current_user: Annotated[User, Depends(get_current_user)],
    model_id: str = "ml_best",
):
    """Devuelve las figuras de evaluacion asociadas a un modelo accesible."""
    try:
        return {
            "figures": service.get_model_figures(
                model_id,
                int(current_user.id),
            )
        }
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
