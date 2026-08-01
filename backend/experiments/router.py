from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.api.responses import NOT_FOUND_RESPONSES
from backend.auth.dependencies import get_current_user
from backend.db.models import User
from backend.experiments import service
from backend.experiments.schemas import (
    ExperimentDetailResponse,
    ExperimentsListResponse,
)


router = APIRouter()


@router.get("/experiments", response_model=ExperimentsListResponse)
def experiments_list(
    current_user: Annotated[User, Depends(get_current_user)],
    model_type: str | None = None,
    model_name: str | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    """Devuelve los experimentos del usuario autenticado."""
    return {
        "experiments": service.list_experiments(
            owner_id=int(current_user.id),
            model_type=model_type,
            model_name=model_name,
            limit=limit,
            offset=offset,
        )
    }


@router.get(
    "/experiments/{experiment_id}",
    response_model=ExperimentDetailResponse,
    responses=NOT_FOUND_RESPONSES,
)
def experiment_detail(
    experiment_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Devuelve un experimento si pertenece al usuario autenticado."""
    experiment = service.get_experiment(experiment_id, int(current_user.id))
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experimento no encontrado.")
    return experiment
