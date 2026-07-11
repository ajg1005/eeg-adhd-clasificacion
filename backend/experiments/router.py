from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from backend.experiments import service
from backend.experiments.schemas import (
    ExperimentDetailResponse,
    ExperimentsListResponse,
)
from backend.routers.responses import NOT_FOUND_RESPONSES


router = APIRouter()


@router.get("/experiments", response_model=ExperimentsListResponse)
def experiments_list(
    model_type: str | None = None,
    model_name: str | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    """Devuelve experimentos paginados, opcionalmente filtrados por modelo."""
    return {
        "experiments": service.list_experiments(
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
def experiment_detail(experiment_id: int):
    """Devuelve un experimento guardado con dataset y resultados por fold."""
    experiment = service.get_experiment(experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experimento no encontrado.")
    return experiment
