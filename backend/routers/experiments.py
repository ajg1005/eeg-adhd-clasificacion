from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from backend.db.repository import get_experiment, list_experiments
from backend.routers.responses import NOT_FOUND_RESPONSES
from backend.schemas import ExperimentDetailResponse, ExperimentsListResponse


router = APIRouter()


@router.get("/experiments", response_model=ExperimentsListResponse)
def experiments_list(
    model_type: str | None = None,
    model_name: str | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    """Return paginated training experiments, optionally filtered by model."""
    return {
        "experiments": list_experiments(
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
    """Return one stored experiment with dataset metadata and fold results."""
    experiment = get_experiment(experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experimento no encontrado.")
    return experiment
