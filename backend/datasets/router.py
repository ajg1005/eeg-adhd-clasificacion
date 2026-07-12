from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.datasets.schemas import (
    SavedTrainingDatasetResponse,
    SavedTrainingDatasetsListResponse,
    TrainingDatasetStatsResponse,
)
from backend.routers.responses import BAD_REQUEST_RESPONSES
from backend.datasets.service import (
    get_dataset_stats,
    get_saved_dataset_stats,
    get_saved_datasets,
    save_training_dataset,
)


router = APIRouter(prefix="/training", tags=["datasets"])


@router.get("/datasets", response_model=SavedTrainingDatasetsListResponse)
def training_datasets():
    """Lista datasets de entrenamiento guardados para reutilizarlos."""
    return {"datasets": get_saved_datasets()}


@router.post(
    "/datasets",
    response_model=SavedTrainingDatasetResponse,
    responses=BAD_REQUEST_RESPONSES,
)
async def upload_training_dataset(file: Annotated[UploadFile, File(...)]):
    """Guarda un CSV de entrenamiento para reutilizarlo en ejecuciones futuras."""
    try:
        return save_training_dataset(
            file_bytes=await file.read(),
            filename=file.filename or "training.csv",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get(
    "/datasets/{dataset_id}/stats",
    response_model=TrainingDatasetStatsResponse,
    responses=BAD_REQUEST_RESPONSES,
)
def saved_training_dataset_stats(dataset_id: int):
    """Calcula la vista previa de un dataset ya guardado."""
    try:
        return get_saved_dataset_stats(dataset_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/dataset/stats",
    response_model=TrainingDatasetStatsResponse,
    responses=BAD_REQUEST_RESPONSES,
)
async def training_dataset_stats(file: Annotated[UploadFile, File(...)]):
    """Lee un CSV de entrenamiento y devuelve estadisticas de vista previa."""
    try:
        return get_dataset_stats(await file.read())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
