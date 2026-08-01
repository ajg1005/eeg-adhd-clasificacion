from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from backend.api.responses import BAD_REQUEST_RESPONSES
from backend.auth.dependencies import get_current_user
from backend.datasets.schemas import (
    DatasetAnalysisTaskResponse,
    SavedTrainingDatasetResponse,
    SavedTrainingDatasetsListResponse,
    TrainingDatasetStatsResponse,
)
from backend.datasets.service import (
    ensure_saved_dataset_access,
    get_dataset_stats,
    get_saved_dataset_stats,
    get_saved_datasets,
    save_training_dataset,
)
from backend.datasets.tasks import analyze_dataset
from backend.db.models import User
from backend.worker.job_service import enqueue_background_job

router = APIRouter(prefix="/training", tags=["datasets"])


@router.get("/datasets", response_model=SavedTrainingDatasetsListResponse)
def training_datasets(
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Lista datasets de entrenamiento accesibles para el usuario."""
    return {"datasets": get_saved_datasets(int(current_user.id))}


@router.post(
    "/datasets",
    response_model=SavedTrainingDatasetResponse,
    responses=BAD_REQUEST_RESPONSES,
)
async def upload_training_dataset(
    file: Annotated[UploadFile, File(...)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Guarda un CSV de entrenamiento para el usuario autenticado."""
    try:
        return save_training_dataset(
            file_bytes=await file.read(),
            filename=file.filename or "training.csv",
            user_id=int(current_user.id),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/datasets/{dataset_id}/analysis",
    response_model=DatasetAnalysisTaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def queue_dataset_analysis(
    dataset_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Encola el analisis si el usuario puede acceder al dataset."""
    try:
        user_id = int(current_user.id)
        ensure_saved_dataset_access(dataset_id, user_id)
        task = enqueue_background_job(
            analyze_dataset,
            user_id,
            args=(dataset_id, user_id),
        )
        return {"task_id": task.id, "status": "PENDING"}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get(
    "/datasets/{dataset_id}/stats",
    response_model=TrainingDatasetStatsResponse,
    responses=BAD_REQUEST_RESPONSES,
)
def saved_training_dataset_stats(
    dataset_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Calcula la vista previa de un dataset accesible para el usuario."""
    try:
        return get_saved_dataset_stats(dataset_id, int(current_user.id))
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
