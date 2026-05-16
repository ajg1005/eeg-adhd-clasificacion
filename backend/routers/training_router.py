import json
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.routers.responses import BAD_REQUEST_RESPONSES, TRAINING_RUN_RESPONSES
from backend.schemas import (
    TrainingDatasetStatsResponse,
    TrainingOptionsResponse,
    TrainingRunResponse,
)
from backend.services.training_service import (
    get_dataset_stats,
    get_training_options,
    run_training,
)


router = APIRouter()


def _json_dict(raw_value: str | None) -> dict[str, Any]:
    if not raw_value:
        return {}
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError("JSON invalido en parametros.") from exc
    if not isinstance(value, dict):
        raise ValueError("El parametro debe ser un objeto JSON.")
    return value


@router.get("/options", response_model=TrainingOptionsResponse)
def training_options():
    return get_training_options()


@router.post("/dataset/stats", response_model=TrainingDatasetStatsResponse, responses=BAD_REQUEST_RESPONSES)
async def training_dataset_stats(file: Annotated[UploadFile, File(...)]):
    try:
        return get_dataset_stats(await file.read())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/run", response_model=TrainingRunResponse, responses=TRAINING_RUN_RESPONSES)
async def training_run(
    file: Annotated[UploadFile, File(...)],
    model_type: Annotated[str, Form()],
    model_name: Annotated[str, Form()],
    eeg_params: Annotated[str | None, Form()] = None,
    model_params: Annotated[str | None, Form()] = None,
    training_params: Annotated[str | None, Form()] = None,
):
    try:
        return run_training(
            file_bytes=await file.read(),
            model_type=model_type,
            model_name=model_name,
            eeg_params=_json_dict(eeg_params),
            model_params=_json_dict(model_params),
            training_params=_json_dict(training_params),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

