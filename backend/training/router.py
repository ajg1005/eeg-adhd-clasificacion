import json
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from backend.api.responses import TRAINING_RUN_RESPONSES
from backend.datasets.service import save_training_dataset
from backend.training.schemas import TrainingOptionsResponse, TrainingTaskResponse
from backend.training.service import get_training_options
from backend.training.tasks import execute_training_task


router = APIRouter(prefix="/training", tags=["training"])


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
    """Devuelve las opciones aceptadas por el formulario de entrenamiento."""
    return get_training_options()


@router.post(
    "/run",
    response_model=TrainingTaskResponse,
    responses=TRAINING_RUN_RESPONSES,
    status_code=status.HTTP_202_ACCEPTED,
)
async def training_run(
    model_type: Annotated[str, Form()],
    model_name: Annotated[str, Form()],
    file: UploadFile | None = File(default=None),
    dataset_id: int | None = Form(default=None),
    eeg_params: Annotated[str | None, Form()] = None,
    model_params: Annotated[str | None, Form()] = None,
    training_params: Annotated[str | None, Form()] = None,
):
    """Encola un entrenamiento desde un dataset y parametros de formulario."""
    try:
        if dataset_id is None:
            if file is None:
                raise ValueError(
                    "Debes subir un CSV o seleccionar un dataset guardado."
                )

            saved_dataset = save_training_dataset(
                file_bytes=await file.read(),
                filename=file.filename or "training.csv",
            )
            dataset_id = saved_dataset["id"]

        task = execute_training_task.delay(
            dataset_id=dataset_id,
            model_type=model_type,
            model_name=model_name,
            eeg_params=_json_dict(eeg_params),
            model_params=_json_dict(model_params),
            training_params=_json_dict(training_params),
        )

        return {
            "task_id": task.id,
            "status": "PENDING",
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
