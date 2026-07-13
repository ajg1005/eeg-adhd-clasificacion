import json
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.api.responses import TRAINING_RUN_RESPONSES
from backend.training.schemas import TrainingOptionsResponse, TrainingRunResponse
from backend.datasets.service import get_saved_dataset_file
from backend.training.service import get_training_options, run_training


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


@router.post("/run", response_model=TrainingRunResponse, responses=TRAINING_RUN_RESPONSES)
async def training_run(
    model_type: Annotated[str, Form()],
    model_name: Annotated[str, Form()],
    file: UploadFile | None = File(default=None),
    dataset_id: int | None = Form(default=None),
    eeg_params: Annotated[str | None, Form()] = None,
    model_params: Annotated[str | None, Form()] = None,
    training_params: Annotated[str | None, Form()] = None,
):
    """Entrena y evalua un modelo desde un dataset y parametros de formulario."""
    try:
        if dataset_id is not None:
            file_bytes, filename = get_saved_dataset_file(dataset_id)
        elif file is not None:
            file_bytes = await file.read()
            filename = file.filename or "training.csv"
        else:
            raise ValueError("Debes subir un CSV o seleccionar un dataset guardado.")

        return run_training(
            file_bytes=file_bytes,
            model_type=model_type,
            model_name=model_name,
            filename=filename,
            eeg_params=_json_dict(eeg_params),
            model_params=_json_dict(model_params),
            training_params=_json_dict(training_params),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

