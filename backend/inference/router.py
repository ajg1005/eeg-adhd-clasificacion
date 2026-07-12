from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.inference.schemas import PredictionResponse, ValidationResponse
from backend.inference.service import (
    predict_dataframe,
    validate_dataframe_for_model,
)
from backend.inference.upload import read_csv_upload
from backend.api.responses import BAD_REQUEST_RESPONSES


router = APIRouter()


# Validar CSV antes de ejecutar la prediccion
@router.post("/validate", response_model=ValidationResponse, responses=BAD_REQUEST_RESPONSES)
async def validate_file(
    file: Annotated[UploadFile, File(...)],
    model_id: str = "ml_best",
):
    """Valida un CSV subido contra los requisitos del modelo seleccionado."""
    try:
        df = read_csv_upload(file)
        validation = validate_dataframe_for_model(df, model_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "valid": True,
        "filename": file.filename,
        **validation,
    }


# Ejecutar prediccion con el modelo seleccionado
@router.post("/predict", response_model=PredictionResponse, responses=BAD_REQUEST_RESPONSES)
async def predict(
    file: Annotated[UploadFile, File(...)],
    model_id: str = "ml_best",
):
    """Ejecuta inferencia sobre un CSV EEG con el modelo exportado seleccionado."""
    try:
        df = read_csv_upload(file)
        return predict_dataframe(df, model_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
