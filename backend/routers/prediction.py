from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.routers.responses import BAD_REQUEST_RESPONSES
from backend.schemas import PredictionResponse, ValidationResponse
from backend.services.csv_service import read_csv_upload
from backend.services.model_service import predict_dataframe, validate_dataframe_for_model


router = APIRouter()


# Validar CSV antes de ejecutar la prediccion
@router.post("/validate", response_model=ValidationResponse, responses=BAD_REQUEST_RESPONSES)
async def validate_file(
    file: Annotated[UploadFile, File(...)],
    model_id: str = "ml_best",
):
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
    try:
        df = read_csv_upload(file)
        return predict_dataframe(df, model_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
