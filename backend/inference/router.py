from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from backend.api.responses import BAD_REQUEST_RESPONSES
from backend.auth.dependencies import get_current_user
from backend.db.models import User
from backend.inference.schemas import PredictionResponse, ValidationResponse
from backend.inference.service import (
    predict_dataframe,
    validate_dataframe_for_model,
)
from backend.inference.upload import read_csv_upload


router = APIRouter()


@router.post(
    "/validate", response_model=ValidationResponse, responses=BAD_REQUEST_RESPONSES
)
async def validate_file(
    file: Annotated[UploadFile, File(...)],
    current_user: Annotated[User, Depends(get_current_user)],
    model_id: str = "ml_best",
):
    """Valida un CSV contra un modelo accesible por el usuario."""
    try:
        df = read_csv_upload(file)
        validation = validate_dataframe_for_model(
            df,
            model_id,
            int(current_user.id),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "valid": True,
        "filename": file.filename,
        **validation,
    }


@router.post(
    "/predict", response_model=PredictionResponse, responses=BAD_REQUEST_RESPONSES
)
async def predict(
    file: Annotated[UploadFile, File(...)],
    current_user: Annotated[User, Depends(get_current_user)],
    model_id: str = "ml_best",
):
    """Ejecuta inferencia con un modelo accesible por el usuario."""
    try:
        df = read_csv_upload(file)
        return predict_dataframe(
            df,
            model_id,
            int(current_user.id),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
