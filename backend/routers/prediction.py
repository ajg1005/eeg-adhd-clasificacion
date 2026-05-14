from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.services.csv_service import read_csv_upload
from backend.services.model_service import predict_dataframe, validate_dataframe_for_model


router = APIRouter()


@router.post("/validate")
async def validate_file(file: UploadFile = File(...), model_id: str = "ml_best"):
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


@router.post("/predict")
async def predict(file: UploadFile = File(...), model_id: str = "ml_best"):
    try:
        df = read_csv_upload(file)
        return predict_dataframe(df, model_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
