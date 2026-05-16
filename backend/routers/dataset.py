from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.routers.responses import BAD_REQUEST_RESPONSES
from backend.schemas import DatasetSummaryResponse
from backend.services.csv_service import read_csv_upload
from backend.services.dataset_service import build_dataset_summary


router = APIRouter()


# Cargar dataset y devolver estadisticas generales y pacientes filtrados
@router.post("/dataset/summary", response_model=DatasetSummaryResponse, responses=BAD_REQUEST_RESPONSES)
async def dataset_summary(
    file: Annotated[UploadFile, File(...)],
    class_filter: str = "all",
    max_patients: int = 10,
):
    try:
        df = read_csv_upload(file)
        return build_dataset_summary(
            df,
            class_filter=class_filter,
            max_patients=max_patients,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
