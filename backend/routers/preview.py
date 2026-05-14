from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.services.csv_service import build_signal_preview, read_csv_upload
from backend.services.model_service import get_model_metadata


router = APIRouter()


@router.post("/preview")
async def preview_signal(
    file: UploadFile = File(...),
    channel: str = "Fp1",
    max_points: int = 1000,
    model_id: str = "ml_best",
):
    try:
        metadata, _, _ = get_model_metadata(model_id)
        df = read_csv_upload(file)
        expected_channels = metadata.get("channels", [])
        return build_signal_preview(df, expected_channels, channel, max_points)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
