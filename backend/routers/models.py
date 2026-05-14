from fastapi import APIRouter, HTTPException

from backend.services.model_service import get_model_figures, get_model_info, list_models


router = APIRouter()


@router.get("/models")
def list_available_models():
    return {"models": list_models()}


@router.get("/model/info")
def model_info(model_id: str = "ml_best"):
    try:
        return get_model_info(model_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/model/figures")
def model_figures(model_id: str = "ml_best"):
    try:
        return {"figures": get_model_figures(model_id)}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
