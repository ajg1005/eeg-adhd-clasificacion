from fastapi import APIRouter

from backend.schemas import HealthResponse


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}
