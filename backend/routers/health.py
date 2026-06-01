from fastapi import APIRouter

from backend.schemas import HealthResponse


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    """Return the API liveness status used by monitoring and smoke tests."""
    return {"status": "ok"}
