from fastapi import APIRouter

from backend.schemas import HealthResponse


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    """Devuelve el estado basico de disponibilidad de la API."""
    return {"status": "ok"}
