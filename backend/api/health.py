from fastapi import APIRouter
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    """Devuelve el estado basico de disponibilidad de la API."""
    return {"status": "ok"}