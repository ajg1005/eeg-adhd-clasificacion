from typing import Any

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    model_name: str | None
    feature_mode: str | None
    sfreq: int | float | None
    epoch_size: int | None
    step_size: int | None
    channels: list[str]
    n_features: int | None
    metrics: dict[str, Any] | None
    metadata: dict[str, Any]


class PredictionResponse(BaseModel):
    prediction: str
    prediction_label: str
    confidence: float
    decision_score: float | None = None
    final_class_epoch_percentage: float | None = None
    n_epochs: int
    epoch_count_by_class: dict[str, int]
    epoch_percentage_by_class: dict[str, float]
    metrics: dict[str, Any] | None
    metadata: dict[str, Any]
