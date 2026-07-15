from datetime import datetime
from typing import Any

from pydantic import BaseModel

from backend.api.schemas import FlexibleSchema


class ModelRegistryItem(FlexibleSchema):
    model_id: str
    display_name: str
    model_family: str
    description: str | None = None
    enabled: bool | None = None


class ModelsResponse(BaseModel):
    models: list[ModelRegistryItem]


class BestAvailableModelResponse(BaseModel):
    model_id: str
    trained_model_id: int
    experiment_id: int
    display_name: str
    model_name: str
    model_type: str
    model_family: str
    created_at: datetime
    balanced_accuracy: float
    f1_score: float
    dataset_filename: str
    n_subjects: int


class ModelInfoResponse(FlexibleSchema):
    model_id: str
    display_name: str
    model_name: str | None = None
    model_family: str
    feature_mode: str | None = None
    sfreq: int | float | None = None
    epoch_size: int | None = None
    step_size: int | None = None
    channels: list[str]
    n_features: int | None = None
    metrics: dict[str, Any] | None = None
    metadata: dict[str, Any]


class FigureItem(BaseModel):
    title: str
    url: str


class FiguresResponse(BaseModel):
    figures: list[FigureItem]
