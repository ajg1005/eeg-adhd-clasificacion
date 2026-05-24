from typing import Any

from pydantic import BaseModel

from backend.schemas.common import FlexibleSchema


class ModelRegistryItem(FlexibleSchema):
    model_id: str
    display_name: str
    model_family: str
    description: str | None = None
    enabled: bool | None = None


class ModelsResponse(BaseModel):
    models: list[ModelRegistryItem]


class ModelCatalogResponse(FlexibleSchema):
    machine_learning: list[dict[str, Any]]
    deep_learning: list[dict[str, Any]]


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
