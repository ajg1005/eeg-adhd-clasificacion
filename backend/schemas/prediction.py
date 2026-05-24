from typing import Any

from backend.schemas.common import FlexibleSchema


class ValidationResponse(FlexibleSchema):
    valid: bool
    filename: str | None = None
    rows: int
    columns: int
    available_channels: list[str]
    expected_channels: list[str]
    has_id: bool
    has_class: bool


class PredictionResponse(FlexibleSchema):
    prediction_label: str
    n_epochs: int
    epoch_count_by_class: dict[str, int]
    epoch_percentage_by_class: dict[str, float]
    metadata: dict[str, Any]
