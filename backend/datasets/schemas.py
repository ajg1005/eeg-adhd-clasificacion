from datetime import datetime
from typing import Any

from pydantic import BaseModel

from backend.schemas.common import FlexibleSchema, OrmSchema


class TrainingDatasetPatient(BaseModel):
    patient_id: str
    class_label: str
    rows: int


class TrainingDatasetStatsResponse(BaseModel):
    rows: int
    columns: int
    n_patients: int
    class_distribution: dict[str, int]
    patients: list[TrainingDatasetPatient]
    eeg_columns: list[str]
    missing_required_columns: list[str]
    preview: list[dict[str, Any]]


class SavedTrainingDatasetResponse(OrmSchema):
    id: int
    dataset_hash: str
    filename: str
    original_filename: str | None = None
    storage_path: str | None = None
    file_size_bytes: int | None = None
    rows: int
    columns: int
    n_subjects: int
    class_distribution: dict[str, int]
    eeg_columns: list[str]
    created_at: datetime
    reusable: bool = False


class SavedTrainingDatasetsListResponse(FlexibleSchema):
    datasets: list[SavedTrainingDatasetResponse]
