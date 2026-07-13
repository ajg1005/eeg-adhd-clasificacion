from datetime import datetime
from typing import Any

from backend.api.schemas import OrmSchema


class ExperimentDatasetResponse(OrmSchema):
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


class ExperimentSummaryResponse(OrmSchema):
    id: int
    created_at: datetime
    model_type: str
    model_name: str
    display_name: str
    evaluation_mode: str
    training_time_seconds: float
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1_score: float
    dataset: ExperimentDatasetResponse


class ExperimentFoldResponse(OrmSchema):
    id: int
    fold: int
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1_score: float
    n_train_subjects: int | None = None
    n_val_subjects: int | None = None
    n_test_subjects: int | None = None
    best_threshold: float | None = None


class ExperimentDetailResponse(ExperimentSummaryResponse):
    eeg_params: dict[str, Any]
    model_params: dict[str, Any]
    training_params: dict[str, Any]
    confusion_matrix: list[list[int]]
    classification_report: dict[str, Any]
    fold_results: list[ExperimentFoldResponse]


class ExperimentsListResponse(OrmSchema):
    experiments: list[ExperimentSummaryResponse]
