from typing import Any

from pydantic import BaseModel

from backend.schemas.common import FlexibleSchema


class TrainingOptionsResponse(FlexibleSchema):
    default_model_type: str
    default_models: dict[str, str]
    default_eeg_params: dict[str, dict[str, Any]]
    default_training_params: dict[str, Any]
    training_params_by_type: dict[str, list[str]]
    model_types: dict[str, Any]
    eeg_params: dict[str, list[Any]]
    training_params: dict[str, list[Any]]


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


class PatientTrainingResult(BaseModel):
    patient_id: str
    true_label: str
    predicted_label: str
    n_epochs: int
    control_epoch_percentage: float
    adhd_epoch_percentage: float
    correct: bool


class FeatureImportanceItem(BaseModel):
    feature: str
    importance_mean: float
    importance_std: float


class FeatureImportanceResponse(BaseModel):
    method: str
    scoring: str
    n_repeats: int
    evaluated_epochs: int
    source: str
    top_features: list[FeatureImportanceItem]
    by_channel: list[FeatureImportanceItem]
    error: str | None = None


class TrainingRunResponse(FlexibleSchema):
    # experiment_id es None si la persistencia en BD fallo; las metricas siguen siendo validas.
    experiment_id: int | None = None
    persisted: bool = True
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    balanced_accuracy: float
    classification_report: dict[str, Any]
    confusion_matrix: list[list[int]]
    patient_results: list[PatientTrainingResult]
    fold_results: list[dict[str, Any]]
    feature_importance: FeatureImportanceResponse | None = None
    configuration: dict[str, Any]
    training_time_seconds: float
