from typing import Any

from pydantic import BaseModel


class FlexibleSchema(BaseModel):
    class Config:
        extra = "allow"


class HealthResponse(BaseModel):
    status: str


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
