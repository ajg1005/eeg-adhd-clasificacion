export type UnknownRecord = Record<string, unknown>;
export type JsonPrimitive = string | number | boolean | null;
export type JsonValue =
  | JsonPrimitive
  | JsonValue[]
  | { [key: string]: JsonValue };
export type TrainingOptionValue = JsonPrimitive;
export type TrainingParameters = Record<string, JsonValue>;
export type TrainingControlValues = Record<string, TrainingOptionValue>;
export type TrainingModelTypeId = "ml" | "dl";
export type ApiStatus = "checking" | "ok" | "error";

export interface MetricChartDatum {
  name: string;
  value: number;
}

export interface SelectOption {
  disabled?: boolean;
  label: string;
  value: string;
}

export type TaskStatus =
  | "PENDING"
  | "RECEIVED"
  | "STARTED"
  | "RETRY"
  | "SUCCESS"
  | "FAILURE"
  | "REVOKED";

export interface HealthResponse {
  status: string;
}

export interface ModelRegistryItem {
  model_id: string;
  display_name: string;
  model_family: string;
  description?: string | null;
  enabled?: boolean | null;
}

export interface BestAvailableModel {
  model_id: string;
  trained_model_id: number;
  experiment_id: number;
  display_name: string;
  model_name: string;
  model_type: string;
  model_family: string;
  created_at: string;
  balanced_accuracy: number;
  f1_score: number;
  dataset_filename: string;
  n_subjects: number;
}

export interface CvMetrics {
  accuracy_epoch_mean?: number;
  balanced_accuracy_epoch_mean?: number;
  precision_epoch_mean?: number;
  recall_epoch_mean?: number;
  f1_epoch_mean?: number;
}

export interface ModelMetrics extends CvMetrics {
  [key: string]: unknown;
  cv_metrics?: CvMetrics;
}

export interface ModelInfo {
  model_id: string;
  display_name: string;
  model_name?: string | null;
  model_family: string;
  feature_mode?: string | null;
  sfreq?: number | null;
  epoch_size?: number | null;
  step_size?: number | null;
  channels: string[];
  n_features?: number | null;
  metrics?: ModelMetrics | null;
  metadata: UnknownRecord;
}

export interface ModelFigure {
  title: string;
  url: string;
}

export interface TrainingDatasetPatient {
  patient_id: string;
  class_label: string;
  rows: number;
}

export interface TrainingDatasetStats {
  rows: number;
  columns: number;
  n_patients: number;
  class_distribution: Record<string, number>;
  patients: TrainingDatasetPatient[];
  eeg_columns: string[];
  missing_required_columns: string[];
  preview: UnknownRecord[];
}

export interface SavedTrainingDataset {
  id: number;
  dataset_hash: string;
  filename: string;
  original_filename?: string | null;
  storage_path?: string | null;
  file_size_bytes?: number | null;
  rows: number;
  columns: number;
  n_subjects: number;
  class_distribution: Record<string, number>;
  eeg_columns: string[];
  created_at: string;
  reusable: boolean;
}

export interface TrainingModelOption {
  display_name: string;
  default_params: TrainingControlValues;
  parameters: Record<string, TrainingOptionValue[]>;
}

export interface TrainingModelType {
  display_name?: string;
  models: Record<string, TrainingModelOption>;
}

export interface TrainingOptions {
  default_model_type: TrainingModelTypeId;
  default_models: Record<TrainingModelTypeId, string>;
  default_eeg_params: Record<TrainingModelTypeId, TrainingControlValues>;
  eeg_params_by_type: Record<TrainingModelTypeId, string[]>;
  default_training_params: TrainingControlValues;
  training_params_by_type: Record<TrainingModelTypeId, string[]>;
  model_types: Record<TrainingModelTypeId, TrainingModelType>;
  eeg_params: Record<string, TrainingOptionValue[]>;
  training_params: Record<string, TrainingOptionValue[]>;
}

export interface AsyncTaskResponse {
  task_id: string;
  status: TaskStatus;
}

export interface TaskStatusResponse<TResult = unknown>
  extends AsyncTaskResponse {
  result?: TResult;
  error?: string;
}

export interface TrainingPayload {
  datasetId?: number | null;
  modelType: TrainingModelTypeId;
  modelName: string;
  eegParams: TrainingParameters;
  modelParams: TrainingParameters;
  trainingParams: TrainingParameters;
}

export interface PatientTrainingResult {
  patient_id: string;
  true_label: string;
  predicted_label: string;
  n_epochs: number;
  control_epoch_percentage: number;
  adhd_epoch_percentage: number;
  correct: boolean;
}

export interface FeatureImportanceItem {
  feature: string;
  importance_mean: number;
  importance_std: number;
}

export interface FeatureImportance {
  method: string;
  scoring: string;
  n_repeats: number;
  evaluated_epochs: number;
  source: string;
  top_features: FeatureImportanceItem[];
  by_channel: FeatureImportanceItem[];
  error?: string | null;
}

export interface TrainingResult {
  experiment_id?: number | null;
  persisted: boolean;
  trained_model_id?: number | null;
  model_saved: boolean;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  balanced_accuracy: number;
  classification_report: UnknownRecord;
  confusion_matrix: number[][];
  patient_results: PatientTrainingResult[];
  fold_results: UnknownRecord[];
  feature_importance?: FeatureImportance | null;
  configuration: TrainingConfiguration;
  training_time_seconds: number;
}

export interface TrainingConfiguration extends UnknownRecord {
  evaluation_mode?: string;
}

export interface ExperimentDataset {
  id: number;
  dataset_hash: string;
  filename: string;
  original_filename?: string | null;
  storage_path?: string | null;
  file_size_bytes?: number | null;
  rows: number;
  columns: number;
  n_subjects: number;
  class_distribution: Record<string, number>;
  eeg_columns: string[];
  created_at: string;
}

export interface ExperimentSummary {
  id: number;
  created_at: string;
  model_type: string;
  model_name: string;
  display_name: string;
  evaluation_mode: string;
  training_time_seconds: number;
  accuracy: number;
  balanced_accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  dataset: ExperimentDataset;
}

export interface ExperimentFold {
  id: number;
  fold: number;
  accuracy: number;
  balanced_accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  n_train_subjects?: number | null;
  n_val_subjects?: number | null;
  n_test_subjects?: number | null;
  best_threshold?: number | null;
}

export interface ExperimentDetail extends ExperimentSummary {
  eeg_params: Record<string, JsonValue>;
  model_params: Record<string, JsonValue>;
  training_params: Record<string, JsonValue>;
  confusion_matrix: number[][];
  classification_report: UnknownRecord;
  fold_results: ExperimentFold[];
}

export interface ValidationResult {
  valid: boolean;
  filename?: string | null;
  rows: number;
  columns: number;
  available_channels: string[];
  expected_channels: string[];
  has_id: boolean;
  has_class: boolean;
}

export interface PredictionResult {
  model_id?: string;
  model_name?: string | null;
  model_family?: string;
  prediction?: string;
  prediction_label: string;
  confidence?: number;
  decision_score?: number;
  final_class_epoch_percentage?: number;
  threshold?: number;
  n_epochs: number;
  epoch_count_by_class: Record<string, number>;
  epoch_percentage_by_class: Record<string, number>;
  metrics?: ModelMetrics | null;
  metadata: UnknownRecord;
}