import type {
  JsonPrimitive,
  JsonValue,
  TaskStatus,
  UnknownRecord,
} from "../../shared/types";

export type TrainingOptionValue = JsonPrimitive;
export type TrainingParameters = Record<string, JsonValue>;
export type TrainingControlValues = Record<string, TrainingOptionValue>;
export type TrainingModelTypeId = "ml" | "dl";
export type TrainingTaskStatus = TaskStatus | "SUBMITTING" | null;

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

export interface TrainingConfiguration extends UnknownRecord {
  evaluation_mode?: string;
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
