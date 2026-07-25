import type { JsonValue, UnknownRecord } from "../../shared/types";

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
