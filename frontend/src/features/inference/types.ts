import type { UnknownRecord } from "../../shared/types";

export interface ModelRegistryItem {
  model_id: string;
  display_name: string;
  model_family: string;
  description?: string | null;
  enabled?: boolean | null;
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
  epoch_predictions?: string[];
  metrics?: ModelMetrics | null;
  metadata: UnknownRecord;
}
