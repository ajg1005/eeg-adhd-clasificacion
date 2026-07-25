import type { UnknownRecord } from "../../shared/types";

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
