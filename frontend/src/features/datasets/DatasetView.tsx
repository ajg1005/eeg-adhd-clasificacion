import { useMemo, type ChangeEvent } from "react";
import { useTranslation } from "react-i18next";

import type {
  SavedTrainingDataset,
  TrainingDatasetPatient,
  TrainingDatasetStats,
} from "./types";

interface DatasetViewProps {
  classFilter: string;
  error: string;
  file: File | null;
  handleAnalyzeDataset: () => Promise<void>;
  handleClassFilterChange: (event: ChangeEvent<HTMLSelectElement>) => void;
  handleFileChange: (event: ChangeEvent<HTMLInputElement>) => void;
  handleMaxPatientsChange: (event: ChangeEvent<HTMLInputElement>) => void;
  handleSavedDatasetChange: (
    event: ChangeEvent<HTMLSelectElement>,
  ) => Promise<void>;
  loadingDatasets: boolean;
  loadingStats: boolean;
  maxPatients: number;
  savedDatasets: SavedTrainingDataset[];
  selectedDataset: SavedTrainingDataset | null;
  stats: TrainingDatasetStats | null;
}

function filterPatients(
  patients: TrainingDatasetPatient[] | undefined,
  classFilter: string,
  maxPatients: number,
): TrainingDatasetPatient[] {
  if (!patients) {
    return [];
  }

  const filtered =
    classFilter === "all"
      ? patients
      : patients.filter(
          (patient) => patient.class_label.toLowerCase() === classFilter,
        );

  return filtered.slice(0, maxPatients);
}

export function DatasetView({
  classFilter,
  error,
  file,
  handleAnalyzeDataset,
  handleClassFilterChange,
  handleFileChange,
  handleMaxPatientsChange,
  handleSavedDatasetChange,
  loadingDatasets,
  loadingStats,
  maxPatients,
  savedDatasets,
  selectedDataset,
  stats,
}: DatasetViewProps) {
  const { t } = useTranslation();
  const filteredPatients = useMemo(
    () => filterPatients(stats?.patients, classFilter, maxPatients),
    [stats, classFilter, maxPatients],
  );

  return (
    <section className="training-layout">
      {error && <div className="alert alert-error">{error}</div>}

      <div className="panel">
        <div className="section-heading-row section-heading-row-end">
          <button
            className="primary-button compact-button"
            disabled={(!file && !selectedDataset) || loadingStats}
            onClick={() => {
              void handleAnalyzeDataset();
            }}
            type="button"
          >
            {loadingStats ? t("dataset.analyzing") : t("dataset.analyze")}
          </button>
        </div>

        {savedDatasets.length > 0 && (
          <div className="controls-row">
            <label>
              {t("dataset.savedDatasets")}
              <select
                disabled={loadingDatasets || loadingStats}
                value={selectedDataset?.id ?? ""}
                onChange={(event) => {
                  void handleSavedDatasetChange(event);
                }}
              >
                <option value="">{t("dataset.newDataset")}</option>
                {savedDatasets.map((dataset) => (
                  <option
                    disabled={!dataset.reusable}
                    key={dataset.id}
                    value={dataset.id}
                  >
                    {dataset.filename} - {dataset.n_subjects}{" "}
                    {t("common.patients")}
                  </option>
                ))}
              </select>
            </label>
          </div>
        )}

        <label className="file-drop">
          <input accept=".csv" onChange={handleFileChange} type="file" />
          {file?.name || selectedDataset?.filename || t("dataset.selectCsv")}
        </label>

        {loadingDatasets && (
          <p className="muted">{t("dataset.loadingSavedDatasets")}</p>
        )}

        {(file || selectedDataset) && !stats && !loadingStats && (
          <div className="alert alert-info">{t("dataset.readyToAnalyze")}</div>
        )}

        {stats && (
          <>
            <div className="metric-grid dataset-summary-grid training-metrics-row">
              <div>
                <span>{t("common.rows")}</span>
                <strong>{stats.rows}</strong>
              </div>
              <div>
                <span>{t("common.columns")}</span>
                <strong>{stats.columns}</strong>
              </div>
              <div>
                <span>{t("common.patients")}</span>
                <strong>{stats.n_patients}</strong>
              </div>
              <div>
                <span>{t("dataset.eegChannels")}</span>
                <strong>{stats.eeg_columns.length}</strong>
              </div>
            </div>

            <div className="class-counts">
              {Object.entries(stats.class_distribution).map(([label, count]) => (
                <div key={label}>
                  <span>{label}</span>
                  <strong>{count}</strong>
                </div>
              ))}
            </div>

            {stats.missing_required_columns.length > 0 && (
              <div className="alert alert-error">
                {t("dataset.missingColumns", {
                  columns: stats.missing_required_columns.join(", "),
                })}
              </div>
            )}
          </>
        )}
      </div>

      {stats && (
        <div className="panel">
          <div className="section-heading-row">
            <h3>{t("common.patients")}</h3>
            <div className="controls-row compact-controls">
              <label>
                {t("dataset.filterClass")}
                <select value={classFilter} onChange={handleClassFilterChange}>
                  <option value="all">{t("dataset.all")}</option>
                  <option value="adhd">TDAH</option>
                  <option value="control">Control</option>
                </select>
              </label>
              <label>
                {t("common.patients")}
                <input
                  max="100"
                  min="1"
                  step="1"
                  type="number"
                  value={maxPatients}
                  onChange={handleMaxPatientsChange}
                />
              </label>
            </div>
          </div>

          {filteredPatients.length > 0 ? (
            <div className="patient-table-wrap">
              <table className="patient-table">
                <thead>
                  <tr>
                    <th>{t("common.patient")}</th>
                    <th>{t("common.class")}</th>
                    <th>{t("common.rows")}</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredPatients.map((patient) => (
                    <tr key={patient.patient_id}>
                      <td>{patient.patient_id}</td>
                      <td>{patient.class_label}</td>
                      <td>{patient.rows}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="muted">{t("dataset.noPatients")}</p>
          )}
        </div>
      )}
    </section>
  );
}
