import { useMemo, type ChangeEvent } from "react";
import { useTranslation } from "react-i18next";

import { formatPercent } from "../../shared/utils/formatters";
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
  onClassFilterChange: (value: string) => void;
  handleFileChange: (event: ChangeEvent<HTMLInputElement>) => Promise<void>;
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
function classVariant(label: string): "adhd" | "control" | "other" {
  const normalized = label.trim().toLowerCase();

  if (normalized === "adhd" || normalized === "tdah") {
    return "adhd";
  }

  return normalized === "control" ? "control" : "other";
}

interface ClassBalance {
  entries: { count: number; label: string; share: number }[];
  ratio: number | null;
}

function buildClassBalance(
  distribution: Record<string, number> | undefined,
): ClassBalance | null {
  const entries = Object.entries(distribution ?? {});
  const total = entries.reduce((sum, [, count]) => sum + count, 0);

  if (entries.length === 0 || total === 0) {
    return null;
  }

  const counts = entries.map(([, count]) => count);
  const smallest = Math.min(...counts);

  return {
    entries: entries.map(([label, count]) => ({
      count,
      label,
      share: count / total,
    })),
    ratio:
      counts.length > 1 && smallest > 0 ? Math.max(...counts) / smallest : null,
  };
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
  onClassFilterChange,
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
  const classBalance = useMemo(
    () => buildClassBalance(stats?.class_distribution),
    [stats],
  );

  function className(label: string): string {
    const variant = classVariant(label);

    if (variant === "other") {
      return label;
    }

    return variant === "adhd" ? t("common.adhd") : t("common.control");
  }

  return (
    <section className="training-layout">
      {error && (
        <div className="alert alert-error" role="alert">
          {error}
        </div>
      )}
      {stats && (
        <div className="panel">
          <div className="metric-grid dataset-summary-grid">
            <div>
              <span>{t("common.patients")}</span>
              <strong>{stats.n_patients}</strong>
            </div>
            <div>
              <span>{t("common.rows")}</span>
              <strong>{stats.rows}</strong>
            </div>
            <div>
              <span>{t("dataset.eegChannels")}</span>
              <strong>{stats.eeg_columns.length}</strong>
            </div>
            <div>
              <span>{t("common.columns")}</span>
              <strong>{stats.columns}</strong>
            </div>
          </div>

          {stats.missing_required_columns.length > 0 && (
            <div
              className="alert alert-error dataset-missing-columns"
              role="alert"
            >
              {t("dataset.missingColumns", {
                columns: stats.missing_required_columns.join(", "),
              })}
            </div>
          )}
        </div>
      )}

      <div className="panel dataset-source-row">
        <div>
          <span className="eyebrow">{t("dataset.source")}</span>

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
            <input
              accept=".csv"
              onChange={(event) => {
                void handleFileChange(event);
              }}
              type="file"
            />
            {file?.name || selectedDataset?.filename || t("dataset.selectCsv")}
          </label>

          {loadingDatasets && (
            <p className="muted">{t("dataset.loadingSavedDatasets")}</p>
          )}

          {loadingStats && (
            <div className="alert alert-info" role="status">
              {t("dataset.analyzing")}
            </div>
          )}
          {(file || selectedDataset) && (
            <button
              className="primary-button compact-button"
              disabled={loadingStats}
              onClick={() => {
                void handleAnalyzeDataset();
              }}
              type="button"
            >
              {loadingStats ? t("dataset.analyzing") : t("dataset.retry")}
            </button>
          )}
        </div>

        <div>
          {classBalance ? (
            <div className="class-balance">
              <span className="eyebrow">{t("dataset.classBalance")}</span>

              <div
                aria-label={t("dataset.classBalance")}
                className="distribution-bar"
                role="img"
              >
                {classBalance.entries.map((entry) => (
                  <span
                    className={`distribution-segment ${classVariant(
                        entry.label,
                      )}`}
                    key={entry.label}
                    style={{ width: `${entry.share * 100}%` }}
                    title={`${className(entry.label)}: ${formatPercent(
                        entry.share,
                      )}`}
                  />
                ))}
              </div>

              <div className="distribution-legend">
                {classBalance.entries.map((entry) => (
                  <div className="distribution-legend-row" key={entry.label}>
                    <span
                      aria-hidden="true"
                      className={`legend-dot ${classVariant(entry.label)}`}
                    />
                    <span className="distribution-label">
                      {className(entry.label)}
                    </span>
                    <strong>{entry.count}</strong>
                    <span>{formatPercent(entry.share)}</span>
                  </div>
                ))}
              </div>

              {classBalance.ratio !== null && (
                <p className="muted class-balance-ratio">
                  {t("dataset.classRatio", {
                    ratio: classBalance.ratio.toFixed(2),
                  })}
                </p>
              )}
            </div>
          ) : (
            <>
              <span className="eyebrow">{t("dataset.classBalance")}</span>
              <p className="muted">{t("dataset.balanceEmpty")}</p>
            </>
          )}
        </div>
      </div>

      {stats && (
        <div className="panel">
          <div className="section-heading-row">
            <span className="eyebrow">{t("common.patients")}</span>
            <div className="patient-controls">
              <div className="filter-links">
                {(
                  [
                    ["all", t("dataset.all")],
                    ["adhd", t("common.adhd")],
                    ["control", t("common.control")],
                  ] as const
                ).map(([value, label]) => (
                  <button
                    aria-pressed={classFilter === value}
                    className={classFilter === value ? "active" : ""}
                    key={value}
                    onClick={() => {
                      onClassFilterChange(value);
                    }}
                    type="button"
                  >
                    {label}
                  </button>
                ))}
              </div>
              <label className="patient-limit">
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

          {stats.patients && (
            <p className="muted patient-count">
              {t("dataset.shownOfTotal", {
                shown: filteredPatients.length,
                total: stats.patients.length,
              })}
            </p>
          )}
        </div>
      )}
    </section>
  );
}
