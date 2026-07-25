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
  handleClassFilterChange: (event: ChangeEvent<HTMLSelectElement>) => void;
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

// El backend normaliza a "ADHD" / "Control", pero tambien puede devolver
// "Sin clase" o la etiqueta original del CSV: esas caen en "other".
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
    // Con una sola clase, o con alguna vacia, el ratio no dice nada.
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
            {/* Con un origen elegido el analisis ya se ha lanzado solo, asi que
                el boton solo puede ser un reintento. */}
            {loadingStats
              ? t("dataset.analyzing")
              : file || selectedDataset
                ? t("dataset.retry")
                : t("dataset.analyze")}
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
          <div className="alert alert-info">{t("dataset.analyzing")}</div>
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

            {classBalance && (
              <div className="class-balance">
                <span className="eyebrow">{t("dataset.classBalance")}</span>

                <div
                  aria-label={t("dataset.classBalance")}
                  className="distribution-bar"
                  role="img"
                >
                  {classBalance.entries.map((entry) => (
                    <span
                      className={`distribution-segment ${classVariant(entry.label)}`}
                      key={entry.label}
                      style={{ width: `${entry.share * 100}%` }}
                      title={`${className(entry.label)}: ${formatPercent(entry.share)}`}
                    />
                  ))}
                </div>

                <div className="distribution-legend">
                  {classBalance.entries.map((entry) => (
                    <div className="distribution-legend-row" key={entry.label}>
                      <span
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
            )}

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
                  <option value="adhd">{t("common.adhd")}</option>
                  <option value="control">{t("common.control")}</option>
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
