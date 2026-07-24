import type {
  ChangeEventHandler,
  CSSProperties,
} from "react";
import type { TFunction } from "i18next";
import { useTranslation } from "react-i18next";

import type {
  FeatureImportanceItem,
  PatientTrainingResult,
  TrainingResult,
  UnknownRecord,
} from "../../../types";
import {
  REPORT_COLUMNS,
  evaluationModeLabel,
  methodLabel,
  reportColumnLabel,
  reportRowLabel,
} from "../trainingLabels";

type ReportColumn = (typeof REPORT_COLUMNS)[number];

interface TrainingResultsPanelProps {
  filteredPatientResults: PatientTrainingResult[];
  onPatientFilterChange: ChangeEventHandler<HTMLInputElement>;
  patientFilter: string;
  result: TrainingResult | null;
}

interface ClassificationReportTableProps {
  report: UnknownRecord;
  t: TFunction;
}

interface ConfusionMatrixProps {
  matrix: number[][];
  t: TFunction;
}

interface NumericValueProps {
  value?: number;
}

interface ImportanceListProps {
  rows: FeatureImportanceItem[];
  title: string;
}

function isReportMetrics(value: unknown): value is UnknownRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function formatReportCell(column: ReportColumn, value: unknown): string {
  if (typeof value !== "number" && typeof value !== "string") {
    return "-";
  }

  // El soporte es un recuento de muestras; el resto son métricas en [0, 1].
  return column === "support" ? String(value) : Number(value).toFixed(3);
}

function ClassificationReportTable({
  report,
  t,
}: ClassificationReportTableProps) {
  const rows = Object.entries(report).filter(
    (entry): entry is [string, UnknownRecord] => isReportMetrics(entry[1]),
  );
  const accuracy = typeof report.accuracy === "number" ? report.accuracy : null;

  return (
    <table className="patient-table report-table">
      <thead>
        <tr>
          <th>{t("common.class")}</th>
          {REPORT_COLUMNS.map((column) => (
            <th key={column}>{reportColumnLabel(t, column)}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map(([name, metrics]) => (
          <tr key={name}>
            <td>{reportRowLabel(t, name)}</td>
            {REPORT_COLUMNS.map((column) => (
              <td key={column}>{formatReportCell(column, metrics[column])}</td>
            ))}
          </tr>
        ))}
        {accuracy !== null && (
          <tr>
            <td>{reportRowLabel(t, "accuracy")}</td>
            <td colSpan={REPORT_COLUMNS.length}>{accuracy.toFixed(3)}</td>
          </tr>
        )}
      </tbody>
    </table>
  );
}

// Sombreado secuencial: cuanto mayor el recuento, más opaca la celda.
function confusionCellStyle(value: number, max: number): CSSProperties {
  const intensity = max > 0 ? value / max : 0;
  const alpha = (0.12 + intensity * 0.68).toFixed(3);
  return { background: `rgba(190, 124, 77, ${alpha})` };
}

function ConfusionMatrix({ matrix, t }: ConfusionMatrixProps) {
  const labels = [t("training.reportRows.Control"), t("training.reportRows.ADHD")];
  const max = Math.max(...matrix.flat(), 1);

  return (
    <div className="confusion-matrix">
      <span className="cm-caption cm-caption-top">{t("training.predicted")}</span>
      <div className="cm-body">
        <span className="cm-caption cm-caption-left">{t("training.actual")}</span>
        <table className="cm-table">
          <thead>
            <tr>
              <td className="cm-corner" aria-hidden="true" />
              {labels.map((label) => (
                <th key={label} scope="col">
                  {label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, rowIndex) => (
              <tr key={labels[rowIndex] ?? rowIndex}>
                <th scope="row">{labels[rowIndex]}</th>
                {row.map((value, colIndex) => (
                  <td
                    className="cm-cell"
                    key={colIndex}
                    style={confusionCellStyle(value, max)}
                  >
                    {value}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Percent({ value }: NumericValueProps) {
  return `${((value ?? 0) * 100).toFixed(1)}%`;
}

function ImportanceValue({ value }: NumericValueProps) {
  return (value ?? 0).toFixed(4);
}

function ImportanceList({ rows, title }: ImportanceListProps) {
  const maxValue = Math.max(
    ...rows.map((row) => Math.max(0, row.importance_mean)),
    0,
  );

  return (
    <div className="importance-card">
      <h3>{title}</h3>
      <div className="importance-list">
        {rows.map((row) => {
          const width =
            maxValue > 0
              ? `${Math.max(4, (Math.max(0, row.importance_mean) / maxValue) * 100)}%`
              : "4%";

          return (
            <div className="importance-row" key={row.feature}>
              <div>
                <strong>{row.feature}</strong>
                <div className="importance-track">
                  <span className="importance-fill" style={{ width }} />
                </div>
              </div>
              <span>
                <ImportanceValue value={row.importance_mean} />
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function TrainingResultsPanel({
  filteredPatientResults,
  onPatientFilterChange,
  patientFilter,
  result,
}: TrainingResultsPanelProps) {
  const { t } = useTranslation();

  if (!result) {
    return null;
  }

  const featureImportance = result.feature_importance;
  const evaluationMode = result.configuration.evaluation_mode;

  return (
    <div className="panel training-section">
      <h2>{t("training.results")}</h2>
      {evaluationMode && (
        <p className="muted">
          {t("training.evaluation", {
            mode: evaluationModeLabel(t, evaluationMode),
          })}
        </p>
      )}
      {result.persisted === false && (
        <div className="alert alert-error">{t("training.persistError")}</div>
      )}
      {result.persisted !== false && result.model_saved === false && (
        <div className="alert alert-warning">{t("training.modelSaveWarning")}</div>
      )}
      {result.model_saved && result.trained_model_id && (
        <div className="alert alert-success">{t("training.modelSaved")}</div>
      )}

      <div className="metric-grid metrics-wide training-result-grid">
        <div>
          <span>{t("metrics.accuracy")}</span>
          <strong>{result.accuracy.toFixed(3)}</strong>
        </div>
        <div>
          <span>{t("metrics.precision")}</span>
          <strong>{result.precision.toFixed(3)}</strong>
        </div>
        <div>
          <span>{t("metrics.recall")}</span>
          <strong>{result.recall.toFixed(3)}</strong>
        </div>
        <div>
          <span>{t("metrics.f1Score")}</span>
          <strong>{result.f1_score.toFixed(3)}</strong>
        </div>
        <div>
          <span>{t("metrics.balancedAccuracy")}</span>
          <strong>{result.balanced_accuracy.toFixed(3)}</strong>
        </div>
      </div>

      <div className="training-result-columns">
        <div>
          <h3>{t("training.confusionMatrix")}</h3>
          <ConfusionMatrix matrix={result.confusion_matrix} t={t} />
        </div>
        <div>
          <h3>{t("training.classificationReport")}</h3>
          <ClassificationReportTable report={result.classification_report} t={t} />
        </div>
      </div>

      {featureImportance && (
        <div className="feature-importance-block">
          <div className="section-heading-row">
            <div>
              <h3>{t("training.featureImportance")}</h3>
              <p className="muted">
                {t("training.importanceSource", {
                  method: methodLabel(t, featureImportance.method),
                  source: featureImportance.source,
                })}
              </p>
            </div>
            <div className="importance-meta">
              <span>{methodLabel(t, featureImportance.scoring)}</span>
              <span>
                {t("training.evaluatedWindows", {
                  count: featureImportance.evaluated_epochs,
                })}
              </span>
            </div>
          </div>

          {featureImportance.error ? (
            <p className="muted">
              {t("training.importanceError", { error: featureImportance.error })}
            </p>
          ) : (
            <div className="feature-importance-grid">
              <ImportanceList
                rows={featureImportance.top_features}
                title={t("training.topFeatures")}
              />
              <ImportanceList
                rows={featureImportance.by_channel}
                title={t("training.byEegChannel")}
              />
            </div>
          )}
        </div>
      )}

      <div className="section-heading-row result-filter-row">
        <div>
          <h3>{t("training.patientResults")}</h3>
          <p className="muted">
            {t("training.trainingTime", {
              seconds: result.training_time_seconds,
            })}
          </p>
        </div>
        <input
          className="patient-filter-input"
          onChange={onPatientFilterChange}
          placeholder={t("training.patientFilter")}
          type="search"
          value={patientFilter}
        />
      </div>

      <div className="patient-table-wrap">
        <table className="patient-table">
          <thead>
            <tr>
              <th>{t("common.patient")}</th>
              <th>{t("training.trueClass")}</th>
              <th>{t("training.predictedClass")}</th>
              <th>{t("training.windows")}</th>
              <th>{t("training.controlPercent")}</th>
              <th>{t("training.adhdPercent")}</th>
              <th>{t("training.correct")}</th>
            </tr>
          </thead>
          <tbody>
            {filteredPatientResults.map((patient) => (
              <tr key={patient.patient_id}>
                <td>{patient.patient_id}</td>
                <td>{patient.true_label}</td>
                <td>{patient.predicted_label}</td>
                <td>{patient.n_epochs}</td>
                <td>
                  <Percent value={patient.control_epoch_percentage} />
                </td>
                <td>
                  <Percent value={patient.adhd_epoch_percentage} />
                </td>
                <td>{patient.correct ? t("common.yes") : t("common.no")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
