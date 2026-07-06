import PropTypes from "prop-types";
import { useTranslation } from "react-i18next";

import { patientResultShape } from "../../propTypes";
import {
  REPORT_COLUMNS,
  evaluationModeLabel,
  methodLabel,
  reportColumnLabel,
  reportRowLabel,
} from "../../utils/trainingLabels";

function formatReportCell(column, value) {
  if (value === undefined || value === null) {
    return "-";
  }
  // El soporte es un recuento de muestras; el resto son metricas en [0, 1].
  return column === "support" ? String(value) : Number(value).toFixed(3);
}

function ClassificationReportTable({ report, t }) {
  const rows = Object.entries(report).filter(
    ([, value]) => value !== null && typeof value === "object",
  );
  const accuracy = typeof report.accuracy === "number" ? report.accuracy : null;

  return (
    <table className="patient-table compact-table">
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

ClassificationReportTable.propTypes = {
  report: PropTypes.object.isRequired,
  t: PropTypes.func.isRequired,
};

const importanceRowShape = PropTypes.shape({
  feature: PropTypes.string.isRequired,
  importance_mean: PropTypes.number,
});

function Percent({ value }) {
  return `${((value || 0) * 100).toFixed(1)}%`;
}

Percent.propTypes = {
  value: PropTypes.number,
};

function ImportanceValue({ value }) {
  return Number(value || 0).toFixed(4);
}

ImportanceValue.propTypes = {
  value: PropTypes.number,
};

function ImportanceList({ rows, title }) {
  const maxValue = Math.max(
    ...rows.map((row) => Math.max(0, row.importance_mean || 0)),
    0
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

ImportanceList.propTypes = {
  rows: PropTypes.arrayOf(importanceRowShape).isRequired,
  title: PropTypes.string.isRequired,
};

export function TrainingResultsPanel({
  filteredPatientResults,
  onPatientFilterChange,
  patientFilter,
  result,
}) {
  const { t } = useTranslation();

  if (!result) {
    return null;
  }

  const featureImportance = result.feature_importance;

  return (
    <div className="panel training-section">
      <h2>{t("training.results")}</h2>
      {result.configuration?.evaluation_mode && (
        <p className="muted">
          {t("training.evaluation", {
            mode: evaluationModeLabel(t, result.configuration.evaluation_mode),
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
        <div className="alert alert-success">
          {t("training.modelSaved", {
            modelId: `trained_model_${result.trained_model_id}`,
          })}
        </div>
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
          <table className="patient-table compact-table">
            <tbody>
              {result.confusion_matrix.map((row, index) => (
                <tr key={index}>
                  {row.map((value, column) => (
                    <td key={column}>{value}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
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
                rows={featureImportance.top_features || []}
                title={t("training.topFeatures")}
              />
              <ImportanceList
                rows={featureImportance.by_channel || []}
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

TrainingResultsPanel.propTypes = {
  filteredPatientResults: PropTypes.arrayOf(patientResultShape).isRequired,
  onPatientFilterChange: PropTypes.func.isRequired,
  patientFilter: PropTypes.string.isRequired,
  result: PropTypes.shape({
    accuracy: PropTypes.number.isRequired,
    balanced_accuracy: PropTypes.number.isRequired,
    classification_report: PropTypes.object.isRequired,
    confusion_matrix: PropTypes.arrayOf(
      PropTypes.arrayOf(PropTypes.number),
    ).isRequired,
    configuration: PropTypes.shape({
      evaluation_mode: PropTypes.string,
    }),
    f1_score: PropTypes.number.isRequired,
    feature_importance: PropTypes.shape({
      by_channel: PropTypes.arrayOf(importanceRowShape),
      error: PropTypes.string,
      evaluated_epochs: PropTypes.number,
      method: PropTypes.string,
      scoring: PropTypes.string,
      source: PropTypes.string,
      top_features: PropTypes.arrayOf(importanceRowShape),
    }),
    model_saved: PropTypes.bool,
    persisted: PropTypes.bool,
    precision: PropTypes.number.isRequired,
    recall: PropTypes.number.isRequired,
    trained_model_id: PropTypes.number,
    training_time_seconds: PropTypes.number.isRequired,
  }),
};
