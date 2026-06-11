import PropTypes from "prop-types";

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
    return "—";
  }
  // El soporte es un recuento de muestras; el resto son métricas en [0, 1].
  return column === "support" ? String(value) : Number(value).toFixed(3);
}

function ClassificationReportTable({ report }) {
  const rows = Object.entries(report).filter(
    ([, value]) => value !== null && typeof value === "object",
  );
  const accuracy = typeof report.accuracy === "number" ? report.accuracy : null;

  return (
    <table className="patient-table compact-table">
      <thead>
        <tr>
          <th>Clase</th>
          {REPORT_COLUMNS.map((column) => (
            <th key={column}>{reportColumnLabel(column)}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map(([name, metrics]) => (
          <tr key={name}>
            <td>{reportRowLabel(name)}</td>
            {REPORT_COLUMNS.map((column) => (
              <td key={column}>{formatReportCell(column, metrics[column])}</td>
            ))}
          </tr>
        ))}
        {accuracy !== null && (
          <tr>
            <td>{reportRowLabel("accuracy")}</td>
            <td colSpan={REPORT_COLUMNS.length}>{accuracy.toFixed(3)}</td>
          </tr>
        )}
      </tbody>
    </table>
  );
}

ClassificationReportTable.propTypes = {
  report: PropTypes.object.isRequired,
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
  if (!result) {
    return null;
  }

  const featureImportance = result.feature_importance;

  return (
    <div className="panel training-section">
      <h2>Resultados del entrenamiento</h2>
      {result.configuration?.evaluation_mode && (
        <p className="muted">
          Evaluación: {evaluationModeLabel(result.configuration.evaluation_mode)}
        </p>
      )}
      {result.persisted === false && (
        <div className="alert alert-error">
          El entrenamiento ha finalizado, pero no se ha podido guardar en la base de datos.
        </div>
      )}

      <div className="metric-grid metrics-wide training-result-grid">
        <div>
          <span>Exactitud</span>
          <strong>{result.accuracy.toFixed(3)}</strong>
        </div>
        <div>
          <span>Precisión</span>
          <strong>{result.precision.toFixed(3)}</strong>
        </div>
        <div>
          <span>Sensibilidad</span>
          <strong>{result.recall.toFixed(3)}</strong>
        </div>
        <div>
          <span>F1-score</span>
          <strong>{result.f1_score.toFixed(3)}</strong>
        </div>
        <div>
          <span>Exactitud balanceada</span>
          <strong>{result.balanced_accuracy.toFixed(3)}</strong>
        </div>
      </div>

      <div className="training-result-columns">
        <div>
          <h3>Matriz de confusión</h3>
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
          <h3>Informe de clasificación</h3>
          <ClassificationReportTable report={result.classification_report} />
        </div>
      </div>

      {featureImportance && (
        <div className="feature-importance-block">
          <div className="section-heading-row">
            <div>
              <h3>Importancia de características</h3>
              <p className="muted">
                {methodLabel(featureImportance.method)} sobre {featureImportance.source}
              </p>
            </div>
            <div className="importance-meta">
              <span>{methodLabel(featureImportance.scoring)}</span>
              <span>{featureImportance.evaluated_epochs} ventanas evaluadas</span>
            </div>
          </div>

          {featureImportance.error ? (
            <p className="muted">No se pudo calcular: {featureImportance.error}</p>
          ) : (
            <div className="feature-importance-grid">
              <ImportanceList
                rows={featureImportance.top_features || []}
                title="Características principales"
              />
              <ImportanceList
                rows={featureImportance.by_channel || []}
                title="Por canal EEG"
              />
            </div>
          )}
        </div>
      )}

      <div className="section-heading-row result-filter-row">
        <div>
          <h3>Resultados por paciente</h3>
          <p className="muted">
            Tiempo de entrenamiento: {result.training_time_seconds}s
          </p>
        </div>
        <input
          className="patient-filter-input"
          onChange={onPatientFilterChange}
          placeholder="Filtrar paciente"
          type="search"
          value={patientFilter}
        />
      </div>

      <div className="patient-table-wrap">
        <table className="patient-table">
          <thead>
            <tr>
              <th>Paciente</th>
              <th>Clase real</th>
              <th>Clase predicha</th>
              <th>Ventanas</th>
              <th>Control (%)</th>
              <th>TDAH (%)</th>
              <th>Correcto</th>
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
                <td>{patient.correct ? "Sí" : "No"}</td>
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
    persisted: PropTypes.bool,
    precision: PropTypes.number.isRequired,
    recall: PropTypes.number.isRequired,
    training_time_seconds: PropTypes.number.isRequired,
  }),
};
