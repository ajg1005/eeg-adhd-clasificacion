import PropTypes from "prop-types";

import { patientResultShape } from "../../propTypes";

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
        <p className="muted">Evaluación: {result.configuration.evaluation_mode}</p>
      )}
      {result.persisted === false && (
        <div className="alert alert-error">
          El entrenamiento ha finalizado, pero no se ha podido guardar en la base de datos.
        </div>
      )}

      <div className="metric-grid metrics-wide training-result-grid">
        <div>
          <span>Accuracy</span>
          <strong>{result.accuracy.toFixed(3)}</strong>
        </div>
        <div>
          <span>Precision</span>
          <strong>{result.precision.toFixed(3)}</strong>
        </div>
        <div>
          <span>Recall</span>
          <strong>{result.recall.toFixed(3)}</strong>
        </div>
        <div>
          <span>F1-score</span>
          <strong>{result.f1_score.toFixed(3)}</strong>
        </div>
        <div>
          <span>Balanced</span>
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
          <pre className="training-log">
            {JSON.stringify(result.classification_report, null, 2)}
          </pre>
        </div>
      </div>

      {featureImportance && (
        <div className="feature-importance-block">
          <div className="section-heading-row">
            <div>
              <h3>Importancia de características</h3>
              <p className="muted">
                {featureImportance.method} sobre {featureImportance.source}
              </p>
            </div>
            <div className="importance-meta">
              <span>{featureImportance.scoring}</span>
              <span>{featureImportance.evaluated_epochs} epochs</span>
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
              <th>Epochs</th>
              <th>control_%</th>
              <th>adhd_%</th>
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
