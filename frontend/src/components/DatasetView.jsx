import { useMemo } from "react";


function filterPatients(patients, classFilter, maxPatients) {
  if (!patients) {
    return [];
  }

  const filtered = classFilter === "all"
    ? patients
    : patients.filter((patient) => patient.class_label.toLowerCase() === classFilter);

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
  loadingStats,
  maxPatients,
  stats,
}) {
  const filteredPatients = useMemo(
    () => filterPatients(stats?.patients, classFilter, maxPatients),
    [stats, classFilter, maxPatients],
  );

  return (
    <section className="training-layout">
      {error && <div className="alert alert-error">{error}</div>}

      <div className="panel">
        <div className="section-heading-row">
          <div>
            <h2>Dataset de entrenamiento</h2>
            <p className="muted">
              Carga un CSV con varios pacientes y revisa las estadísticas antes
              de pasar a la pestaña de entrenamiento.
            </p>
          </div>
          <button
            className="primary-button compact-button"
            disabled={!file || loadingStats}
            onClick={handleAnalyzeDataset}
            type="button"
          >
            {loadingStats ? "Analizando..." : "Analizar dataset"}
          </button>
        </div>

        <label className="file-drop">
          <input accept=".csv" onChange={handleFileChange} type="file" />
          {file ? file.name : "Seleccionar CSV EEG"}
        </label>

        {stats && (
          <>
            <div className="metric-grid dataset-summary-grid training-metrics-row">
              <div>
                <span>Filas</span>
                <strong>{stats.rows}</strong>
              </div>
              <div>
                <span>Columnas</span>
                <strong>{stats.columns}</strong>
              </div>
              <div>
                <span>Pacientes</span>
                <strong>{stats.n_patients}</strong>
              </div>
              <div>
                <span>Canales EEG</span>
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
                Faltan columnas: {stats.missing_required_columns.join(", ")}
              </div>
            )}
          </>
        )}
      </div>

      {stats && (
        <div className="panel">
          <div className="section-heading-row">
            <h3>Pacientes</h3>
            <div className="controls-row compact-controls">
              <label>
                Clase
                <select value={classFilter} onChange={handleClassFilterChange}>
                  <option value="all">Todos</option>
                  <option value="adhd">TDAH</option>
                  <option value="control">Control</option>
                </select>
              </label>
              <label>
                Pacientes
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
                    <th>Paciente</th>
                    <th>Clase</th>
                    <th>Filas</th>
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
            <p className="muted">No hay pacientes para el filtro seleccionado.</p>
          )}
        </div>
      )}
    </section>
  );
}
