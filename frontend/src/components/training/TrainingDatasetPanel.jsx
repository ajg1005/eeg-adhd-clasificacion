export function TrainingDatasetPanel({
  file,
  loadingStats,
  onAnalyzeDataset,
  onFileChange,
  stats,
}) {
  return (
    <div className="panel training-section">
      <div className="section-heading-row">
        <div>
          <h2>Entrenamiento interactivo</h2>
          <p className="muted">
            Carga un CSV, revisa el dataset y evalua el modelo usando todos los
            pacientes a lo largo de validacion cruzada.
          </p>
        </div>
        <button
          className="primary-button compact-button"
          disabled={!file || loadingStats}
          onClick={onAnalyzeDataset}
          type="button"
        >
          {loadingStats ? "Analizando..." : "Analizar dataset"}
        </button>
      </div>

      <label className="file-drop">
        <input accept=".csv" onChange={onFileChange} type="file" />
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

          <div className="patient-table-wrap">
            <table className="patient-table">
              <thead>
                <tr>
                  {Object.keys(stats.preview[0] || {})
                    .slice(0, 8)
                    .map((column) => (
                      <th key={column}>{column}</th>
                    ))}
                </tr>
              </thead>
              <tbody>
                {stats.preview.map((row, index) => (
                  <tr key={index}>
                    {Object.keys(stats.preview[0] || {})
                      .slice(0, 8)
                      .map((column) => (
                        <td key={column}>{String(row[column] ?? "")}</td>
                      ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
